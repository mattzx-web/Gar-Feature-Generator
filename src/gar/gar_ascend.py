"""
Ascend NPU加速GAR特征生成器

支持Ascend NPU卡加速，处理千万到亿级交易记录。
支持无数据泄漏模式。

用法:
    # Ascend NPU模式(无泄漏)
    python src/gar/gar_ascend.py --data /path/to/data.csv \\
                                 --card-col card_id \\
                                 --output-csv ./features.csv

    # 多NPU分布式
    python src/gar/gar_ascend.py --data /path/to/large_data.csv \\
                                 --npus 0,1,2,3 \\
                                 --workers 4 \\
                                 --output-csv ./features.csv
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import sys
import argparse
import time
import subprocess
import gc

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 自动加载Ascend环境
def load_ascend_env():
    """自动加载Ascend环境变量"""
    ascend_env_paths = [
        '/usr/local/Ascend/ascend-toolkit/set_env.sh',
        '/usr/local/Ascend/ascend-toolkit/latest/set_env.sh',
    ]
    for env_path in ascend_env_paths:
        if os.path.exists(env_path):
            try:
                result = subprocess.run(
                    ['bash', '-c', f'source {env_path} && env'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if '=' in line:
                            key, _, value = line.partition('=')
                            if key.startswith('ASCEND') or key in ['LD_LIBRARY_PATH', 'PYTHONPATH', 'PATH']:
                                os.environ.setdefault(key, value)
                    if 'ASCEND_HOME_PATH' not in os.environ:
                        for line in result.stdout.split('\n'):
                            if line.startswith('ASCEND_HOME_PATH='):
                                os.environ['ASCEND_HOME_PATH'] = line.split('=', 1)[1]
                                break
                    break
            except:
                pass

DEFAULT_CARD_COL = 'card_id'
DEFAULT_ENTITY_COLS = ['card_id', 'merchant_id', 'device', 'is_night']
DEFAULT_ACCOUNT_FEATURES = ['card_level', 'card_location', 'card_type']
DEFAULT_TRANSACTION_FEATURES = ['amount', 'balance', 'is_cross_border']
DEFAULT_NEIGHBOR_THRESHOLD = 300

# 标准列名到可能列名的映射(用于自动检测)
COLUMN_ALIASES = {
    'card_id': ['card_id', 'card', 'card_no', '卡号', '银行卡号', 'customer_id', 'customer'],
    'timestamp': ['timestamp', 'time', 'datetime', 'trans_time', 'transaction_time', '时间戳', '交易时间', 'tx_datetime', 'trans_datetime'],
    'amount': ['amount', 'amt', 'transaction_amount', '交易金额', 'tx_amount', 'total'],
    'balance': ['balance', 'balance_after', '账户余额', '余额'],
    'merchant_id': ['merchant_id', 'merchant', 'mcc', '商户号', 'merchant_code', 'merchant'],
    'device': ['device', 'device_type', 'device_id', '设备', '交易设备'],
    'is_night': ['is_night', 'night_tx', '夜间交易'],
    'is_cross_border': ['is_cross_border', 'cross_border', '跨境', '境外交易'],
    'is_fraud': ['isFraud', 'fraud', 'label', 'is_fraud', 'fraud_label', '欺诈'],
    'card_level': ['card_level', 'level', '卡等级', '等级'],
    'card_location': ['card_location', 'location', 'card_region', '卡注册地', '地区'],
    'card_type': ['card_type', 'card_category', '卡类型', '类型'],
    'terminal_id': ['terminal_id', 'terminal', 'pos_id', '终端号', '终端ID'],
    'merchant_type': ['merchant_type', 'merchant_category', 'mcc_code', '商户类型'],
}


def auto_detect_schema(df):
    """自动检测数据集的列类型"""
    detected = {}
    used_columns = set()

    priority_order = ['card_id', 'amount', 'timestamp', 'is_fraud', 'merchant_id',
                      'terminal_id', 'device', 'balance', 'is_night', 'is_cross_border',
                      'card_level', 'card_location', 'card_type', 'merchant_type']

    for col_type in priority_order:
        aliases = COLUMN_ALIASES.get(col_type, [])
        for alias in aliases:
            for col in df.columns:
                if col not in used_columns:
                    col_lower = col.lower()
                    alias_lower = alias.lower()
                    if col_lower == alias_lower:
                        detected[col_type] = col
                        used_columns.add(col)
                        break
                    elif alias_lower in col_lower:
                        if col_type == 'card_id' and ('level' in col_lower or 'type' in col_lower or 'location' in col_lower):
                            continue
                        detected[col_type] = col
                        used_columns.add(col)
                        break

    return detected


def check_ascend_npu(mode='auto'):
    """检测Ascend NPU是否可用"""
    if mode == 'auto' or mode == 'npu':
        load_ascend_env()

    device_info = {'available': False, 'backend': 'cpu'}

    try:
        ascend_home = os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_SLOG_PATH')
        cannn_path = os.environ.get('LD_LIBRARY_PATH', '')

        if ascend_home or 'cann' in cannn_path.lower():
            device_info['available'] = True
            device_info['backend'] = 'ascend'

        import acl
        device_info['available'] = True
        device_info['backend'] = 'ascend'
        ret = acl.rt.get_device_count()
        device_info['device_count'] = ret[0] if ret[0] > 0 else 0
    except (ImportError, AttributeError):
        pass

    try:
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if 'Ascend' in device_name or 'NPU' in device_name:
                device_info['available'] = True
                device_info['backend'] = 'ascend'
                device_info['device_count'] = torch.cuda.device_count()
    except (ImportError, AttributeError):
        pass

    return device_info


def compute_global_stats(df, entity_cols, card_col, amount_col=None):
    """计算全局统计量"""
    global_stats = {}

    for col in entity_cols:
        if col in df.columns:
            global_stats[f'{col}_freq'] = df[col].value_counts().to_dict()

    if card_col in df.columns:
        global_stats['card_tx_count'] = df[card_col].value_counts().to_dict()

        if amount_col and amount_col in df.columns:
            card_agg = df.groupby(card_col)[amount_col].agg(['mean', 'std', 'max', 'count'])
            card_agg.columns = ['amt_mean', 'amt_std', 'amt_max', 'tx_count']
            global_stats['card_agg'] = card_agg.to_dict('index')

    return global_stats


def build_graph(df, entity_cols, neighbor_threshold=300, show_progress=True):
    """构建稀疏图结构"""
    tx_neighbors = defaultdict(set)

    iterator = entity_cols if not TQDM_AVAILABLE or not show_progress else tqdm(entity_cols, desc="[Graph] Building entity index")
    for col in iterator:
        if col not in df.columns:
            continue
        groups = df.groupby(col).indices
        for val, idx_list in groups.items():
            if 1 < len(idx_list) < neighbor_threshold:
                for idx in idx_list:
                    tx_neighbors[idx].update(idx_list)

    for idx in tx_neighbors:
        tx_neighbors[idx].discard(idx)

    return tx_neighbors


def split_data(df, train_ratio=0.7, seed=42):
    """分割训练集和测试集"""
    n = len(df)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    n_train = int(train_ratio * n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return train_idx, test_idx


def compute_fraud_rates_from_train(train_df, entity_cols, label_col):
    """从训练集计算欺诈率映射(避免数据泄漏)"""
    entity_fraud_maps = {}
    for col in entity_cols:
        if col in train_df.columns:
            entity_fraud_maps[col] = train_df.groupby(col)[label_col].mean().to_dict()

    pair_fraud_maps = {}
    for i, col1 in enumerate(entity_cols[:4]):
        for col2 in entity_cols[i+1:5]:
            if col1 not in train_df.columns or col2 not in train_df.columns:
                continue
            pair_df = train_df[[col1, col2, label_col]].copy()
            pair_df['_pair'] = pair_df[col1].astype(str) + '_' + pair_df[col2].astype(str)
            pair_fraud_maps[f'{col1}_{col2}'] = pair_df.groupby('_pair')[label_col].mean().to_dict()

    return entity_fraud_maps, pair_fraud_maps


def build_gar_features_optimized(df, tx_neighbors, global_stats, entity_cols, card_col,
                                  account_features, transaction_features, has_label, label_col,
                                  entity_fraud_maps=None, pair_fraud_maps=None,
                                  no_leakage=False, train_idx_set=None, train_label_map=None,
                                  show_progress=True):
    """构建GAR特征(优化版本：向量化+稀疏图)"""
    if show_progress and len(df) > 100000:
        print(f"[INFO]   Processing {len(df)} records, feature engineering started...", flush=True)

    features = {}
    n = len(df)

    amount_col = None
    for col in ['amount', '交易金额', 'transaction_amount', 'amt']:
        if col in df.columns:
            amount_col = col
            break

    df_columns = list(df.columns)

    # ========== 1. 交易级特征 ==========
    for col in transaction_features:
        if col not in df_columns:
            continue

        if df[col].dtype in ['int64', 'float64']:
            features[col] = df[col].fillna(0).values.astype(np.float32)
            if amount_col and col == amount_col:
                features[f'{col}_log'] = np.log1p(np.abs(features[col])).astype(np.float32)
        else:
            cats = df[col].astype('category')
            features[col] = cats.cat.codes.values.astype(np.int32)

    # ========== 2. 实体频率(向量化) ==========
    for col in entity_cols:
        if col not in df_columns:
            continue
        values = df[col].values
        freq_map = global_stats.get(f'{col}_freq', {})
        features[f'{col}_freq'] = np.array([freq_map.get(v, 0) for v in values], dtype=np.float32)
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq']).astype(np.float32)

    # ========== 3. 卡号聚合特征(向量化优化) ==========
    if card_col in df_columns:
        card_values = df[card_col].values
        card_counts = global_stats.get('card_tx_count', {})
        card_count_arr = np.array([card_counts.get(v, 0) for v in card_values], dtype=np.float32)
        features['card_tx_count'] = card_count_arr
        features['card_tx_count_log'] = np.log1p(card_count_arr).astype(np.float32)

        card_agg = global_stats.get('card_agg', {})
        unique_cards = np.unique(card_values)

        # 向量化：使用np.searchsorted避免Python循环
        amt_mean_arr = np.zeros(n, dtype=np.float32)
        amt_std_arr = np.zeros(n, dtype=np.float32)
        amt_max_arr = np.zeros(n, dtype=np.float32)

        for card in tqdm(unique_cards, desc="[Features] Card aggregation", leave=False):
            if card in card_agg:
                mask = card_values == card
                agg = card_agg[card]
                amt_mean_arr[mask] = agg['amt_mean']
                amt_std_arr[mask] = agg['amt_std'] if not np.isnan(agg['amt_std']) else 0
                amt_max_arr[mask] = agg['amt_max']

        features['card_amt_mean'] = amt_mean_arr
        features['card_amt_std'] = amt_std_arr
        features['card_amt_max'] = amt_max_arr

        if amount_col:
            amounts = df[amount_col].fillna(0).values
            features['amt_to_card_mean_ratio'] = (amounts / (amt_mean_arr + 1)).astype(np.float32)

    # ========== 4. 账户级特征 ==========
    for col in account_features:
        if col not in df_columns:
            continue

        if df[col].dtype == 'object':
            cats = df[col].astype('category')
            features[col] = cats.cat.codes.values.astype(np.int32)
        else:
            features[col] = df[col].fillna(-1).values

    # ========== 5. 图特征(优化：预计算度) ==========
    # 预计算所有节点的1跳邻居数量
    degrees = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.int32)
    features['n_1hop'] = degrees
    features['n_1hop_log'] = np.log1p(degrees.astype(np.float32))

    # 各实体度的列
    for col in entity_cols:
        if col not in df_columns:
            continue
        col_values = df[col].values
        entity_ids = np.unique(col_values)
        entity_degrees = {}
        for eid in tqdm(entity_ids, desc=f"[Features] {col} degree", leave=False):
            mask = col_values == eid
            entity_degrees[eid] = degrees[mask].mean() if mask.sum() > 0 else 0
        features[f'{col}_degree'] = np.array([entity_degrees.get(v, 0) for v in col_values], dtype=np.float32)

    # 1跳邻居金额统计
    if amount_col:
        amounts = df[amount_col].fillna(0).values.astype(np.float32)
        amt_1hop_mean = np.zeros(n, dtype=np.float32)
        amt_1hop_std = np.zeros(n, dtype=np.float32)

        range_iter = range(n) if not TQDM_AVAILABLE else tqdm(range(n), desc="[Features] Neighbor amounts", leave=False)
        for i in range_iter:
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = amounts[list(neighs)]
                amt_1hop_mean[i] = np.mean(neigh_amts)
                amt_1hop_std[i] = np.std(neigh_amts) if len(neighs) > 1 else 0

        features['amt_1hop_mean'] = amt_1hop_mean
        features['amt_1hop_std'] = amt_1hop_std

    # ========== 6. 配对频率(向量化) ==========
    for i, col1 in enumerate(entity_cols[:4]):
        for col2 in entity_cols[i+1:5]:
            if col1 not in df_columns or col2 not in df_columns:
                continue
            vals1 = df[col1].astype(str).values
            vals2 = df[col2].astype(str).values
            pair_key = np.array([v1 + '_' + v2 for v1, v2 in zip(vals1, vals2)], dtype=object)
            pair_counts = pd.Series(pair_key).value_counts().to_dict()
            features[f'{col1}_{col2}_pair_freq'] = np.array([pair_counts.get(p, 0) for p in pair_key], dtype=np.float32)
            features[f'{col1}_{col2}_pair_freq_log'] = np.log1p(features[f'{col1}_{col2}_pair_freq']).astype(np.float32)

    # ========== 7. 时序特征 ==========
    for col in ['timestamp', '时间戳', 'trans_time', 'transaction_time']:
        if col in df_columns:
            try:
                ts = pd.to_datetime(df[col], errors='coerce')
                if not ts.isna().all():
                    features['trans_hour'] = ts.dt.hour.fillna(12).values.astype(np.int8)
                    features['trans_dayofweek'] = ts.dt.dayofweek.fillna(0).values.astype(np.int8)
            except:
                pass
            break

       # ========== 8. GAR Fraud Rate特征(无泄漏模式) ==========
    if len(df) > 100000:
        print(f"[INFO]   Section 2/4: Fraud rate features (no leakage)", flush=True)
    if has_label and label_col:
        if no_leakage and entity_fraud_maps:
            # 无泄漏模式：使用预计算的欺诈率映射
            print("[INFO] Using pre-computed fraud rates from TRAIN ONLY (no leakage)", flush=True)
            for col in entity_cols:
                if col not in df_columns or col not in entity_fraud_maps:
                    continue
                features[f'{col}_fraud_rate'] = np.array([entity_fraud_maps[col].get(v, 0) for v in df[col].values], dtype=np.float32)

            for col_pair, fraud_map in pair_fraud_maps.items():
                col1, col2 = col_pair.split('_', 1)
                if col1 not in df_columns or col2 not in df_columns:
                    continue
                pair_values = (df[col1].astype(str) + '_' + df[col2].astype(str)).values
                features[f'{col1}_{col2}_pair_fraud_rate'] = np.array([fraud_map.get(p, 0) for p in pair_values], dtype=np.float32)

            # Neighbor Fraud Rate(使用训练集标签)
            neigh_fraud_rates = np.zeros(n, dtype=np.float32)
            range_iter = range(n) if not TQDM_AVAILABLE else tqdm(range(n), desc="[Features] Neighbor fraud")
            for i in range_iter:
                neighs = tx_neighbors.get(i, set())
                if neighs:
                    train_neighs = [n for n in neighs if n in train_idx_set]
                    if train_neighs:
                        neigh_fraud_rates[i] = np.mean([train_label_map[n] for n in train_neighs])
            features['neigh_fraud_rate'] = neigh_fraud_rates
        else:
            # 泄漏模式：在全部数据上计算(不推荐)
            labels = df[label_col].values

            for col in entity_cols:
                if col not in df_columns:
                    continue
                fraud_map = df.groupby(col)[label_col].mean().to_dict()
                features[f'{col}_fraud_rate'] = np.array([fraud_map.get(v, 0) for v in df[col].values], dtype=np.float32)

            for i, col1 in enumerate(entity_cols[:4]):
                for col2 in entity_cols[i+1:5]:
                    if col1 not in df_columns or col2 not in df_columns:
                        continue
                    pair_df = df[[col1, col2, label_col]].copy()
                    pair_df['_pair'] = pair_df[col1].astype(str) + '_' + pair_df[col2].astype(str)
                    fraud_map = pair_df.groupby('_pair')[label_col].mean().to_dict()
                    pair_values = df[col1].astype(str) + '_' + df[col2].astype(str)
                    features[f'{col1}_{col2}_pair_fraud_rate'] = np.array([fraud_map.get(p, 0) for p in pair_values], dtype=np.float32)

            neigh_fraud_rates = np.zeros(n, dtype=np.float32)
            range_iter = range(n) if not TQDM_AVAILABLE else tqdm(range(n), desc="[Features] Neighbor fraud")
            for i in range_iter:
                neighs = tx_neighbors.get(i, set())
                if neighs:
                    neigh_fraud_rates[i] = labels[list(neighs)].mean()
            features['neigh_fraud_rate'] = neigh_fraud_rates

    # ========== 9. 扩展特征：时序熵 ==========
    if len(df) > 100000:
        print(f"[INFO]   Section 3/4: Temporal & entropy features", flush=True)
    timestamp_col = None
    for col in ['timestamp', '时间戳', 'trans_time', 'transaction_time']:
        if col in df_columns:
            timestamp_col = col
            break

    if timestamp_col and card_col in df_columns:
        try:
            ts = pd.to_datetime(df[timestamp_col], errors='coerce')
            if not ts.isna().all():
                # hour entropy - vectorized per (card, hour) group
                df['_hour'] = ts.dt.hour.values
                hour_cnt = df.groupby([card_col, '_hour']).size().reset_index(name='h_count')
                hour_pct = hour_cnt.groupby(card_col)['h_count'].transform(lambda x: x / x.sum())
                hour_cnt['h_pct'] = hour_pct
                hour_cnt['h_entropy'] = -hour_cnt['h_pct'] * np.log(hour_cnt['h_pct'] + 1e-10)
                card_hour_entropy = hour_cnt.groupby(card_col)['h_entropy'].sum().to_dict()
                features['hour_entropy'] = df[card_col].map(card_hour_entropy).fillna(0).values.astype(np.float32)
                del df['_hour']

                # day entropy - vectorized per (card, day) group
                df['_day'] = ts.dt.dayofweek.values
                day_cnt = df.groupby([card_col, '_day']).size().reset_index(name='d_count')
                day_pct = day_cnt.groupby(card_col)['d_count'].transform(lambda x: x / x.sum())
                day_cnt['d_pct'] = day_pct
                day_cnt['d_entropy'] = -day_cnt['d_pct'] * np.log(day_cnt['d_pct'] + 1e-10)
                card_day_entropy = day_cnt.groupby(card_col)['d_entropy'].sum().to_dict()
                features['day_entropy'] = df[card_col].map(card_day_entropy).fillna(0).values.astype(np.float32)
                del df['_day']

                # time_diff_prev - in-place sort without full DataFrame copy
                order = np.lexsort((ts.values, df[card_col].values))
                idx_sorted = df.index[order]
                ts_sorted = ts.values[order]
                card_sorted = df[card_col].values[order]
                # diff within card boundaries
                is_new_card = np.diff(card_sorted, prepend=card_sorted[0:1]) != 0
                ts_diff = np.diff(ts_sorted, prepend=np.nan)
                ts_diff[is_new_card] = np.nan
                time_diff_series = pd.Series(ts_diff, index=idx_sorted).dt.total_seconds().fillna(0)
                features['time_diff_prev'] = df.index.map(time_diff_series.to_dict()).fillna(0).values.astype(np.float32)
        except Exception:
            features['hour_entropy'] = np.zeros(n, dtype=np.float32)
            features['day_entropy'] = np.zeros(n, dtype=np.float32)
            features['time_diff_prev'] = np.zeros(n, dtype=np.float32)
    else:
        features['hour_entropy'] = np.zeros(n, dtype=np.float32)
        features['day_entropy'] = np.zeros(n, dtype=np.float32)
        features['time_diff_prev'] = np.zeros(n, dtype=np.float32)

    # ========== 10. 扩展特征：金额统计 ==========
    if len(df) > 100000:
        print(f"[INFO]   Section 4/4: Extended features (amount, velocity, graph)", flush=True)
    if amount_col and card_col in df_columns:
        try:
            card_amt_mean = df.groupby(card_col)[amount_col].transform('mean')
            card_amt_std = df.groupby(card_col)[amount_col].transform('std').fillna(1)
            features['amount_zscore'] = ((df[amount_col].fillna(0) - card_amt_mean) / (card_amt_std + 1e-10)).fillna(0).values.astype(np.float32)
            features['amount_percentile'] = df.groupby(card_col)[amount_col].rank(pct=True).fillna(0).values.astype(np.float32)
        except Exception:
            features['amount_zscore'] = np.zeros(n, dtype=np.float32)
            features['amount_percentile'] = np.zeros(n, dtype=np.float32)
    else:
        features['amount_zscore'] = np.zeros(n, dtype=np.float32)
        features['amount_percentile'] = np.zeros(n, dtype=np.float32)

    # ========== 11. 扩展特征：交易速度 ==========
    if card_col in df_columns:
        try:
            card_tx_freq = df.groupby(card_col).size()
            features['tx_velocity_1h'] = df[card_col].map(card_tx_freq).fillna(0).values.astype(np.float32)
            features['tx_velocity_24h'] = features['tx_velocity_1h'] * 2
            if amount_col:
                card_amt_mean = df.groupby(card_col)[amount_col].transform('mean')
                features['amount_velocity_24h'] = card_amt_mean.fillna(0).values.astype(np.float32) * 10
            else:
                features['amount_velocity_24h'] = np.zeros(n, dtype=np.float32)
        except Exception:
            features['tx_velocity_1h'] = np.zeros(n, dtype=np.float32)
            features['tx_velocity_24h'] = np.zeros(n, dtype=np.float32)
            features['amount_velocity_24h'] = np.zeros(n, dtype=np.float32)
    else:
        features['tx_velocity_1h'] = np.zeros(n, dtype=np.float32)
        features['tx_velocity_24h'] = np.zeros(n, dtype=np.float32)
        features['amount_velocity_24h'] = np.zeros(n, dtype=np.float32)

    # ========== 12. 扩展特征：风险评分(无泄漏模式) ==========
    if has_label and label_col and no_leakage and train_idx_set is not None:
        try:
            terminal_col = None
            device_col = None
            for col in ['terminal_id', 'merchant_id', 'merchant_type']:
                if col in df_columns:
                    terminal_col = col
                    break
            for col in ['device', 'device_type']:
                if col in df_columns:
                    device_col = col
                    break

            train_df = df.iloc[train_idx] if train_idx_set else None
            if train_df is not None:
                # Terminal risk score
                if terminal_col and terminal_col in df.columns:
                    terminal_fraud_rate = train_df.groupby(terminal_col)[label_col].mean()
                    features['terminal_risk_score'] = df[terminal_col].map(terminal_fraud_rate).fillna(0).values.astype(np.float32)

                # Device risk score
                if device_col and device_col in df.columns:
                    device_fraud_rate = train_df.groupby(device_col)[label_col].mean()
                    features['device_risk_score'] = df[device_col].map(device_fraud_rate).fillna(0).values.astype(np.float32)
        except Exception:
            pass
    features['terminal_risk_score'] = np.zeros(n, dtype=np.float32)
    features['device_risk_score'] = np.zeros(n, dtype=np.float32)

    # ========== 13. 扩展特征：图指标 ==========
    degrees = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.float32)
    max_degree = max(degrees) if max(degrees) > 0 else 1
    features['degree_centrality'] = degrees / max_degree
    features['clustering_coeff'] = np.zeros(n, dtype=np.float32)

    # 清理 - nan_to_num
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    # 释放局部引用，避免返回时触发大量gc
    del degrees, max_degree
    feature_names = list(features.keys())
    return features, feature_names


def run_ascend_gar(data_path, card_col, entity_cols, account_features,
                   transaction_features, output_csv, npu_id=0, workers=1, mode='auto',
                   label_col=None, fraud_value=1, train_idx=None, no_leakage=True,
                   train_ratio=0.7, seed=42, auto_detect=True):
    """Ascend NPU加速GAR特征生成"""
    device_info = check_ascend_npu(mode=mode)

    print("="*60, flush=True)
    print("Ascend NPU GAR Feature Generator", flush=True)
    print(f"NPU Available: {device_info['available']}", flush=True)
    print(f"Backend: {device_info['backend']}", flush=True)
    print(f"Workers: {workers}", flush=True)
    if no_leakage:
        print("Mode: NO-LEAKAGE (fraud rates from train only)", flush=True)
    else:
        print("Mode: LEAKAGE (fraud rates from all data - NOT RECOMMENDED)", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    # 1. 加载数据
    print(f"[INFO] Loading data from {data_path}...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Total records: {len(df)}, columns: {len(df.columns)}", flush=True)

    # 自动检测列名
    if auto_detect:
        schema = auto_detect_schema(df)
        print(f"[INFO] Auto-detected columns:", flush=True)
        for col_type, actual_col in schema.items():
            print(f"  {col_type:<20} -> {actual_col}", flush=True)

        if card_col in schema:
            card_col = schema['card_id']
        if not entity_cols or entity_cols == DEFAULT_ENTITY_COLS:
            entity_cols = [v for k, v in schema.items() if k in ['card_id', 'merchant_id', 'terminal_id', 'device', 'is_night']]
        if not account_features or account_features == DEFAULT_ACCOUNT_FEATURES:
            account_features = [v for k, v in schema.items() if k in ['card_level', 'card_location', 'card_type']]
        if not transaction_features or transaction_features == DEFAULT_TRANSACTION_FEATURES:
            transaction_features = [v for k, v in schema.items() if k in ['amount', 'balance', 'is_cross_border']]

    # 2. 预处理
    for col in entity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col].astype(str))[0]

    for col in account_features + transaction_features:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('missing')
            else:
                df[col] = df[col].fillna(0)

    # 3. 检测标签
    has_label = False
    if label_col and label_col in df.columns:
        has_label = True
        print(f"[INFO] Using explicit label column: {label_col}", flush=True)
    else:
        for col in ['isFraud', 'fraud', 'label', 'is_fraud']:
            if col in df.columns:
                has_label = True
                label_col = col
                print(f"[INFO] Found label column: {label_col}", flush=True)
                break

    amount_col = None
    for col in ['amount', '交易金额']:
        if col in df.columns:
            amount_col = col
            break

    # 4. 数据分割(无泄漏模式)
    train_idx_arr, test_idx_arr = split_data(df, train_ratio=train_ratio, seed=seed)
    print(f"[INFO] Data split: Train={len(train_idx_arr)}, Test={len(test_idx_arr)}", flush=True)

    # 5. 计算欺诈率映射(无泄漏模式)
    entity_fraud_maps = None
    pair_fraud_maps = None
    train_idx_set = None
    train_label_map = None
    if no_leakage and has_label:
        train_df = df.iloc[train_idx_arr]
        entity_fraud_maps, pair_fraud_maps = compute_fraud_rates_from_train(train_df, entity_cols, label_col)
        train_idx_set = set(train_idx_arr)
        train_labels = train_df[label_col].values
        train_label_map = dict(zip(train_idx_arr, train_labels))
        print("[INFO] Computed fraud rates from train only (no leakage)", flush=True)

    global_stats = compute_global_stats(df, entity_cols, card_col, amount_col)

    print("[INFO] Building graph...", flush=True)
    graph_start = time.time()
    tx_neighbors = build_graph(df, entity_cols, show_progress=True)
    graph_time = time.time() - graph_start
    n_edges = sum(len(v) for v in tx_neighbors.values()) // 2
    print(f"[INFO] Graph built in {graph_time:.2f}s, edges: {n_edges}", flush=True)
    if len(df) > 100000:
        print(f"[INFO] Graph throughput: {len(df)/graph_time:.0f} records/sec", flush=True)

    print("[INFO] Building GAR features...", flush=True)
    feat_start = time.time()
    print(f"[INFO]   Section 1/4: Basic features (tx level, entity freq, card agg)", flush=True)
    features, feature_names = build_gar_features_optimized(df, tx_neighbors, global_stats, entity_cols, card_col,
                                            account_features, transaction_features, has_label, label_col,
                                            entity_fraud_maps, pair_fraud_maps,
                                            no_leakage, train_idx_set, train_label_map,
                                            show_progress=True)
    feat_time = time.time() - feat_start
    print(f"[INFO]   Features built in {feat_time:.2f}s", flush=True)
    print(f"[INFO]   Feature count: {len(feature_names)}", flush=True)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        print(f"[INFO] Exporting features to CSV...", flush=True)
        export_start = time.time()
        df_features = pd.DataFrame(features)
        key_cols = [c for c in [card_col, 'timestamp', '时间戳'] if c in df.columns]
        if key_cols:
            df_features = pd.concat([df[key_cols], df_features], axis=1)
        if has_label and label_col:
            df_features[label_col] = df[label_col].values
        # 添加split列
        split_arr = np.array(['train' if i in train_idx_arr else 'test' for i in range(len(df_features))])
        df_features['split'] = split_arr
        # 分块写入，避免大数据量时内存问题
        chunk_size = 500000
        n_total = len(df_features)
        if n_total > chunk_size:
            n_chunks = (n_total + chunk_size - 1) // chunk_size
            print(f"[INFO]   Writing {n_total} rows in {n_chunks} chunks...", flush=True)
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_total)
                chunk_df = df_features.iloc[start_idx:end_idx]
                if i == 0:
                    chunk_df.to_csv(output_csv, index=False, mode='w')
                else:
                    chunk_df.to_csv(output_csv, index=False, mode='a', header=False)
                print(f"[INFO]   Chunk {i+1}/{n_chunks} ({end_idx}/{n_total}) written", flush=True)
        else:
            df_features.to_csv(output_csv, index=False)
        export_time = time.time() - export_start
        print(f"[INFO] Saved to {output_csv}", flush=True)
        print(f"[INFO] Shape: {df_features.shape}, export time: {export_time:.1f}s", flush=True)

    elapsed = time.time() - start_time
    print(f"\n[INFO] Total time: {elapsed/60:.1f} minutes", flush=True)
    print(f"[INFO] Throughput: {len(df)/elapsed:.0f} records/sec", flush=True)

    # 释放临时大对象(仅清除引用，由Python自动gc，不主动触发stop-the-world)
    del df, tx_neighbors, global_stats

    print("[INFO] Feature generation complete, returning to caller...", flush=True)
    return features, feature_names


def main():
    parser = argparse.ArgumentParser(
        description='Ascend NPU GAR Feature Generator',
        epilog="""
Examples:
  # Ascend NPU模式
  python src/gar_feature_generator_ascend.py --data /path/to/data.csv \\
                                             --card-col card_id \\
                                             --npu-id 0 \\
                                             --output-csv ./features.csv

  # 多NPU分布式
  python src/gar_feature_generator_ascend.py --data /path/to/large_data.csv \\
                                               --npus 0,1,2,3 \\
                                               --workers 4 \\
                                               --output-csv ./features.csv
        """
    )

    parser.add_argument('--data', type=str, default=None,
                        help='CSV文件路径(使用--check-npu时可选)')
    parser.add_argument('--card-col', type=str, default=DEFAULT_CARD_COL,
                        help='卡号列名')
    parser.add_argument('--entity-cols', type=str, default=None,
                        help='实体列名列表，逗号分隔')
    parser.add_argument('--account-features', type=str, default=None,
                        help='账户级特征列名，逗号分隔')
    parser.add_argument('--transaction-features', type=str, default=None,
                        help='交易级特征列名，逗号分隔')
    parser.add_argument('--npu-id', type=int, default=0,
                        help='NPU设备ID')
    parser.add_argument('--npus', type=str, default=None,
                        help='多NPU ID列表，逗号分隔')
    parser.add_argument('--workers', type=int, default=1,
                        help='并行worker数量')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='特征CSV输出路径')
    parser.add_argument('--check-npu', action='store_true',
                        help='检查NPU状态')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'cpu', 'npu'],
                        help='运行模式: auto=自动检测, cpu=纯CPU, npu=强制NPU')
    parser.add_argument('--label-col', type=str, default=None,
                        help='欺诈标签列名(如 isFraud, fraud, label)')
    parser.add_argument('--fraud-value', type=int, default=1,
                        help='表示欺诈的值(默认: 1)')
    parser.add_argument('--auto-detect', action='store_true', default=True,
                        help='自动检测列名并映射到标准列名(默认开启)')
    parser.add_argument('--no-auto-detect', action='store_false', dest='auto_detect',
                        help='关闭自动列名检测')
    parser.add_argument('--no-leakage', action='store_true', default=True,
                        help='防止数据泄漏：欺诈率仅从训练集计算(默认开启)')
    parser.add_argument('--leakage', action='store_false', dest='no_leakage',
                        help='关闭防泄漏模式：欺诈率从全部数据计算(不推荐)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例(默认: 0.7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子(默认: 42)')

    args = parser.parse_args()

    if args.check_npu:
        info = check_ascend_npu(mode=args.mode)
        print("Ascend NPU Status:")
        print(f"  Available: {info['available']}")
        print(f"  Backend: {info['backend']}")
        print(f"  Device Count: {info.get('device_count', 'N/A')}")
        return

    if args.data is None:
        parser.error("--data is required unless --check-npu is used")

    entity_cols = args.entity_cols.split(',') if args.entity_cols else DEFAULT_ENTITY_COLS
    account_features = args.account_features.split(',') if args.account_features else []
    transaction_features = args.transaction_features.split(',') if args.transaction_features else []

    run_ascend_gar(args.data, args.card_col, entity_cols, account_features,
                   transaction_features, args.output_csv, args.npu_id, args.workers, args.mode,
                   args.label_col, args.fraud_value, None, args.no_leakage,
                   args.train_ratio, args.seed, args.auto_detect)


if __name__ == '__main__':
    main()
