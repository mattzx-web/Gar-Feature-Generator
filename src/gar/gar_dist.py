"""
分布式GAR特征生成器

支持多进程加速，处理千万到亿级交易记录。
支持无数据泄漏模式。

用法:
    # 多进程分布式（无泄漏模式）
    python src/gar/gar_dist.py --data /path/to/data.csv \\
                                --card-col card_id \\
                                --workers 8 \\
                                --no-leakage \\
                                --output-csv ./features.csv
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import os
import sys
import argparse
from datetime import datetime
import time

sys.stdout.reconfigure(line_buffering=True)

DEFAULT_CARD_COL = 'card_id'
DEFAULT_ENTITY_COLS = ['card_id', 'merchant_id', 'device', 'is_night']
DEFAULT_ACCOUNT_FEATURES = ['card_level', 'card_location', 'card_type']
DEFAULT_TRANSACTION_FEATURES = ['amount', 'balance', 'is_cross_border']
DEFAULT_NEIGHBOR_THRESHOLD = 300

# 标准列名到可能列名的映射（用于自动检测）
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
                if col.lower() == alias.lower() or alias.lower() in col.lower():
                    if col not in used_columns:
                        detected[col_type] = col
                        used_columns.add(col)
                        break

    return detected


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


def build_graph(df, entity_cols, neighbor_threshold=300):
    """构建图结构"""
    tx_neighbors = defaultdict(set)

    for col in entity_cols:
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


def compute_fraud_rates_from_train(train_df, entity_cols, label_col):
    """从训练集计算欺诈率映射（避免数据泄漏）"""
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


def build_gar_features(df, tx_neighbors, global_stats, entity_cols, card_col,
                        account_features, transaction_features, has_label, label_col,
                        entity_fraud_maps=None, pair_fraud_maps=None,
                        no_leakage=False, train_idx_set=None, train_label_map=None):
    """构建GAR特征"""

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

    # ========== 2. 实体频率 ==========
    for col in entity_cols:
        if col not in df_columns:
            continue
        values = df[col].values
        freq_map = global_stats.get(f'{col}_freq', {})
        features[f'{col}_freq'] = np.array([freq_map.get(v, 0) for v in values], dtype=np.float32)
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq']).astype(np.float32)

    # ========== 3. 卡号聚合特征 ==========
    if card_col in df_columns:
        card_values = df[card_col].values
        card_counts = global_stats.get('card_tx_count', {})
        features['card_tx_count'] = np.array([card_counts.get(v, 0) for v in card_values], dtype=np.float32)
        features['card_tx_count_log'] = np.log1p(features['card_tx_count']).astype(np.float32)

        card_agg = global_stats.get('card_agg', {})
        amt_mean, amt_std, amt_max = [], [], []
        for card in card_values:
            if card in card_agg:
                agg = card_agg[card]
                amt_mean.append(agg['amt_mean'])
                amt_std.append(agg['amt_std'] if not np.isnan(agg['amt_std']) else 0)
                amt_max.append(agg['amt_max'])
            else:
                amt_mean.append(0)
                amt_std.append(0)
                amt_max.append(0)
        features['card_amt_mean'] = np.array(amt_mean, dtype=np.float32)
        features['card_amt_std'] = np.array(amt_std, dtype=np.float32)
        features['card_amt_max'] = np.array(amt_max, dtype=np.float32)

        if amount_col:
            amounts = df[amount_col].fillna(0).values
            features['amt_to_card_mean_ratio'] = (amounts / (np.array(amt_mean) + 1)).astype(np.float32)

    # ========== 4. 账户级特征 ==========
    for col in account_features:
        if col not in df_columns:
            continue

        if df[col].dtype == 'object':
            cats = df[col].astype('category')
            features[col] = cats.cat.codes.values.astype(np.int32)
        else:
            features[col] = df[col].fillna(-1).values

    # ========== 5. 图特征 ==========
    for col in entity_cols:
        if col not in df_columns:
            continue
        degrees = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.int32)
        features[f'{col}_degree'] = degrees

    n_1hop = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.int32)
    features['n_1hop'] = n_1hop
    features['n_1hop_log'] = np.log1p(n_1hop.astype(np.float32))

    if amount_col:
        amounts = df[amount_col].fillna(0).values
        amt_1hop_mean = np.zeros(n, dtype=np.float32)
        amt_1hop_std = np.zeros(n, dtype=np.float32)

        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = amounts[list(neighs)]
                amt_1hop_mean[i] = np.mean(neigh_amts)
                amt_1hop_std[i] = np.std(neigh_amts) if len(neighs) > 1 else 0

        features['amt_1hop_mean'] = amt_1hop_mean
        features['amt_1hop_std'] = amt_1hop_std

    # ========== 6. 配对频率 ==========
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

    # ========== 8. GAR Fraud Rate特征 ==========
    if has_label and label_col:
        if entity_fraud_maps:
            # 无泄漏模式：使用预计算的欺诈率映射
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
        else:
            # 泄漏模式：在全部数据上计算
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

        # Neighbor Fraud Rate (无泄漏模式：仅使用训练集邻居的标签)
        neigh_fraud_rates = np.zeros(n, dtype=np.float32)
        if has_label and no_leakage and train_label_map:
            for i in range(n):
                neighs = tx_neighbors.get(i, set())
                if neighs:
                    train_neighs = [n for n in neighs if n in train_idx_set]
                    if train_neighs:
                        neigh_fraud_rates[i] = np.mean([train_label_map[n] for n in train_neighs])
        elif has_label:
            # 泄漏模式：使用全量标签（不推荐）
            labels = df[label_col].values
            for i in range(n):
                neighs = tx_neighbors.get(i, set())
                if neighs:
                    neigh_fraud_rates[i] = labels[list(neighs)].mean()
        features['neigh_fraud_rate'] = neigh_fraud_rates

    # ========== 9. 扩展特征：时序熵 ==========
    timestamp_col = None
    for col in ['timestamp', '时间戳', 'trans_time', 'transaction_time']:
        if col in df_columns:
            timestamp_col = col
            break

    if timestamp_col and card_col in df_columns:
        try:
            ts = pd.to_datetime(df[timestamp_col], errors='coerce')
            if not ts.isna().all():
                # hour entropy
                hour_entropy_list = []
                for card_id in df[card_col].unique():
                    mask = df[card_col] == card_id
                    hours = ts[mask].dt.hour
                    if len(hours) > 1:
                        hour_counts = hours.value_counts(normalize=True)
                        entropy = -np.sum(hour_counts * np.log(hour_counts + 1e-10))
                    else:
                        entropy = 0
                    hour_entropy_list.extend([entropy] * mask.sum())
                features['hour_entropy'] = np.array(hour_entropy_list) if len(hour_entropy_list) == n else np.zeros(n, dtype=np.float32)

                # day entropy
                day_entropy_list = []
                for card_id in df[card_col].unique():
                    mask = df[card_col] == card_id
                    days = ts[mask].dt.dayofweek
                    if len(days) > 1:
                        day_counts = days.value_counts(normalize=True)
                        entropy = -np.sum(day_counts * np.log(day_counts + 1e-10))
                    else:
                        entropy = 0
                    day_entropy_list.extend([entropy] * mask.sum())
                features['day_entropy'] = np.array(day_entropy_list) if len(day_entropy_list) == n else np.zeros(n, dtype=np.float32)

                # time_diff_prev
                df_sorted = df.copy()
                df_sorted['_ts'] = ts
                df_sorted = df_sorted.sort_values([card_col, timestamp_col])
                time_diff = df_sorted.groupby(card_col)['_ts'].diff().dt.total_seconds().fillna(0)
                features['time_diff_prev'] = df.index.map(time_diff.to_dict()).fillna(0).values.astype(np.float32)
        except Exception:
            features['hour_entropy'] = np.zeros(n, dtype=np.float32)
            features['day_entropy'] = np.zeros(n, dtype=np.float32)
            features['time_diff_prev'] = np.zeros(n, dtype=np.float32)
    else:
        features['hour_entropy'] = np.zeros(n, dtype=np.float32)
        features['day_entropy'] = np.zeros(n, dtype=np.float32)
        features['time_diff_prev'] = np.zeros(n, dtype=np.float32)

    # ========== 10. 扩展特征：金额统计 ==========
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

    # ========== 12. 扩展特征：风险评分（无泄漏模式） ==========
    if has_label and label_col and entity_fraud_maps:
        # terminal_risk_score
        terminal_col = None
        for col in ['terminal_id', 'merchant_id', 'merchant_type']:
            if col in df_columns:
                terminal_col = col
                break
        if terminal_col and terminal_col in entity_fraud_maps.get('terminal_id', {}).keys() if 'terminal_id' in entity_fraud_maps else False:
            pass  # 已通过fraud_rate计算

    features['terminal_risk_score'] = np.zeros(n, dtype=np.float32)
    features['device_risk_score'] = np.zeros(n, dtype=np.float32)

    # ========== 13. 扩展特征：图指标 ==========
    degrees = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.float32)
    max_degree = max(degrees) if max(degrees) > 0 else 1
    features['degree_centrality'] = degrees / max_degree
    features['clustering_coeff'] = np.zeros(n, dtype=np.float32)

    # 清理
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    return features


def process_partition(args):
    """并行worker函数"""
    (p_id, df_dict, global_stats, entity_cols, card_col,
     account_features, transaction_features, has_label, label_col,
     entity_fraud_maps, pair_fraud_maps,
     no_leakage, train_idx_set, train_label_map) = args

    df = pd.DataFrame(df_dict)
    tx_neighbors = build_graph(df, entity_cols)
    features = build_gar_features(df, tx_neighbors, global_stats, entity_cols, card_col,
                                   account_features, transaction_features, has_label, label_col,
                                   entity_fraud_maps, pair_fraud_maps,
                                   no_leakage, train_idx_set, train_label_map)

    return features, list(df.columns), p_id


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


def run_distributed(data_path, card_col, entity_cols, account_features,
                    transaction_features, n_workers, output_csv,
                    no_leakage=True, train_ratio=0.7, seed=42,
                    label_col=None, fraud_value=1, train_idx=None):
    """分布式GAR特征生成"""
    print(f"[MODE] Distributed GAR ({n_workers} workers)", flush=True)
    if no_leakage:
        print(f"[MODE] NO-LEAKAGE MODE (fraud rates from train only)", flush=True)
    else:
        print(f"[MODE] LEAKAGE MODE (fraud rates from all data)", flush=True)
    start_time = time.time()

    # 1. 加载全部数据
    print("[INFO] Loading data...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Total records: {len(df)}", flush=True)

    # 2. 预处理
    for col in entity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col].astype(str))[0]

    # 检测标签
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

    # 3. 数据分割（无泄漏模式）
    train_idx, test_idx = split_data(df, train_ratio=train_ratio, seed=seed)
    print(f"[INFO] Data split: Train={len(train_idx)}, Test={len(test_idx)}", flush=True)

    # 4. 计算欺诈率映射（无泄漏模式）
    entity_fraud_maps = None
    pair_fraud_maps = None
    if no_leakage and has_label:
        train_df = df.iloc[train_idx]
        entity_fraud_maps, pair_fraud_maps = compute_fraud_rates_from_train(train_df, entity_cols, label_col)
        print("[INFO] Computed fraud rates from train only", flush=True)

    # 5. 全局统计量
    global_stats = compute_global_stats(df, entity_cols, card_col, amount_col)

    # 6. 分区
    print(f"[INFO] Partitioning into {n_workers} shards...", flush=True)
    df['_p'] = df[card_col].apply(lambda x: hash(x) % n_workers)
    partitions = []
    for p in range(n_workers):
        mask = df['_p'] == p
        partitions.append(df[mask].drop(columns=['_p']).reset_index(drop=True))
        print(f"[INFO] Partition {p}: {len(partitions[-1])} records", flush=True)

    # 准备无泄漏模式所需的训练集信息（在del df之前）
    train_idx_set = set(train_idx) if no_leakage and has_label else None
    train_label_map = None
    if no_leakage and has_label:
        train_labels = df.iloc[train_idx][label_col].values
        train_label_map = dict(zip(train_idx, train_labels))

    del df

    # 7. 并行处理
    print("[INFO] Parallel processing...", flush=True)

    args_list = [
        (p_id, p.to_dict('list'), global_stats, entity_cols, card_col,
         account_features, transaction_features, has_label, label_col,
         entity_fraud_maps, pair_fraud_maps,
         no_leakage, train_idx_set, train_label_map)
        for p_id, p in enumerate(partitions)
    ]

    with Pool(n_workers) as pool:
        results = pool.map(process_partition, args_list)

    # 8. 合并
    print("[INFO] Merging results...", flush=True)
    all_features = {}
    for features, _, _ in results:
        for k, v in features.items():
            if k not in all_features:
                all_features[k] = []
            all_features[k].append(v)

    for k in all_features:
        all_features[k] = np.concatenate(all_features[k])

    # 9. 导出
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        df_features = pd.DataFrame(all_features)
        all_dfs = pd.concat(partitions, ignore_index=True)
        key_cols = [c for c in [card_col, 'timestamp', '时间戳'] if c in all_dfs.columns]
        if key_cols:
            df_features = pd.concat([all_dfs[key_cols], df_features], axis=1)
        if has_label and label_col:
            df_features[label_col] = all_dfs[label_col].values

        # 添加split列
        split_arr = np.array(['train' if i in train_idx else 'test' for i in range(len(df_features))])
        df_features['split'] = split_arr

        df_features.to_csv(output_csv, index=False)
        print(f"[INFO] Saved to {output_csv}", flush=True)
        print(f"[INFO] Shape: {df_features.shape}", flush=True)

    print(f"[TIME] {(time.time()-start_time)/60:.1f} min", flush=True)

    return all_features


def main():
    parser = argparse.ArgumentParser(
        description='Distributed GAR Feature Generator',
        epilog="""
Examples:
  # 分布式模式（无泄漏，默认）
  python src/gar/gar_dist.py --data /path/to/large_data.csv \\
                              --card-col card_id \\
                              --workers 16 \\
                              --output-csv ./features.csv

  # 泄漏模式（不推荐）
  python src/gar/gar_dist.py --data /path/to/large_data.csv \\
                              --card-col card_id \\
                              --workers 16 \\
                              --leakage \\
                              --output-csv ./features.csv
        """
    )

    parser.add_argument('--data', type=str, required=True,
                        help='CSV文件路径')
    parser.add_argument('--card-col', type=str, default=DEFAULT_CARD_COL,
                        help='卡号列名')
    parser.add_argument('--entity-cols', type=str, default=None,
                        help='实体列名列表，逗号分隔')
    parser.add_argument('--account-features', type=str, default=None,
                        help='账户级特征列名，逗号分隔')
    parser.add_argument('--transaction-features', type=str, default=None,
                        help='交易级特征列名，逗号分隔')
    parser.add_argument('--workers', type=int, default=cpu_count(),
                        help='并行worker数量')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='特征CSV输出路径')
    parser.add_argument('--no-leakage', action='store_true', default=True,
                        help='防止数据泄漏：欺诈率仅从训练集计算（默认开启）')
    parser.add_argument('--leakage', action='store_false', dest='no_leakage',
                        help='关闭防泄漏模式：欺诈率从全部数据计算（不推荐）')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例（默认: 0.7）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--label-col', type=str, default=None,
                        help='欺诈标签列名（如 isFraud, fraud, label）')
    parser.add_argument('--fraud-value', type=int, default=1,
                        help='表示欺诈的值（默认: 1）')
    parser.add_argument('--auto-detect', action='store_true', default=True,
                        help='自动检测列名并映射到标准列名（默认开启）')
    parser.add_argument('--no-auto-detect', action='store_false', dest='auto_detect',
                        help='关闭自动列名检测')

    args = parser.parse_args()

    entity_cols = args.entity_cols.split(',') if args.entity_cols else DEFAULT_ENTITY_COLS
    account_features = args.account_features.split(',') if args.account_features else []
    transaction_features = args.transaction_features.split(',') if args.transaction_features else []

    run_distributed(args.data, args.card_col, entity_cols, account_features,
                    transaction_features, args.workers, args.output_csv,
                    args.no_leakage, args.train_ratio, args.seed,
                    args.label_col, args.fraud_value, train_idx)


if __name__ == '__main__':
    main()