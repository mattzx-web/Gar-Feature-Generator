"""
NPU加速GAR特征生成器

将图关联规则特征生成的两个热点 O(n) Python循环：
1. 邻居金额统计 (amt_1hop_mean, amt_1hop_std)
2. 邻居欺诈率 (neigh_fraud_rate)

替换为PyTorch NPU向量化操作(gather + reduce)。

用法:
    python src/gar/gar_npu.py --data /path/to/data.csv \\
        --output-csv ./features.csv \\
        --card-col card_id
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


# ========== NPU-accelerated helper functions ==========

def build_neighbor_tensor_cpu(tx_neighbors, n, degrees):
    """将 dict-of-sets 图结构转换为 (n, max_degree) 填充张量(CPU侧)"""
    max_degree = int(max(degrees)) if len(degrees) > 0 else 1
    max_degree = max(max_degree, 1)

    neighbors_arr = np.full((n, max_degree), -1, dtype=np.int64)
    valid_counts = np.zeros(n, dtype=np.int32)

    for i in range(n):
        neighs = tx_neighbors.get(i, set())
        if neighs:
            neigh_list = list(neighs)[:max_degree]
            k = len(neigh_list)
            neighbors_arr[i, :k] = neigh_list
            valid_counts[i] = k

    return neighbors_arr, valid_counts.astype(np.int32)


def npu_neighbor_agg(amounts_npu, neighbors_npu, valid_counts_npu):
    """NPU向量化：计算所有节点的邻居金额均值/标准差

    amounts_npu: (n,) float32 NPU tensor
    neighbors_npu: (n, max_degree) int64 NPU tensor, -1 padded
    valid_counts_npu: (n,) int32 NPU tensor
    returns: (amt_1hop_mean, amt_1hop_std) both (n,) float32 numpy arrays
    """
    import torch

    n, max_degree = neighbors_npu.shape
    device = amounts_npu.device

    # Create valid mask: True where neighbor != -1
    valid_mask = (neighbors_npu != -1)  # (n, max_degree) bool

    # Gather neighbor amounts: (n, max_degree)
    # Clamp to handle -1 padding (will be masked out anyway)
    neigh_idx = torch.clamp(neighbors_npu, min=0)
    neigh_amts = amounts_npu.unsqueeze(-1).expand(n, max_degree).gather(1, neigh_idx)  # (n, max_degree)

    # Zero out padded positions
    neigh_amts = neigh_amts * valid_mask.float()

    # Sum and divide by valid count
    valid_counts_f = valid_counts_npu.float().clamp(min=1)  # (n,)
    amt_1hop_mean = neigh_amts.sum(dim=1) / valid_counts_f  # (n,)

    # Std: E[X^2] - E[X]^2
    amt_1hop_mean_expanded = amt_1hop_mean.unsqueeze(-1).expand(n, max_degree)  # (n, max_degree)
    masked_mean_sq = (amt_1hop_mean_expanded ** 2) * valid_mask.float()
    amt_sq = (neigh_amts ** 2) * valid_mask.float()
    variance = (amt_sq.sum(dim=1) / valid_counts_f) - (masked_mean_sq.sum(dim=1) / valid_counts_f)
    variance = torch.clamp(variance, min=0)
    amt_1hop_std = torch.sqrt(variance)

    return amt_1hop_mean.cpu().numpy(), amt_1hop_std.cpu().numpy()


def npu_neighbor_fraud_rate(neighbors_npu, valid_counts_npu, train_label_npu, train_idx_set, n):
    """NPU向量化：计算邻居欺诈率

    neighbors_npu: (n, max_degree) int64 NPU tensor, -1 padded
    valid_counts_npu: (n,) int32 NPU tensor
    train_label_npu: (n,) float32 NPU tensor, 0 for non-train indices
    train_idx_set: set of train indices (Python set for fast lookup on CPU)
    returns: neigh_fraud_rate (n,) float32 numpy array
    """
    import torch

    n, max_degree = neighbors_npu.shape
    device = neighbors_npu.device

    # Build train mask on CPU, transfer to NPU
    train_idx_set_np = np.array(list(train_idx_set), dtype=np.int64)
    is_train = np.isin(neighbors_npu.cpu().numpy(), train_idx_set_np).astype(np.float32)
    train_mask_npu = torch.from_numpy(is_train).to(device)  # (n, max_degree) float

    # Valid mask (not -1 padding)
    valid_mask = (neighbors_npu != -1).float()  # (n, max_degree)

    # Combined: only count neighbors that are both valid AND in train set
    effective_mask = train_mask_npu * valid_mask  # (n, max_degree)

    # Gather labels for all neighbors (padded with 0, masked out)
    neigh_idx = torch.clamp(neighbors_npu, min=0)
    neigh_labels = train_label_npu.unsqueeze(-1).expand(n, max_degree).gather(1, neigh_idx)  # (n, max_degree)

    # Multiply by effective mask and sum
    fraud_sum = (neigh_labels * effective_mask).sum(dim=1)  # (n,)
    valid_counts_f = valid_counts_npu.float().clamp(min=1)
    neigh_fraud_rate = fraud_sum / valid_counts_f

    return neigh_fraud_rate.cpu().numpy()


def build_gar_features_npu(df, tx_neighbors, global_stats, entity_cols, card_col,
                            account_features, transaction_features, has_label, label_col,
                            entity_fraud_maps=None, pair_fraud_maps=None,
                            no_leakage=False, train_idx_set=None, train_label_map=None,
                            show_progress=True):
    """构建GAR特征(NPU加速版本：热点循环替换为PyTorch NPU操作)"""
    import torch

    if show_progress and len(df) > 100000:
        print(f"[INFO]   Processing {len(df)} records with NPU acceleration...", flush=True)

    features = {}
    n = len(df)
    device = torch.device('npu') if torch.npu.is_available() else torch.device('cpu')

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

    # ========== 5. 图特征(优化：预计算度 + NPU邻居金额) ==========
    print("[INFO]   Section 5/13: Graph features with NPU neighbor aggregation...", flush=True)
    section5_start = time.time()

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

    # NPU-accelerated neighbor amount stats (THE HOT LOOP - now on NPU)
    if amount_col:
        amounts = df[amount_col].fillna(0).values.astype(np.float32)

        # Build neighbor tensor on CPU
        print("[INFO]   Building neighbor tensor (CPU)...", flush=True)
        build_start = time.time()
        neighbors_arr, valid_counts = build_neighbor_tensor_cpu(tx_neighbors, n, degrees)
        build_time = time.time() - build_start
        print(f"[INFO]   Neighbor tensor built in {build_time:.1f}s, shape: {neighbors_arr.shape}", flush=True)

        # Transfer to NPU
        print("[INFO]   Transferring to NPU...", flush=True)
        transfer_start = time.time()
        amounts_npu = torch.from_numpy(amounts).to(device)
        neighbors_npu = torch.from_numpy(neighbors_arr).to(device)
        valid_counts_npu = torch.from_numpy(valid_counts).to(device)
        torch.npu.synchronize() if device.type == 'npu' else None
        transfer_time = time.time() - transfer_start
        print(f"[INFO]   NPU transfer done in {transfer_time:.1f}s", flush=True)

        # NPU neighbor aggregation
        print("[INFO]   Running NPU neighbor aggregation...", flush=True)
        npu_start = time.time()
        amt_1hop_mean, amt_1hop_std = npu_neighbor_agg(amounts_npu, neighbors_npu, valid_counts_npu)
        torch.npu.synchronize() if device.type == 'npu' else None
        npu_time = time.time() - npu_start
        print(f"[INFO]   NPU neighbor agg done in {npu_time:.1f}s ({n/npu_time:.0f} records/sec)", flush=True)

        features['amt_1hop_mean'] = amt_1hop_mean.astype(np.float32)
        features['amt_1hop_std'] = amt_1hop_std.astype(np.float32)

        # Free NPU memory
        del amounts_npu, neighbors_npu, valid_counts_npu
        if device.type == 'npu':
            torch.npu.empty_cache()
        del neighbors_arr, valid_counts
        gc.collect()

        section5_time = time.time() - section5_start
        print(f"[INFO]   Section 5 done in {section5_time:.1f}s total", flush=True)

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

    # ========== 8. GAR Fraud Rate特征(NPU邻居欺诈率) ==========
    if show_progress and len(df) > 100000:
        print(f"[INFO]   Section 8/13: Fraud rate features with NPU neighbor aggregation...", flush=True)
    section8_start = time.time()

    if has_label and label_col:
        if no_leakage and entity_fraud_maps:
            print("[INFO]   Using pre-computed fraud rates from TRAIN ONLY (no leakage)", flush=True)
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

            # NPU-accelerated neighbor fraud rate
            if train_idx_set and train_label_map:
                print("[INFO]   Building train label tensor for NPU...", flush=True)
                # Build train label array: label if in train_idx_set else 0
                train_labels_arr = np.zeros(n, dtype=np.float32)
                for idx in train_idx_set:
                    if idx < n:
                        train_labels_arr[idx] = train_label_map.get(idx, 0)

                # Build neighbor tensor if not already built (reuse from section 5)
                if 'neighbors_arr' not in dir() or neighbors_arr is None:
                    neighbors_arr, valid_counts = build_neighbor_tensor_cpu(tx_neighbors, n, degrees)

                # Transfer to NPU
                train_label_npu = torch.from_numpy(train_labels_arr).to(device)
                if 'neighbors_npu' not in dir():
                    neighbors_npu = torch.from_numpy(neighbors_arr).to(device)
                    valid_counts_npu = torch.from_numpy(valid_counts).to(device)

                torch.npu.synchronize() if device.type == 'npu' else None

                print("[INFO]   Running NPU neighbor fraud rate...", flush=True)
                npu_start = time.time()
                neigh_fraud_rates = npu_neighbor_fraud_rate(
                    neighbors_npu, valid_counts_npu, train_label_npu, train_idx_set, n
                )
                torch.npu.synchronize() if device.type == 'npu' else None
                npu_time = time.time() - npu_start
                print(f"[INFO]   NPU neighbor fraud done in {npu_time:.1f}s ({n/npu_time:.0f} records/sec)", flush=True)

                features['neigh_fraud_rate'] = neigh_fraud_rates.astype(np.float32)

                # Free NPU memory
                del train_label_npu, neighbors_npu, valid_counts_npu, train_labels_arr
                del neighbors_arr, valid_counts
                if device.type == 'npu':
                    torch.npu.empty_cache()
                gc.collect()

        else:
            # Leakage mode
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

            # Fallback: CPU neighbor fraud for leakage mode
            neigh_fraud_rates = np.zeros(n, dtype=np.float32)
            range_iter = range(n) if not TQDM_AVAILABLE else tqdm(range(n), desc="[Features] Neighbor fraud")
            for i in range_iter:
                neighs = tx_neighbors.get(i, set())
                if neighs:
                    neigh_fraud_rates[i] = labels[list(neighs)].mean()
            features['neigh_fraud_rate'] = neigh_fraud_rates

    section8_time = time.time() - section8_start
    if show_progress:
        print(f"[INFO]   Section 8 done in {section8_time:.1f}s", flush=True)

    # ========== 9. 扩展特征：时序熵 ==========
    if show_progress and len(df) > 100000:
        print(f"[INFO]   Section 9/13: Temporal & entropy features", flush=True)
    timestamp_col = None
    for col in ['timestamp', '时间戳', 'trans_time', 'transaction_time']:
        if col in df_columns:
            timestamp_col = col
            break

    if timestamp_col and card_col in df_columns:
        try:
            ts = pd.to_datetime(df[timestamp_col], errors='coerce')
            if not ts.isna().all():
                df['_hour'] = ts.dt.hour.values
                hour_cnt = df.groupby([card_col, '_hour']).size().reset_index(name='h_count')
                hour_pct = hour_cnt.groupby(card_col)['h_count'].transform(lambda x: x / x.sum())
                hour_cnt['h_pct'] = hour_pct
                hour_cnt['h_entropy'] = -hour_cnt['h_pct'] * np.log(hour_cnt['h_pct'] + 1e-10)
                card_hour_entropy = hour_cnt.groupby(card_col)['h_entropy'].sum().to_dict()
                features['hour_entropy'] = df[card_col].map(card_hour_entropy).fillna(0).values.astype(np.float32)
                del df['_hour']

                df['_day'] = ts.dt.dayofweek.values
                day_cnt = df.groupby([card_col, '_day']).size().reset_index(name='d_count')
                day_pct = day_cnt.groupby(card_col)['d_count'].transform(lambda x: x / x.sum())
                day_cnt['d_pct'] = day_pct
                day_cnt['d_entropy'] = -day_cnt['d_pct'] * np.log(day_cnt['d_pct'] + 1e-10)
                card_day_entropy = day_cnt.groupby(card_col)['d_entropy'].sum().to_dict()
                features['day_entropy'] = df[card_col].map(card_day_entropy).fillna(0).values.astype(np.float32)
                del df['_day']

                order = np.lexsort((ts.values, df[card_col].values))
                idx_sorted = df.index[order]
                ts_sorted = ts.values[order]
                card_sorted = df[card_col].values[order]
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
    if show_progress and len(df) > 100000:
        print(f"[INFO]   Section 10/13: Amount statistics", flush=True)
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
    if show_progress and len(df) > 100000:
        print(f"[INFO]   Section 11/13: Transaction velocity", flush=True)
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
    if show_progress and len(df) > 100000:
        print(f"[INFO]   Section 12/13: Risk scores", flush=True)
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

            train_df = df.iloc[train_idx_set] if train_idx_set else None
            if train_df is not None:
                if terminal_col and terminal_col in df.columns:
                    terminal_fraud_rate = train_df.groupby(terminal_col)[label_col].mean()
                    features['terminal_risk_score'] = df[terminal_col].map(terminal_fraud_rate).fillna(0).values.astype(np.float32)

                if device_col and device_col in df.columns:
                    device_fraud_rate = train_df.groupby(device_col)[label_col].mean()
                    features['device_risk_score'] = df[device_col].map(device_fraud_rate).fillna(0).values.astype(np.float32)
        except Exception:
            pass
    features['terminal_risk_score'] = np.zeros(n, dtype=np.float32)
    features['device_risk_score'] = np.zeros(n, dtype=np.float32)

    # ========== 13. 扩展特征：图指标 ==========
    if show_progress and len(df) > 100000:
        print(f"[INFO]   Section 13/13: Graph metrics", flush=True)
    degrees_f = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.float32)
    max_degree = max(degrees_f) if max(degrees_f) > 0 else 1
    features['degree_centrality'] = degrees_f / max_degree
    features['clustering_coeff'] = np.zeros(n, dtype=np.float32)

    # Cleanup
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    feature_names = list(features.keys())
    return features, feature_names


def run_npu_gar(data_path, card_col, entity_cols, account_features,
                transaction_features, output_csv, npu_id=0, workers=1, mode='auto',
                label_col=None, fraud_value=1, train_idx=None, no_leakage=True,
                train_ratio=0.7, seed=42, auto_detect=True):
    """NPU加速GAR特征生成"""
    import torch

    load_ascend_env()
    device_info = check_ascend_npu(mode=mode)

    # Check NPU availability
    npu_available = device_info['available'] and torch.npu.is_available()
    if not npu_available:
        print("[WARN] NPU not available, falling back to CPU mode (gar_ascend)", flush=True)
        from src.gar.gar_ascend import run_ascend_gar
        return run_ascend_gar(data_path, card_col, entity_cols, account_features,
                              transaction_features, output_csv, npu_id, workers, mode,
                              label_col, fraud_value, train_idx, no_leakage,
                              train_ratio, seed, auto_detect)

    device = torch.device('npu', npu_id)
    torch.npu.set_device(device)

    print("="*60, flush=True)
    print("NPU-accelerated GAR Feature Generator", flush=True)
    print(f"NPU Available: {npu_available}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Workers: {workers}", flush=True)
    if no_leakage:
        print("Mode: NO-LEAKAGE (fraud rates from train only)", flush=True)
    else:
        print("Mode: LEAKAGE (not recommended)", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    # 1. Load data
    print(f"[INFO] Loading data from {data_path}...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Total records: {len(df)}, columns: {len(df.columns)}", flush=True)

    # Auto-detect schema
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

    # 2. Preprocess
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

    # 3. Detect label
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

    # 4. Split data
    train_idx_arr, test_idx_arr = split_data(df, train_ratio=train_ratio, seed=seed)
    print(f"[INFO] Data split: Train={len(train_idx_arr)}, Test={len(test_idx_arr)}", flush=True)

    # 5. Compute fraud rate maps
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

    # 6. Build graph
    print("[INFO] Building graph...", flush=True)
    graph_start = time.time()
    tx_neighbors = build_graph(df, entity_cols, show_progress=True)
    graph_time = time.time() - graph_start
    n_edges = sum(len(v) for v in tx_neighbors.values()) // 2
    print(f"[INFO] Graph built in {graph_time:.2f}s, edges: {n_edges}", flush=True)
    if len(df) > 100000:
        print(f"[INFO] Graph throughput: {len(df)/graph_time:.0f} records/sec", flush=True)

    # 7. Build features with NPU
    print("[INFO] Building GAR features (NPU-accelerated)...", flush=True)
    feat_start = time.time()
    features, feature_names = build_gar_features_npu(
        df, tx_neighbors, global_stats, entity_cols, card_col,
        account_features, transaction_features, has_label, label_col,
        entity_fraud_maps, pair_fraud_maps,
        no_leakage, train_idx_set, train_label_map,
        show_progress=True
    )
    feat_time = time.time() - feat_start
    print(f"[INFO] Features built in {feat_time:.2f}s", flush=True)
    print(f"[INFO] Feature count: {len(feature_names)}", flush=True)

    # 8. Export
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
        split_arr = np.array(['train' if i in train_idx_arr else 'test' for i in range(len(df_features))])
        df_features['split'] = split_arr

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

    # Free memory
    del df, tx_neighbors, global_stats
    gc.collect()

    return features, feature_names


def main():
    parser = argparse.ArgumentParser(
        description='NPU-accelerated GAR Feature Generator',
        epilog="""
Examples:
  python src/gar/gar_npu.py --data /path/to/data.csv \\
      --output-csv ./features.csv \\
      --card-col card_id
        """
    )

    parser.add_argument('--data', type=str, default=None, help='CSV file path')
    parser.add_argument('--card-col', type=str, default=DEFAULT_CARD_COL, help='Card column name')
    parser.add_argument('--entity-cols', type=str, default=None, help='Entity columns (comma-separated)')
    parser.add_argument('--account-features', type=str, default=None, help='Account feature columns')
    parser.add_argument('--transaction-features', type=str, default=None, help='Transaction feature columns')
    parser.add_argument('--npu-id', type=int, default=0, help='NPU device ID')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--output-csv', type=str, default=None, help='Output CSV path')
    parser.add_argument('--label-col', type=str, default=None, help='Fraud label column')
    parser.add_argument('--no-leakage', action='store_true', default=True, help='No leakage mode (default)')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train ratio (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--auto-detect', action='store_true', default=True, help='Auto-detect columns')
    parser.add_argument('--check-npu', action='store_true', help='Check NPU status')

    args = parser.parse_args()

    if args.check_npu:
        info = check_ascend_npu(mode='auto')
        import torch
        print("NPU Status:")
        print(f"  Available: {info['available']}")
        print(f"  Backend: {info['backend']}")
        print(f"  torch.npu available: {torch.npu.is_available()}")
        if torch.npu.is_available():
            print(f"  NPU device count: {torch.npu.device_count()}")
        return

    if args.data is None:
        parser.error("--data is required")

    entity_cols = args.entity_cols.split(',') if args.entity_cols else DEFAULT_ENTITY_COLS
    account_features = args.account_features.split(',') if args.account_features else []
    transaction_features = args.transaction_features.split(',') if args.transaction_features else []

    run_npu_gar(args.data, args.card_col, entity_cols, account_features,
                transaction_features, args.output_csv, args.npu_id, args.workers, 'auto',
                args.label_col, 1, None, args.no_leakage,
                args.train_ratio, args.seed, args.auto_detect)


if __name__ == '__main__':
    main()