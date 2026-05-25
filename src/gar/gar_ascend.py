"""
Ascend NPU加速GAR特征生成器

支持Ascend NPU卡加速，处理千万到亿级交易记录。

用法:
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

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import sys
import argparse
import time
import subprocess

sys.stdout.reconfigure(line_buffering=True)

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
DEFAULT_ENTITY_COLS = ['card_id', 'merchant_id', 'device_type', 'transaction_type']
DEFAULT_ACCOUNT_FEATURES = ['card_level', 'issuing_bank']
DEFAULT_TRANSACTION_FEATURES = ['amount', 'balance_after', 'timestamp', 'is_pos', 'is_cross_border']
DEFAULT_NEIGHBOR_THRESHOLD = 300


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
        device_info['device_count'] = ret if ret > 0 else 0
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


def build_graph(df, entity_cols, neighbor_threshold=300):
    """构建稀疏图结构"""
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


def build_gar_features_optimized(df, tx_neighbors, global_stats, entity_cols, card_col,
                                  account_features, transaction_features, has_label, label_col):
    """构建GAR特征（优化版本：向量化+稀疏图）"""

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

    # ========== 2. 实体频率（向量化） ==========
    for col in entity_cols:
        if col not in df_columns:
            continue
        values = df[col].values
        freq_map = global_stats.get(f'{col}_freq', {})
        features[f'{col}_freq'] = np.array([freq_map.get(v, 0) for v in values], dtype=np.float32)
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq']).astype(np.float32)

    # ========== 3. 卡号聚合特征（向量化优化） ==========
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

        for card in unique_cards:
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

    # ========== 5. 图特征（优化：预计算度） ==========
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
        for eid in entity_ids:
            mask = col_values == eid
            entity_degrees[eid] = degrees[mask].mean() if mask.sum() > 0 else 0
        features[f'{col}_degree'] = np.array([entity_degrees.get(v, 0) for v in col_values], dtype=np.float32)

    # 1跳邻居金额统计
    if amount_col:
        amounts = df[amount_col].fillna(0).values.astype(np.float32)
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

    # ========== 6. 配对频率（向量化） ==========
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
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_fraud_rates[i] = labels[list(neighs)].mean()
        features['neigh_fraud_rate'] = neigh_fraud_rates

    # 清理
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    return features


def run_ascend_gar(data_path, card_col, entity_cols, account_features,
                   transaction_features, output_csv, npu_id=0, workers=1, mode='auto'):
    """Ascend NPU加速GAR特征生成"""
    device_info = check_ascend_npu(mode=mode)

    print("="*60, flush=True)
    print("Ascend NPU GAR Feature Generator", flush=True)
    print(f"NPU Available: {device_info['available']}", flush=True)
    print(f"Backend: {device_info['backend']}", flush=True)
    print(f"Workers: {workers}", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    print(f"[INFO] Loading data from {data_path}...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Total records: {len(df)}", flush=True)

    for col in entity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col].astype(str))[0]

    has_label = False
    label_col = None
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

    global_stats = compute_global_stats(df, entity_cols, card_col, amount_col)

    print("[INFO] Building graph...", flush=True)
    graph_start = time.time()
    tx_neighbors = build_graph(df, entity_cols)
    graph_time = time.time() - graph_start
    n_edges = sum(len(v) for v in tx_neighbors.values()) // 2
    print(f"[INFO] Graph built in {graph_time:.2f}s, edges: {n_edges}", flush=True)

    print("[INFO] Building GAR features...", flush=True)
    feat_start = time.time()
    features = build_gar_features_optimized(df, tx_neighbors, global_stats, entity_cols, card_col,
                                            account_features, transaction_features, has_label, label_col)
    feat_time = time.time() - feat_start
    print(f"[INFO] Features built in {feat_time:.2f}s", flush=True)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        df_features = pd.DataFrame(features)
        key_cols = [c for c in [card_col, 'timestamp', '时间戳'] if c in df.columns]
        if key_cols:
            df_features = pd.concat([df[key_cols], df_features], axis=1)
        if has_label and label_col:
            df_features[label_col] = df[label_col].values
        df_features.to_csv(output_csv, index=False)
        print(f"[INFO] Saved to {output_csv}", flush=True)
        print(f"[INFO] Shape: {df_features.shape}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n[INFO] Total time: {elapsed/60:.1f} minutes", flush=True)
    print(f"[INFO] Throughput: {len(df)/elapsed:.0f} records/sec", flush=True)

    return features


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
                        help='CSV文件路径（使用--check-npu时可选）')
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
                   transaction_features, args.output_csv, args.npu_id, args.workers, args.mode)


if __name__ == '__main__':
    main()