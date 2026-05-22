"""
统一特征生成入口

自动选择最佳模式：CPU / GPU / 分布式

用法:
    # 自动选择模式（小数据用CPU，大数据用多进程）
    python src/feature_generator.py --data /path/to/data.csv --card-col card_id

    # 强制使用特定模式
    python src/feature_generator.py --data /path/to/data.csv --card-col card_id --mode gpu
    python src/feature_generator.py --data /path/to/data.csv --card-col card_id --mode distributed --workers 16
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

# 默认配置
DEFAULT_CARD_COL = 'card_id'
DEFAULT_ENTITY_COLS = ['card_id', 'merchant_id', 'device_type', 'transaction_type']
DEFAULT_ACCOUNT_FEATURES = ['card_level', 'issuing_bank']
DEFAULT_TRANSACTION_FEATURES = ['amount', 'balance_after', 'timestamp', 'is_pos', 'is_cross_border']

# 数据规模阈值
LARGE_DATA_THRESHOLD = 500000   # 50万以上使用分布式
GPU_MEMORY_THRESHOLD = 4         # 4GB以上考虑GPU

# 分布式分区数
DISTRIBUTED_PARTITIONS = 8


def check_gpu():
    """检测GPU可用性"""
    try:
        import cupy as cp
        if cp.cuda.is_available():
            return True, cp.cuda.Device(0).meminfo._get_limit() / (1024**3)
    except (ImportError, AttributeError):
        pass
    return False, 0


def compute_global_stats(df, entity_cols, card_col, amount_col=None):
    """计算全局统计量"""
    global_stats = {}

    # 实体频率
    for col in entity_cols:
        if col in df.columns:
            global_stats[f'{col}_freq'] = df[col].value_counts().to_dict()

    # 卡号聚合统计
    if card_col in df.columns:
        global_stats['card_tx_count'] = df[card_col].value_counts().to_dict()

        if amount_col and amount_col in df.columns:
            card_agg = df.groupby(card_col)[amount_col].agg(['mean', 'std', 'max', 'count'])
            card_agg.columns = ['amt_mean', 'amt_std', 'amt_max', 'tx_count']
            global_stats['card_agg'] = card_agg.to_dict('index')

    return global_stats


def build_features_single(df, tx_neighbors, global_stats, entity_cols, card_col,
                          account_features, transaction_features):
    """在单个进程/分区上构建特征"""

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
        values = df[col].values

        if df[col].dtype in ['int64', 'float64']:
            features[col] = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
            if amount_col and col == amount_col:
                features[f'{col}_log'] = np.log1p(np.abs(features[col]))
        else:
            unique_vals = np.unique(values)
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            features[col] = np.array([val_to_idx.get(v, 0) for v in values])

    # ========== 2. 实体频率 ==========
    for col in entity_cols:
        if col not in df_columns:
            continue
        values = df[col].values
        freq_map = global_stats.get(f'{col}_freq', {}) if global_stats else {}
        features[f'{col}_freq'] = np.array([freq_map.get(v, 0) for v in values])
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq'])

    # ========== 3. 卡号聚合特征 ==========
    if card_col in df_columns:
        card_values = df[card_col].values

        card_counts = global_stats.get('card_tx_count', {}) if global_stats else {}
        features['card_tx_count'] = np.array([card_counts.get(v, 0) for v in card_values])
        features['card_tx_count_log'] = np.log1p(features['card_tx_count'])

        card_agg = global_stats.get('card_agg', {}) if global_stats else {}
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
        features['card_amt_mean'] = np.array(amt_mean)
        features['card_amt_std'] = np.array(amt_std)
        features['card_amt_max'] = np.array(amt_max)

        if amount_col:
            amounts = df[amount_col].values
            features['amt_to_card_mean_ratio'] = amounts / (np.array(amt_mean) + 1)

    # ========== 4. 账户级特征 ==========
    for col in account_features:
        if col not in df_columns:
            continue
        values = df[col].values

        if df[col].dtype == 'object':
            unique_vals = np.unique(values)
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            features[col] = np.array([val_to_idx.get(v, 0) for v in values])
        else:
            features[col] = np.nan_to_num(values, nan=-1, posinf=0, neginf=0)

    # ========== 5. 图特征 ==========
    # 度
    for col in entity_cols:
        if col not in df_columns:
            continue
        degrees = [len(tx_neighbors.get(i, set())) for i in range(n)]
        features[f'{col}_degree'] = np.array(degrees)

    # 1-hop
    n_1hop = [len(tx_neighbors.get(i, set())) for i in range(n)]
    features['n_1hop'] = np.array(n_1hop)
    features['n_1hop_log'] = np.log1p(features['n_1hop'])

    # 邻居金额统计
    if amount_col:
        amounts = df[amount_col].values
        amt_1hop_mean, amt_1hop_std = [], []
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = amounts[list(neighs)]
                amt_1hop_mean.append(np.mean(neigh_amts))
                amt_1hop_std.append(np.std(neigh_amts) if len(neighs) > 1 else 0)
            else:
                amt_1hop_mean.append(0)
                amt_1hop_std.append(0)
        features['amt_1hop_mean'] = np.array(amt_1hop_mean)
        features['amt_1hop_std'] = np.array(amt_1hop_std)

    # 2-hop
    tx_2hop = defaultdict(set)
    for tx, neighs in tx_neighbors.items():
        for neighbor in neighs:
            tx_2hop[tx].update(tx_neighbors.get(neighbor, set()))
    tx_2hop[tx].discard(tx)

    n_2hop = [len(tx_2hop.get(i, set())) for i in range(n)]
    features['n_2hop'] = np.array(n_2hop)
    features['2hop_1hop_ratio'] = features['n_2hop'] / (features['n_1hop'] + 1)

    # ========== 6. 配对频率 ==========
    for i, col1 in enumerate(entity_cols[:4]):
        for col2 in entity_cols[i+1:5]:
            if col1 not in df_columns or col2 not in df_columns:
                continue
            vals1 = df[col1].astype(str).values
            vals2 = df[col2].astype(str).values
            pairs = np.char.add(np.char.add(vals1, '_'), vals2)
            pair_counts = pd.Series(pairs).value_counts().to_dict()
            features[f'{col1}_{col2}_pair_freq'] = np.array([pair_counts.get(p, 0) for p in pairs])
            features[f'{col1}_{col2}_pair_freq_log'] = np.log1p(features[f'{col1}_{col2}_pair_freq'])

    # ========== 7. 时序特征 ==========
    for col in ['timestamp', '时间戳', 'trans_time', 'transaction_time']:
        if col in df_columns:
            try:
                ts = pd.to_datetime(df[col], errors='coerce')
                if not ts.isna().all():
                    features['trans_hour'] = ts.dt.hour.fillna(12).values
                    features['trans_dayofweek'] = ts.dt.dayofweek.fillna(0).values
            except:
                pass
            break

    # ========== 8. 欺诈率特征 ==========
    label_col = None
    for col in ['isFraud', 'fraud', 'label', 'is_fraud']:
        if col in df_columns:
            label_col = col
            break

    if label_col:
        labels = df[label_col].values

        for col in entity_cols:
            if col not in df_columns:
                continue
            fraud_map = df.groupby(col)[label_col].mean().to_dict()
            features[f'{col}_fraud_rate'] = np.array([fraud_map.get(v, 0) for v in df[col].values])

        neigh_fraud_rates = []
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_fraud_rates.append(labels[list(neighs)].mean())
            else:
                neigh_fraud_rates.append(0)
        features['neigh_fraud_rate'] = np.array(neigh_fraud_rates)

    # 清理
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    return features


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


def run_mode(args):
    """并行worker函数"""
    (partition_id, df_dict, global_stats, entity_cols, card_col,
     account_features, transaction_features) = args

    df = pd.DataFrame(df_dict)
    tx_neighbors = build_graph(df, entity_cols)
    features = build_features_single(df, tx_neighbors, global_stats, entity_cols, card_col,
                                     account_features, transaction_features)

    return features, list(df.columns), partition_id


def run_cpu(data_path, card_col, entity_cols, account_features, transaction_features, output_csv):
    """CPU模式"""
    print("[MODE] CPU (single process)", flush=True)

    start_time = time.time()

    # 加载数据
    print(f"[INFO] Loading data...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Total records: {len(df)}", flush=True)

    # 预处理
    for col in entity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col].astype(str))[0]

    # 全局统计量
    amount_col = None
    for col in ['amount', '交易金额']:
        if col in df.columns:
            amount_col = col
            break

    global_stats = compute_global_stats(df, entity_cols, card_col, amount_col)

    # 构建图和特征
    print("[INFO] Building graph...", flush=True)
    tx_neighbors = build_graph(df, entity_cols)

    print("[INFO] Building features...", flush=True)
    features = build_features_single(df, tx_neighbors, global_stats, entity_cols, card_col,
                                     account_features, transaction_features)

    # 导出
    if output_csv:
        df_features = pd.DataFrame(features)
        key_cols = [c for c in [card_col, 'timestamp', '时间戳'] if c in df.columns]
        if key_cols:
            df_features = pd.concat([df[key_cols], df_features], axis=1)
        label_col = next((c for c in ['isFraud', 'fraud', 'label'] if c in df.columns), None)
        if label_col:
            df_features[label_col] = df[label_col].values
        df_features.to_csv(output_csv, index=False)
        print(f"[INFO] Saved to {output_csv}", flush=True)

    print(f"[TIME] CPU mode: {(time.time()-start_time)/60:.1f} min", flush=True)
    return features


def run_distributed(data_path, card_col, entity_cols, account_features, transaction_features,
                    n_workers, output_csv):
    """分布式模式（多进程）"""
    print(f"[MODE] Distributed ({n_workers} workers)", flush=True)

    start_time = time.time()

    # 加载全部数据（用于全局统计）
    print("[INFO] Loading data for global stats...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Total records: {len(df)}", flush=True)

    # 预处理
    for col in entity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col].astype(str))[0]

    amount_col = None
    for col in ['amount', '交易金额']:
        if col in df.columns:
            amount_col = col
            break

    # 全局统计量
    global_stats = compute_global_stats(df, entity_cols, card_col, amount_col)

    # 分区
    print(f"[INFO] Partitioning into {n_workers} shards...", flush=True)
    df['_p'] = df[card_col].apply(lambda x: hash(x) % n_workers)
    partitions = []
    for p in range(n_workers):
        mask = df['_p'] == p
        partitions.append(df[mask].drop(columns=['_p']).reset_index(drop=True))
        print(f"[INFO] Partition {p}: {len(partitions[-1])} records", flush=True)

    del df

    # 并行处理
    print("[INFO] Starting parallel processing...", flush=True)
    args_list = [
        (p_id, p.to_dict('list'), global_stats, entity_cols, card_col,
         account_features, transaction_features)
        for p_id, p in enumerate(partitions)
    ]

    with Pool(n_workers) as pool:
        results = pool.map(run_mode, args_list)

    # 合并
    print("[INFO] Merging results...", flush=True)
    all_features = {}
    for features, _, _ in results:
        for k, v in features.items():
            if k not in all_features:
                all_features[k] = []
            all_features[k].append(v)

    for k in all_features:
        all_features[k] = np.concatenate(all_features[k])

    # 导出
    if output_csv:
        df_features = pd.DataFrame(all_features)
        # 添加关键列
        all_dfs = pd.concat(partitions, ignore_index=True)
        key_cols = [c for c in [card_col, 'timestamp', '时间戳'] if c in all_dfs.columns]
        if key_cols:
            df_features = pd.concat([all_dfs[key_cols], df_features], axis=1)
        label_col = next((c for c in ['isFraud', 'fraud', 'label'] if c in all_dfs.columns), None)
        if label_col:
            df_features[label_col] = all_dfs[label_col].values
        df_features.to_csv(output_csv, index=False)
        print(f"[INFO] Saved to {output_csv}", flush=True)

    print(f"[TIME] Distributed mode: {(time.time()-start_time)/60:.1f} min", flush=True)
    return all_features


def main():
    parser = argparse.ArgumentParser(
        description='Knowledge Graph Feature Generator - Auto Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 自动选择模式
  python src/feature_generator.py --data /path/to/data.csv --card-col card_id

  # 强制使用分布式
  python src/feature_generator.py --data /path/to/large_data.csv --card-col card_id \\
                                   --mode distributed --workers 16

  # 指定输出
  python src/feature_generator.py --data /path/to/data.csv --card-col card_id \\
                                   --output-csv ./features.csv

  # 自定义特征
  python src/feature_generator.py --data /path/to/data.csv --card-col card_id \\
                                   --entity-cols card_id,merchant_id,device_type \\
                                   --account-features card_level,issuing_bank \\
                                   --transaction-features amount,balance_after \\
                                   --output-csv ./features.csv
        """
    )

    parser.add_argument('--data', type=str, required=True,
                        help='CSV文件路径')
    parser.add_argument('--card-col', type=str, default=DEFAULT_CARD_COL,
                        help=f'卡号列名（默认: {DEFAULT_CARD_COL}）')
    parser.add_argument('--entity-cols', type=str, default=None,
                        help='实体列名列表，逗号分隔')
    parser.add_argument('--account-features', type=str, default=None,
                        help='账户级特征列名，逗号分隔')
    parser.add_argument('--transaction-features', type=str, default=None,
                        help='交易级特征列名，逗号分隔')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'cpu', 'gpu', 'distributed'],
                        help='运行模式: auto/cpu/gpu/distributed（默认: auto）')
    parser.add_argument('--workers', type=int, default=None,
                        help='分布式worker数量（默认: CPU核心数）')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU设备ID（默认: 0）')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='特征CSV输出路径')

    args = parser.parse_args()

    entity_cols = args.entity_cols.split(',') if args.entity_cols else DEFAULT_ENTITY_COLS
    account_features = args.account_features.split(',') if args.account_features else []
    transaction_features = args.transaction_features.split(',') if args.transaction_features else []
    n_workers = args.workers or cpu_count()

    print("="*60, flush=True)
    print("Knowledge Graph Feature Generator", flush=True)
    print(f"Data: {args.data}", flush=True)
    print(f"Card column: {args.card_col}", flush=True)
    print("="*60, flush=True)

    # 估计数据规模
    n_rows = sum(1 for _ in open(args.data)) - 1

    # 自动选择模式
    if args.mode == 'auto':
        if n_rows > LARGE_DATA_THRESHOLD:
            mode = 'distributed'
            print(f"[AUTO] Data size {n_rows:,} > {LARGE_DATA_THRESHOLD:, }, using distributed mode", flush=True)
        else:
            mode = 'cpu'
            print(f"[AUTO] Data size {n_rows:,} <= {LARGE_DATA_THRESHOLD:, }, using CPU mode", flush=True)
    else:
        mode = args.mode

    # 运行
    if mode == 'distributed':
        run_distributed(args.data, args.card_col, entity_cols, account_features,
                        transaction_features, n_workers, args.output_csv)
    elif mode == 'gpu':
        print("[WARN] GPU mode not fully implemented, falling back to CPU", flush=True)
        run_cpu(args.data, args.card_col, entity_cols, account_features,
                transaction_features, args.output_csv)
    else:
        run_cpu(args.data, args.card_col, entity_cols, account_features,
                transaction_features, args.output_csv)


if __name__ == '__main__':
    main()