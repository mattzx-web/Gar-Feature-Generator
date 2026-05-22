"""
分布式GAR特征生成器

支持多进程加速，处理千万到亿级交易记录。

用法:
    # 多进程分布式
    python src/gar_feature_generator_dist.py --data /path/to/data.csv \
                                              --card-col card_id \
                                              --workers 8 \
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
DEFAULT_ENTITY_COLS = ['card_id', 'merchant_id', 'device_type', 'transaction_type']
DEFAULT_ACCOUNT_FEATURES = ['card_level', 'issuing_bank']
DEFAULT_TRANSACTION_FEATURES = ['amount', 'balance_after', 'timestamp', 'is_pos', 'is_cross_border']
DEFAULT_NEIGHBOR_THRESHOLD = 300


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


def build_gar_features(df, tx_neighbors, global_stats, entity_cols, card_col,
                        account_features, transaction_features, has_label, label_col):
    """构建GAR特征（通用版本）"""

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
            pairs = np.char.add(np.char.add(vals1, '_'), vals2)
            pair_counts = pd.Series(pairs).value_counts().to_dict()
            features[f'{col1}_{col2}_pair_freq'] = np.array([pair_counts.get(p, 0) for p in pairs], dtype=np.float32)
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

    # ========== 8. GAR Fraud Rate特征（仅当有标签时） ==========
    if has_label and label_col:
        labels = df[label_col].values

        # Entity Fraud Rates
        for col in entity_cols:
            if col not in df_columns:
                continue
            fraud_map = df.groupby(col)[label_col].mean().to_dict()
            features[f'{col}_fraud_rate'] = np.array([fraud_map.get(v, 0) for v in df[col].values], dtype=np.float32)

        # Pair Fraud Rates
        for i, col1 in enumerate(entity_cols[:4]):
            for col2 in entity_cols[i+1:5]:
                if col1 not in df_columns or col2 not in df_columns:
                    continue
                pair_df = df[[col1, col2, label_col]].copy()
                pair_df['_pair'] = pair_df[col1].astype(str) + '_' + pair_df[col2].astype(str)
                fraud_map = pair_df.groupby('_pair')[label_col].mean().to_dict()
                pair_values = df[col1].astype(str) + '_' + df[col2].astype(str)
                features[f'{col1}_{col2}_pair_fraud_rate'] = np.array([fraud_map.get(p, 0) for p in pair_values], dtype=np.float32)

        # Neighbor Fraud Rate
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


def process_partition(args):
    """并行worker函数"""
    (p_id, df_dict, global_stats, entity_cols, card_col,
     account_features, transaction_features, has_label, label_col) = args

    df = pd.DataFrame(df_dict)
    tx_neighbors = build_graph(df, entity_cols)
    features = build_gar_features(df, tx_neighbors, global_stats, entity_cols, card_col,
                                   account_features, transaction_features, has_label, label_col)

    return features, list(df.columns), p_id


def run_distributed(data_path, card_col, entity_cols, account_features,
                    transaction_features, n_workers, output_csv):
    """分布式GAR特征生成"""
    print(f"[MODE] Distributed GAR ({n_workers} workers)", flush=True)
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

    # 3. 全局统计量
    global_stats = compute_global_stats(df, entity_cols, card_col, amount_col)

    # 4. 分区
    print(f"[INFO] Partitioning into {n_workers} shards...", flush=True)
    df['_p'] = df[card_col].apply(lambda x: hash(x) % n_workers)
    partitions = []
    for p in range(n_workers):
        mask = df['_p'] == p
        partitions.append(df[mask].drop(columns=['_p']).reset_index(drop=True))
        print(f"[INFO] Partition {p}: {len(partitions[-1])} records", flush=True)
    del df

    # 5. 并行处理
    print("[INFO] Parallel processing...", flush=True)
    args_list = [
        (p_id, p.to_dict('list'), global_stats, entity_cols, card_col,
         account_features, transaction_features, has_label, label_col)
        for p_id, p in enumerate(partitions)
    ]

    with Pool(n_workers) as pool:
        results = pool.map(process_partition, args_list)

    # 6. 合并
    print("[INFO] Merging results...", flush=True)
    all_features = {}
    for features, _, _ in results:
        for k, v in features.items():
            if k not in all_features:
                all_features[k] = []
            all_features[k].append(v)

    for k in all_features:
        all_features[k] = np.concatenate(all_features[k])

    # 7. 导出
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        df_features = pd.DataFrame(all_features)
        all_dfs = pd.concat(partitions, ignore_index=True)
        key_cols = [c for c in [card_col, 'timestamp', '时间戳'] if c in all_dfs.columns]
        if key_cols:
            df_features = pd.concat([all_dfs[key_cols], df_features], axis=1)
        if has_label and label_col:
            df_features[label_col] = all_dfs[label_col].values
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
  # 分布式模式
  python src/gar_feature_generator_dist.py --data /path/to/large_data.csv \\
                                            --card-col card_id \\
                                            --workers 16 \\
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

    args = parser.parse_args()

    entity_cols = args.entity_cols.split(',') if args.entity_cols else DEFAULT_ENTITY_COLS
    account_features = args.account_features.split(',') if args.account_features else []
    transaction_features = args.transaction_features.split(',') if args.transaction_features else []

    run_distributed(args.data, args.card_col, entity_cols, account_features,
                    transaction_features, args.workers, args.output_csv)


if __name__ == '__main__':
    main()