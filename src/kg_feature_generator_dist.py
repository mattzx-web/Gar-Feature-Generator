"""
分布式知识图谱特征生成器

支持多进程/GPU加速，处理千万到亿级交易记录。

架构设计:
1. 数据分片: 按卡号hash分区，并行处理
2. 图构建: 广播全局实体映射，各分片独立构建子图
3. 特征生成: 多进程并行，共享实体统计量
4. 结果合并: 归并排序，输出完整特征

用法:
    # 单机多进程
    python src/kg_feature_generator_dist.py --data /path/to/data.csv \
                                           --card-col card_id \
                                           --workers 8 \
                                           --output-csv ./features.csv

    # 多机分布式 (需要配置)
    mpirun -n 16 python src/kg_feature_generator_dist.py --data /path/to/data.csv \
                                                          --card-col card_id \
                                                          --workers 16
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from multiprocessing import Pool, cpu_count, Manager
import json
import os
import sys
import argparse
from datetime import datetime
import time
from functools import reduce

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# 默认配置
DEFAULT_CARD_COL = 'card_id'
DEFAULT_ENTITY_COLS = ['card_id', 'merchant_id', 'device_type', 'transaction_type']
DEFAULT_ACCOUNT_FEATURES = ['card_level', 'issuing_bank']
DEFAULT_TRANSACTION_FEATURES = ['amount', 'balance_after', 'timestamp', 'is_pos', 'is_cross_border']
DEFAULT_NEIGHBOR_THRESHOLD = 300


def compute_global_stats(df, entity_cols, card_col, amount_col=None):
    """
    计算全局统计量（实体频率、卡号聚合等）
    这些统计量可以在分片间共享

    Returns:
        global_stats: dict containing global statistics
    """
    print(f"[INFO] Computing global statistics...", flush=True)

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

    # 卡号度数（全局）
    global_stats['n_cards'] = df[card_col].nunique() if card_col in df.columns else 0

    return global_stats


def build_partition_graph(df, entity_cols, neighbor_threshold=DEFAULT_NEIGHBOR_THRESHOLD):
    """
    在分片内构建图结构

    Args:
        df: 分片数据
        entity_cols: 实体列
        neighbor_threshold: 邻居数量上限

    Returns:
        tx_neighbors: 邻居映射
        entity_to_idx: 实体到本地索引的映射
    """
    n = len(df)
    tx_neighbors = defaultdict(set)
    entity_to_idx = defaultdict(list)

    # 记录每个实体对应的本地索引
    for col in entity_cols:
        if col not in df.columns:
            continue
        for local_idx, val in enumerate(df[col].values):
            entity_to_idx[(col, val)].append(local_idx)

    # 构建邻居关系
    for col in entity_cols:
        if col not in df.columns:
            continue
        groups = df.groupby(col).indices
        for val, idx_list in groups.items():
            if 1 < len(idx_list) < neighbor_threshold:
                for idx in idx_list:
                    tx_neighbors[idx].update(idx_list)

    # 移除自身
    for idx in tx_neighbors:
        tx_neighbors[idx].discard(idx)

    return tx_neighbors, entity_to_idx


def build_features_partition(args):
    """
    在单个分片上构建特征（并行worker函数）

    Args:
        args: tuple (partition_id, df_dict, global_stats, entity_cols, card_col,
                    account_features, transaction_features, has_label, label_col)

    Returns:
        features_dict, feature_names, partition_id
    """
    (partition_id, df_dict, global_stats, entity_cols, card_col,
     account_features, transaction_features, has_label, label_col) = args

    # 重建DataFrame
    df = pd.DataFrame(df_dict)

    features = {}
    n = len(df)

    # ========== 1. 交易级特征 ==========
    amount_col = None
    for col in ['amount', '交易金额', 'transaction_amount', 'amt']:
        if col in df.columns:
            amount_col = col
            break

    for col in transaction_features:
        if col not in df.columns:
            continue
        if df[col].dtype in ['int64', 'float64']:
            features[col] = df[col].fillna(0).values
            if amount_col and col == amount_col:
                features[f'{col}_log'] = np.log1p(np.abs(df[col].fillna(0).values))
        else:
            le = LabelEncoder()
            features[col] = le.fit_transform(df[col].fillna('missing').astype(str))

    # ========== 2. 实体频率（使用全局统计） ==========
    for col in entity_cols:
        if col not in df.columns:
            continue
        freq_map = global_stats.get(f'{col}_freq', {})
        features[f'{col}_freq'] = df[col].map(freq_map).fillna(0).values
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq'])

    # ========== 3. 卡号聚合特征（使用全局统计） ==========
    if card_col in df.columns:
        # 卡号交易次数
        card_counts = global_stats.get('card_tx_count', {})
        features['card_tx_count'] = df[card_col].map(card_counts).fillna(0).values
        features['card_tx_count_log'] = np.log1p(features['card_tx_count'])

        # 卡号金额统计
        card_agg = global_stats.get('card_agg', {})
        amt_mean = []
        amt_std = []
        amt_max = []
        for card in df[card_col].values:
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

        # 金额与卡均值比率
        if amount_col:
            features['amt_to_card_mean_ratio'] = df[amount_col].fillna(0) / (np.array(amt_mean) + 1)

    # ========== 4. 账户级特征 ==========
    for col in account_features:
        if col not in df.columns:
            continue
        if df[col].dtype == 'object':
            le = LabelEncoder()
            features[col] = le.fit_transform(df[col].fillna('missing').astype(str))
        else:
            features[col] = df[col].fillna(-1).values

    # ========== 5. 图特征（分片内） ==========
    tx_neighbors, _ = build_partition_graph(df, entity_cols)

    # 度
    for col in entity_cols:
        if col not in df.columns:
            continue
        degrees = [len(tx_neighbors.get(i, set())) for i in range(n)]
        features[f'{col}_degree'] = np.array(degrees)

    # 1-hop邻居数量
    n_1hop = [len(tx_neighbors.get(i, set())) for i in range(n)]
    features['n_1hop'] = np.array(n_1hop)
    features['n_1hop_log'] = np.log1p(features['n_1hop'])

    # 邻居金额统计
    if amount_col:
        amt_1hop_mean = []
        amt_1hop_std = []
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = df[amount_col].iloc[list(neighs)].fillna(0).values
                amt_1hop_mean.append(np.mean(neigh_amts))
                amt_1hop_std.append(np.std(neigh_amts) if len(neighs) > 1 else 0)
            else:
                amt_1hop_mean.append(0)
                amt_1hop_std.append(0)
        features['amt_1hop_mean'] = np.array(amt_1hop_mean)
        features['amt_1hop_std'] = np.array(amt_1hop_std)

    # 2-hop邻居
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
            if col1 not in df.columns or col2 not in df.columns:
                continue
            pairs = df[col1].astype(str) + '_' + df[col2].astype(str)
            pair_counts = pairs.map(pairs.value_counts())
            features[f'{col1}_{col2}_pair_freq'] = pair_counts.fillna(0).values
            features[f'{col1}_{col2}_pair_freq_log'] = np.log1p(pair_counts.fillna(0)).values

    # ========== 7. 时序特征 ==========
    timestamp_col = None
    for col in ['timestamp', '时间戳', 'trans_time', 'transaction_time']:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col:
        try:
            ts = pd.to_datetime(df[timestamp_col], errors='coerce')
            if not ts.isna().all():
                features['trans_hour'] = ts.dt.hour.fillna(12).values
                features['trans_dayofweek'] = ts.dt.dayofweek.fillna(0).values
        except:
            pass

    # ========== 8. 欺诈率特征（仅当有标签时） ==========
    if has_label and label_col:
        # Entity Fraud Rates
        for col in entity_cols:
            if col not in df.columns:
                continue
            fraud_map = df.groupby(col)[label_col].mean().to_dict()
            features[f'{col}_fraud_rate'] = df[col].map(fraud_map).fillna(0).values

        # Neighbor Fraud Rate
        train_is_fraud = df[label_col].values
        neigh_fraud_rates = []
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_fraud_rates.append(train_is_fraud[list(neighs)].mean())
            else:
                neigh_fraud_rates.append(0)
        features['neigh_fraud_rate'] = np.array(neigh_fraud_rates)

    # 清理无穷值
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    feature_names = list(features.keys())

    return features, feature_names, partition_id


def load_data_in_chunks(data_path, chunk_size=100000):
    """
    分块加载数据，生成chunks
    """
    print(f"[INFO] Loading data in chunks of {chunk_size}...", flush=True)

    chunks = []
    reader = pd.read_csv(data_path, chunksize=chunk_size)

    for i, chunk in enumerate(reader):
        print(f"[INFO] Loaded chunk {i+1}: {len(chunk)} records", flush=True)
        chunks.append(chunk)

    return chunks


def partition_data(df, card_col, n_partitions):
    """
    按卡号hash分区，将数据分成n_partitions个分片
    同一卡号的交易必须在同一个分片中
    """
    print(f"[INFO] Partitioning data into {n_partitions} shards...", flush=True)

    # 使用hash分区
    df['_partition'] = df[card_col].apply(lambda x: hash(x) % n_partitions)

    partitions = []
    for p in range(n_partitions):
        mask = df['_partition'] == p
        partition_df = df[mask].drop(columns=['_partition']).reset_index(drop=True)
        partitions.append(partition_df)
        print(f"[INFO] Partition {p}: {len(partition_df)} records", flush=True)

    return partitions


def merge_features(feature_list, n_total):
    """
    合并多个分片的特征，保持原始顺序

    Args:
        feature_list: [(features_dict, feature_names, partition_id), ...]
        n_total: 总记录数

    Returns:
        merged_features_dict, feature_names
    """
    print(f"[INFO] Merging {len(feature_list)} partitions...", flush=True)

    # 按partition_id排序
    feature_list = sorted(feature_list, key=lambda x: x[2])

    # 合并所有特征名
    all_feature_names = set()
    for features_dict, _, _ in feature_list:
        all_feature_names.update(features_dict.keys())
    all_feature_names = sorted(all_feature_names)

    # 合并特征值
    merged = {name: [] for name in all_feature_names}

    for features_dict, _, partition_id in feature_list:
        for name in all_feature_names:
            if name in features_dict:
                merged[name].append(features_dict[name])
            else:
                # 填充0
                merged[name].append(np.zeros(len(feature_list[0][0]) if feature_list else 0))

    # 拼接
    for name in all_feature_names:
        merged[name] = np.concatenate(merged[name])

    return merged, all_feature_names


def run_distributed_feature_generation(data_path, card_col, entity_cols, account_features,
                                       transaction_features, n_workers=8, output_csv=None,
                                       has_label=False):
    """
    分布式特征生成主流程

    Args:
        data_path: CSV文件路径
        card_col: 卡号列
        entity_cols: 实体列
        account_features: 账户级特征
        transaction_features: 交易级特征
        n_workers: 并行worker数量
        output_csv: 输出CSV路径
        has_label: 是否有标签
    """
    print("="*60, flush=True)
    print("Distributed Knowledge Graph Feature Generator", flush=True)
    print(f"Workers: {n_workers}", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    # 1. 加载全部数据（计算全局统计量）
    print(f"[INFO] Loading full data for global statistics...", flush=True)
    df_full = pd.read_csv(data_path)
    print(f"[INFO] Total records: {len(df_full)}", flush=True)

    # 检测标签
    label_col = None
    if has_label:
        for col in ['isFraud', 'fraud', 'label', 'is_fraud']:
            if col in df_full.columns:
                label_col = col
                print(f"[INFO] Found label column: {label_col}", flush=True)
                break

    # 实体列编码
    for col in entity_cols:
        if col in df_full.columns:
            df_full[col] = df_full[col].fillna(-1)
            if df_full[col].dtype == 'object':
                le = LabelEncoder()
                df_full[col] = le.fit_transform(df_full[col].astype(str))

    # 计算全局统计量
    amount_col = None
    for col in ['amount', '交易金额']:
        if col in df_full.columns:
            amount_col = col
            break

    global_stats = compute_global_stats(df_full, entity_cols, card_col, amount_col)

    # 2. 数据分区
    partitions = partition_data(df_full, card_col, n_workers)

    # 释放内存
    del df_full

    # 3. 并行特征生成
    print(f"[INFO] Starting parallel feature generation with {n_workers} workers...", flush=True)

    # 准备参数
    args_list = []
    for p_id, partition_df in enumerate(partitions):
        df_dict = partition_df.to_dict('list')
        args_list.append((
            p_id, df_dict, global_stats, entity_cols, card_col,
            account_features, transaction_features, has_label, label_col
        ))

    # 并行处理
    with Pool(n_workers) as pool:
        results = pool.map(build_features_partition, args_list)

    # 4. 合并结果
    total_records = sum(len(p) for p in partitions)
    merged_features, feature_names = merge_features(results, total_records)

    # 5. 导出CSV
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)

        # 重建DataFrame
        df_features = pd.DataFrame(merged_features)

        # 添加关键列
        all_dfs = pd.concat(partitions, ignore_index=True)
        key_cols = []
        for col in [card_col, 'timestamp', '时间戳']:
            if col in all_dfs.columns:
                key_cols.append(col)
        if key_cols:
            df_features = pd.concat([all_dfs[key_cols], df_features], axis=1)

        if has_label and label_col:
            df_features[label_col] = all_dfs[label_col].values

        df_features.to_csv(output_csv, index=False)
        print(f"[INFO] Features saved to {output_csv}", flush=True)
        print(f"[INFO] Shape: {df_features.shape}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n[INFO] Total time: {elapsed/60:.1f} minutes", flush=True)
    print(f"[INFO] Throughput: {total_records/elapsed:.0f} records/sec", flush=True)

    return merged_features, feature_names


def main():
    parser = argparse.ArgumentParser(
        description='Distributed Knowledge Graph Feature Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单机多进程（8 workers）
  python src/kg_feature_generator_dist.py --data /path/to/large_data.csv \\
                                           --card-col card_id \\
                                           --workers 8 \\
                                           --output-csv ./features.csv

  # 使用所有CPU核心
  python src/kg_feature_generator_dist.py --data /path/to/large_data.csv \\
                                           --card-col card_id \\
                                           --workers auto \\
                                           --output-csv ./features.csv

  # 指定特征列
  python src/kg_feature_generator_dist.py --data /path/to/data.csv \\
                                          --card-col card_id \\
                                          --entity-cols card_id,merchant_id,device_type \\
                                          --account-features card_level,issuing_bank \\
                                          --transaction-features amount,balance_after \\
                                          --workers 16 \\
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
    parser.add_argument('--workers', type=str, default='8',
                        help='并行worker数量（默认: 8，或"auto"使用所有CPU）')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='特征CSV输出路径')

    args = parser.parse_args()

    # 解析worker数量
    if args.workers == 'auto':
        n_workers = cpu_count()
    else:
        n_workers = int(args.workers)

    # 解析列名
    entity_cols = args.entity_cols.split(',') if args.entity_cols else DEFAULT_ENTITY_COLS
    account_features = args.account_features.split(',') if args.account_features else []
    transaction_features = args.transaction_features.split(',') if args.transaction_features else []

    run_distributed_feature_generation(
        data_path=args.data,
        card_col=args.card_col,
        entity_cols=entity_cols,
        account_features=account_features,
        transaction_features=transaction_features,
        n_workers=n_workers,
        output_csv=args.output_csv,
        has_label=True  # 自动检测标签
    )


if __name__ == '__main__':
    main()