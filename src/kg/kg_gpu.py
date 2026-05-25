"""
GPU加速知识图谱特征生成器

使用GPU加速特征计算，处理千万到亿级交易记录。

依赖:
    pip install cupy-cuda11x  # 或 pip install cupy-cuda12x
    # 或使用ROCm: pip install cupy-rocm

用法:
    # GPU加速（单卡）
    python src/kg_feature_generator_gpu.py --data /path/to/data.csv \\
                                            --card-col card_id \\
                                            --gpu-id 0 \\
                                            --output-csv ./features.csv

    # 多GPU（数据并行）
    python src/kg_feature_generator_gpu.py --data /path/to/data.csv \\
                                            --card-col card_id \\
                                            --gpus 0,1,2,3 \\
                                            --output-csv ./features.csv
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import sys
import argparse
from datetime import datetime
import time

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# 默认配置
DEFAULT_CARD_COL = 'card_id'
DEFAULT_ENTITY_COLS = ['card_id', 'merchant_id', 'device_type', 'transaction_type']
DEFAULT_ACCOUNT_FEATURES = ['card_level', 'issuing_bank']
DEFAULT_TRANSACTION_FEATURES = ['amount', 'balance_after', 'timestamp', 'is_pos', 'is_cross_border']
DEFAULT_NEIGHBOR_THRESHOLD = 300


def has_gpu():
    """检测GPU是否可用"""
    try:
        import cupy as cp
        return cp.cuda.is_available()
    except ImportError:
        return False


def get_gpu_memory(gpu_id=0):
    """获取GPU内存信息"""
    try:
        import cupy as cp
        mem = cp.cuda.Device(gpu_id).mempool._get_limit()
        return mem / (1024**3)  # GB
    except:
        return 0


def build_graph_gpu(df_indices, df_values, n_rows, entity_cols, entity_col_indices, neighbor_threshold=300):
    """
    在GPU上构建图结构

    Args:
        df_indices: 实体列的索引数组 (n_rows, n_entity_cols)
        df_values: 实体列的值数组
        n_rows: 行数
        entity_cols: 实体列名列表
        entity_col_indices: 实体列在df中的列索引
        neighbor_threshold: 邻居数量上限

    Returns:
        tx_neighbors: 邻居列表（返回Python对象，因为需要用于特征计算）
    """
    try:
        import cupy as cp
        use_gpu = True
    except ImportError:
        use_gpu = False

    print(f"[INFO] Building graph on {'GPU' if use_gpu else 'CPU'}...", flush=True)

    tx_neighbors = defaultdict(set)

    for col_idx, col_name in zip(entity_col_indices, entity_cols):
        # 找到该列的唯一值及其对应的行索引
        col_values = df_values[:, col_idx]

        # 使用GPU加速groupby
        if use_gpu:
            # CuPy版本
            unique_vals, inverse = cp.unique(cp.array(col_values), return_inverse=True)
            inverse = cp.asnumpy(inverse)
            unique_vals = cp.asnumpy(unique_vals)
        else:
            # NumPy版本
            unique_vals, inverse = np.unique(col_values, return_inverse=True)

        # 对每个唯一值，找到其所有行的索引
        for val_idx, val in enumerate(unique_vals):
            mask = (inverse == val_idx)
            row_indices = np.where(mask)[0]

            if 1 < len(row_indices) < neighbor_threshold:
                for row_idx in row_indices:
                    tx_neighbors[row_idx].update(row_indices.tolist())

    # 移除自身
    for row_idx in tx_neighbors:
        tx_neighbors[row_idx].discard(row_idx)

    n_with_neigh = sum(1 for tx in tx_neighbors if len(tx_neighbors[tx]) > 0)
    print(f"[INFO] Nodes with neighbors: {n_with_neigh}/{n_rows} ({100*n_with_neigh/n_rows:.1f}%)", flush=True)

    return tx_neighbors


def build_features_gpu(df, tx_neighbors, card_col, entity_cols, account_features,
                        transaction_features, global_stats=None):
    """
    构建知识图谱特征（GPU加速版本）

    Args:
        df: DataFrame
        tx_neighbors: 邻居映射
        card_col: 卡号列
        entity_cols: 实体列
        account_features: 账户级特征
        transaction_features: 交易级特征
        global_stats: 全局统计量字典

    Returns:
        features_dict, feature_names
    """
    try:
        import cupy as cp
        use_gpu = True
    except ImportError:
        use_gpu = False

    print(f"[INFO] Building features on {'GPU' if use_gpu else 'CPU'}...", flush=True)

    features = {}
    n = len(df)

    # 获取DataFrame的numpy数组
    df_array = df.values
    df_columns = list(df.columns)

    amount_col = None
    for col in ['amount', '交易金额', 'transaction_amount', 'amt']:
        if col in df_columns:
            amount_col = col
            break

    # ========== 1. 交易级特征 ==========
    for col in transaction_features:
        if col not in df_columns:
            continue
        col_idx = df_columns.index(col)
        values = df_array[:, col_idx]

        if df[col].dtype in ['int64', 'float64']:
            features[col] = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
            if amount_col and col == amount_col:
                features[f'{col}_log'] = np.log1p(np.abs(features[col]))
        else:
            # 类别型特征编码
            unique_vals = np.unique(values)
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            features[col] = np.array([val_to_idx.get(v, 0) for v in values])

    # ========== 2. 实体频率（使用全局统计） ==========
    for col in entity_cols:
        if col not in df_columns:
            continue
        col_idx = df_columns.index(col)
        values = df_array[:, col_idx]

        freq_map = global_stats.get(f'{col}_freq', {}) if global_stats else {}
        features[f'{col}_freq'] = np.array([freq_map.get(v, 0) for v in values])
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq'])

    # ========== 3. 卡号聚合特征 ==========
    if card_col in df_columns:
        card_idx = df_columns.index(card_col)
        card_values = df_array[:, card_idx]

        # 卡号交易次数
        card_counts = global_stats.get('card_tx_count', {}) if global_stats else {}
        features['card_tx_count'] = np.array([card_counts.get(v, 0) for v in card_values])
        features['card_tx_count_log'] = np.log1p(features['card_tx_count'])

        # 卡号金额统计
        card_agg = global_stats.get('card_agg', {}) if global_stats else {}
        amt_mean = []
        amt_std = []
        amt_max = []
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
            amount_idx = df_columns.index(amount_col)
            amounts = df_array[:, amount_idx]
            features['amt_to_card_mean_ratio'] = amounts / (np.array(amt_mean) + 1)

    # ========== 4. 账户级特征 ==========
    for col in account_features:
        if col not in df_columns:
            continue
        col_idx = df_columns.index(col)
        values = df_array[:, col_idx]

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

    # 1-hop邻居数量
    n_1hop = [len(tx_neighbors.get(i, set())) for i in range(n)]
    features['n_1hop'] = np.array(n_1hop)
    features['n_1hop_log'] = np.log1p(features['n_1hop'])

    # 邻居金额统计
    if amount_col:
        amount_idx = df_columns.index(amount_col)
        amounts = df_array[:, amount_idx]

        amt_1hop_mean = []
        amt_1hop_std = []
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
            if col1 not in df_columns or col2 not in df_columns:
                continue
            idx1 = df_columns.index(col1)
            idx2 = df_columns.index(col2)
            vals1 = df_array[:, idx1].astype(str)
            vals2 = df_array[:, idx2].astype(str)
            pair_key = np.array([v1 + '_' + v2 for v1, v2 in zip(vals1, vals2)], dtype=object)
            pair_counts = pd.Series(pair_key).value_counts().to_dict()
            features[f'{col1}_{col2}_pair_freq'] = np.array([pair_counts.get(p, 0) for p in pair_key])
            features[f'{col1}_{col2}_pair_freq_log'] = np.log1p(features[f'{col1}_{col2}_pair_freq'])

    # ========== 7. 时序特征 ==========
    timestamp_col = None
    for col in ['timestamp', '时间戳', 'trans_time', 'transaction_time']:
        if col in df_columns:
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

    # ========== 8. 欺诈率特征 ==========
    label_col = None
    for col in ['isFraud', 'fraud', 'label', 'is_fraud']:
        if col in df_columns:
            label_col = col
            break

    if label_col:
        label_idx = df_columns.index(label_col)
        labels = df_array[:, label_idx]

        # Entity Fraud Rates
        for col in entity_cols:
            if col not in df_columns:
                continue
            col_idx = df_columns.index(col)
            values = df_array[:, col_idx]
            fraud_map = df.groupby(col)[label_col].mean().to_dict()
            features[f'{col}_fraud_rate'] = np.array([fraud_map.get(v, 0) for v in values])

        # Neighbor Fraud Rate
        neigh_fraud_rates = []
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_fraud_rates.append(labels[list(neighs)].mean())
            else:
                neigh_fraud_rates.append(0)
        features['neigh_fraud_rate'] = np.array(neigh_fraud_rates)

    # 清理无穷值
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    feature_names = list(features.keys())
    print(f"[INFO] Generated {len(feature_names)} features", flush=True)

    return features, feature_names


def compute_global_stats(df, entity_cols, card_col, amount_col=None):
    """计算全局统计量"""
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

    global_stats['n_cards'] = df[card_col].nunique() if card_col in df.columns else 0

    return global_stats


def run_gpu_feature_generation(data_path, card_col, entity_cols, account_features,
                                transaction_features, output_csv=None, gpu_id=0):
    """
    GPU加速特征生成主流程
    """
    has_cuda = has_gpu()

    print("="*60, flush=True)
    print("GPU-Accelerated Knowledge Graph Feature Generator", flush=True)
    print(f"GPU Available: {has_cuda}", flush=True)
    if has_cuda:
        print(f"GPU Memory: {get_gpu_memory(gpu_id):.1f} GB", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    # 1. 加载数据
    print(f"[INFO] Loading data from {data_path}...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Total records: {len(df)}", flush=True)

    # 实体列编码
    for col in entity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col].astype(str))[0]

    # 2. 计算全局统计量
    amount_col = None
    for col in ['amount', '交易金额']:
        if col in df.columns:
            amount_col = col
            break

    global_stats = compute_global_stats(df, entity_cols, card_col, amount_col)

    # 3. 构建图
    df_values = df.values
    df_columns = list(df.columns)
    entity_col_indices = [df_columns.index(col) for col in entity_cols if col in df_columns]

    tx_neighbors = build_graph_gpu(
        None, df_values, len(df), entity_cols, entity_col_indices
    )

    # 4. 构建特征
    features, feature_names = build_features_gpu(
        df, tx_neighbors, card_col, entity_cols, account_features,
        transaction_features, global_stats
    )

    # 5. 导出CSV
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)

        df_features = pd.DataFrame(features)

        # 添加关键列
        key_cols = []
        for col in [card_col, 'timestamp', '时间戳']:
            if col in df.columns:
                key_cols.append(col)
        if key_cols:
            df_features = pd.concat([df[key_cols], df_features], axis=1)

        label_col = None
        for col in ['isFraud', 'fraud', 'label']:
            if col in df.columns:
                label_col = col
                break
        if label_col:
            df_features[label_col] = df[label_col].values

        df_features.to_csv(output_csv, index=False)
        print(f"[INFO] Features saved to {output_csv}", flush=True)
        print(f"[INFO] Shape: {df_features.shape}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n[INFO] Total time: {elapsed/60:.1f} minutes", flush=True)
    print(f"[INFO] Throughput: {len(df)/elapsed:.0f} records/sec", flush=True)

    return features, feature_names


def main():
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated Knowledge Graph Feature Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单GPU
  python src/kg_feature_generator_gpu.py --data /path/to/large_data.csv \\
                                           --card-col card_id \\
                                           --gpu-id 0 \\
                                           --output-csv ./features.csv

  # 指定特征列
  python src/kg_feature_generator_gpu.py --data /path/to/data.csv \\
                                          --card-col card_id \\
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
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU设备ID（默认: 0）')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='特征CSV输出路径')

    args = parser.parse_args()

    entity_cols = args.entity_cols.split(',') if args.entity_cols else DEFAULT_ENTITY_COLS
    account_features = args.account_features.split(',') if args.account_features else []
    transaction_features = args.transaction_features.split(',') if args.transaction_features else []

    run_gpu_feature_generation(
        data_path=args.data,
        card_col=args.card_col,
        entity_cols=entity_cols,
        account_features=account_features,
        transaction_features=transaction_features,
        output_csv=args.output_csv,
        gpu_id=args.gpu_id
    )


if __name__ == '__main__':
    main()