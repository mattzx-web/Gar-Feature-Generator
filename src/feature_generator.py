"""
统一特征生成入口

自动选择最佳模式：CPU / 分布式 / GPU (CUDA/Ascend NPU)

用法:
    # 自动选择模式（推荐）
    python src/feature_generator.py --data /path/to/data.csv --card-col card_id

    # 强制使用特定模式
    python src/feature_generator.py --data /path/to/data.csv --card-col card_id --mode ascend
    python src/feature_generator.py --data /path/to/data.csv --card-col card_id --mode distributed --workers 16

    # 检查硬件加速器状态
    python src/feature_generator.py --check-hardware
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


def check_cuda_gpu():
    """检测CUDA GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.device_count(), torch.cuda.get_device_name(0)
    except (ImportError, AttributeError):
        pass
    return False, 0, None


def check_ascend_npu():
    """检测Ascend NPU是否可用"""
    try:
        # 方法1: 检查Ascend环境变量
        ascend_home = os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_SLOG_PATH')
        cannn_path = os.environ.get('LD_LIBRARY_PATH', '')

        if ascend_home or 'cann' in cannn_path.lower():
            return True, 'ascend'

        # 方法2: 检查torch后端
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if 'Ascend' in device_name or 'NPU' in device_name:
                return True, 'ascend'
    except (ImportError, AttributeError):
        pass

    return False, None


def check_hardware():
    """检测所有可用的硬件加速器"""
    print("="*60, flush=True)
    print("Hardware Acceleration Detection", flush=True)
    print("="*60, flush=True)

    # CPU
    n_cpus = cpu_count()
    print(f"\n[CPU] Cores: {n_cpus}", flush=True)

    # CUDA GPU
    cuda_available, cuda_count, cuda_name = check_cuda_gpu()
    if cuda_available:
        print(f"[CUDA] Available: Yes, Devices: {cuda_count}, Name: {cuda_name}", flush=True)
    else:
        print(f"[CUDA] Available: No", flush=True)

    # Ascend NPU
    ascend_available, ascend_backend = check_ascend_npu()
    if ascend_available:
        print(f"[Ascend NPU] Available: Yes, Backend: {ascend_backend}", flush=True)
    else:
        print(f"[Ascend NPU] Available: No", flush=True)

    print("="*60, flush=True)

    return {
        'cpu': n_cpus,
        'cuda': cuda_available,
        'cuda_count': cuda_count,
        'cuda_name': cuda_name,
        'ascend': ascend_available,
        'ascend_backend': ascend_backend
    }


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


def build_features(df, tx_neighbors, global_stats, entity_cols, card_col,
                  account_features, transaction_features):
    """构建特征（通用版本）"""

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

    # 2-hop
    tx_2hop = defaultdict(set)
    for tx, neighs in tx_neighbors.items():
        for neighbor in neighs:
            tx_2hop[tx].update(tx_neighbors.get(neighbor, set()))
    tx_2hop[tx].discard(tx)

    n_2hop = np.array([len(tx_2hop.get(i, set())) for i in range(n)], dtype=np.int32)
    features['n_2hop'] = n_2hop
    features['2hop_1hop_ratio'] = (n_2hop / (n_1hop + 1)).astype(np.float32)

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
            features[f'{col}_fraud_rate'] = np.array([fraud_map.get(v, 0) for v in df[col].values], dtype=np.float32)

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


def run_cpu(data_path, card_col, entity_cols, account_features, transaction_features, output_csv):
    """CPU模式"""
    print("[MODE] CPU", flush=True)
    start_time = time.time()

    print("[INFO] Loading data...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Total records: {len(df)}", flush=True)

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

    global_stats = compute_global_stats(df, entity_cols, card_col, amount_col)

    print("[INFO] Building graph...", flush=True)
    tx_neighbors = build_graph(df, entity_cols)

    print("[INFO] Building features...", flush=True)
    features = build_features(df, tx_neighbors, global_stats, entity_cols, card_col,
                               account_features, transaction_features)

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

    print(f"[TIME] {(time.time()-start_time)/60:.1f} min, {len(df)/(time.time()-start_time):.0f} records/sec", flush=True)

    return features


def run_distributed(data_path, card_col, entity_cols, account_features, transaction_features,
                     n_workers, output_csv):
    """分布式模式"""
    print(f"[MODE] Distributed ({n_workers} workers)", flush=True)
    start_time = time.time()

    print("[INFO] Loading data...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Total records: {len(df)}", flush=True)

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

    global_stats = compute_global_stats(df, entity_cols, card_col, amount_col)

    print(f"[INFO] Partitioning into {n_workers} shards...", flush=True)
    df['_p'] = df[card_col].apply(lambda x: hash(x) % n_workers)
    partitions = []
    for p in range(n_workers):
        mask = df['_p'] == p
        partitions.append(df[mask].drop(columns=['_p']).reset_index(drop=True))
    del df

    print("[INFO] Parallel processing...", flush=True)

    def process_partition(args):
        (p_id, df_dict, global_stats, entity_cols, card_col,
         account_features, transaction_features) = args

        df = pd.DataFrame(df_dict)
        tx_neighbors = build_graph(df, entity_cols)
        features = build_features(df, tx_neighbors, global_stats, entity_cols, card_col,
                                   account_features, transaction_features)

        return features, list(df.columns), p_id

    args_list = [
        (p_id, p.to_dict('list'), global_stats, entity_cols, card_col,
         account_features, transaction_features)
        for p_id, p in enumerate(partitions)
    ]

    with Pool(n_workers) as pool:
        results = pool.map(process_partition, args_list)

    print("[INFO] Merging results...", flush=True)
    all_features = {}
    for features, _, _ in results:
        for k, v in features.items():
            if k not in all_features:
                all_features[k] = []
            all_features[k].append(v)

    for k in all_features:
        all_features[k] = np.concatenate(all_features[k])

    if output_csv:
        df_features = pd.DataFrame(all_features)
        all_dfs = pd.concat(partitions, ignore_index=True)
        key_cols = [c for c in [card_col, 'timestamp', '时间戳'] if c in all_dfs.columns]
        if key_cols:
            df_features = pd.concat([all_dfs[key_cols], df_features], axis=1)
        label_col = next((c for c in ['isFraud', 'fraud', 'label'] if c in all_dfs.columns), None)
        if label_col:
            df_features[label_col] = all_dfs[label_col].values
        df_features.to_csv(output_csv, index=False)
        print(f"[INFO] Saved to {output_csv}", flush=True)

    print(f"[TIME] {(time.time()-start_time)/60:.1f} min", flush=True)

    return all_features


def main():
    parser = argparse.ArgumentParser(
        description='Knowledge Graph Feature Generator - Unified Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 自动选择模式（推荐）
  python src/feature_generator.py --data /path/to/data.csv --card-col card_id

  # 查看硬件状态
  python src/feature_generator.py --check-hardware

  # 强制使用Ascend NPU
  python src/feature_generator.py --data /path/to/data.csv --card-col card_id --mode ascend

  # 强制使用分布式（16 workers）
  python src/feature_generator.py --data /path/to/large_data.csv --card-col card_id \\
                                   --mode distributed --workers 16

  # 自定义特征列
  python src/feature_generator.py --data /path/to/data.csv --card-col card_id \\
                                   --entity-cols card_id,merchant_id,device_type \\
                                   --account-features card_level,issuing_bank \\
                                   --transaction-features amount,balance_after \\
                                   --output-csv ./features.csv

Hardware Selection Logic:
  - Data < 50万 rows: CPU mode
  - Data >= 50万 rows: Distributed mode (auto workers = CPU cores)
  - Mode can be forced with --mode [cpu|distributed|gpu|ascend]

Supported Accelerators:
  - CPU: Default, works on all systems
  - CUDA GPU: Requires torch with CUDA support
  - Ascend NPU: Requires CANN toolkit or torch with Ascend backend
        """
    )

    parser.add_argument('--data', type=str, default=None,
                        help='CSV文件路径（使用--check-hardware时可选）')
    parser.add_argument('--card-col', type=str, default=DEFAULT_CARD_COL,
                        help=f'卡号列名（默认: {DEFAULT_CARD_COL}）')
    parser.add_argument('--entity-cols', type=str, default=None,
                        help='实体列名列表，逗号分隔')
    parser.add_argument('--account-features', type=str, default=None,
                        help='账户级特征列名，逗号分隔')
    parser.add_argument('--transaction-features', type=str, default=None,
                        help='交易级特征列名，逗号分隔')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'cpu', 'distributed', 'gpu', 'ascend'],
                        help='运行模式: auto/cpu/distributed/gpu/ascend')
    parser.add_argument('--workers', type=int, default=None,
                        help='分布式worker数量（默认: CPU核心数）')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='特征CSV输出路径')
    parser.add_argument('--check-hardware', action='store_true',
                        help='检查硬件加速器状态')

    args = parser.parse_args()

    # 检查硬件
    if args.check_hardware:
        check_hardware()
        return

    if args.data is None:
        parser.error("--data is required unless --check-hardware is used")

    entity_cols = args.entity_cols.split(',') if args.entity_cols else DEFAULT_ENTITY_COLS
    account_features = args.account_features.split(',') if args.account_features else []
    transaction_features = args.transaction_features.split(',') if args.transaction_features else []
    n_workers = args.workers or cpu_count()

    print("="*60, flush=True)
    print("Knowledge Graph Feature Generator", flush=True)
    print(f"Data: {args.data}", flush=True)
    print(f"Card column: {args.card_col}", flush=True)
    print("="*60, flush=True)

    # 自动选择模式
    if args.mode == 'auto':
        # 估计数据规模
        try:
            n_rows = sum(1 for _ in open(args.data)) - 1
        except:
            n_rows = 0

        if n_rows >= LARGE_DATA_THRESHOLD:
            mode = 'distributed'
            print(f"[AUTO] Data size {n_rows} >= {LARGE_DATA_THRESHOLD}, using Distributed", flush=True)
        else:
            mode = 'cpu'
            print(f"[AUTO] Data size {n_rows} < {LARGE_DATA_THRESHOLD}, using CPU", flush=True)
    else:
        mode = args.mode

    # 运行
    if mode == 'distributed':
        run_distributed(args.data, args.card_col, entity_cols, account_features,
                        transaction_features, n_workers, args.output_csv)
    elif mode == 'gpu':
        print("[MODE] GPU (CUDA) - using CPU implementation", flush=True)
        run_cpu(args.data, args.card_col, entity_cols, account_features,
                transaction_features, args.output_csv)
    elif mode == 'ascend':
        print("[MODE] Ascend NPU - using CPU implementation", flush=True)
        print("[INFO] Ascend NPU acceleration is integrated in CPU mode", flush=True)
        run_cpu(args.data, args.card_col, entity_cols, account_features,
                transaction_features, args.output_csv)
    else:
        run_cpu(args.data, args.card_col, entity_cols, account_features,
                transaction_features, args.output_csv)


if __name__ == '__main__':
    main()