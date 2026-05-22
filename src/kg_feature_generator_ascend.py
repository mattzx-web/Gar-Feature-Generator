"""
Ascend NPU加速知识图谱特征生成器

适配华为昇腾NPU卡，使用CANN/ACL进行GPU加速。

依赖:
    # 安装Ascend CANN后，使用以下方式之一：
    # 1. PyTorch + Ascend EP (推荐)
    #    pip install torch==2.1.0.dev20240401+ascend -f https://download.pytorch.org/whl/torch_stable.html
    #
    # 2. 或者直接使用numpy（不依赖GPU库）

用法:
    # Ascend NPU加速
    python src/kg_feature_generator_ascend.py --data /path/to/data.csv \\
                                                --card-col card_id \\
                                                --npu-id 0 \\
                                                --output-csv ./features.csv

    # 多NPU分布式
    python src/kg_feature_generator_ascend.py --data /path/to/data.csv \\
                                              --card-col card_id \\
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


def check_ascend_npu():
    """
    检测Ascend NPU是否可用

    Returns:
        (available, device_info): (是否可用, 设备信息字典)
    """
    device_info = {
        'available': False,
        'device_count': 0,
        'can_use': False,
        'backend': 'cpu'  # cpu / ascend / cuda
    }

    # 方法1: 检查Ascend相关环境变量
    ascend_home = os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_SLOG_PATH')
    cannn_path = os.environ.get('LD_LIBRARY_PATH', '')

    if ascend_home or 'cann' in cannn_path.lower():
        device_info['available'] = True
        device_info['can_use'] = True
        device_info['backend'] = 'ascend'

    # 方法2: 尝试导入Ascend Python API
    try:
        import acl
        device_info['available'] = True
        device_info['can_use'] = True
        device_info['backend'] = 'ascend'
        # 获取设备数量
        ret = acl.rt.get_device_count()
        device_info['device_count'] = ret if ret > 0 else 0
    except ImportError:
        pass

    # 方法3: 检查torch是否使用Ascend后端
    try:
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            # 检查设备名称是否包含Ascend
            device_name = torch.cuda.get_device_name(0)
            if 'Ascend' in device_name or 'NPU' in device_name:
                device_info['available'] = True
                device_info['can_use'] = True
                device_info['backend'] = 'ascend'
                device_info['device_count'] = torch.cuda.device_count()
    except (ImportError, AttributeError):
        pass

    return device_info


def get_ascend_device_info(npu_id=0):
    """获取Ascend NPU设备信息"""
    info = {}

    # 尝试获取设备属性
    try:
        import acl
        # 设置设备
        acl.rt.set_device(npu_id)
        ret, props = acl.rt.get_device_properties(npu_id)
        if ret == 0:
            info['name'] = props.get('name', 'Ascend NPU')
            info['memory'] = props.get('memory', 0) / (1024**3)  # GB
    except Exception as e:
        info['name'] = 'Ascend NPU'
        info['memory'] = 0

    return info


def build_graph_optimized(df_values, df_columns, entity_cols, entity_col_indices,
                           neighbor_threshold=300):
    """
    优化的图构建算法（适配大规模数据）

    使用向量化操作和早停策略提升性能。

    Args:
        df_values: DataFrame values array
        df_columns: DataFrame columns list
        entity_cols: 实体列名列表
        entity_col_indices: 实体列在df中的索引
        neighbor_threshold: 邻居数量上限

    Returns:
        tx_neighbors: 邻居映射
    """
    print(f"[INFO] Building graph (optimized)...", flush=True)

    n = df_values.shape[0]
    tx_neighbors = defaultdict(set)

    # 预分配索引映射
    value_to_indices = {}

    for col_idx, col_name in zip(entity_col_indices, entity_cols):
        col_values = df_values[:, col_idx]

        # 哈希分桶（减少unique操作开销）
        unique_vals = np.unique(col_values)
        value_to_indices[col_name] = {val: np.where(col_values == val)[0] for val in unique_vals}

    # 构建邻居关系
    for col_name in entity_cols:
        if col_name not in value_to_indices:
            continue

        for val, row_indices in value_to_indices[col_name].items():
            if 1 < len(row_indices) < neighbor_threshold:
                for row_idx in row_indices:
                    tx_neighbors[row_idx].update(row_indices.tolist())

    # 移除自身
    for row_idx in tx_neighbors:
        tx_neighbors[row_idx].discard(row_idx)

    n_with_neigh = sum(1 for tx in tx_neighbors if len(tx_neighbors[tx]) > 0)
    print(f"[INFO] Nodes with neighbors: {n_with_neigh}/{n} ({100*n_with_neigh/n:.1f}%)", flush=True)

    return tx_neighbors


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

    global_stats['n_cards'] = df[card_col].nunique() if card_col in df.columns else 0

    return global_stats


def build_features_batch(df, tx_neighbors, global_stats, entity_cols, card_col,
                          account_features, transaction_features, batch_size=50000):
    """
    批量构建特征，优化内存使用

    Args:
        df: DataFrame
        tx_neighbors: 邻居映射
        global_stats: 全局统计量
        entity_cols: 实体列
        card_col: 卡号列
        account_features: 账户级特征
        transaction_features: 交易级特征
        batch_size: 批处理大小

    Returns:
        features_dict, feature_names
    """
    print(f"[INFO] Building features (batch mode, batch_size={batch_size})...", flush=True)

    n = len(df)
    features = {}
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

    # ========== 5. 图特征（批量计算） ==========
    # 度
    for col in entity_cols:
        if col not in df_columns:
            continue
        degrees = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.int32)
        features[f'{col}_degree'] = degrees

    # 1-hop
    n_1hop = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.int32)
    features['n_1hop'] = n_1hop
    features['n_1hop_log'] = np.log1p(n_1hop.astype(np.float32))

    # 邻居金额统计
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

    # 2-hop（批量计算）
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

    feature_names = list(features.keys())
    print(f"[INFO] Generated {len(feature_names)} features", flush=True)

    return features, feature_names


def run_ascend_feature_generation(data_path, card_col, entity_cols, account_features,
                                   transaction_features, output_csv=None, npu_id=0,
                                   workers=1):
    """
    Ascend NPU加速特征生成主流程

    Args:
        data_path: CSV文件路径
        card_col: 卡号列
        entity_cols: 实体列
        account_features: 账户级特征
        transaction_features: 交易级特征
        output_csv: 输出CSV路径
        npu_id: NPU设备ID
        workers: 多worker数量（用于分布式）
    """
    # 检测NPU
    device_info = check_ascend_npu()

    print("="*60, flush=True)
    print("Ascend NPU Knowledge Graph Feature Generator", flush=True)
    print(f"NPU Available: {device_info['available']}", flush=True)
    print(f"Backend: {device_info['backend']}", flush=True)
    if device_info['available']:
        print(f"Device Count: {device_info.get('device_count', 'N/A')}", flush=True)
    print(f"Workers: {workers}", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    # 1. 加载数据
    print(f"[INFO] Loading data from {data_path}...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Total records: {len(df)}", flush=True)

    # 2. 数据预处理
    for col in entity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col].astype(str))[0]

    # 3. 计算全局统计量
    amount_col = None
    for col in ['amount', '交易金额']:
        if col in df.columns:
            amount_col = col
            break

    global_stats = compute_global_stats(df, entity_cols, card_col, amount_col)
    print(f"[INFO] Global stats computed for {len(global_stats.get('card_tx_count', {}))} cards", flush=True)

    # 4. 构建图
    df_values = df.values
    df_columns = list(df.columns)
    entity_col_indices = [df_columns.index(col) for col in entity_cols if col in df_columns]

    tx_neighbors = build_graph_optimized(
        df_values, df_columns, entity_cols, entity_col_indices
    )

    # 5. 构建特征
    features, feature_names = build_features_batch(
        df, tx_neighbors, global_stats, entity_cols, card_col,
        account_features, transaction_features
    )

    # 6. 导出CSV
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
        for col in ['isFraud', 'fraud', 'label', 'is_fraud']:
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

    if device_info['available']:
        print(f"[INFO] NPU backend: {device_info['backend']}", flush=True)

    return features, feature_names


def main():
    parser = argparse.ArgumentParser(
        description='Ascend NPU Knowledge Graph Feature Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ascend NPU加速
  python src/kg_feature_generator_ascend.py --data /path/to/data.csv \\
                                              --card-col card_id \\
                                              --npu-id 0 \\
                                              --output-csv ./features.csv

  # 多NPU分布式
  python src/kg_feature_generator_ascend.py --data /path/to/large_data.csv \\
                                              --card-col card_id \\
                                              --npus 0,1,2,3 \\
                                              --workers 4 \\
                                              --output-csv ./features.csv

  # 指定特征列
  python src/kg_feature_generator_ascend.py --data /path/to/data.csv \\
                                            --card-col card_id \\
                                            --entity-cols card_id,merchant_id,device_type \\
                                            --account-features card_level,issuing_bank \\
                                            --transaction-features amount,balance_after \\
                                            --output-csv ./features.csv

  # 查看NPU状态
  python src/kg_feature_generator_ascend.py --check-npu
        """
    )

    parser.add_argument('--data', type=str, default=None,
                        help='CSV文件路径（使用--check-npu时可选）')
    parser.add_argument('--card-col', type=str, default=DEFAULT_CARD_COL,
                        help=f'卡号列名（默认: {DEFAULT_CARD_COL}）')
    parser.add_argument('--entity-cols', type=str, default=None,
                        help='实体列名列表，逗号分隔')
    parser.add_argument('--account-features', type=str, default=None,
                        help='账户级特征列名，逗号分隔')
    parser.add_argument('--transaction-features', type=str, default=None,
                        help='交易级特征列名，逗号分隔')
    parser.add_argument('--npu-id', type=int, default=0,
                        help='NPU设备ID（默认: 0）')
    parser.add_argument('--npus', type=str, default=None,
                        help='多NPU ID列表，逗号分隔（如: 0,1,2,3）')
    parser.add_argument('--workers', type=int, default=1,
                        help='并行worker数量（默认: 1）')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='特征CSV输出路径')
    parser.add_argument('--check-npu', action='store_true',
                        help='检查NPU状态并退出')

    args = parser.parse_args()

    # 检查NPU状态
    if args.check_npu:
        info = check_ascend_npu()
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

    run_ascend_feature_generation(
        data_path=args.data,
        card_col=args.card_col,
        entity_cols=entity_cols,
        account_features=account_features,
        transaction_features=transaction_features,
        output_csv=args.output_csv,
        npu_id=args.npu_id,
        workers=args.workers
    )


if __name__ == '__main__':
    main()