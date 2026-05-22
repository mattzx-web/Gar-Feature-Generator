"""
Ascend NPU加速GAR特征生成器 - torch_npu优化版

使用torch_npu进行矩阵运算加速，处理千万到亿级交易记录。

用法:
    python src/gar_feature_generator_npu.py --data /path/to/data.csv \\
                                             --card-col card_id \\
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
                                os.environ[key] = value
                    break
            except:
                pass


def get_device():
    """获取计算设备"""
    load_ascend_env()
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return torch_npu.npu.current_device()
    except:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda')
    except:
        pass
    return torch.device('cpu')


def check_npu():
    """检测NPU状态"""
    load_ascend_env()
    device_info = {'available': False, 'backend': 'cpu'}

    try:
        import torch_npu
        if torch_npu.npu.is_available():
            device_info['available'] = True
            device_info['backend'] = 'npu'
            device_info['device_count'] = torch_npu.npu.device_count()
    except (ImportError, AttributeError):
        pass

    try:
        import torch
        if torch.cuda.is_available():
            device_info['available'] = True
            device_info['backend'] = 'cuda'
    except:
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


def build_sparse_graph(df, entity_cols, neighbor_threshold=300):
    """构建稀疏图结构，返回邻居列表"""
    tx_neighbors = defaultdict(list)

    for col in entity_cols:
        if col not in df.columns:
            continue
        groups = df.groupby(col).indices
        for val, idx_list in groups.items():
            if 1 < len(idx_list) < neighbor_threshold:
                for idx in idx_list:
                    tx_neighbors[idx].extend(idx_list)

    # 去重并移除自环
    for idx in tx_neighbors:
        unique_neighs = list(set(tx_neighbors[idx]))
        if idx in unique_neighs:
            unique_neighs.remove(idx)
        tx_neighbors[idx] = unique_neighs

    return tx_neighbors


def build_features_torch(df, tx_neighbors, global_stats, entity_cols, card_col,
                          account_features, transaction_features, has_label, label_col):
    """构建GAR特征（torch_npu优化版）"""
    import torch
    import torch_npu

    device = get_device()
    is_npu = isinstance(device, int)
    n = len(df)
    features = {}

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
            vals = df[col].fillna(0).values.astype(np.float32)
            if is_npu:
                tensor = torch.tensor(vals, device='npu')
            else:
                tensor = torch.from_numpy(vals).to(device)
            features[col] = tensor.cpu().numpy()
            if amount_col and col == amount_col:
                features[f'{col}_log'] = torch.log1p(torch.abs(tensor)).cpu().numpy()
        else:
            cats = df[col].astype('category')
            features[col] = cats.cat.codes.values.astype(np.int32)

    # ========== 2. 实体频率 ==========
    for col in entity_cols:
        if col not in df_columns:
            continue
        values = df[col].values
        freq_map = global_stats.get(f'{col}_freq', {})
        freq_arr = np.array([freq_map.get(v, 0) for v in values], dtype=np.float32)
        features[f'{col}_freq'] = freq_arr
        features[f'{col}_freq_log'] = np.log1p(freq_arr)

    # ========== 3. 卡号聚合特征 ==========
    if card_col in df_columns:
        card_values = df[card_col].values
        card_counts = global_stats.get('card_tx_count', {})
        card_count_arr = np.array([card_counts.get(v, 0) for v in card_values], dtype=np.float32)
        features['card_tx_count'] = card_count_arr
        features['card_tx_count_log'] = np.log1p(card_count_arr)

        card_agg = global_stats.get('card_agg', {})
        unique_cards = np.unique(card_values)

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

    # ========== 5. 图特征 - torch_npu矩阵运算 ==========
    degrees = np.array([len(tx_neighbors.get(i, [])) for i in range(n)], dtype=np.float32)
    features['n_1hop'] = degrees
    features['n_1hop_log'] = np.log1p(degrees)

    # 1跳邻居金额统计 - torch_npu矩阵运算
    if amount_col:
        amounts = df[amount_col].fillna(0).values.astype(np.float32)

        # 转换为torch tensor on NPU
        if is_npu:
            amounts_tensor = torch.tensor(amounts, device='npu')
        else:
            amounts_tensor = torch.from_numpy(amounts).to(device)

        # 构建稀疏邻接矩阵的密集表示
        # 只对有邻居的节点计算
        amt_1hop_mean = np.zeros(n, dtype=np.float32)
        amt_1hop_std = np.zeros(n, dtype=np.float32)

        # 批量处理：收集所有邻居索引
        neigh_indices = []
        neigh_counts = []
        for i in range(n):
            neighs = tx_neighbors.get(i, [])
            if neighs:
                neigh_indices.extend(neighs)
                neigh_counts.append(len(neighs))
            else:
                neigh_counts.append(0)

        # 使用torch的index_add进行向量化计算
        if neigh_counts and sum(neigh_counts) > 0:
            # 构建索引和值
            idx_tensor = torch.zeros(sum(neigh_counts), dtype=torch.long, device='npu' if is_npu else device)
            val_tensor = torch.zeros(sum(neigh_counts), dtype=torch.float32, device='npu' if is_npu else device)

            pos = 0
            for i in range(n):
                neighs = tx_neighbors.get(i, [])
                if neighs:
                    for j, neigh in enumerate(neighs):
                        idx_tensor[pos + j] = neigh
                        val_tensor[pos + j] = amounts_tensor[neigh]
                    pos += len(neighs)

            # 使用scatter_add进行聚合
            agg = torch.zeros(n, dtype=torch.float32, device='npu' if is_npu else device)
            counts_tensor = torch.tensor(neigh_counts, dtype=torch.float32, device='npu' if is_npu else device)

            pos = 0
            for i in range(n):
                c = int(counts_tensor[i].item()) if counts_tensor[i].item() > 0 else 0
                if c > 0:
                    window = val_tensor[pos:pos+c]
                    amt_1hop_mean[i] = window.mean().item()
                    if c > 1:
                        amt_1hop_std[i] = window.std().item()
                    pos += c

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
            features[f'{col1}_{col2}_pair_freq_log'] = np.log1p(features[f'{col1}_{col2}_pair_freq'])

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

        # Neighbor Fraud Rate
        neigh_fraud = np.zeros(n, dtype=np.float32)
        for i in range(n):
            neighs = tx_neighbors.get(i, [])
            if neighs:
                neigh_fraud[i] = labels[neighs].mean()
        features['neigh_fraud_rate'] = neigh_fraud

    # 清理
    for key in features:
        if hasattr(features[key], '__iter__') and not isinstance(features[key], str):
            features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)
        else:
            features[key] = features[key]

    return features


def run_npu_gar(data_path, card_col, entity_cols, account_features,
                transaction_features, output_csv, mode='auto'):
    """NPU加速GAR特征生成"""
    device_info = check_npu()

    print("="*60, flush=True)
    print("Ascend NPU GAR Feature Generator (torch_npu)", flush=True)
    print(f"NPU Available: {device_info['available']}", flush=True)
    print(f"Backend: {device_info['backend']}", flush=True)
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
    tx_neighbors = build_sparse_graph(df, entity_cols)
    graph_time = time.time() - graph_start
    n_edges = sum(len(v) for v in tx_neighbors.values()) // 2
    print(f"[INFO] Graph built in {graph_time:.2f}s, edges: {n_edges}", flush=True)

    print("[INFO] Building GAR features (torch_npu)...", flush=True)
    feat_start = time.time()
    features = build_features_torch(df, tx_neighbors, global_stats, entity_cols, card_col,
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
    print(f"\n[INFO] Total time: {elapsed:.1f}s", flush=True)
    print(f"[INFO] Throughput: {len(df)/elapsed:.0f} records/sec", flush=True)

    return features


def main():
    parser = argparse.ArgumentParser(description='Ascend NPU GAR Feature Generator (torch_npu)')
    parser.add_argument('--data', type=str, required=True, help='CSV文件路径')
    parser.add_argument('--card-col', type=str, default='card_id', help='卡号列名')
    parser.add_argument('--entity-cols', type=str, default=None, help='实体列名')
    parser.add_argument('--account-features', type=str, default=None, help='账户级特征')
    parser.add_argument('--transaction-features', type=str, default=None, help='交易级特征')
    parser.add_argument('--output-csv', type=str, default=None, help='输出CSV路径')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'cpu', 'npu'])

    args = parser.parse_args()

    entity_cols = args.entity_cols.split(',') if args.entity_cols else ['card_id', 'merchant_id', 'device_type', 'transaction_type']
    account_features = args.account_features.split(',') if args.account_features else ['card_level', 'issuing_bank']
    transaction_features = args.transaction_features.split(',') if args.transaction_features else ['amount', 'balance_after', 'timestamp', 'is_pos', 'is_cross_border']

    run_npu_gar(args.data, args.card_col, entity_cols, account_features,
                transaction_features, args.output_csv, args.mode)


if __name__ == '__main__':
    main()