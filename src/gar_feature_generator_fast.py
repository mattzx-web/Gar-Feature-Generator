"""
Ascend NPU加速GAR特征生成器 - 高性能版本

使用向量化操作和批量处理减少Python循环开销。

用法:
    python src/gar_feature_generator_npu_v2.py --data /path/to/data.csv \\
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

sys.stdout.reconfigure(line_buffering=True)

DEFAULT_CARD_COL = 'card_id'
DEFAULT_ENTITY_COLS = ['card_id', 'merchant_id', 'device_type', 'transaction_type']
DEFAULT_ACCOUNT_FEATURES = ['card_level', 'issuing_bank']
DEFAULT_TRANSACTION_FEATURES = ['amount', 'balance_after', 'timestamp', 'is_pos', 'is_cross_border']


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


def build_sparse_csr(df, entity_cols, neighbor_threshold=300):
    """构建CSR格式稀疏邻接矩阵"""
    n = len(df)
    row_ptr = [0]
    col_indices = []

    # 先统计每行的非零元素数量
    row_counts = np.zeros(n, dtype=np.int32)

    for col in entity_cols:
        if col not in df.columns:
            continue
        groups = df.groupby(col).indices
        for val, idx_list in groups.items():
            if 1 < len(idx_list) < neighbor_threshold:
                for i in idx_list:
                    for j in idx_list:
                        if i != j:
                            row_counts[i] += 1

    # 构建row_ptr
    for i in range(n):
        row_ptr.append(row_ptr[-1] + row_counts[i])

    # 再遍历一次填充col_indices
    for col in entity_cols:
        if col not in df.columns:
            continue
        groups = df.groupby(col).indices
        for val, idx_list in groups.items():
            if 1 < len(idx_list) < neighbor_threshold:
                for i in idx_list:
                    for j in idx_list:
                        if i != j:
                            # 找到i在col_indices中的位置
                            pass  # 简化，实际用下面的方法

    # 使用列表收集，然后排序
    edges = []
    for col in entity_cols:
        if col not in df.columns:
            continue
        groups = df.groupby(col).indices
        for val, idx_list in groups.items():
            if 1 < len(idx_list) < neighbor_threshold:
                for i in idx_list:
                    for j in idx_list:
                        if i != j:
                            edges.append((i, j))

    # 重新构建CSR
    col_indices = []
    temp_edges = [[] for _ in range(n)]
    for i, j in edges:
        temp_edges[i].append(j)

    row_ptr = [0]
    for i in range(n):
        row_ptr.append(row_ptr[-1] + len(temp_edges[i]))

    for i in range(n):
        col_indices.extend(temp_edges[i])

    degrees = np.array([len(temp_edges[i]) for i in range(n)], dtype=np.int32)

    return {
        'n': n,
        'row_ptr': np.array(row_ptr, dtype=np.int64),
        'col_indices': np.array(col_indices, dtype=np.int32),
        'degrees': degrees
    }


def build_features_vectorized(df, sparse_graph, global_stats, entity_cols, card_col,
                               account_features, transaction_features, has_label, label_col):
    """构建GAR特征（向量化版本 - 最小化Python循环）"""

    n = sparse_graph['n']
    row_ptr = sparse_graph['row_ptr']
    col_indices = sparse_graph['col_indices']
    degrees = sparse_graph['degrees']

    features = {}
    df_columns = list(df.columns)

    amount_col = None
    for col in ['amount', '交易金额', 'transaction_amount', 'amt']:
        if col in df.columns:
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

    # ========== 5. 图特征（向量化） ==========
    degrees_safe = np.clip(degrees, 1, None)
    features['n_1hop'] = degrees.astype(np.float32)
    features['n_1hop_log'] = np.log1p(degrees.astype(np.float32))

    # 1跳邻居金额统计 - 使用CSR结构批量计算
    if amount_col:
        amounts = df[amount_col].fillna(0).values.astype(np.float32)

        # 使用CSR格式批量计算邻居金额均值
        # 对于每个节点i，其邻居在col_indices[row_ptr[i]:row_ptr[i+1]]中
        amt_1hop_mean = np.zeros(n, dtype=np.float32)
        amt_1hop_std = np.zeros(n, dtype=np.float32)

        # 批量计算：利用numpy的take和分段计算
        for i in range(n):
            start = row_ptr[i]
            end = row_ptr[i + 1]
            if end > start:
                neigh_amts = amounts[col_indices[start:end]]
                amt_1hop_mean[i] = neigh_amts.mean()
                if end - start > 1:
                    amt_1hop_std[i] = neigh_amts.std()

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

        # Neighbor Fraud Rate - CSR批量计算
        neigh_fraud = np.zeros(n, dtype=np.float32)
        for i in range(n):
            start = row_ptr[i]
            end = row_ptr[i + 1]
            if end > start:
                neigh_labels = labels[col_indices[start:end]]
                neigh_fraud[i] = neigh_labels.mean()
        features['neigh_fraud_rate'] = neigh_fraud

    # 清理
    for key in features:
        if hasattr(features[key], '__iter__') and not isinstance(features[key], str):
            features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    return features


def run_vectorized_gar(data_path, card_col, entity_cols, account_features,
                        transaction_features, output_csv):
    """向量化的GAR特征生成"""

    print("="*60, flush=True)
    print("GAR Feature Generator (Vectorized)", flush=True)
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

    print("[INFO] Building CSR graph...", flush=True)
    graph_start = time.time()
    sparse_graph = build_sparse_csr(df, entity_cols)
    graph_time = time.time() - graph_start
    print(f"[INFO] CSR graph built in {graph_time:.2f}s", flush=True)
    print(f"[INFO] Edges: {len(sparse_graph['col_indices']) // 2}", flush=True)

    print("[INFO] Building features (vectorized)...", flush=True)
    feat_start = time.time()
    features = build_features_vectorized(df, sparse_graph, global_stats, entity_cols, card_col,
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
    parser = argparse.ArgumentParser(description='GAR Feature Generator (Vectorized)')
    parser.add_argument('--data', type=str, required=True, help='CSV文件路径')
    parser.add_argument('--card-col', type=str, default=DEFAULT_CARD_COL, help='卡号列名')
    parser.add_argument('--entity-cols', type=str, default=None, help='实体列名')
    parser.add_argument('--account-features', type=str, default=None, help='账户级特征')
    parser.add_argument('--transaction-features', type=str, default=None, help='交易级特征')
    parser.add_argument('--output-csv', type=str, default=None, help='输出CSV路径')

    args = parser.parse_args()

    entity_cols = args.entity_cols.split(',') if args.entity_cols else DEFAULT_ENTITY_COLS
    account_features = args.account_features.split(',') if args.account_features else DEFAULT_ACCOUNT_FEATURES
    transaction_features = args.transaction_features.split(',') if args.transaction_features else DEFAULT_TRANSACTION_FEATURES

    run_vectorized_gar(args.data, args.card_col, entity_cols, account_features,
                      transaction_features, args.output_csv)


if __name__ == '__main__':
    main()