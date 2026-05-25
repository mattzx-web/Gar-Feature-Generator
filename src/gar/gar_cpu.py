"""
通用GAR特征生成器

支持白样本（无标签）数据，支持账户级和交易级特征。
基于图关联规则的欺诈率特征工程。

数据格式:
- 每个CSV包含交易记录
- 账户级特征（如卡等级）每个账号重复
- 交易级特征（如交易金额）每条记录不同

用法:
    # 白样本特征生成
    python src/gar_feature_generator.py --data /path/to/data.csv \\
                                        --card-col card_id \\
                                        --export-features-only \\
                                        --output-csv ./features/gar_features.csv

    # 指定账户级和交易级特征列
    python src/gar_feature_generator.py --data /path/to/data.csv \\
                                        --card-col card_id \\
                                        --account-features card_level,issuing_bank \\
                                        --transaction-features amount,balance,timestamp \\
                                        --entity-cols card_id,merchant_id,device_type
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import json
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
DEFAULT_TRANSACTION_FEATURES = ['timestamp', 'amount', 'balance_after', 'is_frequent_contact',
                                'transaction_channel', 'device_type', 'is_pos', 'is_cross_border']
DEFAULT_NEIGHBOR_THRESHOLD = 300


def load_and_preprocess_data(data_path, card_col, entity_cols, account_features, transaction_features):
    """
    加载并预处理数据
    """
    print(f"[INFO] Loading data from {data_path}...", flush=True)

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded {len(df)} records", flush=True)

    # 检测是否有标签
    has_label = False
    label_col = None
    for col in ['isFraud', 'fraud', 'label', 'is_fraud']:
        if col in df.columns:
            has_label = True
            label_col = col
            print(f"[INFO] Found label column: {label_col}", flush=True)
            break

    # 实体列编码
    for col in entity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

    # 填充特征缺失值
    for col in account_features + transaction_features:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('missing')
            else:
                df[col] = df[col].fillna(0)

    return df, card_col, entity_cols, account_features, transaction_features, has_label, label_col


def build_graph(df, entity_cols, neighbor_threshold=DEFAULT_NEIGHBOR_THRESHOLD):
    """构建交易图结构"""
    print(f"[INFO] Building graph...", flush=True)

    n = len(df)
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

    n_with_neigh = sum(1 for tx in tx_neighbors if len(tx_neighbors[tx]) > 0)
    print(f"[INFO] Nodes with neighbors: {n_with_neigh}/{n} ({100*n_with_neigh/n:.1f}%)", flush=True)

    return tx_neighbors


def build_gar_features(df, tx_neighbors, card_col, entity_cols, account_features,
                       transaction_features, has_label, label_col=None):
    """
    构建GAR特征（通用版本，有数据泄漏风险）

    特征类型:
    1. Transaction Features (交易级特征)
    2. Entity Frequency (实体频率)
    3. Card Aggregation (卡号聚合特征)
    4. Pair Frequency (配对频率)
    5. Neighbor Features (邻居特征)
    6. Account Features (账户级特征)
    7. GAR Fraud Rates (欺诈率特征 - 仅在有标签时可用)
    """
    print(f"[INFO] Building GAR features...", flush=True)

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

    # ========== 2. Entity Frequency特征 ==========
    for col in entity_cols:
        if col not in df.columns:
            continue
        freq_map = df[col].value_counts().to_dict()
        features[f'{col}_freq'] = df[col].map(freq_map).fillna(0).values
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq'])

    # ========== 3. Card聚合特征 ==========
    if card_col in df.columns:
        card_counts = df[card_col].value_counts().to_dict()
        features['card_tx_count'] = df[card_col].map(card_counts).fillna(0).values
        features['card_tx_count_log'] = np.log1p(features['card_tx_count'])

        if amount_col:
            card_amt_mean = df.groupby(card_col)[amount_col].transform('mean')
            card_amt_std = df.groupby(card_col)[amount_col].transform('std').fillna(0)
            card_amt_max = df.groupby(card_col)[amount_col].transform('max')
            features['card_amt_mean'] = card_amt_mean.fillna(0).values
            features['card_amt_std'] = card_amt_std.fillna(0).values
            features['card_amt_max'] = card_amt_max.fillna(0).values
            features['amt_to_card_mean_ratio'] = df[amount_col].fillna(0) / (card_amt_mean.fillna(1) + 1)

        card_degree_map = df.groupby(card_col).apply(
            lambda x: np.mean([len(tx_neighbors.get(idx, set())) for idx in x.index]),
            include_groups=False
        ).to_dict()
        features['card_avg_degree'] = df[card_col].map(card_degree_map).fillna(0).values

    # ========== 4. Pair Frequency特征 ==========
    for i, col1 in enumerate(entity_cols[:4]):
        for col2 in entity_cols[i+1:5]:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            pairs = df[col1].astype(str) + '_' + df[col2].astype(str)
            pair_counts = pairs.map(pairs.value_counts())
            features[f'{col1}_{col2}_pair_freq'] = pair_counts.fillna(0).values
            features[f'{col1}_{col2}_pair_freq_log'] = np.log1p(pair_counts.fillna(0)).values

    # ========== 5. Neighbor特征 ==========
    n_1hop = [len(tx_neighbors.get(i, set())) for i in range(n)]
    features['n_1hop'] = np.array(n_1hop)
    features['n_1hop_log'] = np.log1p(features['n_1hop'])

    if amount_col:
        amt_1hop_mean = []
        amt_1hop_std = []
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = df[amount_col].iloc[list(neighs)].fillna(0).values
                amt_1hop_mean.append(np.mean(neigh_amts))
                amt_1hop_std.append(np.std(neigh_amts) if len(neigh_amts) > 1 else 0)
            else:
                amt_1hop_mean.append(0)
                amt_1hop_std.append(0)
        features['amt_1hop_mean'] = np.array(amt_1hop_mean)
        features['amt_1hop_std'] = np.array(amt_1hop_std)

    # ========== 6. Account级特征 ==========
    for col in account_features:
        if col not in df.columns:
            continue
        if df[col].dtype == 'object':
            le = LabelEncoder()
            features[col] = le.fit_transform(df[col].fillna('missing').astype(str))
        else:
            features[col] = df[col].fillna(-1).values

    # ========== 7. GAR Fraud Rate特征 (仅当有标签时) ==========
    if has_label and label_col:
        print(f"[INFO] Computing fraud rate features using label column: {label_col}", flush=True)

        # Entity Fraud Rates
        for col in entity_cols:
            if col not in df.columns:
                continue
            fraud_map = df.groupby(col)[label_col].mean().to_dict()
            features[f'{col}_fraud_rate'] = df[col].map(fraud_map).fillna(0).values

        # Pair Fraud Rates
        for i, col1 in enumerate(entity_cols[:4]):
            for col2 in entity_cols[i+1:5]:
                if col1 not in df.columns or col2 not in df.columns:
                    continue
                pair_df = df[[col1, col2, label_col]].copy()
                pair_df['_pair'] = pair_df[col1].astype(str) + '_' + pair_df[col2].astype(str)
                fraud_map = pair_df.groupby('_pair')[label_col].mean().to_dict()
                pair_values = df[col1].astype(str) + '_' + df[col2].astype(str)
                features[f'{col1}_{col2}_pair_fraud_rate'] = pair_values.map(fraud_map).fillna(0).values

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

    # ========== 8. 时序特征 ==========
    timestamp_col = None
    for col in ['timestamp', '时间戳', 'trans_time', 'transaction_time', 'trans_date']:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col:
        try:
            ts = pd.to_datetime(df[timestamp_col], errors='coerce')
            if not ts.isna().all():
                features['trans_hour'] = ts.dt.hour.fillna(12).values
                features['trans_dayofweek'] = ts.dt.dayofweek.fillna(0).values
                df_sorted = df.copy()
                df_sorted['_ts'] = ts
                df_sorted = df_sorted.sort_values([card_col, timestamp_col])
                time_diff = df_sorted.groupby(card_col)['_ts'].diff().dt.total_seconds().fillna(0)
                features['time_diff_prev'] = df.index.map(time_diff.to_dict()).fillna(0).values
        except Exception as e:
            print(f"[WARN] Failed to extract temporal features: {e}", flush=True)

    # 清理无穷值
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    feature_names = list(features.keys())
    print(f"[INFO] Generated {len(feature_names)} features", flush=True)

    return features, feature_names


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
    """
    从训练集计算欺诈率映射（避免数据泄漏）

    Returns:
        entity_fraud_maps: dict of {col: {entity_value: fraud_rate}}
        pair_fraud_maps: dict of {col_pair: {pair_key: fraud_rate}}
    """
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


def build_gar_features_no_leakage(df, train_idx, tx_neighbors, card_col,
                                   entity_cols, account_features, transaction_features,
                                   has_label, label_col, entity_fraud_maps, pair_fraud_maps):
    """
    构建GAR特征（无数据泄漏版本）

    关键区别：
    - 欺诈率仅从训练集计算
    - 训练集和测试集使用相同的欺诈率映射
    """
    print(f"[INFO] Building GAR features (no leakage mode)...", flush=True)

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

    # ========== 2. Entity Frequency特征 ==========
    for col in entity_cols:
        if col not in df.columns:
            continue
        freq_map = df[col].value_counts().to_dict()
        features[f'{col}_freq'] = df[col].map(freq_map).fillna(0).values
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq'])

    # ========== 3. Card聚合特征 ==========
    if card_col in df.columns:
        card_counts = df[card_col].value_counts().to_dict()
        features['card_tx_count'] = df[card_col].map(card_counts).fillna(0).values
        features['card_tx_count_log'] = np.log1p(features['card_tx_count'])

        if amount_col:
            card_amt_mean = df.groupby(card_col)[amount_col].transform('mean')
            card_amt_std = df.groupby(card_col)[amount_col].transform('std').fillna(0)
            card_amt_max = df.groupby(card_col)[amount_col].transform('max')
            features['card_amt_mean'] = card_amt_mean.fillna(0).values
            features['card_amt_std'] = card_amt_std.fillna(0).values
            features['card_amt_max'] = card_amt_max.fillna(0).values
            features['amt_to_card_mean_ratio'] = df[amount_col].fillna(0) / (card_amt_mean.fillna(1) + 1)

        card_degree_map = df.groupby(card_col).apply(
            lambda x: np.mean([len(tx_neighbors.get(idx, set())) for idx in x.index]),
            include_groups=False
        ).to_dict()
        features['card_avg_degree'] = df[card_col].map(card_degree_map).fillna(0).values

    # ========== 4. Pair Frequency特征 ==========
    for i, col1 in enumerate(entity_cols[:4]):
        for col2 in entity_cols[i+1:5]:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            pairs = df[col1].astype(str) + '_' + df[col2].astype(str)
            pair_counts = pairs.map(pairs.value_counts())
            features[f'{col1}_{col2}_pair_freq'] = pair_counts.fillna(0).values
            features[f'{col1}_{col2}_pair_freq_log'] = np.log1p(pair_counts.fillna(0)).values

    # ========== 5. Neighbor特征 ==========
    n_1hop = [len(tx_neighbors.get(i, set())) for i in range(n)]
    features['n_1hop'] = np.array(n_1hop)
    features['n_1hop_log'] = np.log1p(features['n_1hop'])

    if amount_col:
        amt_1hop_mean = []
        amt_1hop_std = []
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = df[amount_col].iloc[list(neighs)].fillna(0).values
                amt_1hop_mean.append(np.mean(neigh_amts))
                amt_1hop_std.append(np.std(neigh_amts) if len(neigh_amts) > 1 else 0)
            else:
                amt_1hop_mean.append(0)
                amt_1hop_std.append(0)
        features['amt_1hop_mean'] = np.array(amt_1hop_mean)
        features['amt_1hop_std'] = np.array(amt_1hop_std)

    # ========== 6. Account级特征 ==========
    for col in account_features:
        if col not in df.columns:
            continue
        if df[col].dtype == 'object':
            le = LabelEncoder()
            features[col] = le.fit_transform(df[col].fillna('missing').astype(str))
        else:
            features[col] = df[col].fillna(-1).values

    # ========== 7. GAR Fraud Rate特征（仅从训练集计算） ==========
    if has_label and label_col and entity_fraud_maps:
        print(f"[INFO] Computing fraud rate features from TRAIN ONLY (no leakage)", flush=True)

        # Entity Fraud Rates（使用训练集计算的映射）
        for col in entity_cols:
            if col not in df.columns or col not in entity_fraud_maps:
                continue
            features[f'{col}_fraud_rate'] = df[col].map(entity_fraud_maps[col]).fillna(0).values

        # Pair Fraud Rates（使用训练集计算的映射）
        for col_pair, fraud_map in pair_fraud_maps.items():
            col1, col2 = col_pair.split('_', 1)
            if col1 not in df.columns or col2 not in df.columns:
                continue
            pair_values = df[col1].astype(str) + '_' + df[col2].astype(str)
            features[f'{col1}_{col2}_pair_fraud_rate'] = pair_values.map(fraud_map).fillna(0).values

        # Neighbor Fraud Rate（使用训练集的标签）
        train_is_fraud = df.iloc[train_idx][label_col].values if label_col in df.columns else None
        if train_is_fraud is not None:
            train_label_map = dict(zip(train_idx, train_is_fraud))
            neigh_fraud_rates = []
            for i in range(n):
                neighs = tx_neighbors.get(i, set())
                if neighs:
                    train_neighs = [n for n in neighs if n in train_label_map]
                    if train_neighs:
                        neigh_fraud_rates.append(np.mean([train_label_map[n] for n in train_neighs]))
                    else:
                        neigh_fraud_rates.append(0)
                else:
                    neigh_fraud_rates.append(0)
            features['neigh_fraud_rate'] = np.array(neigh_fraud_rates)

    # ========== 8. 时序特征 ==========
    timestamp_col = None
    for col in ['timestamp', '时间戳', 'trans_time', 'transaction_time', 'trans_date']:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col:
        try:
            ts = pd.to_datetime(df[timestamp_col], errors='coerce')
            if not ts.isna().all():
                features['trans_hour'] = ts.dt.hour.fillna(12).values
                features['trans_dayofweek'] = ts.dt.dayofweek.fillna(0).values
                df_sorted = df.copy()
                df_sorted['_ts'] = ts
                df_sorted = df_sorted.sort_values([card_col, timestamp_col])
                time_diff = df_sorted.groupby(card_col)['_ts'].diff().dt.total_seconds().fillna(0)
                features['time_diff_prev'] = df.index.map(time_diff.to_dict()).fillna(0).values
        except Exception as e:
            print(f"[WARN] Failed to extract temporal features: {e}", flush=True)

    # 清理无穷值
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    feature_names = list(features.keys())
    print(f"[INFO] Generated {len(feature_names)} features", flush=True)

    return features, feature_names


def export_features_to_csv(features_dict, feature_names, output_path, original_df=None, has_label=False, split_col=None):
    """导出特征到CSV"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    df_features = pd.DataFrame({name: features_dict[name] for name in feature_names})

    # 保留关键列
    if original_df is not None:
        key_cols = []
        for col in ['card_id', '卡号', 'TransactionID', 'transaction_id', 'timestamp', '时间戳']:
            if col in original_df.columns:
                key_cols.append(col)

        if key_cols:
            df_features = pd.concat([original_df[key_cols], df_features], axis=1)

    # 保留标签
    if has_label:
        for col in ['isFraud', 'fraud', 'label', 'is_fraud']:
            if col in original_df.columns:
                df_features[col] = original_df[col].values
                break

    # 添加split列
    if split_col is not None:
        df_features['split'] = split_col

    df_features.to_csv(output_path, index=False)
    print(f"[INFO] Features saved to {output_path}", flush=True)
    print(f"[INFO] Shape: {df_features.shape}", flush=True)

    return output_path


def train_classifier(features_dict, feature_names, has_label, label_col, split_col=None, train_ratio=0.7, seed=42):
    """训练分类器"""
    if not has_label:
        print("[INFO] White sample mode - skipping model training", flush=True)
        return None

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    n = len(feature_names[0])

    if split_col is not None:
        # 使用预定义的分割（无泄漏模式）
        train_mask = np.array(split_col) == 'train'
        test_mask = np.array(split_col) == 'test'
        X = np.column_stack([features_dict[name] for name in feature_names])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_train, X_test = X[train_mask], X[test_mask]
        y_train = features_dict[label_col][train_mask]
        y_test = features_dict[label_col][test_mask]
        print(f"[INFO] Using predefined split: Train={X_train.shape[0]}, Test={X_test.shape[0]}", flush=True)
    else:
        # 随机分割（有泄漏风险）
        n_train = int(train_ratio * n)
        indices = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(indices)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        X = np.column_stack([features_dict[name] for name in feature_names])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        y = features_dict[label_col]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                     subsample=0.8, random_state=seed)
    gb.fit(X_train, y_train)

    train_proba = gb.predict_proba(X_train)[:, 1]
    test_proba = gb.predict_proba(X_test)[:, 1]

    results = {
        'train_auc': float(roc_auc_score(y_train, train_proba)),
        'test_auc': float(roc_auc_score(y_test, test_proba)),
        'feature_importance': list(zip(feature_names, gb.feature_importances_.tolist()))
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='GAR Feature Generator for White Samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 白样本特征生成（无标签）
  python src/gar_feature_generator.py --data /path/to/transactions.csv \\
                                      --card-col card_id \\
                                      --export-features-only \\
                                      --output-csv ./features/gar_features.csv

  # 有标签数据（默认无泄漏模式）
  python src/gar_feature_generator.py --data /path/to/transactions.csv \\
                                      --card-col card_id

  # 关闭无泄漏模式（不推荐）
  python src/gar_feature_generator.py --data /path/to/transactions.csv \\
                                      --leakage \\
                                      --card-col card_id
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
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='输出目录（默认: ./outputs）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--export-features-only', action='store_true',
                        help='仅生成特征，不训练模型')
    parser.add_argument('--feature-only', action='store_true',
                        help='与--export-features-only相同')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='特征CSV输出路径')
    parser.add_argument('--no-leakage', action='store_true', default=True,
                        help='防止数据泄漏：欺诈率仅从训练集计算（默认开启）')
    parser.add_argument('--leakage', action='store_false', dest='no_leakage',
                        help='关闭防泄漏模式：欺诈率从全部数据计算（不推荐）')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例（默认: 0.7）')

    args = parser.parse_args()

    export_only = args.export_features_only or args.feature_only

    entity_cols = args.entity_cols.split(',') if args.entity_cols else DEFAULT_ENTITY_COLS
    account_features = args.account_features.split(',') if args.account_features else []
    transaction_features = args.transaction_features.split(',') if args.transaction_features else []

    print("="*60, flush=True)
    if args.no_leakage:
        print("GAR Feature Generator (NO-LEAKAGE MODE)", flush=True)
    else:
        print("GAR Feature Generator (LEAKAGE MODE - NOT RECOMMENDED)", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    # 1. 加载数据
    df, card_col, entity_cols, account_features, transaction_features, has_label, label_col = load_and_preprocess_data(
        args.data, args.card_col, entity_cols, account_features, transaction_features
    )

    # 2. 分割数据
    train_idx, test_idx = split_data(df, train_ratio=args.train_ratio, seed=args.seed)
    print(f"[INFO] Data split: Train={len(train_idx)}, Test={len(test_idx)}", flush=True)

    # 3. 构建图（从完整数据构建，以获取所有邻居关系）
    tx_neighbors = build_graph(df, entity_cols)

    # 4. 根据模式构建特征
    if args.no_leakage and has_label:
        # 无泄漏模式：从训练集计算欺诈率
        train_df = df.iloc[train_idx]
        entity_fraud_maps, pair_fraud_maps = compute_fraud_rates_from_train(train_df, entity_cols, label_col)

        features_dict, feature_names = build_gar_features_no_leakage(
            df, train_idx, tx_neighbors, card_col,
            entity_cols, account_features, transaction_features,
            has_label, label_col, entity_fraud_maps, pair_fraud_maps
        )

        # 添加split列标记
        split标记 = np.array(['train' if i in train_idx else 'test' for i in range(len(df))])
    else:
        # 原始模式（有泄漏）
        features_dict, feature_names = build_gar_features(
            df, tx_neighbors, card_col, entity_cols, account_features,
            transaction_features, has_label, label_col
        )
        split标记 = None

    # 5. 导出或训练
    if export_only:
        if args.output_csv:
            export_features_to_csv(features_dict, feature_names, args.output_csv, df, has_label, split标记)
        else:
            print("[ERROR] --output-csv is required when using --export-features-only", flush=True)
    else:
        if has_label:
            results = train_classifier(features_dict, feature_names, has_label, label_col, split标记, seed=args.seed)
            if results:
                print(f"\nTrain AUC: {results['train_auc']:.4f}", flush=True)
                print(f"Test AUC: {results['test_auc']:.4f}", flush=True)
                print("\nTop 10 Features:", flush=True)
                for i, (name, imp) in enumerate(sorted(results['feature_importance'], key=lambda x: x[1], reverse=True)[:10]):
                    print(f"  {i+1:2d}. {name:<40} {imp:.4f}", flush=True)
        else:
            print("[INFO] White sample mode - use --export-features-only to save features", flush=True)
            os.makedirs(args.output_dir, exist_ok=True)
            output_csv = args.output_csv or f"{args.output_dir}/gar_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            export_features_to_csv(features_dict, feature_names, output_csv, df, has_label, split标记)

    print(f"\nTotal time: {(time.time()-start_time)/60:.1f} minutes", flush=True)


if __name__ == '__main__':
    main()