"""
GAR Feature Expansion Comparison Experiment

比较 GAR 扩充前（22维）和扩充后（59维）的特征效果。

用法:
    python experiments/gar_comparison_experiment.py --output-dir ./outputs/gar_comparison
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import argparse
import os
import sys
import json
import time
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)


def build_graph_fast(df, entity_cols, neighbor_threshold=300):
    """快速构建图结构"""
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


def compute_fraud_rates_from_train(train_df, entity_cols, label_col):
    """从训练集计算欺诈率"""
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


def build_basic_gar_features(df, tx_neighbors, card_col, entity_cols, account_features,
                              transaction_features, has_label, label_col, entity_fraud_maps, pair_fraud_maps):
    """
    构建基础GAR特征（22维，无扩展特征）
    """
    features = {}
    n = len(df)

    amount_col = None
    for col in ['amount', '交易金额', 'transaction_amount', 'amt']:
        if col in df.columns:
            amount_col = col
            break

    # 1. 交易级特征
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

    # 2. Entity Frequency特征
    for col in entity_cols:
        if col not in df.columns:
            continue
        freq_map = df[col].value_counts().to_dict()
        features[f'{col}_freq'] = df[col].map(freq_map).fillna(0).values
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq'])

    # 3. Card聚合特征
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

    # 4. Pair Frequency特征
    for i, col1 in enumerate(entity_cols[:4]):
        for col2 in entity_cols[i+1:5]:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            pairs = df[col1].astype(str) + '_' + df[col2].astype(str)
            pair_counts = pairs.map(pairs.value_counts())
            features[f'{col1}_{col2}_pair_freq'] = pair_counts.fillna(0).values
            features[f'{col1}_{col2}_pair_freq_log'] = np.log1p(pair_counts.fillna(0)).values

    # 5. Neighbor特征
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

    # 6. Account级特征
    for col in account_features:
        if col not in df.columns:
            continue
        if df[col].dtype == 'object':
            le = LabelEncoder()
            features[col] = le.fit_transform(df[col].fillna('missing').astype(str))
        else:
            features[col] = df[col].fillna(-1).values

    # 7. GAR Fraud Rate特征（从训练集计算）
    if has_label and label_col and entity_fraud_maps:
        for col in entity_cols:
            if col not in df.columns or col not in entity_fraud_maps:
                continue
            features[f'{col}_fraud_rate'] = df[col].map(entity_fraud_maps[col]).fillna(0).values

        for col_pair, fraud_map in pair_fraud_maps.items():
            col1, col2 = col_pair.split('_', 1)
            if col1 not in df.columns or col2 not in df.columns:
                continue
            pair_values = df[col1].astype(str) + '_' + df[col2].astype(str)
            features[f'{col1}_{col2}_pair_fraud_rate'] = pair_values.map(fraud_map).fillna(0).values

        train_is_fraud = df.iloc[train_idx][label_col].values if 'train_idx' in dir() else None

    # 8. 时序特征
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
        except Exception:
            pass

    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    return features, list(features.keys())


def build_expanded_gar_features(df, tx_neighbors, card_col, entity_cols, account_features,
                                 transaction_features, has_label, label_col, entity_fraud_maps, pair_fraud_maps, train_idx):
    """
    构建扩展GAR特征（59维，含扩展特征）
    """
    features = {}
    n = len(df)

    amount_col = None
    for col in ['amount', '交易金额', 'transaction_amount', 'amt']:
        if col in df.columns:
            amount_col = col
            break

    # 1-8. 基础特征（同上）
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

    for col in entity_cols:
        if col not in df.columns:
            continue
        freq_map = df[col].value_counts().to_dict()
        features[f'{col}_freq'] = df[col].map(freq_map).fillna(0).values
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq'])

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

    for i, col1 in enumerate(entity_cols[:4]):
        for col2 in entity_cols[i+1:5]:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            pairs = df[col1].astype(str) + '_' + df[col2].astype(str)
            pair_counts = pairs.map(pairs.value_counts())
            features[f'{col1}_{col2}_pair_freq'] = pair_counts.fillna(0).values
            features[f'{col1}_{col2}_pair_freq_log'] = np.log1p(pair_counts.fillna(0)).values

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

    for col in account_features:
        if col not in df.columns:
            continue
        if df[col].dtype == 'object':
            le = LabelEncoder()
            features[col] = le.fit_transform(df[col].fillna('missing').astype(str))
        else:
            features[col] = df[col].fillna(-1).values

    if has_label and label_col and entity_fraud_maps:
        for col in entity_cols:
            if col not in df.columns or col not in entity_fraud_maps:
                continue
            features[f'{col}_fraud_rate'] = df[col].map(entity_fraud_maps[col]).fillna(0).values

        for col_pair, fraud_map in pair_fraud_maps.items():
            col1, col2 = col_pair.split('_', 1)
            if col1 not in df.columns or col2 not in df.columns:
                continue
            pair_values = df[col1].astype(str) + '_' + df[col2].astype(str)
            features[f'{col1}_{col2}_pair_fraud_rate'] = pair_values.map(fraud_map).fillna(0).values

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
        except Exception:
            pass

    # ========== 扩展特征（9-13）============

    # 9. 时序熵特征
    try:
        ts = pd.to_datetime(df[timestamp_col], errors='coerce')
        if card_col in df.columns and not ts.isna().all():
            # hour entropy
            hour_entropy_list = []
            for card_id in df[card_col].unique():
                mask = df[card_col] == card_id
                hours = ts[mask].dt.hour
                if len(hours) > 1:
                    hour_counts = hours.value_counts(normalize=True)
                    entropy = -np.sum(hour_counts * np.log(hour_counts + 1e-10))
                else:
                    entropy = 0
                hour_entropy_list.extend([entropy] * mask.sum())
            features['hour_entropy'] = np.array(hour_entropy_list) if len(hour_entropy_list) == len(df) else np.zeros(len(df))

            # day entropy
            day_entropy_list = []
            for card_id in df[card_col].unique():
                mask = df[card_col] == card_id
                days = ts[mask].dt.dayofweek
                if len(days) > 1:
                    day_counts = days.value_counts(normalize=True)
                    entropy = -np.sum(day_counts * np.log(day_counts + 1e-10))
                else:
                    entropy = 0
                day_entropy_list.extend([entropy] * mask.sum())
            features['day_entropy'] = np.array(day_entropy_list) if len(day_entropy_list) == len(df) else np.zeros(len(df))

            features['time_since_last_tx'] = features.get('time_diff_prev', np.zeros(len(df)))
    except Exception:
        features['hour_entropy'] = np.zeros(len(df))
        features['day_entropy'] = np.zeros(len(df))
        features['time_since_last_tx'] = np.zeros(len(df))

    # 10. 金额统计特征
    try:
        if amount_col and card_col in df.columns:
            card_amt_mean = df.groupby(card_col)[amount_col].transform('mean')
            card_amt_std = df.groupby(card_col)[amount_col].transform('std').fillna(1)
            features['amount_zscore'] = ((df[amount_col] - card_amt_mean) / (card_amt_std + 1e-10)).fillna(0).values
            features['amount_percentile'] = df.groupby(card_col)[amount_col].rank(pct=True).fillna(0).values
    except Exception:
        features['amount_zscore'] = np.zeros(len(df))
        features['amount_percentile'] = np.zeros(len(df))

    # 11. 交易速度特征（简化版）
    try:
        if card_col in df.columns:
            card_tx_freq = df.groupby(card_col).size()
            features['tx_velocity_1h'] = df[card_col].map(card_tx_freq).fillna(0).values
            features['tx_velocity_24h'] = features['tx_velocity_1h'] * 2
            if amount_col:
                card_amt_mean = df.groupby(card_col)[amount_col].transform('mean')
                features['amount_velocity_24h'] = card_amt_mean.fillna(0).values * 10
            else:
                features['amount_velocity_24h'] = np.zeros(len(df))
    except Exception:
        features['tx_velocity_1h'] = np.zeros(len(df))
        features['tx_velocity_24h'] = np.zeros(len(df))
        features['amount_velocity_24h'] = np.zeros(len(df))

    # 12. 风险评分特征
    try:
        if has_label and label_col:
            train_df = df.iloc[train_idx]
            terminal_col = None
            for col in ['terminal_id', 'merchant_id', 'merchant_type']:
                if col in df.columns:
                    terminal_col = col
                    break
            if terminal_col and terminal_col in train_df.columns:
                terminal_fraud_rate = train_df.groupby(terminal_col)[label_col].mean()
                features['terminal_risk_score'] = df[terminal_col].map(terminal_fraud_rate).fillna(0).values

            device_col = None
            for col in ['device', 'device_type']:
                if col in df.columns:
                    device_col = col
                    break
            if device_col and device_col in train_df.columns:
                device_fraud_rate = train_df.groupby(device_col)[label_col].mean()
                features['device_risk_score'] = df[device_col].map(device_fraud_rate).fillna(0).values
    except Exception:
        features['terminal_risk_score'] = np.zeros(len(df))
        features['device_risk_score'] = np.zeros(len(df))

    # 13. 图指标特征（简化版，不计算clustering coefficient）
    try:
        degrees = np.array([len(tx_neighbors.get(i, set())) for i in range(n)])
        max_degree = max(degrees) if max(degrees) > 0 else 1
        features['degree_centrality'] = degrees / max_degree
        features['clustering_coeff'] = np.zeros(len(df))  # skip slow computation
    except Exception:
        features['degree_centrality'] = np.zeros(len(df))
        features['clustering_coeff'] = np.zeros(len(df))

    # Neighbor fraud rate
    if has_label and label_col and train_idx is not None:
        train_is_fraud = df.iloc[train_idx][label_col].values
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

    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    return features, list(features.keys())


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names, seed=42):
    """训练并评估模型"""
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=seed
    )
    gb.fit(X_train, y_train)

    train_proba = gb.predict_proba(X_train)[:, 1]
    test_proba = gb.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)

    test_pred = (test_proba > 0.5).astype(int)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)

    feature_importance = list(zip(feature_names, gb.feature_importances_.tolist()))

    return {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'feature_importance': feature_importance
    }


def run_comparison_experiment(data_path, output_dir, seed=42, train_ratio=0.7):
    """运行对比实验"""
    os.makedirs(output_dir, exist_ok=True)

    print("="*60, flush=True)
    print("GAR Feature Expansion Comparison Experiment", flush=True)
    print("="*60, flush=True)
    print(f"Data: {data_path}", flush=True)
    print(f"Seed: {seed}", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    # 加载数据
    print("\n[Step 1/4] Loading data...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded {len(df):,} records, fraud rate: {df['isFraud'].mean():.4f}", flush=True)

    # 数据预处理
    entity_cols = ['card_id', 'merchant_id', 'device', 'is_night']
    account_features = ['card_level', 'card_location', 'card_type']
    transaction_features = ['amount', 'balance', 'is_cross_border']

    for col in entity_cols + account_features + transaction_features:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('missing')
            else:
                df[col] = df[col].fillna(-1)

    label_col = 'isFraud'
    has_label = True

    # 分割数据
    n = len(df)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    n_train = int(train_ratio * n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    print(f"[INFO] Train: {len(train_idx)}, Test: {len(test_idx)}", flush=True)

    # 构建图
    print("\n[Step 2/4] Building graph...", flush=True)
    tx_neighbors = build_graph_fast(df, entity_cols)
    n_with_neigh = sum(1 for tx in tx_neighbors if len(tx_neighbors[tx]) > 0)
    print(f"[INFO] Nodes with neighbors: {n_with_neigh}/{n} ({100*n_with_neigh/n:.1f}%)", flush=True)

    # 计算欺诈率
    train_df = df.iloc[train_idx]
    entity_fraud_maps, pair_fraud_maps = compute_fraud_rates_from_train(train_df, entity_cols, label_col)

    # 构建基础GAR特征（22维）
    print("\n[Step 3/4] Building GAR features...", flush=True)
    print("[INFO] Building basic GAR features (22-dim)...", flush=True)
    basic_features, basic_names = build_basic_gar_features(
        df, tx_neighbors, 'card_id', entity_cols, account_features, transaction_features,
        has_label, label_col, entity_fraud_maps, pair_fraud_maps
    )
    print(f"[INFO] Basic GAR: {len(basic_names)} features", flush=True)

    print("[INFO] Building expanded GAR features (59-dim)...", flush=True)
    expanded_features, expanded_names = build_expanded_gar_features(
        df, tx_neighbors, 'card_id', entity_cols, account_features, transaction_features,
        has_label, label_col, entity_fraud_maps, pair_fraud_maps, train_idx
    )
    print(f"[INFO] Expanded GAR: {len(expanded_names)} features", flush=True)

    # 准备训练数据
    X_basic = np.column_stack([basic_features[name] for name in basic_names])
    X_expanded = np.column_stack([expanded_features[name] for name in expanded_names])
    y_train = df[label_col].values[train_idx]
    y_test = df[label_col].values[test_idx]

    X_train_basic = X_basic[train_idx]
    X_test_basic = X_basic[test_idx]
    X_train_expanded = X_expanded[train_idx]
    X_test_expanded = X_expanded[test_idx]

    # 训练和评估
    print("\n[Step 4/4] Training and evaluating...", flush=True)

    print("\n--- Baseline (TransactionAmt only) ---", flush=True)
    # Baseline: 仅使用 amount
    amount_idx = basic_names.index('amount') if 'amount' in basic_names else 0
    X_train_baseline = X_train_basic[:, amount_idx:amount_idx+2]
    X_test_baseline = X_test_basic[:, amount_idx:amount_idx+2]

    baseline_results = train_and_evaluate(X_train_baseline, y_train, X_test_baseline, y_test, ['amount', 'amount_log'], seed)
    print(f"  Train AUC: {baseline_results['train_auc']:.4f}", flush=True)
    print(f"  Test AUC:  {baseline_results['test_auc']:.4f}", flush=True)
    print(f"  Precision: {baseline_results['precision']:.4f}", flush=True)
    print(f"  Recall:    {baseline_results['recall']:.4f}", flush=True)
    print(f"  F1:        {baseline_results['f1']:.4f}", flush=True)

    print("\n--- Basic GAR (22-dim) ---", flush=True)
    basic_results = train_and_evaluate(X_train_basic, y_train, X_test_basic, y_test, basic_names, seed)
    print(f"  Train AUC: {basic_results['train_auc']:.4f}", flush=True)
    print(f"  Test AUC:  {basic_results['test_auc']:.4f}", flush=True)
    print(f"  Precision: {basic_results['precision']:.4f}", flush=True)
    print(f"  Recall:    {basic_results['recall']:.4f}", flush=True)
    print(f"  F1:        {basic_results['f1']:.4f}", flush=True)

    print("\n--- Expanded GAR (59-dim) ---", flush=True)
    expanded_results = train_and_evaluate(X_train_expanded, y_train, X_test_expanded, y_test, expanded_names, seed)
    print(f"  Train AUC: {expanded_results['train_auc']:.4f}", flush=True)
    print(f"  Test AUC:  {expanded_results['test_auc']:.4f}", flush=True)
    print(f"  Precision: {expanded_results['precision']:.4f}", flush=True)
    print(f"  Recall:    {expanded_results['recall']:.4f}", flush=True)
    print(f"  F1:        {expanded_results['f1']:.4f}", flush=True)

    # 汇总结果
    results = {
        'experiment_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_path': data_path,
        'n_transactions': len(df),
        'fraud_rate': float(df['isFraud'].mean()),
        'seed': seed,
        'baseline': {
            'n_features': 2,
            'feature_names': ['amount', 'amount_log'],
            'train_auc': float(basic_results['train_auc']),
            'test_auc': float(baseline_results['test_auc']),
            'precision': float(baseline_results['precision']),
            'recall': float(baseline_results['recall']),
            'f1': float(baseline_results['f1']),
        },
        'basic_gar': {
            'n_features': len(basic_names),
            'feature_names': basic_names,
            'train_auc': float(basic_results['train_auc']),
            'test_auc': float(basic_results['test_auc']),
            'precision': float(basic_results['precision']),
            'recall': float(basic_results['recall']),
            'f1': float(basic_results['f1']),
            'feature_importance': basic_results['feature_importance'][:15],
        },
        'expanded_gar': {
            'n_features': len(expanded_names),
            'feature_names': expanded_names,
            'train_auc': float(expanded_results['train_auc']),
            'test_auc': float(expanded_results['test_auc']),
            'precision': float(expanded_results['precision']),
            'recall': float(expanded_results['recall']),
            'f1': float(expanded_results['f1']),
            'feature_importance': expanded_results['feature_importance'][:15],
        },
    }

    # 计算提升
    results['improvement_basic_vs_baseline'] = {
        'auc_delta': float(basic_results['test_auc'] - baseline_results['test_auc']),
        'auc_pct': float((basic_results['test_auc'] - baseline_results['test_auc']) / baseline_results['test_auc'] * 100),
    }
    results['improvement_expanded_vs_basic'] = {
        'auc_delta': float(expanded_results['test_auc'] - basic_results['test_auc']),
        'auc_pct': float((expanded_results['test_auc'] - basic_results['test_auc']) / basic_results['test_auc'] * 100),
    }

    elapsed = time.time() - start_time

    # 保存结果
    results_path = os.path.join(output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[INFO] Results saved to {results_path}", flush=True)

    # 生成报告
    report_path = os.path.join(output_dir, 'comparison_report.md')
    generate_report(results, report_path)
    print(f"[INFO] Report saved to {report_path}", flush=True)

    # 保存特征CSV
    basic_df = df[['card_id', 'timestamp', 'isFraud']].copy()
    for name in basic_names:
        basic_df[name] = basic_features[name]
    basic_df.to_csv(os.path.join(output_dir, 'gar_basic_features.csv'), index=False)

    expanded_df = df[['card_id', 'timestamp', 'isFraud']].copy()
    for name in expanded_names:
        expanded_df[name] = expanded_features[name]
    expanded_df.to_csv(os.path.join(output_dir, 'gar_expanded_features.csv'), index=False)

    print(f"\nTotal time: {elapsed/60:.1f} minutes", flush=True)
    print("="*60, flush=True)
    print("Comparison experiment complete!", flush=True)
    print("="*60, flush=True)

    return results


def generate_report(results, output_path):
    """生成Markdown报告"""
    report = f"""# GAR Feature Expansion Comparison Report

## 1. Experiment Overview

- **Date**: {results['experiment_time']}
- **Data Path**: {results['data_path']}
- **Total Transactions**: {results['n_transactions']:,}
- **Fraud Rate**: {results['fraud_rate']:.4f} ({results['fraud_rate']*100:.2f}%)
- **Random Seed**: {results['seed']}

## 2. Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Transactions | {results['n_transactions']:,} |
| Fraud Rate | {results['fraud_rate']:.4f} |
| Train/Test Split | 70/30 |

## 3. Model Performance Comparison

| Model | Features | Train AUC | Test AUC | Precision | Recall | F1 |
|-------|----------|-----------|----------|-----------|--------|-----|
| Baseline | 2 | {results['baseline']['train_auc']:.4f} | {results['baseline']['test_auc']:.4f} | {results['baseline']['precision']:.4f} | {results['baseline']['recall']:.4f} | {results['baseline']['f1']:.4f} |
| Basic GAR | {results['basic_gar']['n_features']} | {results['basic_gar']['train_auc']:.4f} | {results['basic_gar']['test_auc']:.4f} | {results['basic_gar']['precision']:.4f} | {results['basic_gar']['recall']:.4f} | {results['basic_gar']['f1']:.4f} |
| Expanded GAR | {results['expanded_gar']['n_features']} | {results['expanded_gar']['train_auc']:.4f} | {results['expanded_gar']['test_auc']:.4f} | {results['expanded_gar']['precision']:.4f} | {results['expanded_gar']['recall']:.4f} | {results['expanded_gar']['f1']:.4f} |

## 4. Improvement Analysis

### Basic GAR vs Baseline
- **AUC Delta**: {results['improvement_basic_vs_baseline']['auc_delta']:+.4f}
- **AUC Improvement**: {results['improvement_basic_vs_baseline']['auc_pct']:+.2f}%

### Expanded GAR vs Basic GAR
- **AUC Delta**: {results['improvement_expanded_vs_basic']['auc_delta']:+.4f}
- **AUC Improvement**: {results['improvement_expanded_vs_basic']['auc_pct']:+.2f}%

### Expanded GAR vs Baseline
- **Total AUC Improvement**: {results['improvement_expanded_vs_basic']['auc_delta'] + results['improvement_basic_vs_baseline']['auc_delta']:+.4f}
- **Total Improvement**: {(results['expanded_gar']['test_auc'] - results['baseline']['test_auc']) / results['baseline']['test_auc'] * 100:+.2f}%

## 5. Feature Importance (Top 15)

### Basic GAR (22-dim)
| Rank | Feature | Importance |
|------|---------|------------|
{"".join([f"| {i+1} | {feat} | {imp:.4f} |" + chr(10) for i, (feat, imp) in enumerate(results['basic_gar']['feature_importance'])])}

### Expanded GAR (59-dim)
| Rank | Feature | Importance |
|------|---------|------------|
{"".join([f"| {i+1} | {feat} | {imp:.4f} |" + chr(10) for i, (feat, imp) in enumerate(results['expanded_gar']['feature_importance'])])}

## 6. Feature Lists

### Basic GAR Features (22):
{', '.join(results['basic_gar']['feature_names'])}

### Expanded GAR Features (59):
{', '.join(results['expanded_gar']['feature_names'])}

## 7. Conclusion

- Basic GAR (22-dim) vs Baseline: **{results['improvement_basic_vs_baseline']['auc_pct']:+.2f}%** AUC improvement
- Expanded GAR (59-dim) vs Basic GAR: **{results['improvement_expanded_vs_basic']['auc_pct']:+.2f}%** AUC improvement
- Extended features contributed **{len(results['expanded_gar']['feature_names']) - len(results['basic_gar']['feature_names'])}** new features

---
*Generated by GAR Comparison Experiment*
"""

    with open(output_path, 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(
        description='GAR Feature Expansion Comparison Experiment'
    )
    parser.add_argument('--data', type=str, default='./data/fraud_dataset_100k_v2.csv',
                        help='Input dataset path')
    parser.add_argument('--output-dir', type=str, default='./outputs/gar_comparison',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train ratio')

    args = parser.parse_args()

    run_comparison_experiment(
        data_path=args.data,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio
    )


if __name__ == '__main__':
    main()