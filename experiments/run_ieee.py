"""
IEEE-CIS Fraud Detection 实验
快速多方法对比：Baseline vs KG vs GAR
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import sys
import os
import json
import time

sys.stdout.reconfigure(line_buffering=True)

SEEDS = [42, 123, 456, 789, 999]
TRAIN_RATIO = 0.7


def build_graph_from_df(df, entity_cols, threshold=300):
    from collections import defaultdict
    tx_neighbors = defaultdict(set)
    n = len(df)

    for col in entity_cols:
        if col not in df.columns:
            continue
        groups = df.groupby(col).indices
        for val, idx_list in groups.items():
            if 1 < len(idx_list) < threshold:
                for idx in idx_list:
                    tx_neighbors[idx].update(idx_list)

    for idx in tx_neighbors:
        tx_neighbors[idx].discard(idx)

    return tx_neighbors


def build_kg_features(df, tx_neighbors, entity_cols):
    features = {}
    n = len(df)

    if 'TransactionAmt' in df.columns:
        amounts = df['TransactionAmt'].fillna(0).values.astype(np.float32)
        features['TransactionAmt'] = amounts
        features['TransactionAmt_log'] = np.log1p(np.abs(amounts))

    for col in entity_cols:
        if col not in df.columns:
            continue
        degree_map = df[col].value_counts().to_dict()
        features[f'{col}_degree'] = df[col].map(degree_map).fillna(0).values.astype(np.float32)

    n_1hop = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.float32)
    features['n_1hop'] = n_1hop
    features['n_1hop_log'] = np.log1p(n_1hop)

    if 'TransactionAmt' in df.columns:
        amt_1hop_mean = np.zeros(n, dtype=np.float32)
        amt_1hop_std = np.zeros(n, dtype=np.float32)
        amounts = df['TransactionAmt'].fillna(0).values
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = amounts[list(neighs)]
                amt_1hop_mean[i] = np.mean(neigh_amts)
                amt_1hop_std[i] = np.std(neigh_amts) if len(neighs) > 1 else 0
        features['amt_1hop_mean'] = amt_1hop_mean
        features['amt_1hop_std'] = amt_1hop_std

    return features


def build_gar_features(df, tx_neighbors, entity_cols, label_col, train_idx, entity_fraud_maps, pair_fraud_maps):
    features = {}
    n = len(df)

    if 'TransactionAmt' in df.columns:
        amounts = df['TransactionAmt'].fillna(0).values.astype(np.float32)
        features['TransactionAmt'] = amounts
        features['TransactionAmt_log'] = np.log1p(np.abs(amounts))

    for col in entity_cols:
        if col not in df.columns:
            continue
        freq_map = df[col].value_counts().to_dict()
        features[f'{col}_freq'] = df[col].map(freq_map).fillna(0).values.astype(np.float32)
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq'])

    n_1hop = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.float32)
    features['n_1hop'] = n_1hop
    features['n_1hop_log'] = np.log1p(n_1hop)

    if 'TransactionAmt' in df.columns:
        amt_1hop_mean = np.zeros(n, dtype=np.float32)
        amt_1hop_std = np.zeros(n, dtype=np.float32)
        amounts = df['TransactionAmt'].fillna(0).values
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = amounts[list(neighs)]
                amt_1hop_mean[i] = np.mean(neigh_amts)
                amt_1hop_std[i] = np.std(neigh_amts) if len(neighs) > 1 else 0
        features['amt_1hop_mean'] = amt_1hop_mean
        features['amt_1hop_std'] = amt_1hop_std

    if entity_fraud_maps:
        for col in entity_cols:
            if col not in df.columns or col not in entity_fraud_maps:
                continue
            features[f'{col}_fraud_rate'] = df[col].map(entity_fraud_maps[col]).fillna(0).values.astype(np.float32)

    if pair_fraud_maps:
        for col_pair, fraud_map in pair_fraud_maps.items():
            col1, col2 = col_pair.split('_', 1)
            if col1 not in df.columns or col2 not in df.columns:
                continue
            pair_values = (df[col1].astype(str) + '_' + df[col2].astype(str)).values
            features[f'{col1}_{col2}_pair_fraud_rate'] = np.array([fraud_map.get(p, 0) for p in pair_values], dtype=np.float32)

    if label_col and train_idx is not None:
        train_idx_set = set(train_idx)
        train_labels = df.iloc[train_idx][label_col].values
        train_label_map = dict(zip(train_idx, train_labels))
        neigh_fraud_rates = np.zeros(n, dtype=np.float32)
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                train_neighs = [n for n in neighs if n in train_idx_set]
                if train_neighs:
                    neigh_fraud_rates[i] = np.mean([train_label_map[n] for n in train_neighs])
        features['neigh_fraud_rate'] = neigh_fraud_rates

    return features


def compute_fraud_rates_from_train(train_df, entity_cols, label_col):
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


def main():
    print("="*60)
    print("IEEE-CIS Fraud Detection Experiments")
    print("="*60)

    start_time = time.time()

    # Load IEEE-CIS data
    print("\n[INFO] Loading train_transaction.csv...")
    df = pd.read_csv('/Users/matt/ieee-fraud-detection/train_transaction.csv',
                     usecols=['TransactionID', 'isFraud', 'TransactionAmt',
                             'card1', 'card2', 'card3', 'card4', 'addr1', 'addr2',
                             'P_emaildomain', 'R_emaildomain'])
    print(f"[INFO] Loaded {len(df)} rows, fraud rate: {df['isFraud'].mean():.4f}")

    # Try to merge identity
    identity_path = '/Users/matt/ieee-fraud-detection/train_identity.csv'
    if os.path.exists(identity_path):
        print("[INFO] Merging with train_identity.csv...")
        identity = pd.read_csv(identity_path, usecols=['TransactionID', 'id_01', 'id_02', 'id_05', 'id_06'])
        df = df.merge(identity, on='TransactionID', how='left')
        print(f"[INFO] After merge: {len(df)} rows")

    entity_cols = ['card1', 'card2', 'card3', 'card4', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
    label_col = 'isFraud'

    # Preprocess
    for col in entity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)

    print(f"\n[INFO] Entity columns: {entity_cols}")

    all_results = []

    for seed in SEEDS:
        seed_start = time.time()
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)

        n = len(df)
        indices = np.arange(n)
        np.random.shuffle(indices)
        n_train = int(TRAIN_RATIO * n)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        y_train = train_df[label_col].values
        y_test = test_df[label_col].values

        valid_entity_cols = [c for c in entity_cols if c in df.columns]
        print(f"[Seed {seed}] Building graph...", flush=True)
        tx_neighbors = build_graph_from_df(df, valid_entity_cols)

        results = []

        # Baseline
        print(f"[Seed {seed}] Baseline...", end=" ", flush=True)
        X_train = train_df[['TransactionAmt']].fillna(0).values
        X_test = test_df[['TransactionAmt']].fillna(0).values
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed)
        gb.fit(X_train, y_train)
        y_pred_proba = gb.predict_proba(X_test)[:, 1]
        y_pred = gb.predict(X_test)
        results.append({'method': 'Baseline', 'auc': roc_auc_score(y_test, y_pred_proba),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall': recall_score(y_test, y_pred, zero_division=0),
                        'f1': f1_score(y_test, y_pred, zero_division=0)})
        print(f"AUC={results[-1]['auc']:.4f}")

        # KG
        print(f"[Seed {seed}] KG...", end=" ", flush=True)
        kg_features = build_kg_features(df, tx_neighbors, valid_entity_cols)
        feat_names = list(kg_features.keys())
        X = np.column_stack([kg_features[f] for f in feat_names])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_train = X[train_idx]
        X_test = X[test_idx]
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=seed)
        gb.fit(X_train, y_train)
        y_pred_proba = gb.predict_proba(X_test)[:, 1]
        y_pred = gb.predict(X_test)
        results.append({'method': 'KG', 'auc': roc_auc_score(y_test, y_pred_proba),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall': recall_score(y_test, y_pred, zero_division=0),
                        'f1': f1_score(y_test, y_pred, zero_division=0)})
        print(f"AUC={results[-1]['auc']:.4f}")

        # GAR
        print(f"[Seed {seed}] GAR...", end=" ", flush=True)
        entity_fraud_maps, pair_fraud_maps = compute_fraud_rates_from_train(train_df, valid_entity_cols, label_col)
        gar_features = build_gar_features(df, tx_neighbors, valid_entity_cols, label_col,
                                          train_idx, entity_fraud_maps, pair_fraud_maps)
        gar_names = list(gar_features.keys())
        X = np.column_stack([gar_features[f] for f in gar_names])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_train = X[train_idx]
        X_test = X[test_idx]
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=seed)
        gb.fit(X_train, y_train)
        y_pred_proba = gb.predict_proba(X_test)[:, 1]
        y_pred = gb.predict(X_test)
        results.append({'method': 'GAR', 'auc': roc_auc_score(y_test, y_pred_proba),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall': recall_score(y_test, y_pred, zero_division=0),
                        'f1': f1_score(y_test, y_pred, zero_division=0)})
        print(f"AUC={results[-1]['auc']:.4f}")

        for r in results:
            r['seed'] = seed
            r['dataset'] = 'IEEE_CIS'
        all_results.extend(results)

        print(f"[Seed {seed}] Time: {time.time()-seed_start:.1f}s")

    # Aggregate
    print(f"\n{'='*60}")
    print("IEEE-CIS FINAL RESULTS")
    print(f"{'='*60}")

    output = {'IEEE_CIS': {}}

    for method in ['Baseline', 'KG', 'GAR']:
        method_results = [r for r in all_results if r['method'] == method]
        if not method_results:
            continue

        aucs = [r['auc'] for r in method_results]
        precs = [r['precision'] for r in method_results]
        recs = [r['recall'] for r in method_results]
        f1s = [r['f1'] for r in method_results]

        output['IEEE_CIS'][method] = {
            'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
            'precision_mean': float(np.mean(precs)), 'recall_mean': float(np.mean(recs)),
            'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
            'seeds': [r['seed'] for r in method_results]
        }

        print(f"{method:10s}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
              f"P={np.mean(precs):.4f} R={np.mean(recs):.4f} F1={np.mean(f1s):.4f}")

    os.makedirs('/Users/matt/Gar-Feature-Generator/outputs', exist_ok=True)
    output_path = '/Users/matt/Gar-Feature-Generator/outputs/ieee_cis_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nTotal time: {(time.time()-start_time)/60:.1f} minutes")
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()