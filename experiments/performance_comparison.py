"""
GAR CPU vs Dist Performance Comparison

比较 GAR CPU 和 Dist 版本的性能差异。

用法:
    python experiments/performance_comparison.py --data ./data/test.csv --workers 4
"""

import pandas as pd
import numpy as np
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gar.gar_cpu import (
    load_and_preprocess_data, build_graph, split_data,
    compute_fraud_rates_from_train, build_gar_features_no_leakage
)
from src.gar.gar_dist import run_distributed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def train_and_eval(X, y, split_arr, seed=42):
    """训练并评估模型"""
    train_mask = np.array(split_arr) == 'train'
    test_mask = np.array(split_arr) == 'test'

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=seed
    )
    gb.fit(X_train, y_train)

    train_proba = gb.predict_proba(X_train)[:, 1]
    test_proba = gb.predict_proba(X_test)[:, 1]

    return {
        'train_auc': roc_auc_score(y_train, train_proba),
        'test_auc': roc_auc_score(y_test, test_proba),
        'feature_importance': list(zip(range(X.shape[1]), gb.feature_importances_))
    }


def main():
    parser = argparse.ArgumentParser(description='GAR CPU vs Dist Comparison')
    parser.add_argument('--data', type=str, required=True, help='CSV file path')
    parser.add_argument('--workers', type=int, default=4, help='Dist workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    print("="*60, flush=True)
    print("GAR CPU vs Dist Performance Comparison", flush=True)
    print("="*60, flush=True)

    # ===== CPU Version =====
    print("\n[Step 1/4] Running CPU version...", flush=True)
    start = time.time()

    entity_cols = ['card_id', 'merchant_id', 'device', 'is_night']
    account_features = ['card_level', 'card_location', 'card_type']
    transaction_features = ['amount', 'balance', 'is_cross_border']

    df, card_col, entity_cols, account_features, transaction_features, has_label, label_col = \
        load_and_preprocess_data(args.data, 'card_id', entity_cols, account_features, transaction_features)

    train_idx, test_idx = split_data(df, train_ratio=0.7, seed=args.seed)
    tx_neighbors = build_graph(df, entity_cols)

    train_df = df.iloc[train_idx]
    entity_fraud_maps, pair_fraud_maps = compute_fraud_rates_from_train(train_df, entity_cols, label_col)

    features_cpu, feature_names = build_gar_features_no_leakage(
        df, train_idx, tx_neighbors, 'card_id',
        entity_cols, account_features, transaction_features,
        has_label, label_col, entity_fraud_maps, pair_fraud_maps
    )

    split_arr = np.array(['train' if i in train_idx else 'test' for i in range(len(df))])
    features_cpu[label_col] = df[label_col].values

    X_cpu = np.column_stack([features_cpu[name] for name in feature_names])
    X_cpu = np.nan_to_num(X_cpu, nan=0, posinf=0, neginf=0)
    y_cpu = features_cpu[label_col]

    cpu_time = time.time() - start
    print(f"[CPU] Time: {cpu_time:.2f}s, Features: {len(feature_names)}", flush=True)

    # ===== Dist Version =====
    print("\n[Step 2/4] Running Dist version...", flush=True)
    start = time.time()

    dist_features = run_distributed(
        args.data, 'card_id', entity_cols, account_features,
        transaction_features, args.workers, None,
        no_leakage=True, train_ratio=0.7, seed=args.seed
    )

    # 获取Dist特征
    dist_df = pd.read_csv(args.data.replace('.csv', '') + '_dist_output.csv') if os.path.exists(args.data.replace('.csv', '') + '_dist_output.csv') else None

    dist_time = time.time() - start
    print(f"[Dist] Time: {dist_time:.2f}s", flush=True)

    # ===== Train & Eval =====
    print("\n[Step 3/4] Training models...", flush=True)

    results_cpu = train_and_eval(X_cpu, y_cpu, split_arr, args.seed)
    print(f"[CPU] Train AUC: {results_cpu['train_auc']:.4f}, Test AUC: {results_cpu['test_auc']:.4f}", flush=True)

    # ===== Summary =====
    print("\n[Step 4/4] Summary", flush=True)
    print("="*60, flush=True)
    print(f"{'Metric':<30} {'CPU':>15} {'Dist':>15}", flush=True)
    print("-"*60, flush=True)
    print(f"{'Time (s)':<30} {cpu_time:>15.2f} {dist_time:>15.2f}", flush=True)
    print(f"{'Speedup':<30} {'-':>15} {cpu_time/dist_time:>15.2f}x", flush=True)
    print(f"{'Features':<30} {len(feature_names):>15} {'-':>15}", flush=True)
    print(f"{'Test AUC':<30} {results_cpu['test_auc']:>15.4f} {'-':>15}", flush=True)
    print("="*60, flush=True)

    return results_cpu


if __name__ == '__main__':
    main()
