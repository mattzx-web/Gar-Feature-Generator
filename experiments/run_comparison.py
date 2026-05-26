"""
多方法对比实验：Baseline vs KG vs GAR
使用多个随机种子进行重复实验

支持数据集：
- Synthetic Financial (CSV)
- PaySim (CSV)
- Amazon (.mat) - CARE-GNN格式
- YelpChi (.mat) - CARE-GNN格式
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import sys
import os
import time
import json
from scipy.io import loadmat

sys.stdout.reconfigure(line_buffering=True)

SEEDS = [42, 123, 456, 789, 999]
TRAIN_RATIO = 0.7


def build_graph_from_df(df, entity_cols, threshold=300):
    """从DataFrame构建图"""
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


def build_graph_from_adj(adj_matrix):
    """从邻接矩阵构建图（每个节点为一条交易）"""
    from collections import defaultdict
    n = adj_matrix.shape[0]
    tx_neighbors = defaultdict(set)

    # Convert sparse to dense if needed
    if hasattr(adj_matrix, 'toarray'):
        adj_dense = adj_matrix.toarray()
    else:
        adj_dense = adj_matrix

    # 对于每个节点，找出所有邻居
    for i in range(n):
        # 找出与i相连的所有节点（边的两端都是交易）
        # homo是同构网络，节点i和j之间有边表示它们有关联
        neighbors = np.where(adj_dense[i, :] > 0)[0]
        if len(neighbors) < 300:  # threshold
            for j in neighbors:
                if i != j:
                    tx_neighbors[i].add(j)
                    tx_neighbors[j].add(i)

    return tx_neighbors


def build_kg_features(df, tx_neighbors, entity_cols, amount_col=None):
    """KG Brute Force 特征"""
    features = {}
    n = len(df)

    # Amount features
    if amount_col and amount_col in df.columns:
        amounts = df[amount_col].fillna(0).values.astype(np.float32)
        features['amount'] = amounts
        features['amount_log'] = np.log1p(np.abs(amounts))

    # Entity degree features
    for col in entity_cols:
        if col not in df.columns:
            continue
        degree_map = df[col].value_counts().to_dict()
        features[f'{col}_degree'] = df[col].map(degree_map).fillna(0).values.astype(np.float32)

    # 1-hop neighbor features
    n_1hop = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.float32)
    features['n_1hop'] = n_1hop
    features['n_1hop_log'] = np.log1p(n_1hop)

    if amount_col and amount_col in df.columns:
        amt_1hop_mean = np.zeros(n, dtype=np.float32)
        amt_1hop_std = np.zeros(n, dtype=np.float32)
        amounts = df[amount_col].fillna(0).values
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = amounts[list(neighs)]
                amt_1hop_mean[i] = np.mean(neigh_amts)
                amt_1hop_std[i] = np.std(neigh_amts) if len(neighs) > 1 else 0
        features['amt_1hop_mean'] = amt_1hop_mean
        features['amt_1hop_std'] = amt_1hop_std

    # Pair count features
    for i, col1 in enumerate(entity_cols[:3]):
        for col2 in entity_cols[i+1:4]:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            pairs = df[col1].astype(str) + '_' + df[col2].astype(str)
            pair_counts = pairs.map(pairs.value_counts())
            features[f'{col1}_{col2}_pair_count'] = pair_counts.fillna(0).values.astype(np.float32)

    return features


def build_gar_features_simple(df, tx_neighbors, entity_cols, amount_col, label_col,
                               train_idx, entity_fraud_maps, pair_fraud_maps):
    """简化GAR特征（无泄漏模式）"""
    features = {}
    n = len(df)

    # Amount features
    if amount_col and amount_col in df.columns:
        amounts = df[amount_col].fillna(0).values.astype(np.float32)
        features['amount'] = amounts
        features['amount_log'] = np.log1p(np.abs(amounts))

    # Entity frequency features
    for col in entity_cols:
        if col not in df.columns:
            continue
        freq_map = df[col].value_counts().to_dict()
        features[f'{col}_freq'] = df[col].map(freq_map).fillna(0).values.astype(np.float32)
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq'])

    # Neighbor features
    n_1hop = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.float32)
    features['n_1hop'] = n_1hop
    features['n_1hop_log'] = np.log1p(n_1hop)

    if amount_col and amount_col in df.columns:
        amt_1hop_mean = np.zeros(n, dtype=np.float32)
        amt_1hop_std = np.zeros(n, dtype=np.float32)
        amounts = df[amount_col].fillna(0).values
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = amounts[list(neighs)]
                amt_1hop_mean[i] = np.mean(neigh_amts)
                amt_1hop_std[i] = np.std(neigh_amts) if len(neighs) > 1 else 0
        features['amt_1hop_mean'] = amt_1hop_mean
        features['amt_1hop_std'] = amt_1hop_std

    # GAR fraud rate features (from train only)
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

    # Neighbor fraud rate (no leakage: only use train neighbors)
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
    """从训练集计算欺诈率映射"""
    entity_fraud_maps = {}
    for col in entity_cols:
        if col in train_df.columns:
            entity_fraud_maps[col] = train_df.groupby(col)[label_col].mean().to_dict()

    pair_fraud_maps = {}
    for i, col1 in enumerate(entity_cols[:3]):
        for col2 in entity_cols[i+1:4]:
            if col1 not in train_df.columns or col2 not in train_df.columns:
                continue
            pair_df = train_df[[col1, col2, label_col]].copy()
            pair_df['_pair'] = pair_df[col1].astype(str) + '_' + pair_df[col2].astype(str)
            pair_fraud_maps[f'{col1}_{col2}'] = pair_df.groupby('_pair')[label_col].mean().to_dict()

    return entity_fraud_maps, pair_fraud_maps


def run_single_experiment(df, entity_cols, label_col, seed):
    """运行单次实验"""
    np.random.seed(seed)

    # Split
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

    # Amount column
    amount_col = None
    for col in ['amount', 'Amount', 'transaction_amount', 'TransactionAmt']:
        if col in df.columns:
            amount_col = col
            break

    # Build graph
    valid_entity_cols = [c for c in entity_cols if c in df.columns]
    tx_neighbors = build_graph_from_df(df, valid_entity_cols)

    results = []

    # === Baseline ===
    if amount_col:
        X_train = train_df[[amount_col]].fillna(0).values
        X_test = test_df[[amount_col]].fillna(0).values
    else:
        X_train = np.zeros((len(train_df), 1))
        X_test = np.zeros((len(test_df), 1))

    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed)
    gb.fit(X_train, y_train)
    y_pred_proba = gb.predict_proba(X_test)[:, 1]
    y_pred = gb.predict(X_test)

    results.append({
        'method': 'Baseline',
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    })

    # === KG ===
    kg_features = build_kg_features(df, tx_neighbors, valid_entity_cols, amount_col)
    feat_names = list(kg_features.keys())
    X = np.column_stack([kg_features[f] for f in feat_names])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_train = X[train_idx]
    X_test = X[test_idx]

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=seed)
    gb.fit(X_train, y_train)
    y_pred_proba = gb.predict_proba(X_test)[:, 1]
    y_pred = gb.predict(X_test)

    results.append({
        'method': 'KG',
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    })

    # === GAR ===
    entity_fraud_maps, pair_fraud_maps = compute_fraud_rates_from_train(train_df, valid_entity_cols, label_col)
    gar_features = build_gar_features_simple(df, tx_neighbors, valid_entity_cols, amount_col, label_col,
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

    results.append({
        'method': 'GAR',
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    })

    return results


def run_mat_experiment(mat_path, dataset_name, net_key, seed):
    """运行基于邻接矩阵的实验（Amazon/YelpChi）"""
    np.random.seed(seed)

    mat = loadmat(mat_path)
    adj = mat.get(net_key, mat.get('homo'))
    if adj is None:
        return None

    n = adj.shape[0]
    labels = mat['label'].flatten()

    # Create dataframe (each node is a sample)
    df = pd.DataFrame({'node_id': np.arange(n)})
    df['label'] = labels

    # Convert sparse to dense if needed
    if hasattr(adj, 'toarray'):
        adj_dense = adj.toarray()
    else:
        adj_dense = adj

    # Features from adjacency degree
    degrees = np.array([np.sum(adj_dense[i, :] > 0) for i in range(n)], dtype=np.float32)
    df['degree'] = degrees
    df['degree_log'] = np.log1p(degrees)

    # Build graph from adjacency
    tx_neighbors = build_graph_from_adj(adj)

    # 2-hop neighbors (using dense for row access)
    n_2hop = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neighs_1 = set(np.where(adj_dense[i, :] > 0)[0])
        for j in neighs_1:
            neighs_2 = set(np.where(adj_dense[j, :] > 0)[0])
            n_2hop[i] += len(neighs_2 - neighs_1 - {i})
    df['n_2hop'] = n_2hop
    df['n_2hop_log'] = np.log1p(n_2hop)

    label_col = 'label'
    entity_cols = ['node_id']

    # Split
    indices = np.arange(n)
    np.random.shuffle(indices)
    n_train = int(TRAIN_RATIO * n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    y_train = df.iloc[train_idx]['label'].values
    y_test = df.iloc[test_idx]['label'].values

    results = []

    # === Baseline ===
    X_train = df[['degree']].iloc[train_idx].values
    X_test = df[['degree']].iloc[test_idx].values

    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed)
    gb.fit(X_train, y_train)
    y_pred_proba = gb.predict_proba(X_test)[:, 1]
    y_pred = gb.predict(X_test)

    results.append({
        'method': 'Baseline',
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    })

    # === KG ===
    feat_cols = ['degree', 'degree_log', 'n_2hop', 'n_2hop_log']
    X = df[feat_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_train = X[train_idx]
    X_test = X[test_idx]

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=seed)
    gb.fit(X_train, y_train)
    y_pred_proba = gb.predict_proba(X_test)[:, 1]
    y_pred = gb.predict(X_test)

    results.append({
        'method': 'KG',
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    })

    # === GAR (neighbor fraud rate from train only) ===
    train_idx_set = set(train_idx)
    train_label_map = dict(zip(train_idx, y_train))
    neigh_fraud_rates = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neighs = tx_neighbors.get(i, set())
        if neighs:
            train_neighs = [n for n in neighs if n in train_idx_set]
            if train_neighs:
                neigh_fraud_rates[i] = np.mean([train_label_map[n] for n in train_neighs])

    df['neigh_fraud_rate'] = neigh_fraud_rates
    feat_cols = ['degree', 'degree_log', 'n_2hop', 'n_2hop_log', 'neigh_fraud_rate']
    X = df[feat_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_train = X[train_idx]
    X_test = X[test_idx]

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=seed)
    gb.fit(X_train, y_train)
    y_pred_proba = gb.predict_proba(X_test)[:, 1]
    y_pred = gb.predict(X_test)

    results.append({
        'method': 'GAR',
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    })

    return results


def run_dataset_experiments(df, dataset_name, entity_cols, label_col):
    """在单个CSV数据集上运行多种子实验"""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} ({len(df)} rows, fraud_rate={df[label_col].mean():.4f})")
    print(f"{'='*70}")

    all_results = []

    for seed in SEEDS:
        try:
            results = run_single_experiment(df, entity_cols, label_col, seed)
            for r in results:
                r['seed'] = seed
                r['dataset'] = dataset_name
            all_results.extend(results)
            print(f"  Seed {seed}: Baseline={results[0]['auc']:.4f}, KG={results[1]['auc']:.4f}, GAR={results[2]['auc']:.4f}")
        except Exception as e:
            print(f"  [ERROR] Seed {seed}: {e}")

    return all_results


def run_mat_dataset_experiments(mat_path, dataset_name, net_key):
    """在单个.mat数据集上运行多种子实验"""
    mat = loadmat(mat_path)
    n = mat['label'].shape[1]
    fraud_rate = mat['label'].flatten().mean()

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} ({n} nodes, fraud_rate={fraud_rate:.4f})")
    print(f"{'='*70}")

    all_results = []

    for seed in SEEDS:
        try:
            results = run_mat_experiment(mat_path, dataset_name, net_key, seed)
            if results is None:
                continue
            for r in results:
                r['seed'] = seed
                r['dataset'] = dataset_name
            all_results.extend(results)
            print(f"  Seed {seed}: Baseline={results[0]['auc']:.4f}, KG={results[1]['auc']:.4f}, GAR={results[2]['auc']:.4f}")
        except Exception as e:
            print(f"  [ERROR] Seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def main():
    datasets = []

    # 1. Synthetic Financial
    try:
        df = pd.read_csv('/Users/matt/data/financial_fraud/synthetic_fraud_dataset.csv')
        datasets.append({
            'type': 'csv',
            'name': 'Synthetic_Financial',
            'df': df,
            'entity_cols': ['user_id', 'merchant_category', 'country', 'transaction_type'],
            'label_col': 'is_fraud'
        })
        print(f"[OK] Synthetic Financial: {len(df)} rows, fraud={df['is_fraud'].mean():.4f}")
    except Exception as e:
        print(f"[SKIP] Synthetic Financial: {e}")

    # 2. PaySim
    try:
        df = pd.read_csv('/Users/matt/data/paysim/paysim.csv')
        datasets.append({
            'type': 'csv',
            'name': 'PaySim',
            'df': df,
            'entity_cols': ['customer', 'merchant', 'type'],
            'label_col': 'isFraud'
        })
        print(f"[OK] PaySim: {len(df)} rows, fraud={df['isFraud'].mean():.4f}")
    except Exception as e:
        print(f"[SKIP] PaySim: {e}")

    # 3. Amazon.mat
    try:
        mat_path = '/Users/matt/data/CARE-GNN/data/Amazon.mat'
        if os.path.exists(mat_path):
            mat = loadmat(mat_path)
            n = mat['label'].shape[1]
            datasets.append({
                'type': 'mat',
                'name': 'Amazon',
                'mat_path': mat_path,
                'net_key': 'homo',
                'nodes': n,
                'fraud_rate': mat['label'].flatten().mean()
            })
            print(f"[OK] Amazon: {n} nodes, fraud={mat['label'].flatten().mean():.4f}")
    except Exception as e:
        print(f"[SKIP] Amazon: {e}")

    # 4. YelpChi.mat
    try:
        mat_path = '/Users/matt/data/CARE-GNN/data/YelpChi.mat'
        if os.path.exists(mat_path):
            mat = loadmat(mat_path)
            n = mat['label'].shape[1]
            datasets.append({
                'type': 'mat',
                'name': 'YelpChi',
                'mat_path': mat_path,
                'net_key': 'homo',
                'nodes': n,
                'fraud_rate': mat['label'].flatten().mean()
            })
            print(f"[OK] YelpChi: {n} nodes, fraud={mat['label'].flatten().mean():.4f}")
    except Exception as e:
        print(f"[SKIP] YelpChi: {e}")

    # Run experiments
    all_results = []

    for ds in datasets:
        try:
            if ds['type'] == 'csv':
                results = run_dataset_experiments(ds['df'], ds['name'], ds['entity_cols'], ds['label_col'])
            elif ds['type'] == 'mat':
                results = run_mat_dataset_experiments(ds['mat_path'], ds['name'], ds['net_key'])
            all_results.extend(results)
        except Exception as e:
            print(f"[ERROR] {ds['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate and print results
    print(f"\n{'='*70}")
    print("FINAL RESULTS (5 seeds each)")
    print(f"{'='*70}")

    output = {}

    for dataset in sorted(set([r['dataset'] for r in all_results])):
        print(f"\n--- {dataset} ---")
        ds_results = [r for r in all_results if r['dataset'] == dataset]
        output[dataset] = {}

        for method in ['Baseline', 'KG', 'GAR']:
            method_results = [r for r in ds_results if r['method'] == method]
            if not method_results:
                continue

            aucs = [r['auc'] for r in method_results]
            precs = [r['precision'] for r in method_results]
            recs = [r['recall'] for r in method_results]
            f1s = [r['f1'] for r in method_results]

            output[dataset][method] = {
                'auc_mean': float(np.mean(aucs)),
                'auc_std': float(np.std(aucs)),
                'precision_mean': float(np.mean(precs)),
                'recall_mean': float(np.mean(recs)),
                'f1_mean': float(np.mean(f1s)),
                'f1_std': float(np.std(f1s)),
                'seeds': SEEDS
            }

            print(f"  {method:10s}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
                  f"P={np.mean(precs):.4f} R={np.mean(recs):.4f} F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    # Save
    os.makedirs('/Users/matt/Gar-Feature-Generator/outputs', exist_ok=True)
    output_path = '/Users/matt/Gar-Feature-Generator/outputs/multi_seed_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()