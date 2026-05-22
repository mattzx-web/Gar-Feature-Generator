"""
KG Brute Force Feature Generator
基于知识图谱暴力特征扩展的欺诈检测特征工程方法

功能:
- 从原始CSV文件加载交易数据
- 构建实体图结构 (card1, card2, card3, card4, addr1等)
- 生成枚举式图特征 (度/计数/VCD统计/配对计数等)
- 输出增强特征集用于下游分类器

作者: Matt
日期: 2026-05
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
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
DEFAULT_ENTITY_COLS = ['card1', 'card2', 'card3', 'card4', 'addr1', 'addr2',
                       'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo']
DEFAULT_NEIGHBOR_THRESHOLD = 100  # 邻居节点数量上限（避免过多邻居）
DEFAULT_TRAIN_RATIO = 0.7         # 训练集比例
DEFAULT_N_ESTIMATORS = 200        # GBDT树数量
DEFAULT_MAX_DEPTH = 6             # GBDT深度


def load_and_preprocess_data(data_dir, entity_cols):
    """
    加载并预处理数据

    Args:
        data_dir: 数据目录路径
        entity_cols: 实体列名列表

    Returns:
        train_data, test_data, y_train, y_test, train_idx, test_idx
    """
    print(f"[INFO] Loading data from {data_dir}...", flush=True)

    # 加载交易数据和身份数据
    train_trans = pd.read_csv(f"{data_dir}/train_transaction.csv")
    train_identity = pd.read_csv(f"{data_dir}/train_identity.csv")
    train = train_trans.merge(train_identity, on='TransactionID', how='left')
    del train_trans, train_identity

    n_full = len(train)
    print(f"[INFO] Full dataset: {n_full}", flush=True)

    # 实体列编码
    for col in entity_cols:
        if col in train.columns:
            train[col] = train[col].fillna(-1)
            if train[col].dtype == 'object':
                le = LabelEncoder()
                train[col] = le.fit_transform(train[col].astype(str))

    n = len(train)
    n_train = int(DEFAULT_TRAIN_RATIO * n)

    # 随机划分训练/测试集
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_data = train.iloc[train_indices].copy()
    test_data = train.iloc[test_indices].copy()

    y_train = train_data['isFraud'].values
    y_test = test_data['isFraud'].values

    print(f"[INFO] Train: {len(train_data)}, Test: {len(test_data)}", flush=True)
    print(f"[INFO] Fraud rates: train={y_train.mean():.4f}, test={y_test.mean():.4f}", flush=True)

    return train_data, test_data, y_train, y_test, train_indices, test_indices


def build_graph(train_data, entity_cols, neighbor_threshold=DEFAULT_NEIGHBOR_THRESHOLD):
    """
    构建交易图结构

    Args:
        train_data: 训练数据DataFrame
        entity_cols: 实体列名列表
        neighbor_threshold: 每个实体组合的邻居数量上限

    Returns:
        tx_neighbors: 交易ID到邻居集合的映射
        tx_2hop_neighbors: 2跳邻居映射
    """
    print(f"[INFO] Building graph...", flush=True)

    # Entity → Transaction mapping
    entity_to_tx = defaultdict(list)
    for col in entity_cols:
        if col in train_data.columns:
            for i, val in enumerate(train_data[col].values):
                entity_to_tx[(col, val)].append(i)

    # Build 1-hop neighbors
    tx_neighbors = defaultdict(set)
    for col in entity_cols:
        if col not in train_data.columns:
            continue
        groups = train_data.groupby(col).indices
        for val, idx_list in groups.items():
            if 1 < len(idx_list) < neighbor_threshold:
                for i in idx_list:
                    tx_neighbors[i].update(idx_list)

    for tx in tx_neighbors:
        tx_neighbors[tx].discard(tx)

    # Build 2-hop neighbors
    tx_2hop_neighbors = defaultdict(set)
    for tx, neighs in tx_neighbors.items():
        for n in neighs:
            tx_2hop_neighbors[tx].update(tx_neighbors.get(n, set()))
    tx_2hop_neighbors[tx].discard(tx)

    n_with_neigh = sum(1 for tx in tx_neighbors if len(tx_neighbors[tx]) > 0)
    print(f"[INFO] Nodes with neighbors: {n_with_neigh} / {len(train_data)}", flush=True)

    return tx_neighbors, tx_2hop_neighbors


def build_features(df, tx_neighbors, tx_2hop_neighbors, train_data, is_train=True):
    """
    构建KG Brute Force特征

    Features:
        1. Entity degree features (实体度特征)
        2. Entity count features (实体计数特征)
        3. 1-hop neighbor features (1跳邻居特征)
        4. 2-hop neighbor features (2跳邻居特征)
        5. Pair combination features (配对组合特征)
        6. V/C/D statistics (V/C/D统计特征)
        7. TransactionAmt features

    Args:
        df: 输入数据（训练或测试）
        tx_neighbors: 1跳邻居映射
        tx_2hop_neighbors: 2跳邻居映射
        train_data: 训练数据（用于计算统计量）
        is_train: 是否为训练集

    Returns:
        features: 特征DataFrame
    """
    features = {}

    # 1. Entity degree features
    for col in DEFAULT_ENTITY_COLS[:5]:  # card1-card4, addr1
        if col not in df.columns:
            continue
        degrees = []
        for i in range(len(df)):
            local_idx = i
            degrees.append(len(tx_neighbors.get(local_idx, set())))
        features[f'{col}_degree'] = np.array(degrees)

    # 2. Entity count features
    for col in DEFAULT_ENTITY_COLS[:5]:
        if col not in df.columns:
            continue
        val_counts = train_data[col].value_counts().to_dict()
        features[f'{col}_count'] = df[col].map(val_counts).fillna(0).values
        features[f'{col}_count_log'] = np.log1p(features[f'{col}_count'])

    # 3. 1-hop neighbor features
    n_1hop = []
    amt_1hop_mean = []
    amt_1hop_std = []
    for i in range(len(df)):
        neighs = tx_neighbors.get(i, set())
        n_1hop.append(len(neighs))
        if neighs:
            neigh_amts = train_data['TransactionAmt'].iloc[list(neighs)].values
            amt_1hop_mean.append(np.mean(neigh_amts))
            amt_1hop_std.append(np.std(neigh_amts))
        else:
            amt_1hop_mean.append(0)
            amt_1hop_std.append(0)
    features['n_1hop'] = np.array(n_1hop)
    features['amt_1hop_mean'] = np.array(amt_1hop_mean)
    features['amt_1hop_std'] = np.array(amt_1hop_std)

    # 4. 2-hop neighbor features
    n_2hop = []
    for i in range(len(df)):
        neighs = tx_2hop_neighbors.get(i, set())
        n_2hop.append(len(neighs))
    features['n_2hop'] = np.array(n_2hop)
    features['2hop_1hop_ratio'] = features['n_2hop'] / (features['n_1hop'] + 1)

    # 5. Pair combination features
    for i, col1 in enumerate(DEFAULT_ENTITY_COLS[:4]):
        for col2 in DEFAULT_ENTITY_COLS[i+1:5]:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            pairs = df[col1].astype(str) + '_' + df[col2].astype(str)
            pair_counts = pairs.map(pairs.value_counts())
            features[f'{col1}_{col2}_pair_count'] = pair_counts.values
            features[f'{col1}_{col2}_pair_count_log'] = np.log1p(pair_counts).values

    # 6. V/C/D statistics
    v_cols = [f'V{i}' for i in range(1, 340) if f'V{i}' in df.columns]
    c_cols = [f'C{i}' for i in range(1, 14) if f'C{i}' in df.columns]
    d_cols = [f'D{i}' for i in range(1, 15) if f'D{i}' in df.columns]

    if v_cols:
        v_data = df[v_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        features['V_mean'] = v_data.mean(axis=1).values
        features['V_std'] = v_data.std(axis=1).fillna(0).values
        features['V_sum'] = v_data.sum(axis=1).values
        features['V_max'] = v_data.max(axis=1).values
        features['V_nan_count'] = df[v_cols].isna().sum(axis=1).values

    if c_cols:
        c_data = df[c_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        features['C_mean'] = c_data.mean(axis=1).values
        features['C_std'] = c_data.std(axis=1).fillna(0).values
        features['C_sum'] = c_data.sum(axis=1).values

    if d_cols:
        d_data = df[d_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        features['D_mean'] = d_data.mean(axis=1).values
        features['D_std'] = d_data.std(axis=1).fillna(0).values

    # 7. Transaction features
    features['TransactionAmt'] = df['TransactionAmt'].fillna(0).values
    features['TransactionAmt_log'] = np.log1p(df['TransactionAmt'].fillna(0).values)

    return pd.DataFrame(features)


def train_and_evaluate(X_train, y_train, X_test, y_test, seed=42):
    """
    训练GBDT模型并评估

    Args:
        X_train, y_train, X_test, y_test: 训练/测试数据
        seed: 随机种子

    Returns:
        results: 包含各模型AUC的字典
    """
    results = {}

    # KG Brute Force Full
    gb_full = GradientBoostingClassifier(
        n_estimators=DEFAULT_N_ESTIMATORS, max_depth=DEFAULT_MAX_DEPTH,
        learning_rate=0.1, subsample=0.8, random_state=seed
    )
    gb_full.fit(X_train, y_train)

    train_proba = gb_full.predict_proba(X_train)[:, 1]
    test_proba = gb_full.predict_proba(X_test)[:, 1]
    results['kg_brute_force'] = {
        'train_auc': float(roc_auc_score(y_train, train_proba)),
        'test_auc': float(roc_auc_score(y_test, test_proba)),
        'feature_importance': list(zip(
            [f'f{i}' for i in range(X_train.shape[1])],
            gb_full.feature_importances_.tolist()
        ))
    }

    # Without neighbor fraud rate (ablation)
    # Note: KG Brute Force doesn't have neigh_fraud_rate, this is for consistency

    # Baseline
    gb_base = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=seed
    )
    gb_base.fit(X_train[:, :2], y_train)  # Just TransactionAmt and log
    results['baseline'] = {
        'test_auc': float(roc_auc_score(y_test, gb_base.predict_proba(X_test[:, :2])[:, 1]))
    }

    return results


def run_experiment(data_dir, seed=42, output_dir='./outputs'):
    """
    运行完整的KG Brute Force特征实验

    Args:
        data_dir: IEEE-CIS数据集根目录
        seed: 随机种子
        output_dir: 输出目录

    Returns:
        results: 实验结果字典
    """
    np.random.seed(seed)
    print(f"\n[Seed {seed}] Starting...", flush=True)

    # 1. 加载数据
    train_data, test_data, y_train, y_test, train_indices, test_indices = load_and_preprocess_data(
        data_dir, DEFAULT_ENTITY_COLS
    )

    # 2. 构建图
    tx_neighbors, tx_2hop_neighbors = build_graph(train_data, DEFAULT_ENTITY_COLS)

    # 3. 构建特征
    train_features = build_features(train_data, tx_neighbors, tx_2hop_neighbors, train_data, is_train=True)
    test_features = build_features(test_data, tx_neighbors, tx_2hop_neighbors, train_data, is_train=False)

    train_features = train_features.fillna(0).replace([np.inf, -np.inf], 0)
    test_features = test_features.fillna(0).replace([np.inf, -np.inf], 0)

    print(f"[Seed {seed}] Features: {train_features.shape[1]}", flush=True)

    # 4. 训练和评估
    X_train = train_features.values
    X_test = test_features.values

    results = train_and_evaluate(X_train, y_train, X_test, y_test, seed)

    print(f"[Seed {seed}] KG Brute Force: Train={results['kg_brute_force']['train_auc']:.4f}, "
          f"Test={results['kg_brute_force']['test_auc']:.4f}", flush=True)
    print(f"[Seed {seed}] Baseline: {results['baseline']['test_auc']:.4f}", flush=True)

    results['feature_names'] = list(train_features.columns)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='KG Brute Force Feature Generator for Fraud Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用默认数据路径运行
  python src/kg_brute_force_generator.py --data-dir /path/to/ieee-fraud-detection

  # 指定输出目录和种子
  python src/kg_brute_force_generator.py --data-dir /path/to/ieee-fraud-detection \\
                                         --output-dir ./results --seed 42

  # 多种子验证
  python src/kg_brute_force_generator.py --data-dir /path/to/ieee-fraud-detection \\
                                          --seeds 42 123 456
        """
    )

    parser.add_argument('--data-dir', type=str, required=True,
                        help='IEEE-CIS数据集根目录（包含train_transaction.csv和train_identity.csv）')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='输出目录（默认: ./outputs）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='多种子验证模式（如: --seeds 42 123 456）')

    args = parser.parse_args()

    print("="*60, flush=True)
    print("KG Brute Force Feature Generator", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    if args.seeds:
        # 多种子模式
        all_results = []
        for seed in args.seeds:
            result = run_experiment(args.data_dir, seed, args.output_dir)
            result['seed'] = seed
            all_results.append(result)

        # 聚合结果
        print("\n" + "="*60, flush=True)
        print("AGGREGATED RESULTS", flush=True)
        print("="*60, flush=True)

        for model in ['kg_brute_force', 'baseline']:
            if model in all_results[0]:
                aucs = [r[model]['test_auc'] for r in all_results]
                print(f"{model}: Test={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)
        out_file = f"{args.output_dir}/kg_brute_force_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output = {
            'experiment': 'KG Brute Force Feature Generator',
            'seeds': args.seeds,
            'aggregated': {},
            'feature_names': all_results[0].get('feature_names', [])
        }

        for model in ['kg_brute_force', 'baseline']:
            if model in all_results[0]:
                aucs = [r[model]['test_auc'] for r in all_results]
                output['aggregated'][model] = {
                    'test_auc_mean': float(np.mean(aucs)),
                    'test_auc_std': float(np.std(aucs)),
                    'individual': [float(x) for x in aucs]
                }
                if model == 'kg_brute_force' and 'train_auc' in all_results[0][model]:
                    train_aucs = [r[model]['train_auc'] for r in all_results]
                    output['aggregated'][model]['train_auc_mean'] = float(np.mean(train_aucs))
                    output['aggregated'][model]['train_auc_std'] = float(np.std(train_aucs))

        with open(out_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {out_file}", flush=True)
    else:
        # 单种子模式
        result = run_experiment(args.data_dir, args.seed, args.output_dir)

        print(f"\nKG Brute Force: Train={result['kg_brute_force']['train_auc']:.4f}, "
              f"Test={result['kg_brute_force']['test_auc']:.4f}", flush=True)
        print(f"Baseline: {result['baseline']['test_auc']:.4f}", flush=True)

    print(f"\nTotal time: {(time.time()-start_time)/60:.1f} minutes", flush=True)


if __name__ == '__main__':
    main()