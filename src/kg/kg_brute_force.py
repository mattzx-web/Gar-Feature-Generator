"""
KG Brute Force Feature Generator
基于知识图谱暴力特征扩展的欺诈检测特征工程方法

功能:
- 从原始CSV文件加载交易数据
- 构建实体图结构 (card1, card2, card3, card4, addr1等)
- 生成枚举式图特征 (度/计数/VCD统计/配对计数等)
- 导出增强特征集为CSV文件，或直接训练分类器

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
DEFAULT_NEIGHBOR_THRESHOLD = 100
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 6


def load_and_preprocess_data(data_dir, entity_cols):
    """加载并预处理数据"""
    print(f"[INFO] Loading data from {data_dir}...", flush=True)

    train_trans = pd.read_csv(f"{data_dir}/train_transaction.csv")
    train_identity = pd.read_csv(f"{data_dir}/train_identity.csv")
    train = train_trans.merge(train_identity, on='TransactionID', how='left')
    del train_trans, train_identity

    n_full = len(train)
    print(f"[INFO] Full dataset: {n_full}", flush=True)

    for col in entity_cols:
        if col in train.columns:
            train[col] = train[col].fillna(-1)
            if train[col].dtype == 'object':
                le = LabelEncoder()
                train[col] = le.fit_transform(train[col].astype(str))

    n = len(train)
    n_train = int(DEFAULT_TRAIN_RATIO * n)

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
    """构建交易图结构"""
    print(f"[INFO] Building graph...", flush=True)

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


def build_kg_features(df, tx_neighbors, tx_2hop_neighbors, train_data, is_train=True):
    """
    构建KG Brute Force特征

    Args:
        df: 输入数据（训练或测试）
        tx_neighbors: 1跳邻居映射
        tx_2hop_neighbors: 2跳邻居映射
        train_data: 训练数据（用于计算统计量）
        is_train: 是否为训练集

    Returns:
        features: 特征字典
    """
    features = {}

    # 1. Entity degree features
    for col in DEFAULT_ENTITY_COLS[:5]:
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

    return features


def export_features_to_csv(train_feat, test_feat, feature_names, y_train, y_test,
                            train_idx, test_idx, output_path):
    """
    将特征导出为CSV文件

    Args:
        train_feat, test_feat: 训练/测试特征字典
        feature_names: 特征名列表
        y_train, y_test: 标签
        train_idx, test_idx: 索引
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # 构建训练集DataFrame
    train_df = pd.DataFrame({name: train_feat[name] for name in feature_names})
    train_df['isFraud'] = y_train
    train_df['split'] = 'train'
    train_df['original_idx'] = train_idx

    # 构建测试集DataFrame
    test_df = pd.DataFrame({name: test_feat[name] for name in feature_names})
    test_df['isFraud'] = y_test
    test_df['split'] = 'test'
    test_df['original_idx'] = test_idx

    # 合并
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"[INFO] Features exported to {output_path}", flush=True)
    print(f"[INFO] Shape: {df.shape} (train: {len(train_df)}, test: {len(test_df)})", flush=True)

    return output_path


def train_kg_classifier(X_train, y_train, X_test, y_test, feature_names, seed=42):
    """
    训练KG分类器

    Args:
        X_train, y_train, X_test, y_test: 训练/测试数据
        feature_names: 特征名列表
        seed: 随机种子

    Returns:
        results: 包含AUC和特征重要性的字典
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
        'feature_importance': list(zip(feature_names, gb_full.feature_importances_.tolist()))
    }

    # Baseline
    gb_base = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                        subsample=0.8, random_state=seed)
    gb_base.fit(X_train[:, :2], y_train)
    results['baseline'] = {
        'test_auc': float(roc_auc_score(y_test, gb_base.predict_proba(X_test[:, :2])[:, 1]))
    }

    return results


def run_full_experiment(data_dir, seed=42, output_dir='./outputs'):
    """运行完整流程：特征生成 + 模型训练"""
    np.random.seed(seed)
    print(f"\n[Seed {seed}] Starting KG Brute Force experiment...", flush=True)

    # 1. 加载数据
    train_data, test_data, y_train, y_test, train_indices, test_indices = load_and_preprocess_data(
        data_dir, DEFAULT_ENTITY_COLS
    )

    # 2. 构建图
    tx_neighbors, tx_2hop_neighbors = build_graph(train_data, DEFAULT_ENTITY_COLS)

    # 3. 构建特征
    train_features = build_kg_features(train_data, tx_neighbors, tx_2hop_neighbors, train_data, is_train=True)
    test_features = build_kg_features(test_data, tx_neighbors, tx_2hop_neighbors, train_data, is_train=False)

    train_features = pd.DataFrame(train_features).fillna(0).replace([np.inf, -np.inf], 0)
    test_features = pd.DataFrame(test_features).fillna(0).replace([np.inf, -np.inf], 0)

    feature_names = list(train_features.columns)
    print(f"[Seed {seed}] KG Features: {len(feature_names)} dimensions", flush=True)

    # 4. 转换为numpy数组
    X_train = train_features.values
    X_test = test_features.values

    # 5. 训练和评估
    results = train_kg_classifier(X_train, y_train, X_test, y_test, feature_names, seed)

    print(f"[Seed {seed}] KG Brute Force: Train={results['kg_brute_force']['train_auc']:.4f}, "
          f"Test={results['kg_brute_force']['test_auc']:.4f}", flush=True)
    print(f"[Seed {seed}] Baseline: {results['baseline']['test_auc']:.4f}", flush=True)

    results['feature_names'] = feature_names

    return results, train_features, test_features, train_indices, test_indices


def main():
    parser = argparse.ArgumentParser(
        description='KG Brute Force Feature Generator for Fraud Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 完整流程（特征生成 + 模型训练）
  python src/kg_brute_force_generator.py --data-dir /path/to/data

  # 仅生成特征并导出CSV
  python src/kg_brute_force_generator.py --data-dir /path/to/data \\
                                          --export-features-only \\
                                          --output-csv ./features/kg_train.csv

  # 多种子验证
  python src/kg_brute_force_generator.py --data-dir /path/to/data --seeds 42 123 456

  # 导出特征后用独立脚本训练
  python src/kg_brute_force_generator.py --data-dir /path/to/data --export-features-only --output-csv ./features.csv
  python src/train_classifier.py --features ./features.csv --model kg
        """
    )

    parser.add_argument('--data-dir', type=str, required=True,
                        help='IEEE-CIS数据集根目录')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='输出目录（默认: ./outputs）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='多种子验证模式')
    parser.add_argument('--export-features-only', action='store_true',
                        help='仅生成特征，不训练模型')
    parser.add_argument('--feature-only', action='store_true',
                        help='与--export-features-only相同')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='特征CSV输出路径')

    args = parser.parse_args()

    export_only = args.export_features_only or args.feature_only

    print("="*60, flush=True)
    print("KG Brute Force Feature Generator", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    if args.seeds:
        # 多种子模式
        all_results = []
        all_feature_names = None

        for seed in args.seeds:
            result, train_features, test_features, train_indices, test_indices = run_full_experiment(
                args.data_dir, seed, args.output_dir
            )
            result['seed'] = seed
            all_results.append(result)
            all_feature_names = result.get('feature_names')

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
            'feature_names': all_feature_names
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
        # 1. 加载数据
        train_data, test_data, y_train, y_test, train_indices, test_indices = load_and_preprocess_data(
            args.data_dir, DEFAULT_ENTITY_COLS
        )

        # 2. 构建图
        tx_neighbors, tx_2hop_neighbors = build_graph(train_data, DEFAULT_ENTITY_COLS)

        # 3. 构建特征
        train_feat = build_kg_features(train_data, tx_neighbors, tx_2hop_neighbors, train_data, is_train=True)
        test_feat = build_kg_features(test_data, tx_neighbors, tx_2hop_neighbors, train_data, is_train=False)

        train_feat = pd.DataFrame(train_feat).fillna(0).replace([np.inf, -np.inf], 0)
        test_feat = pd.DataFrame(test_feat).fillna(0).replace([np.inf, -np.inf], 0)

        feature_names = list(train_feat.columns)

        if export_only:
            # 仅导出特征
            if args.output_csv:
                export_features_to_csv(
                    train_feat.to_dict('list'), test_feat.to_dict('list'),
                    feature_names, y_train, y_test, train_indices, test_indices, args.output_csv
                )
            else:
                print("[ERROR] --output-csv is required when using --export-features-only", flush=True)
        else:
            # 完整流程
            X_train = train_feat.values
            X_test = test_feat.values

            results = train_kg_classifier(X_train, y_train, X_test, y_test, feature_names, args.seed)

            print(f"\nKG Brute Force: Train={results['kg_brute_force']['train_auc']:.4f}, "
                  f"Test={results['kg_brute_force']['test_auc']:.4f}", flush=True)
            print(f"Baseline: {results['baseline']['test_auc']:.4f}", flush=True)

    print(f"\nTotal time: {(time.time()-start_time)/60:.1f} minutes", flush=True)


if __name__ == '__main__':
    main()