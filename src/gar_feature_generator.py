"""
GAR-Inspired Feature Generator
基于图关联规则的欺诈检测特征工程方法

功能:
- 从原始CSV文件加载交易数据
- 构建实体图结构 (card1, card2, addr1, P_emaildomain等)
- 计算欺诈率特征 (Entity Fraud Rates, Pair Fraud Rates, Neighbor Fraud Rate)
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
DEFAULT_ENTITY_COLS = ['card1', 'card2', 'addr1', 'P_emaildomain']
DEFAULT_NEIGHBOR_THRESHOLD = 300
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 6


def load_and_preprocess_data(data_dir, entity_cols):
    """加载并预处理数据"""
    print(f"[INFO] Loading data from {data_dir}...", flush=True)

    train_trans = pd.read_csv(f"{data_dir}/train_transaction.csv",
                              usecols=['TransactionID', 'TransactionAmt', 'card1', 'card2',
                                       'addr1', 'P_emaildomain', 'isFraud'])
    train_identity = pd.read_csv(f"{data_dir}/train_identity.csv",
                                 usecols=['TransactionID', 'DeviceInfo', 'DeviceType'])
    train = train_trans.merge(train_identity, on='TransactionID', how='left')
    del train_trans, train_identity

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
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_data = train.iloc[train_idx].copy()
    test_data = train.iloc[test_idx].copy()
    del train

    y_train = train_data['isFraud'].values
    y_test = test_data['isFraud'].values

    print(f"[INFO] Train: {len(train_data)}, Test: {len(test_data)}", flush=True)
    print(f"[INFO] Fraud rates: train={y_train.mean():.4f}, test={y_test.mean():.4f}", flush=True)

    return train_data, test_data, y_train, y_test, train_idx, test_idx


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

    return tx_neighbors


def build_gar_features(train_data, test_data, tx_neighbors, train_idx, y_train, entity_cols):
    """
    构建GAR特征 (仅特征生成，不含模型训练)

    Args:
        train_data, test_data: 训练/测试数据
        tx_neighbors: 图邻居映射
        train_idx, y_train: 训练集索引和标签
        entity_cols: 实体列

    Returns:
        train_features_dict, test_features_dict, feature_names
    """
    print(f"[INFO] Building GAR features...", flush=True)

    train_feat = {}
    test_feat = {}

    # 1. TransactionAmt特征
    train_feat['TransactionAmt'] = train_data['TransactionAmt'].fillna(0).values
    test_feat['TransactionAmt'] = test_data['TransactionAmt'].fillna(0).values
    train_feat['TransactionAmt_log'] = np.log1p(train_data['TransactionAmt'].fillna(0).values)
    test_feat['TransactionAmt_log'] = np.log1p(test_data['TransactionAmt'].fillna(0).values)

    # 2. degree特征
    train_feat['degree'] = np.array([len(tx_neighbors.get(i, set())) for i in range(len(train_data))])
    test_global_start = len(train_data)
    test_feat['degree'] = np.array([len(tx_neighbors.get(test_global_start + i, set())) for i in range(len(test_data))])

    # 3. Entity frequency特征
    for col in entity_cols[:4]:
        if col not in train_data.columns:
            continue
        freq_map = train_data[col].value_counts().to_dict()
        train_feat[f'{col}_freq'] = train_data[col].map(freq_map).fillna(0).values
        test_feat[f'{col}_freq'] = test_data[col].map(freq_map).fillna(0).values

    # 4. Entity fraud rates特征
    for col in entity_cols[:4]:
        if col not in train_data.columns:
            continue
        fraud_map = train_data.groupby(col)['isFraud'].mean().to_dict()
        train_feat[f'{col}_fraud_rate'] = train_data[col].map(fraud_map).fillna(0).values
        test_feat[f'{col}_fraud_rate'] = test_data[col].map(fraud_map).fillna(0).values

    # 5. Pair fraud rates特征
    for i, col1 in enumerate(entity_cols[:3]):
        for col2 in entity_cols[i+1:4]:
            if col1 not in train_data.columns or col2 not in train_data.columns:
                continue
            train_pair = train_data[col1].astype(str) + '_' + train_data[col2].astype(str)
            test_pair = test_data[col1].astype(str) + '_' + test_data[col2].astype(str)
            pair_map = train_data.assign(_pair=train_pair).groupby('_pair')['isFraud'].mean().to_dict()
            train_feat[f'{col1}_{col2}_pair_fraud'] = train_pair.map(pair_map).fillna(0).values
            test_feat[f'{col1}_{col2}_pair_fraud'] = test_pair.map(pair_map).fillna(0).values

    # 6. Neighbor fraud rate特征
    train_is_fraud = y_train
    train_indices_set = set(train_idx)

    train_neigh_fraud = []
    for i in range(len(train_data)):
        neighs = tx_neighbors.get(i, set())
        if neighs:
            train_neigh_fraud.append(train_is_fraud[list(neighs)].mean())
        else:
            train_neigh_fraud.append(0)
    train_feat['neigh_fraud_rate'] = np.array(train_neigh_fraud)

    test_neigh_fraud = []
    for i in range(len(test_data)):
        global_i = test_idx[i]
        neighs = tx_neighbors.get(global_i, set())
        train_neighs = [n for n in neighs if n in train_indices_set]
        if train_neighs:
            local_neighs = [np.where(train_idx == n)[0][0] for n in train_neighs]
            test_neigh_fraud.append(train_is_fraud[local_neighs].mean())
        else:
            test_neigh_fraud.append(0)
    test_feat['neigh_fraud_rate'] = np.array(test_neigh_fraud)

    feature_names = list(train_feat.keys())
    print(f"[INFO] GAR Features: {len(feature_names)} dimensions", flush=True)

    return train_feat, test_feat, feature_names


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


def train_gar_classifier(X_train, y_train, X_test, y_test, feature_names, seed=42):
    """
    训练GAR分类器

    Args:
        X_train, y_train, X_test, y_test: 训练/测试数据
        feature_names: 特征名列表
        seed: 随机种子

    Returns:
        results: 包含AUC和特征重要性的字典
    """
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    results = {}

    # GAR Full
    gb_full = GradientBoostingClassifier(
        n_estimators=DEFAULT_N_ESTIMATORS, max_depth=DEFAULT_MAX_DEPTH,
        learning_rate=0.1, subsample=0.8, random_state=seed
    )
    gb_full.fit(X_train, y_train)
    train_proba = gb_full.predict_proba(X_train)[:, 1]
    test_proba = gb_full.predict_proba(X_test)[:, 1]
    results['gar_full'] = {
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
    print(f"\n[Seed {seed}] Starting GAR experiment...", flush=True)

    # 1. 加载数据
    train_data, test_data, y_train, y_test, train_idx, test_idx = load_and_preprocess_data(
        data_dir, DEFAULT_ENTITY_COLS
    )

    # 2. 构建图
    tx_neighbors = build_graph(train_data, DEFAULT_ENTITY_COLS)

    # 3. 构建特征
    train_feat, test_feat, feature_names = build_gar_features(
        train_data, test_data, tx_neighbors, train_idx, y_train, DEFAULT_ENTITY_COLS
    )

    # 4. 转换为numpy数组
    X_train = np.column_stack([train_feat[k] for k in feature_names])
    X_test = np.column_stack([test_feat[k] for k in feature_names])
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    # 5. 训练和评估
    results = train_gar_classifier(X_train, y_train, X_test, y_test, feature_names, seed)

    print(f"[Seed {seed}] GAR Full: Train={results['gar_full']['train_auc']:.4f}, "
          f"Test={results['gar_full']['test_auc']:.4f}", flush=True)
    print(f"[Seed {seed}] Baseline: {results['baseline']['test_auc']:.4f}", flush=True)

    return results, feature_names


def main():
    parser = argparse.ArgumentParser(
        description='GAR-Inspired Feature Generator for Fraud Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 完整流程（特征生成 + 模型训练）
  python src/gar_feature_generator.py --data-dir /path/to/data

  # 仅生成特征并导出CSV
  python src/gar_feature_generator.py --data-dir /path/to/data \\
                                      --export-features-only \\
                                      --output-csv ./features/gar_train.csv

  # 多种子验证
  python src/gar_feature_generator.py --data-dir /path/to/data --seeds 42 123 456

  # 导出特征后用独立脚本训练
  python src/gar_feature_generator.py --data-dir /path/to/data --export-features-only --output-csv ./features.csv
  python src/train_classifier.py --features ./features.csv --model gar
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

    # 统一 --feature-only 和 --export-features-only
    export_only = args.export_features_only or args.feature_only

    print("="*60, flush=True)
    print("GAR-Inspired Feature Generator", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    if args.seeds:
        # 多种子模式
        all_results = []
        all_feature_names = None

        for seed in args.seeds:
            result, feature_names = run_full_experiment(args.data_dir, seed, args.output_dir)
            result['seed'] = seed
            all_results.append(result)
            all_feature_names = feature_names

        # 聚合结果
        print("\n" + "="*60, flush=True)
        print("AGGREGATED RESULTS", flush=True)
        print("="*60, flush=True)

        for model in ['gar_full', 'baseline']:
            if model in all_results[0]:
                aucs = [r[model]['test_auc'] for r in all_results]
                print(f"{model}: Test={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)
        out_file = f"{args.output_dir}/gar_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output = {
            'experiment': 'GAR-Inspired Feature Generator',
            'seeds': args.seeds,
            'aggregated': {},
            'feature_names': all_feature_names
        }

        for model in ['gar_full', 'baseline']:
            if model in all_results[0]:
                aucs = [r[model]['test_auc'] for r in all_results]
                output['aggregated'][model] = {
                    'test_auc_mean': float(np.mean(aucs)),
                    'test_auc_std': float(np.std(aucs)),
                    'individual': [float(x) for x in aucs]
                }
                if model == 'gar_full' and 'train_auc' in all_results[0][model]:
                    train_aucs = [r[model]['train_auc'] for r in all_results]
                    output['aggregated'][model]['train_auc_mean'] = float(np.mean(train_aucs))
                    output['aggregated'][model]['train_auc_std'] = float(np.std(train_aucs))

        with open(out_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {out_file}", flush=True)
    else:
        # 单种子模式
        # 1. 加载数据
        train_data, test_data, y_train, y_test, train_idx, test_idx = load_and_preprocess_data(
            args.data_dir, DEFAULT_ENTITY_COLS
        )

        # 2. 构建图
        tx_neighbors = build_graph(train_data, DEFAULT_ENTITY_COLS)

        # 3. 构建特征
        train_feat, test_feat, feature_names = build_gar_features(
            train_data, test_data, tx_neighbors, train_idx, y_train, DEFAULT_ENTITY_COLS
        )

        if export_only:
            # 仅导出特征
            if args.output_csv:
                export_features_to_csv(train_feat, test_feat, feature_names,
                                      y_train, y_test, train_idx, test_idx, args.output_csv)
            else:
                print("[ERROR] --output-csv is required when using --export-features-only", flush=True)
        else:
            # 完整流程
            X_train = np.column_stack([train_feat[k] for k in feature_names])
            X_test = np.column_stack([test_feat[k] for k in feature_names])
            X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
            X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

            results = train_gar_classifier(X_train, y_train, X_test, y_test, feature_names, args.seed)

            print(f"\nGAR Full: Train={results['gar_full']['train_auc']:.4f}, "
                  f"Test={results['gar_full']['test_auc']:.4f}", flush=True)
            print(f"Baseline: {results['baseline']['test_auc']:.4f}", flush=True)

    print(f"\nTotal time: {(time.time()-start_time)/60:.1f} minutes", flush=True)


if __name__ == '__main__':
    main()