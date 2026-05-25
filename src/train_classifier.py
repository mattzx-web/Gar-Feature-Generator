"""
独立模型训练器

从CSV文件加载已生成的特征，训练分类器并评估结果。

用法:
    # 使用GAR特征训练
    python src/train_classifier.py --features ./features/gar_features.csv --model gar

    # 使用KG Brute Force特征训练
    python src/train_classifier.py --features ./features/kg_features.csv --model kg

    # 指定输出目录和种子
    python src/train_classifier.py --features ./features/gar_features.csv --model gar \\
                                   --output-dir ./results --seed 42

    # 多种子验证
    python src/train_classifier.py --features ./features/gar_features.csv --model gar \\
                                   --seeds 42 123 456
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import json
import os
import sys
import argparse
from datetime import datetime
import time

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# 默认配置
DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 6


def load_features_from_csv(csv_path, train_ratio=0.7, seed=42):
    """
    从CSV文件加载特征

    Args:
        csv_path: 特征CSV文件路径
        train_ratio: 训练集比例（默认0.7）
        seed: 随机种子

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print(f"[INFO] Loading features from {csv_path}...", flush=True)
    df = pd.read_csv(csv_path)

    # 检测是否有split列
    has_split = 'split' in df.columns
    has_label = 'isFraud' in df.columns or 'is_fraud' in df.columns

    # 获取特征列（排除meta列和非数值列）
    meta_cols = ['split', 'original_idx', 'isFraud', 'is_fraud']
    key_cols = ['card_id', 'timestamp', '时间戳', 'TransactionID', 'transaction_id']
    exclude_cols = meta_cols + key_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 过滤非数值列
    feature_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    if has_split and has_label:
        train_mask = df['split'] == 'train'
        test_mask = df['split'] == 'test'
        label_col = 'isFraud' if 'isFraud' in df.columns else 'is_fraud'
        X_train = df.loc[train_mask, feature_cols].values
        X_test = df.loc[test_mask, feature_cols].values
        y_train = df.loc[train_mask, label_col].values
        y_test = df.loc[test_mask, label_col].values
    elif has_label:
        # 确定标签列名
        label_col = 'isFraud' if 'isFraud' in df.columns else 'is_fraud'
        # 无split列，随机划分
        n = len(df)
        indices = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(indices)
        n_train = int(train_ratio * n)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        X_train = df[feature_cols].iloc[train_idx].values
        X_test = df[feature_cols].iloc[test_idx].values
        y_train = df[label_col].iloc[train_idx].values
        y_test = df[label_col].iloc[test_idx].values
    else:
        # 无标签，白样本模式不支持训练
        raise ValueError("No label column found. Please provide features with 'isFraud' column.")

    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}", flush=True)
    print(f"[INFO] Features: {len(feature_cols)}", flush=True)

    return X_train, X_test, y_train, y_test, feature_cols


def train_gar_classifier(X_train, y_train, X_test, y_test, feature_names, seed=42):
    """训练GAR分类器"""
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


def train_kg_classifier(X_train, y_train, X_test, y_test, feature_names, seed=42):
    """训练KG Brute Force分类器"""
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


def train_baseline_classifier(X_train, y_train, X_test, y_test, seed=42):
    """训练仅用TransactionAmt的Baseline分类器"""
    results = {}

    gb_base = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                        subsample=0.8, random_state=seed)
    gb_base.fit(X_train[:, :2], y_train)
    results['baseline'] = {
        'train_auc': float(roc_auc_score(y_train, gb_base.predict_proba(X_train[:, :2])[:, 1])),
        'test_auc': float(roc_auc_score(y_test, gb_base.predict_proba(X_test[:, :2])[:, 1]))
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train Classifier from Feature CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用GAR特征训练
  python src/train_classifier.py --features ./features/gar_features.csv --model gar

  # 使用KG Brute Force特征训练
  python src/train_classifier.py --features ./features/kg_features.csv --model kg

  # 多种子验证
  python src/train_classifier.py --features ./features/gar_features.csv --model gar \\
                                 --seeds 42 123 456
        """
    )

    parser.add_argument('--features', type=str, required=True,
                        help='特征CSV文件路径（必需）')
    parser.add_argument('--model', type=str, default='gar',
                        choices=['gar', 'kg', 'baseline'],
                        help='模型类型: gar | kg | baseline（默认: gar）')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='输出目录（默认: ./outputs）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='多种子验证模式')

    args = parser.parse_args()

    print("="*60, flush=True)
    print(f"Train Classifier: {args.model.upper()}", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    # 加载特征
    X_train, X_test, y_train, y_test, feature_names = load_features_from_csv(args.features)

    if args.seeds:
        # 多种子模式
        all_results = []

        for seed in args.seeds:
            np.random.seed(seed)
            print(f"\n[Seed {seed}] Training...", flush=True)

            if args.model == 'gar':
                results = train_gar_classifier(X_train, y_train, X_test, y_test, feature_names, seed)
            elif args.model == 'kg':
                results = train_kg_classifier(X_train, y_train, X_test, y_test, feature_names, seed)
            else:
                results = train_baseline_classifier(X_train, y_train, X_test, y_test, seed)

            results['seed'] = seed
            all_results.append(results)

            model_key = args.model if args.model != 'baseline' else 'baseline'
            if model_key in results:
                print(f"[Seed {seed}] Test AUC: {results[model_key]['test_auc']:.4f}", flush=True)

        # 聚合结果
        print("\n" + "="*60, flush=True)
        print("AGGREGATED RESULTS", flush=True)
        print("="*60, flush=True)

        model_key = args.model if args.model != 'baseline' else 'baseline'
        if model_key in all_results[0]:
            aucs = [r[model_key]['test_auc'] for r in all_results]
            print(f"{args.model}: Test={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

            if 'train_auc' in all_results[0][model_key]:
                train_aucs = [r[model_key]['train_auc'] for r in all_results]
                print(f"{args.model} Train: {np.mean(train_aucs):.4f}±{np.std(train_aucs):.4f}")

        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)
        out_file = f"{args.output_dir}/train_results_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output = {
            'experiment': f'Train {args.model.upper()} Classifier from CSV',
            'features_file': args.features,
            'seeds': args.seeds,
            'aggregated': {},
            'feature_names': feature_names
        }

        if model_key in all_results[0]:
            aucs = [r[model_key]['test_auc'] for r in all_results]
            output['aggregated'][model_key] = {
                'test_auc_mean': float(np.mean(aucs)),
                'test_auc_std': float(np.std(aucs)),
                'individual': [float(x) for x in aucs]
            }
            if 'train_auc' in all_results[0][model_key]:
                train_aucs = [r[model_key]['train_auc'] for r in all_results]
                output['aggregated'][model_key]['train_auc_mean'] = float(np.mean(train_aucs))
                output['aggregated'][model_key]['train_auc_std'] = float(np.std(train_aucs))

        with open(out_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {out_file}", flush=True)
    else:
        # 单种子模式
        np.random.seed(args.seed)

        if args.model == 'gar':
            results = train_gar_classifier(X_train, y_train, X_test, y_test, feature_names, args.seed)
        elif args.model == 'kg':
            results = train_kg_classifier(X_train, y_train, X_test, y_test, feature_names, args.seed)
        else:
            results = train_baseline_classifier(X_train, y_train, X_test, y_test, args.seed)

        model_key = args.model if args.model != 'baseline' else 'baseline'
        # Map short names to actual keys
        key_map = {'gar': 'gar_full', 'kg': 'kg_brute_force', 'baseline': 'baseline'}
        actual_key = key_map.get(model_key, model_key)
        if actual_key in results:
            print(f"\n{actual_key.upper()} Results:", flush=True)
            print(f"  Train AUC: {results[actual_key]['train_auc']:.4f}", flush=True)
            print(f"  Test AUC:  {results[actual_key]['test_auc']:.4f}", flush=True)

            # 打印Top 10特征重要性
            if 'feature_importance' in results[actual_key]:
                print("\nTop 10 Feature Importance:", flush=True)
                feat_imp = sorted(results[actual_key]['feature_importance'], key=lambda x: x[1], reverse=True)[:10]
                for i, (name, imp) in enumerate(feat_imp):
                    print(f"  {i+1:2d}. {name:<40} {imp:.4f}", flush=True)

        if 'baseline' in results and actual_key != 'baseline':
            print(f"\nBaseline (TransactionAmt only) Test AUC: {results['baseline']['test_auc']:.4f}", flush=True)

    print(f"\nTotal time: {(time.time()-start_time)/60:.1f} minutes", flush=True)


if __name__ == '__main__':
    main()