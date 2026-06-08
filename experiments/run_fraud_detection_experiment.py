"""
Fraud Detection Experiment Pipeline

完整的实验流水线：从数据集生成到GAR特征生成、特征筛选、模型训练和报告生成。

用法:
    python experiments/run_fraud_detection_experiment.py \\
        --data ./data/simulated_fraud_dataset.csv \\
        --output-dir ./outputs/experiment_20260608
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
import json
import time
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gar.gar_cpu import (
    load_and_preprocess_data, build_graph, split_data,
    compute_fraud_rates_from_train, build_gar_features_no_leakage,
    export_features_to_csv
)
from src.gar.gar_feature_selector import select_high_fraud_rate_features, export_selected_features
from src.train_classifier import load_features_from_csv, train_gar_classifier


def run_experiment_pipeline(
    data_path: str,
    output_dir: str,
    seed: int = 42,
    train_ratio: float = 0.7,
    top_k_features: int = 20
):
    """
    运行完整的实验流水线

    Args:
        data_path: 输入数据集路径
        output_dir: 输出目录
        seed: 随机种子
        train_ratio: 训练集比例
        top_k_features: 选择的特征数量
    """
    os.makedirs(output_dir, exist_ok=True)

    print("="*60, flush=True)
    print("Fraud Detection Experiment Pipeline", flush=True)
    print("="*60, flush=True)
    print(f"Data: {data_path}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Seed: {seed}", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    results = {
        'experiment_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_path': data_path,
        'seed': seed,
        'train_ratio': train_ratio,
        'top_k_features': top_k_features,
        'dataset_stats': {},
        'feature_stats': {},
        'model_results': {},
        'feature_importance': []
    }

    # ========== Step 1: 数据集统计 ==========
    print("\n[Step 1/5] Analyzing dataset...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded {len(df):,} records", flush=True)

    # 数据集统计
    results['dataset_stats'] = {
        'total_transactions': len(df),
        'fraud_count': int(df['isFraud'].sum()) if 'isFraud' in df.columns else 0,
        'fraud_rate': float(df['isFraud'].mean()) if 'isFraud' in df.columns else 0,
        'n_cards': int(df['card_id'].nunique()) if 'card_id' in df.columns else 0,
        'n_merchants': int(df['merchant_id'].nunique()) if 'merchant_id' in df.columns else 0,
        'amount_mean': float(df['amount'].mean()) if 'amount' in df.columns else 0,
        'amount_std': float(df['amount'].std()) if 'amount' in df.columns else 0,
        'amount_min': float(df['amount'].min()) if 'amount' in df.columns else 0,
        'amount_max': float(df['amount'].max()) if 'amount' in df.columns else 0,
    }

    for key, value in results['dataset_stats'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}", flush=True)
        else:
            print(f"  {key}: {value}", flush=True)

    # ========== Step 2: GAR特征生成 ==========
    print("\n[Step 2/5] Generating GAR features...", flush=True)

    entity_cols = ['card_id', 'merchant_id', 'device', 'is_night']
    account_features = ['card_level', 'card_location', 'card_type']
    transaction_features = ['amount', 'balance', 'is_cross_border']

    df, card_col, entity_cols, account_features, transaction_features, has_label, label_col = load_and_preprocess_data(
        data_path, 'card_id', entity_cols, account_features, transaction_features
    )

    train_idx, test_idx = split_data(df, train_ratio=train_ratio, seed=seed)
    print(f"[INFO] Data split: Train={len(train_idx)}, Test={len(test_idx)}", flush=True)

    tx_neighbors = build_graph(df, entity_cols)

    if has_label:
        train_df = df.iloc[train_idx]
        entity_fraud_maps, pair_fraud_maps = compute_fraud_rates_from_train(train_df, entity_cols, label_col)

        features_dict, feature_names = build_gar_features_no_leakage(
            df, train_idx, tx_neighbors, 'card_id',
            entity_cols, account_features, transaction_features,
            has_label, label_col, entity_fraud_maps, pair_fraud_maps
        )

        split标记 = np.array(['train' if i in train_idx else 'test' for i in range(len(df))])

        if has_label and label_col:
            features_dict[label_col] = df[label_col].values

        gar_features_path = os.path.join(output_dir, 'gar_features.csv')
        export_features_to_csv(features_dict, feature_names, gar_features_path, df, has_label, split标记)
    else:
        gar_features_path = data_path  # 无标签模式直接使用原数据

    results['feature_stats'] = {
        'n_features': len(feature_names) if has_label else 0,
        'feature_names': feature_names if has_label else []
    }
    print(f"[INFO] Generated {len(feature_names)} GAR features", flush=True)

    # ========== Step 3: 特征筛选 ==========
    print("\n[Step 3/5] Selecting high fraud rate features...", flush=True)

    if has_label:
        # 加载特征并筛选
        X_train, X_test, y_train, y_test, feat_names = load_features_from_csv(gar_features_path, train_ratio, seed)

        selected_features, corr_df = select_high_fraud_rate_features(
            X_train, y_train, feat_names,
            top_k=top_k_features,
            correlation_threshold=0.03
        )

        results['feature_stats']['selected_features'] = selected_features
        results['feature_stats']['n_selected'] = len(selected_features)

        # 导出筛选后的特征
        df_features = pd.read_csv(gar_features_path)
        selected_path = os.path.join(output_dir, 'selected_features.csv')
        export_selected_features(df_features, selected_features, selected_path)

        print(f"[INFO] Selected {len(selected_features)} features", flush=True)

    # ========== Step 4: 模型训练 ==========
    print("\n[Step 4/5] Training classifier...", flush=True)

    if has_label:
        X_train, X_test, y_train, y_test, feat_names = load_features_from_csv(gar_features_path, train_ratio, seed)

        # 训练GAR分类器
        model_results = train_gar_classifier(X_train, y_train, X_test, y_test, feat_names, seed)

        results['model_results'] = {
            'gar_full': {
                'train_auc': float(model_results['gar_full']['train_auc']),
                'test_auc': float(model_results['gar_full']['test_auc'])
            },
            'baseline': {
                'test_auc': float(model_results['baseline']['test_auc'])
            }
        }

        # 特征重要性
        if 'feature_importance' in model_results['gar_full']:
            importance = sorted(
                model_results['gar_full']['feature_importance'],
                key=lambda x: x[1],
                reverse=True
            )
            results['feature_importance'] = [
                {'feature': f, 'importance': float(i)}
                for f, i in importance[:20]
            ]

        print(f"\n[Results]", flush=True)
        print(f"  Baseline Test AUC: {results['model_results']['baseline']['test_auc']:.4f}", flush=True)
        print(f"  GAR Full Test AUC: {results['model_results']['gar_full']['test_auc']:.4f}", flush=True)
        print(f"  Improvement: +{(results['model_results']['gar_full']['test_auc'] - results['model_results']['baseline']['test_auc'])*100:.2f}%", flush=True)

    # ========== Step 5: 生成报告 ==========
    print("\n[Step 5/5] Generating experiment report...", flush=True)

    elapsed = time.time() - start_time

    # 保存结果JSON
    results_path = os.path.join(output_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[INFO] Results saved to {results_path}", flush=True)

    # 生成Markdown报告
    report_path = os.path.join(output_dir, 'experiment_report.md')
    generate_markdown_report(results, report_path)
    print(f"[INFO] Report saved to {report_path}", flush=True)

    print(f"\nTotal time: {elapsed/60:.1f} minutes", flush=True)
    print("="*60, flush=True)
    print("Experiment complete!", flush=True)
    print("="*60, flush=True)

    return results


def generate_markdown_report(results: dict, output_path: str):
    """生成Markdown格式的实验报告"""

    report = """# Fraud Detection Experiment Report

## 1. Experiment Overview

- **Date**: {date}
- **Data Path**: {data_path}
- **Random Seed**: {seed}
- **Train Ratio**: {train_ratio}
- **Top-K Features**: {top_k}

## 2. Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Transactions | {total_transactions:,} |
| Fraud Count | {fraud_count:,} |
| Fraud Rate | {fraud_rate:.4f} ({fraud_rate_pct:.2f}%) |
| Unique Cards | {n_cards:,} |
| Unique Merchants | {n_merchants:,} |
| Amount Mean | {amount_mean:.2f} |
| Amount Std | {amount_std:.2f} |
| Amount Range | [{amount_min:.2f}, {amount_max:.2f}] |

## 3. Feature Engineering

| Metric | Value |
|--------|-------|
| Total GAR Features | {n_features} |
| Selected Features | {n_selected} |

### Selected Features (Top-{n_selected}):
{selected_features_list}

## 4. Model Performance

| Model | Test AUC | Improvement |
|-------|----------|-------------|
| Baseline | {baseline_auc:.4f} | - |
| GAR Full | {gar_auc:.4f} | +{improvement:.2f}% |

## 5. Feature Importance (Top 20)

| Rank | Feature | Importance |
|------|---------|------------|
{feature_importance_table}

## 6. Conclusion

- GAR特征工程将欺诈检测AUC从 {baseline_auc:.4f} 提升至 {gar_auc:.4f}
- 提升幅度: +{improvement:.2f}%
- 特征筛选从 {n_features} 维特征中选出了 {n_selected} 个高相关特征

---
*Generated by Fraud Detection Experiment Pipeline*
""".format(
        date=results.get('experiment_time', 'N/A'),
        data_path=results.get('data_path', 'N/A'),
        seed=results.get('seed', 'N/A'),
        train_ratio=results.get('train_ratio', 'N/A'),
        top_k=results.get('top_k_features', 'N/A'),
        total_transactions=results['dataset_stats'].get('total_transactions', 0),
        fraud_count=results['dataset_stats'].get('fraud_count', 0),
        fraud_rate=results['dataset_stats'].get('fraud_rate', 0),
        fraud_rate_pct=results['dataset_stats'].get('fraud_rate', 0) * 100,
        n_cards=results['dataset_stats'].get('n_cards', 0),
        n_merchants=results['dataset_stats'].get('n_merchants', 0),
        amount_mean=results['dataset_stats'].get('amount_mean', 0),
        amount_std=results['dataset_stats'].get('amount_std', 0),
        amount_min=results['dataset_stats'].get('amount_min', 0),
        amount_max=results['dataset_stats'].get('amount_max', 0),
        n_features=results['feature_stats'].get('n_features', 0),
        n_selected=results['feature_stats'].get('n_selected', 0),
        selected_features_list='\n'.join([f"- {f}" for f in results['feature_stats'].get('selected_features', [])[:20]]),
        baseline_auc=results['model_results'].get('baseline', {}).get('test_auc', 0),
        gar_auc=results['model_results'].get('gar_full', {}).get('test_auc', 0),
        improvement=(results['model_results'].get('gar_full', {}).get('test_auc', 0) - results['model_results'].get('baseline', {}).get('test_auc', 0)) * 100,
        feature_importance_table='\n'.join([
            f"| {i+1} | {feat['feature']} | {feat['importance']:.4f} |"
            for i, feat in enumerate(results.get('feature_importance', [])[:20])
        ])
    )

    with open(output_path, 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(
        description='Fraud Detection Experiment Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用生成的数据集运行完整实验
  python experiments/run_fraud_detection_experiment.py \\
      --data ./data/simulated_fraud_dataset.csv \\
      --output-dir ./outputs/experiment_20260608

  # 使用IEEE数据集
  python experiments/run_fraud_detection_experiment.py \\
      --data ./data/IEEE_fraud_dataset.csv \\
      --output-dir ./outputs/ieee_experiment
        """
    )

    parser.add_argument('--data', type=str, required=True,
                        help='输入数据集路径（CSV格式）')
    parser.add_argument('--output-dir', type=str, default='./outputs/experiment',
                        help='输出目录（默认: ./outputs/experiment）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例（默认: 0.7）')
    parser.add_argument('--top-k', type=int, default=20,
                        help='选择的特征数量（默认: 20）')

    args = parser.parse_args()

    run_experiment_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        top_k_features=args.top_k
    )


if __name__ == '__main__':
    main()