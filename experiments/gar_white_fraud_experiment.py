"""
GAR白样本+欺诈样本实验

处理两个独立文件:
- fraud_data.csv: 欺诈交易记录 (包含 isFraud 标签)
- white_data.csv: 白样本交易记录 (无标签或 isFraud=0)

工作流程:
1. 合并数据构建图结构（获取完整邻居关系）
2. 从欺诈样本计算欺诈率
3. 生成GAR特征
4. 训练分类器

用法:
python experiments/gar_white_fraud_experiment.py \
    --fraud-data ./data/fraud_transactions.csv \
    --white-data ./data/white_transactions.csv \
    --output-dir ./outputs/white_fraud_exp
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gar.gar_cpu import (
    auto_detect_schema, build_graph, split_data,
    compute_fraud_rates_from_train, build_gar_features_no_leakage,
    export_features_to_csv
)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def load_and_preprocess_data(data_path, card_col, entity_cols, account_features,
                            transaction_features, explicit_label_col=None, auto_detect=True):
    """加载并预处理单个数据文件"""
    print(f"[INFO] Loading data from {data_path}...", flush=True)

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded {len(df)} records, {len(df.columns)} columns", flush=True)

    # 自动检测列名
    if auto_detect:
        schema = auto_detect_schema(df)
        print(f"[INFO] Auto-detected columns:", flush=True)
        for col_type, actual_col in schema.items():
            print(f"  {col_type:<20} -> {actual_col}", flush=True)

        if card_col in schema:
            card_col = schema['card_id']
        if not entity_cols or entity_cols == ['card_id', 'merchant_id', 'device', 'is_night']:
            entity_cols = [v for k, v in schema.items() if k in ['card_id', 'merchant_id', 'terminal_id', 'device', 'is_night']]
        if not account_features or account_features == ['card_level', 'card_location', 'card_type']:
            account_features = [v for k, v in schema.items() if k in ['card_level', 'card_location', 'card_type']]
        if not transaction_features or transaction_features == ['amount', 'balance', 'is_cross_border']:
            transaction_features = [v for k, v in schema.items() if k in ['amount', 'balance', 'is_cross_border']]

    # 检测是否有标签
    has_label = False
    label_col = None
    if explicit_label_col and explicit_label_col in df.columns:
        has_label = True
        label_col = explicit_label_col
    else:
        for col in ['isFraud', 'fraud', 'label', 'is_fraud', '是否欺诈']:
            if col in df.columns:
                has_label = True
                label_col = col
                print(f"[INFO] Found label column: {label_col}", flush=True)
                break

    # 实体列编码
    from sklearn.preprocessing import LabelEncoder
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


def run_white_fraud_experiment(fraud_data_path, white_data_path, output_dir,
                              card_col='card_id', entity_cols=None, account_features=None,
                              transaction_features=None, train_ratio=0.7, seed=42):
    """
    执行白样本+欺诈样本实验

    Args:
        fraud_data_path: 欺诈数据文件路径 (包含 isFraud=1 的记录)
        white_data_path: 白样本数据文件路径 (无标签或 isFraud=0)
        output_dir: 输出目录
        card_col: 卡号列名
        entity_cols: 实体列名列表
        account_features: 账户级特征列表
        transaction_features: 交易级特征列表
        train_ratio: 训练集比例
        seed: 随机种子
    """
    print("="*60, flush=True)
    print("GAR White+Fraud Experiment", flush=True)
    print("="*60, flush=True)

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    # 默认值
    if entity_cols is None:
        entity_cols = ['card_id', 'merchant_id', 'device', 'is_night']
    if account_features is None:
        account_features = ['card_level', 'card_location', 'card_type']
    if transaction_features is None:
        transaction_features = ['amount', 'balance', 'is_cross_border']

    # ========== 1. 加载数据 ==========
    print("\n[Step 1/6] Loading data...", flush=True)

    # 加载欺诈数据
    fraud_df, card_col, entity_cols, account_features, transaction_features, has_label_fraud, label_col = \
        load_and_preprocess_data(fraud_data_path, card_col, entity_cols, account_features,
                               transaction_features, None, True)

    # 加载白样本数据
    white_df, _, _, _, _, has_label_white, _ = \
        load_and_preprocess_data(white_data_path, card_col, entity_cols, account_features,
                                 transaction_features, None, True)

    # ========== 2. 统一标签列 ==========
    print("\n[Step 2/6] Processing labels...", flush=True)

    # 确保欺诈数据有标签
    if label_col not in fraud_df.columns:
        print("[WARN] No label column found, adding isFraud=1 to fraud data")
        fraud_df['isFraud'] = 1
        label_col = 'isFraud'

    # 白样本数据标记为0
    if 'isFraud' not in white_df.columns:
        white_df['isFraud'] = 0

    # ========== 3. 合并数据 ==========
    print("\n[Step 3/6] Merging data...", flush=True)

    # 确保两边的列一致
    common_cols = list(set(fraud_df.columns) & set(white_df.columns))
    print(f"[INFO] Common columns: {len(common_cols)}", flush=True)

    # 按统一顺序排列列
    all_cols = list(set(fraud_df.columns) | set(white_df.columns))
    for col in all_cols:
        if col not in fraud_df.columns:
            fraud_df[col] = np.nan
        if col not in white_df.columns:
            white_df[col] = np.nan

    # 合并
    combined_df = pd.concat([fraud_df, white_df], ignore_index=True)
    print(f"[INFO] Combined records: {len(combined_df)} (fraud={len(fraud_df)}, white={len(white_df)})", flush=True)

    # 欺诈率统计
    fraud_count = combined_df[label_col].sum()
    total_count = len(combined_df)
    print(f"[INFO] Fraud rate: {fraud_count}/{total_count} ({100*fraud_count/total_count:.2f}%)", flush=True)

    # ========== 4. 分割数据（无泄漏模式） ==========
    print("\n[Step 4/6] Splitting data...", flush=True)

    train_idx, test_idx = split_data(combined_df, train_ratio=train_ratio, seed=seed)
    print(f"[INFO] Data split: Train={len(train_idx)}, Test={len(test_idx)}", flush=True)

    # ========== 5. 构建图和特征 ==========
    print("\n[Step 5/6] Building GAR features...", flush=True)

    # 构建图（从完整合并数据构建）
    tx_neighbors = build_graph(combined_df, entity_cols)

    # 从训练集计算欺诈率
    train_df = combined_df.iloc[train_idx]
    entity_fraud_maps, pair_fraud_maps = compute_fraud_rates_from_train(train_df, entity_cols, label_col)
    print("[INFO] Computed fraud rates from train data only (no leakage)", flush=True)

    # 构建特征
    has_label = True
    features_dict, feature_names = build_gar_features_no_leakage(
        combined_df, train_idx, tx_neighbors, card_col,
        entity_cols, account_features, transaction_features,
        has_label, label_col, entity_fraud_maps, pair_fraud_maps
    )

    # 添加split列和标签
    split_arr = np.array(['train' if i in train_idx else 'test' for i in range(len(combined_df))])
    features_dict[label_col] = combined_df[label_col].values

    # ========== 6. 导出特征 ==========
    print("\n[Step 6/6] Exporting features...", flush=True)

    features_csv = os.path.join(output_dir, 'gar_white_fraud_features.csv')
    export_features_to_csv(features_dict, feature_names, features_csv, combined_df, has_label, split_arr, label_col)

    # 保存元信息
    meta = {
        'fraud_data_path': fraud_data_path,
        'white_data_path': white_data_path,
        'fraud_count': len(fraud_df),
        'white_count': len(white_df),
        'total_count': len(combined_df),
        'train_count': len(train_idx),
        'test_count': len(test_idx),
        'fraud_rate': float(fraud_count/total_count),
        'feature_count': len(feature_names),
        'label_col': label_col
    }

    import json
    meta_path = os.path.join(output_dir, 'experiment_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n[INFO] Total time: {elapsed/60:.1f} minutes", flush=True)
    print(f"[INFO] Features saved to: {features_csv}", flush=True)

    return features_dict, feature_names, split_arr, label_col


def train_and_evaluate(features_dict, feature_names, label_col, split_arr, seed=42):
    """训练并评估分类器"""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    train_mask = np.array(split_arr) == 'train'
    test_mask = np.array(split_arr) == 'test'

    X = np.column_stack([features_dict[name] for name in feature_names])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train = features_dict[label_col][train_mask]
    y_test = features_dict[label_col][test_mask]

    print(f"[INFO] Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples", flush=True)

    # 训练
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=seed
    )
    gb.fit(X_train, y_train)

    # 预测
    train_proba = gb.predict_proba(X_train)[:, 1]
    test_proba = gb.predict_proba(X_test)[:, 1]

    # 评估
    results = {
        'train_auc': roc_auc_score(y_train, train_proba),
        'test_auc': roc_auc_score(y_test, test_proba),
        'feature_importance': list(zip(feature_names, gb.feature_importances_.tolist()))
    }

    # 二分类指标
    test_pred = (test_proba > 0.5).astype(int)
    results['precision'] = precision_score(y_test, test_pred)
    results['recall'] = recall_score(y_test, test_pred)
    results['f1'] = f1_score(y_test, test_pred)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='GAR White+Fraud Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基础用法
  python experiments/gar_white_fraud_experiment.py \\
      --fraud-data ./data/fraud_transactions.csv \\
      --white-data ./data/white_transactions.csv \\
      --output-dir ./outputs/white_fraud_exp

  # 指定列名
  python experiments/gar_white_fraud_experiment.py \\
      --fraud-data ./data/fraud.csv \\
      --white-data ./data/normal.csv \\
      --card-col card_id \\
      --entity-cols card_id,merchant_id,device \\
      --output-dir ./outputs/exp1
        """
    )

    parser.add_argument('--fraud-data', type=str, required=True,
                        help='欺诈数据文件路径（包含 isFraud标签）')
    parser.add_argument('--white-data', type=str, required=True,
                        help='白样本数据文件路径（无标签或 isFraud=0）')
    parser.add_argument('--output-dir', type=str, default='./outputs/white_fraud_exp',
                        help='输出目录')
    parser.add_argument('--card-col', type=str, default='card_id',
                        help='卡号列名')
    parser.add_argument('--entity-cols', type=str, default=None,
                        help='实体列名列表，逗号分隔')
    parser.add_argument('--account-features', type=str, default=None,
                        help='账户级特征列名，逗号分隔')
    parser.add_argument('--transaction-features', type=str, default=None,
                        help='交易级特征列名，逗号分隔')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例（默认: 0.7）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--train-only', action='store_true',
                        help='仅生成特征，不训练模型')

    args = parser.parse_args()

    entity_cols = args.entity_cols.split(',') if args.entity_cols else None
    account_features = args.account_features.split(',') if args.account_features else None
    transaction_features = args.transaction_features.split(',') if args.transaction_features else None

    # 运行实验
    features_dict, feature_names, split_arr, label_col = run_white_fraud_experiment(
        args.fraud_data, args.white_data, args.output_dir,
        args.card_col, entity_cols, account_features, transaction_features,
        args.train_ratio, args.seed
    )

    # 训练模型
    if not args.train_only:
        print("\n" + "="*60, flush=True)
        print("Training and Evaluation", flush=True)
        print("="*60, flush=True)

        results = train_and_evaluate(features_dict, feature_names, label_col, split_arr, args.seed)

        print(f"\nTrain AUC: {results['train_auc']:.4f}", flush=True)
        print(f"Test AUC: {results['test_auc']:.4f}", flush=True)
        print(f"Precision: {results['precision']:.4f}", flush=True)
        print(f"Recall: {results['recall']:.4f}", flush=True)
        print(f"F1: {results['f1']:.4f}", flush=True)

        print("\nTop 10 Features:", flush=True)
        for i, (name, imp) in enumerate(sorted(results['feature_importance'], key=lambda x: x[1], reverse=True)[:10]):
            print(f"  {i+1:2d}. {name:<40} {imp:.4f}", flush=True)


if __name__ == '__main__':
    main()