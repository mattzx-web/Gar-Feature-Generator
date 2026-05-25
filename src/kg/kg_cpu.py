"""
通用知识图谱特征生成器

支持白样本（无标签）数据，支持账户级和交易级特征。

数据格式:
- 每个CSV包含交易记录
- 账户级特征（如卡等级）每个账号重复
- 交易级特征（如交易金额）每条记录不同

用法:
    # 白样本特征生成
    python src/kg_feature_generator.py --data-dir /path/to/data \\
                                        --card-col card_id \\
                                        --export-features-only \\
                                        --output-csv ./features/kg_features.csv

    # 指定账户级和交易级特征列
    python src/kg_feature_generator.py --data-dir /path/to/data \\
                                        --card-col card_id \\
                                        --account-features card_level,issuing_bank \\
                                        --transaction-features amount,balance,timestamp \\
                                        --entity-cols card_id,merchant_id,device_type
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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
DEFAULT_CARD_COL = 'card_id'
DEFAULT_ENTITY_COLS = ['card_id', 'merchant_id', 'device_type', 'transaction_type']
DEFAULT_ACCOUNT_FEATURES = ['card_level', 'issuing_bank']
DEFAULT_TRANSACTION_FEATURES = ['timestamp', 'amount', 'balance_after', 'is_frequent_contact',
                                'transaction_channel', 'device_type', 'is_pos', 'is_cross_border']
DEFAULT_NEIGHBOR_THRESHOLD = 300
DEFAULT_TRAIN_RATIO = 0.7


def load_and_preprocess_data(data_path, card_col, entity_cols, account_features, transaction_features):
    """
    加载并预处理数据

    Args:
        data_path: CSV文件路径
        card_col: 卡号列名
        entity_cols: 实体列名列表（用于构建图）
        account_features: 账户级特征列名列表
        transaction_features: 交易级特征列名列表

    Returns:
        df: 预处理后的DataFrame
        card_col, entity_cols, account_features, transaction_features
    """
    print(f"[INFO] Loading data from {data_path}...", flush=True)

    # 读取CSV
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded {len(df)} records", flush=True)

    # 如果有标签列，保存但不使用（白样本模式）
    has_label = 'isFraud' in df.columns or 'fraud' in df.columns or 'label' in df.columns
    if has_label:
        label_col = 'isFraud' if 'isFraud' in df.columns else ('fraud' if 'fraud' in df.columns else 'label')
        print(f"[INFO] Found label column: {label_col} (will be preserved but not used for feature engineering)", flush=True)

    # 实体列编码
    for col in entity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

    # 填充交易级特征的缺失值
    for col in transaction_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"[INFO] Card column: {card_col}", flush=True)
    print(f"[INFO] Entity columns: {entity_cols}", flush=True)
    print(f"[INFO] Account features: {account_features}", flush=True)
    print(f"[INFO] Transaction features: {transaction_features}", flush=True)

    return df, card_col, entity_cols, account_features, transaction_features, has_label


def build_graph(df, entity_cols, neighbor_threshold=DEFAULT_NEIGHBOR_THRESHOLD):
    """
    构建交易图结构

    基于实体列（卡号、商户、设备等）构建交易之间的邻居关系。

    Args:
        df: DataFrame
        entity_cols: 实体列名列表
        neighbor_threshold: 邻居数量上限

    Returns:
        tx_neighbors: 交易ID到邻居集合的映射
        entity_to_tx: 实体到交易列表的映射
    """
    print(f"[INFO] Building graph...", flush=True)

    n = len(df)
    tx_neighbors = defaultdict(set)
    entity_to_tx = defaultdict(list)

    # 记录每个实体对应的交易
    for col in entity_cols:
        if col not in df.columns:
            continue
        for i, val in enumerate(df[col].values):
            entity_to_tx[(col, val)].append(i)

    # 构建邻居关系：同一实体值的交易互为邻居
    for col in entity_cols:
        if col not in df.columns:
            continue
        groups = df.groupby(col).indices
        for val, idx_list in groups.items():
            if 1 < len(idx_list) < neighbor_threshold:
                for idx in idx_list:
                    tx_neighbors[idx].update(idx_list)

    # 移除自身
    for idx in tx_neighbors:
        tx_neighbors[idx].discard(idx)

    n_with_neigh = sum(1 for tx in tx_neighbors if len(tx_neighbors[tx]) > 0)
    print(f"[INFO] Nodes with neighbors: {n_with_neigh}/{n} ({100*n_with_neigh/n:.1f}%)", flush=True)

    return tx_neighbors, entity_to_tx


def build_features(df, tx_neighbors, card_col, entity_cols, account_features, transaction_features, has_label):
    """
    构建知识图谱特征

    特征类型:
    1. 实体度特征 (Entity Degree)
    2. 实体计数特征 (Entity Count)
    3. 账户级特征 (Account Features) - 每个卡号重复的值
    4. 交易级特征 (Transaction Features)
    5. 1-hop邻居特征 (1-hop Neighbor Features)
    6. 2-hop邻居特征 (2-hop Neighbor Features)
    7. 配对计数特征 (Pair Count Features)
    8. 时序特征 (Temporal Features) - 如果有时间戳

    Args:
        df: DataFrame
        tx_neighbors: 邻居映射
        card_col: 卡号列名
        entity_cols: 实体列
        account_features: 账户级特征
        transaction_features: 交易级特征
        has_label: 是否有标签

    Returns:
        features_dict: 特征字典
        feature_names: 特征名列表
    """
    print(f"[INFO] Building features...", flush=True)

    features = {}
    n = len(df)

    # ========== 1. 实体度特征 ==========
    for col in entity_cols:
        if col not in df.columns:
            continue
        degrees = [len(tx_neighbors.get(i, set())) for i in range(n)]
        features[f'{col}_degree'] = np.array(degrees)

    # ========== 2. 实体计数特征 ==========
    for col in entity_cols[:4]:  # 限制数量避免维度爆炸
        if col not in df.columns:
            continue
        val_counts = df[col].value_counts().to_dict()
        features[f'{col}_count'] = df[col].map(val_counts).fillna(0).values
        features[f'{col}_count_log'] = np.log1p(features[f'{col}_count'])

    # ========== 3. 账户级特征 ==========
    # 账户级特征每个账号重复，直接使用
    for col in account_features:
        if col not in df.columns:
            continue
        # 填充缺失值
        features[col] = df[col].fillna(-1).values
        if df[col].dtype == 'object':
            le = LabelEncoder()
            features[col] = le.fit_transform(df[col].astype(str))

    # ========== 4. 交易级特征 ==========
    for col in transaction_features:
        if col not in df.columns:
            continue
        if col in df.columns:
            # 数值型特征
            if df[col].dtype in ['int64', 'float64']:
                features[col] = df[col].fillna(0).values
                # 添加log变换（对于金额类特征）
                if 'amount' in col.lower() or 'balance' in col.lower():
                    features[f'{col}_log'] = np.log1p(np.abs(df[col].fillna(0).values))
            else:
                # 类别型特征编码
                le = LabelEncoder()
                features[col] = le.fit_transform(df[col].fillna('missing').astype(str))

    # ========== 5. 1-hop邻居特征 ==========
    n_1hop = []
    for i in range(n):
        n_1hop.append(len(tx_neighbors.get(i, set())))
    features['n_1hop'] = np.array(n_1hop)

    # 邻居金额统计
    amount_col = None
    for col in ['amount', '交易金额', 'transaction_amount']:
        if col in df.columns:
            amount_col = col
            break

    if amount_col:
        amt_1hop_mean = []
        amt_1hop_std = []
        amt_1hop_max = []
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = df[amount_col].iloc[list(neighs)].fillna(0).values
                amt_1hop_mean.append(np.mean(neigh_amts))
                amt_1hop_std.append(np.std(neigh_amts) if len(neigh_amts) > 1 else 0)
                amt_1hop_max.append(np.max(neigh_amts))
            else:
                amt_1hop_mean.append(0)
                amt_1hop_std.append(0)
                amt_1hop_max.append(0)
        features['amt_1hop_mean'] = np.array(amt_1hop_mean)
        features['amt_1hop_std'] = np.array(amt_1hop_std)
        features['amt_1hop_max'] = np.array(amt_1hop_max)

    # ========== 6. 2-hop邻居特征 ==========
    tx_2hop_neighbors = defaultdict(set)
    for tx, neighs in tx_neighbors.items():
        for neighbor in neighs:
            tx_2hop_neighbors[tx].update(tx_neighbors.get(neighbor, set()))
    tx_2hop_neighbors[tx].discard(tx)

    n_2hop = []
    for i in range(n):
        n_2hop.append(len(tx_2hop_neighbors.get(i, set())))
    features['n_2hop'] = np.array(n_2hop)
    features['2hop_1hop_ratio'] = features['n_2hop'] / (features['n_1hop'] + 1)

    # ========== 7. 配对计数特征 ==========
    for i, col1 in enumerate(entity_cols[:4]):
        for col2 in entity_cols[i+1:5]:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            pairs = df[col1].astype(str) + '_' + df[col2].astype(str)
            pair_counts = pairs.map(pairs.value_counts())
            features[f'{col1}_{col2}_pair_count'] = pair_counts.values
            features[f'{col1}_{col2}_pair_count_log'] = np.log1p(pair_counts).values

    # ========== 8. 卡号级别特征（聚合） ==========
    if card_col in df.columns:
        # 每个卡号的交易数量
        card_tx_counts = df[card_col].value_counts().to_dict()
        features['card_tx_count'] = df[card_col].map(card_tx_counts).fillna(0).values
        features['card_tx_count_log'] = np.log1p(features['card_tx_count'])

        # 每个卡号的金额统计
        if amount_col:
            card_amt_mean = df.groupby(card_col)[amount_col].transform('mean')
            card_amt_std = df.groupby(card_col)[amount_col].transform('std').fillna(0)
            card_amt_max = df.groupby(card_col)[amount_col].transform('max')
            features['card_amt_mean'] = card_amt_mean.fillna(0).values
            features['card_amt_std'] = card_amt_std.fillna(0).values
            features['card_amt_max'] = card_amt_max.fillna(0).values

        # 每个卡号的平均度数
        card_degree = df.groupby(card_col).apply(
            lambda x: np.mean([len(tx_neighbors.get(idx, set())) for idx in x.index]),
            include_groups=False
        ).to_dict()
        features['card_avg_degree'] = df[card_col].map(card_degree).fillna(0).values

    # ========== 9. 时序特征（如果有时间戳） ==========
    timestamp_col = None
    for col in ['timestamp', '时间戳', 'trans_time', 'transaction_time']:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col:
        try:
            # 解析时间戳
            ts = pd.to_datetime(df[timestamp_col], errors='coerce')
            if not ts.isna().all():
                # 交易时间在一天中的位置（小时）
                features['trans_hour'] = ts.dt.hour.fillna(12).values
                features['trans_dayofweek'] = ts.dt.dayofweek.fillna(0).values

                # 时间差特征：与同一卡号上一笔交易的时间差（秒）
                df_sorted = df.copy()
                df_sorted['_ts_numeric'] = ts
                df_sorted = df_sorted.sort_values([card_col, timestamp_col])
                time_diff = df_sorted.groupby(card_col)['_ts_numeric'].diff().dt.total_seconds().fillna(0)
                # 按原始索引映射回去
                features['time_diff_prev'] = df.index.map(time_diff.to_dict()).fillna(0).values
        except Exception as e:
            print(f"[WARN] Failed to extract temporal features: {e}", flush=True)

    # 清理无穷值
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    feature_names = list(features.keys())
    print(f"[INFO] Generated {len(feature_names)} features", flush=True)

    return features, feature_names


def export_features_to_csv(features_dict, feature_names, output_path, original_df=None, has_label=False):
    """
    将特征导出为CSV文件

    Args:
        features_dict: 特征字典
        feature_names: 特征名列表
        output_path: 输出路径
        original_df: 原始DataFrame（用于保留原始列）
        has_label: 是否有标签列
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # 构建特征DataFrame
    df_features = pd.DataFrame({name: features_dict[name] for name in feature_names})

    # 保留原始关键列（如果提供）
    if original_df is not None:
        # 保留卡号等关键标识列
        key_cols = []
        for col in ['card_id', '卡号', 'TransactionID', 'transaction_id', 'timestamp', '时间戳']:
            if col in original_df.columns:
                key_cols.append(col)

        if key_cols:
            df_key = original_df[key_cols].copy()
            df_features = pd.concat([df_key, df_features], axis=1)

    # 保留标签（如果有）
    if has_label:
        for col in ['isFraud', 'fraud', 'label']:
            if col in original_df.columns:
                df_features[col] = original_df[col].values
                break

    df_features.to_csv(output_path, index=False)
    print(f"[INFO] Features saved to {output_path}", flush=True)
    print(f"[INFO] Shape: {df_features.shape}", flush=True)

    return output_path


def train_classifier(features_dict, feature_names, has_label=False, seed=42, train_ratio=0.7):
    """
    训练分类器（如果有标签）

    Args:
        features_dict: 特征字典
        feature_names: 特征名列表
        has_label: 是否有标签
        seed: 随机种子
        train_ratio: 训练集比例

    Returns:
        results: 结果字典
    """
    if not has_label:
        print("[INFO] White sample mode - skipping model training", flush=True)
        print("[INFO] Use --export-features-only to save features to CSV", flush=True)
        return None

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    n = len(feature_names[0])
    n_train = int(train_ratio * n)

    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # 构建特征矩阵
    X = np.column_stack([features_dict[name] for name in feature_names])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # 获取标签
    label_col = None
    for col in ['isFraud', 'fraud', 'label']:
        if col in features_dict:
            label_col = col
            break

    if label_col is None:
        print("[ERROR] Label column not found", flush=True)
        return None

    y = features_dict[label_col]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 训练
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                     subsample=0.8, random_state=seed)
    gb.fit(X_train, y_train)

    train_proba = gb.predict_proba(X_train)[:, 1]
    test_proba = gb.predict_proba(X_test)[:, 1]

    results = {
        'train_auc': float(roc_auc_score(y_train, train_proba)),
        'test_auc': float(roc_auc_score(y_test, test_proba)),
        'feature_importance': list(zip(feature_names, gb.feature_importances_.tolist()))
    }

    print(f"[RESULT] Train AUC: {results['train_auc']:.4f}", flush=True)
    print(f"[RESULT] Test AUC: {results['test_auc']:.4f}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Knowledge Graph Feature Generator for White Samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 白样本特征生成（无标签）
  python src/kg_feature_generator.py --data /path/to/transactions.csv \\
                                      --card-col card_id \\
                                      --export-features-only \\
                                      --output-csv ./features/kg_features.csv

  # 有标签数据（完整流程）
  python src/kg_feature_generator.py --data /path/to/transactions.csv \\
                                      --card-col card_id \\
                                      --account-features card_level,issuing_bank \\
                                      --entity-cols card_id,merchant_id,device_type

  # 自定义特征列
  python src/kg_feature_generator.py --data /path/to/data.csv \\
                                      --card-col card_id \\
                                      --account-features card_level,issuing_bank \\
                                      --transaction-features amount,balance,timestamp \\
                                      --entity-cols card_id,merchant_id,device_type \\
                                      --export-features-only \\
                                      --output-csv ./features.csv
        """
    )

    parser.add_argument('--data', type=str, required=True,
                        help='CSV文件路径（包含交易记录）')
    parser.add_argument('--card-col', type=str, default=DEFAULT_CARD_COL,
                        help=f'卡号列名（默认: {DEFAULT_CARD_COL}）')
    parser.add_argument('--entity-cols', type=str, default=None,
                        help='实体列名列表，逗号分隔（用于构建图关系）')
    parser.add_argument('--account-features', type=str, default=None,
                        help='账户级特征列名列表，逗号分隔（如: card_level,issuing_bank）')
    parser.add_argument('--transaction-features', type=str, default=None,
                        help='交易级特征列名列表，逗号分隔（如: amount,balance,timestamp）')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='输出目录（默认: ./outputs）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--export-features-only', action='store_true',
                        help='仅生成特征，不训练模型')
    parser.add_argument('--feature-only', action='store_true',
                        help='与--export-features-only相同')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='特征CSV输出路径')

    args = parser.parse_args()

    export_only = args.export_features_only or args.feature_only

    # 解析列名列表
    entity_cols = args.entity_cols.split(',') if args.entity_cols else DEFAULT_ENTITY_COLS
    account_features = args.account_features.split(',') if args.account_features else []
    transaction_features = args.transaction_features.split(',') if args.transaction_features else []

    print("="*60, flush=True)
    print("Knowledge Graph Feature Generator (White Sample Mode)", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    # 1. 加载数据
    df, card_col, entity_cols, account_features, transaction_features, has_label = load_and_preprocess_data(
        args.data, args.card_col, entity_cols, account_features, transaction_features
    )

    # 2. 构建图
    tx_neighbors, entity_to_tx = build_graph(df, entity_cols)

    # 3. 构建特征
    features_dict, feature_names = build_features(
        df, tx_neighbors, card_col, entity_cols, account_features, transaction_features, has_label
    )

    # 4. 导出或训练
    if export_only:
        if args.output_csv:
            export_features_to_csv(features_dict, feature_names, args.output_csv, df, has_label)
        else:
            print("[ERROR] --output-csv is required when using --export-features-only", flush=True)
    else:
        if has_label:
            results = train_classifier(features_dict, feature_names, has_label, args.seed)
            if results:
                print(f"\nTrain AUC: {results['train_auc']:.4f}", flush=True)
                print(f"Test AUC: {results['test_auc']:.4f}", flush=True)
                print("\nTop 10 Features:", flush=True)
                for i, (name, imp) in enumerate(sorted(results['feature_importance'], key=lambda x: x[1], reverse=True)[:10]):
                    print(f"  {i+1:2d}. {name:<40} {imp:.4f}", flush=True)
        else:
            print("[INFO] White sample mode - use --export-features-only to save features", flush=True)

            # 自动导出到outputs
            os.makedirs(args.output_dir, exist_ok=True)
            output_csv = args.output_csv or f"{args.output_dir}/kg_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            export_features_to_csv(features_dict, feature_names, output_csv, df, has_label)

    print(f"\nTotal time: {(time.time()-start_time)/60:.1f} minutes", flush=True)


if __name__ == '__main__':
    main()