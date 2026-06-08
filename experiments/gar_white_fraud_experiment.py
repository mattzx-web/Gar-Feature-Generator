"""
GAR白样本+欺诈样本实验（优化版）

支持大规模不均衡数据（1:20~1:40），特征生成与分类器训练分离。

用法:
    # Step 1: 生成特征
    python experiments/gar_white_fraud_experiment.py \
        --fraud-data ./data/fraud.csv \
        --white-data ./data/white.csv \
        --output-dir ./outputs/exp \
        --step generate

    # Step 2: 训练分类器（支持类别权重调整）
    python experiments/gar_white_fraud_experiment.py \
        --fraud-data ./data/fraud.csv \
        --white-data ./data/white.csv \
        --output-dir ./outputs/exp \
        --step train \
        --class-weight20
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import time
import json
import gc
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 默认配置
DEFAULT_CARD_COL = 'card_id'
DEFAULT_ENTITY_COLS = ['card_id', 'merchant_id', 'device', 'is_night']
DEFAULT_ACCOUNT_FEATURES = ['card_level', 'card_location', 'card_type']
DEFAULT_TRANSACTION_FEATURES = ['amount', 'balance', 'is_cross_border']


def auto_detect_schema(df):
    """自动检测数据集的列类型"""
    COLUMN_ALIASES = {
        'card_id': ['card_id', 'card', 'card_no', '卡号', '银行卡号', 'customer_id'],
        'amount': ['amount', 'amt', '交易金额', 'tx_amount', 'total'],
        'timestamp': ['timestamp', 'time', 'datetime', '交易时间'],
        'is_fraud': ['isFraud', 'fraud', 'label', 'is_fraud', '欺诈'],
        'merchant_id': ['merchant_id', 'merchant', '商户号'],
        'device': ['device', 'device_type', '设备'],
        'balance': ['balance', '账户余额', '余额'],
        'is_night': ['is_night', '夜间交易'],
        'is_cross_border': ['is_cross_border', '跨境'],
        'card_level': ['card_level', '卡等级'],
        'card_location': ['card_location', '卡注册地'],
        'card_type': ['card_type', '卡类型'],
    }

    detected = {}
    used_columns = set()
    priority_order = ['card_id', 'amount', 'timestamp', 'is_fraud', 'merchant_id',
                      'device', 'balance', 'is_night', 'is_cross_border',
                      'card_level', 'card_location', 'card_type']

    for col_type in priority_order:
        aliases = COLUMN_ALIASES.get(col_type, [])
        for alias in aliases:
            for col in df.columns:
                if col not in used_columns:
                    col_lower = col.lower()
                    alias_lower = alias.lower()
                    if col_lower == alias_lower or alias_lower in col_lower:
                        if col_type == 'card_id' and ('level' in col_lower or 'type' in col_lower or 'location' in col_lower):
                            continue
                        detected[col_type] = col
                        used_columns.add(col)
                        break
    return detected


def load_and_preprocess(data_path, card_col, entity_cols, account_features,
                       transaction_features, auto_detect=True):
    """加载并预处理数据"""
    print(f"[INFO] Loading {data_path}...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO]   Records: {len(df)}, Columns: {len(df.columns)}", flush=True)

    if auto_detect:
        schema = auto_detect_schema(df)
        if card_col in schema:
            card_col = schema['card_id']
        if not entity_cols or entity_cols == DEFAULT_ENTITY_COLS:
            entity_cols = [v for k, v in schema.items() if k in ['card_id', 'merchant_id', 'device', 'is_night']]
        if not account_features or account_features == DEFAULT_ACCOUNT_FEATURES:
            account_features = [v for k, v in schema.items() if k in ['card_level', 'card_location', 'card_type']]
        if not transaction_features or transaction_features == DEFAULT_TRANSACTION_FEATURES:
            transaction_features = [v for k, v in schema.items() if k in ['amount', 'balance', 'is_cross_border']]

    # 检测标签
    label_col = None
    for col in ['isFraud', 'fraud', 'label', 'is_fraud', '是否欺诈']:
        if col in df.columns:
            label_col = col
            print(f"[INFO]   Label column: {label_col}", flush=True)
            break

    # 实体列编码
    from sklearn.preprocessing import LabelEncoder
    for col in entity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

    # 填充缺失值
    for col in account_features + transaction_features:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('missing')
            else:
                df[col] = df[col].fillna(0)

    return df, card_col, entity_cols, account_features, transaction_features, label_col


def merge_fraud_white(fraud_df, white_df, card_col, label_col='isFraud'):
    """合并欺诈和白样本数据"""
    print(f"[INFO] Merging data...", flush=True)
    print(f"[INFO]   Fraud records: {len(fraud_df)}, White records: {len(white_df)}", flush=True)

    # 确保白样本有标签列
    if label_col not in white_df.columns:
        white_df[label_col] = 0

    # 确保欺诈数据有标签列
    if label_col not in fraud_df.columns:
        fraud_df[label_col] = 1

    # 统一列
    all_cols = list(set(fraud_df.columns) | set(white_df.columns))
    for col in all_cols:
        if col not in fraud_df.columns:
            fraud_df[col] = np.nan
        if col not in white_df.columns:
            white_df[col] = np.nan

    # 合并
    combined_df = pd.concat([fraud_df, white_df], ignore_index=True)

    fraud_count = combined_df[label_col].sum()
    total_count = len(combined_df)
    white_count = total_count - fraud_count

    print(f"[INFO] Combined: {total_count} records", flush=True)
    print(f"[INFO]   White: {white_count} ({100*white_count/total_count:.1f}%)", flush=True)
    print(f"[INFO]   Fraud: {fraud_count} ({100*fraud_count/total_count:.1f}%)", flush=True)
    print(f"[INFO]   Imbalance ratio: 1:{white_count/fraud_count:.1f}", flush=True)

    return combined_df


def split_data(df, train_ratio=0.7, seed=42):
    """分割训练集和测试集"""
    n = len(df)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    n_train = int(train_ratio * n)
    return indices[:n_train], indices[n_train:]


def build_graph(df, entity_cols, neighbor_threshold=300):
    """构建交易图结构"""
    from collections import defaultdict

    n = len(df)
    tx_neighbors = defaultdict(set)

    iterator = entity_cols if not TQDM_AVAILABLE else tqdm(entity_cols, desc="[Graph] Building")
    for col in iterator:
        if col not in df.columns:
            continue
        groups = df.groupby(col).indices
        for val, idx_list in groups.items():
            if 1 < len(idx_list) < neighbor_threshold:
                for idx in idx_list:
                    tx_neighbors[idx].update(idx_list)

    for idx in tx_neighbors:
        tx_neighbors[idx].discard(idx)

    return tx_neighbors


def compute_fraud_rates_from_train(train_df, entity_cols, label_col):
    """从训练集计算欺诈率映射"""
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


def build_gar_features(df, train_idx, tx_neighbors, card_col, entity_cols,
                     account_features, transaction_features, label_col,
                     entity_fraud_maps, pair_fraud_maps, show_progress=True):
    """构建GAR特征"""
    from sklearn.preprocessing import LabelEncoder

    features = {}
    n = len(df)

    # 找金额列
    amount_col = None
    for col in ['amount', '交易金额', 'amt']:
        if col in df.columns:
            amount_col = col
            break

    df_columns = list(df.columns)

    # ========== 1. 交易级特征 ==========
    for col in transaction_features:
        if col not in df_columns:
            continue
        if df[col].dtype in ['int64', 'float64']:
            features[col] = df[col].fillna(0).values
            if amount_col and col == amount_col:
                features[f'{col}_log'] = np.log1p(np.abs(df[col].fillna(0).values))
        else:
            le = LabelEncoder()
            features[col] = le.fit_transform(df[col].fillna('missing').astype(str))

    # ========== 2. 实体频率 ==========
    for col in entity_cols:
        if col not in df_columns:
            continue
        freq_map = df[col].value_counts().to_dict()
        features[f'{col}_freq'] = df[col].map(freq_map).fillna(0).values
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq'])

    # ========== 3. 卡号聚合 ==========
    if card_col in df_columns:
        card_counts = df[card_col].value_counts().to_dict()
        features['card_tx_count'] = df[card_col].map(card_counts).fillna(0).values
        features['card_tx_count_log'] = np.log1p(features['card_tx_count'])

        if amount_col:
            card_amt_mean = df.groupby(card_col)[amount_col].transform('mean')
            card_amt_std = df.groupby(card_col)[amount_col].transform('std').fillna(0)
            card_amt_max = df.groupby(card_col)[amount_col].transform('max')
            features['card_amt_mean'] = card_amt_mean.fillna(0).values
            features['card_amt_std'] = card_amt_std.fillna(0).values
            features['card_amt_max'] = card_amt_max.fillna(0).values
            features['amt_to_card_mean_ratio'] = df[amount_col].fillna(0) / (card_amt_mean.fillna(1) + 1)

    # ========== 4. 配对频率 ==========
    for i, col1 in enumerate(entity_cols[:4]):
        for col2 in entity_cols[i+1:5]:
            if col1 not in df_columns or col2 not in df_columns:
                continue
            pairs = df[col1].astype(str) + '_' + df[col2].astype(str)
            pair_counts = pairs.map(pairs.value_counts())
            features[f'{col1}_{col2}_pair_freq'] = pair_counts.fillna(0).values
            features[f'{col1}_{col2}_pair_freq_log'] = np.log1p(pair_counts.fillna(0)).values

    # ========== 5. 邻居特征 ==========
    n_1hop = [len(tx_neighbors.get(i, set())) for i in range(n)]
    features['n_1hop'] = np.array(n_1hop)
    features['n_1hop_log'] = np.log1p(features['n_1hop'])

    if amount_col:
        amt_1hop_mean = []
        amt_1hop_std = []
        range_iter = range(n) if not TQDM_AVAILABLE or not show_progress else tqdm(range(n), desc="[Features] Neighbor amount")
        for i in range_iter:
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = df[amount_col].iloc[list(neighs)].fillna(0).values
                amt_1hop_mean.append(np.mean(neigh_amts))
                amt_1hop_std.append(np.std(neigh_amts) if len(neigh_amts) > 1 else 0)
            else:
                amt_1hop_mean.append(0)
                amt_1hop_std.append(0)
        features['amt_1hop_mean'] = np.array(amt_1hop_mean)
        features['amt_1hop_std'] = np.array(amt_1hop_std)

    # ========== 6. 账户级特征 ==========
    for col in account_features:
        if col not in df_columns:
            continue
        if df[col].dtype == 'object':
            le = LabelEncoder()
            features[col] = le.fit_transform(df[col].fillna('missing').astype(str))
        else:
            features[col] = df[col].fillna(-1).values

    # ========== 7. GAR Fraud Rate (无泄漏) ==========
    if entity_fraud_maps:
        print("[INFO] Computing fraud rates from train only...", flush=True)
        for col in entity_cols:
            if col not in df_columns or col not in entity_fraud_maps:
                continue
            features[f'{col}_fraud_rate'] = df[col].map(entity_fraud_maps[col]).fillna(0).values

        for col_pair, fraud_map in pair_fraud_maps.items():
            col1, col2 = col_pair.split('_', 1)
            if col1 not in df_columns or col2 not in df_columns:
                continue
            pair_values = df[col1].astype(str) + '_' + df[col2].astype(str)
            features[f'{col1}_{col2}_pair_fraud_rate'] = pair_values.map(fraud_map).fillna(0).values

        # Neighbor Fraud Rate
        train_label_map = dict(zip(train_idx, df.iloc[train_idx][label_col].values))
        neigh_fraud_rates = []
        range_iter = range(n) if not TQDM_AVAILABLE or not show_progress else tqdm(range(n), desc="[Features] Neighbor fraud")
        for i in range_iter:
            neighs = tx_neighbors.get(i, set())
            if neighs:
                train_neighs = [n for n in neighs if n in train_label_map]
                if train_neighs:
                    neigh_fraud_rates.append(np.mean([train_label_map[n] for n in train_neighs]))
                else:
                    neigh_fraud_rates.append(0)
            else:
                neigh_fraud_rates.append(0)
        features['neigh_fraud_rate'] = np.array(neigh_fraud_rates)

    # ========== 8. 时序特征 ==========
    timestamp_col = None
    for col in ['timestamp', '时间戳', 'trans_time']:
        if col in df_columns:
            timestamp_col = col
            break

    if timestamp_col:
        try:
            ts = pd.to_datetime(df[timestamp_col], errors='coerce')
            features['trans_hour'] = ts.dt.hour.fillna(12).values
            features['trans_dayofweek'] = ts.dt.dayofweek.fillna(0).values
        except:
            pass

    # ========== 9. 扩展特征 ==========
    if amount_col and card_col in df_columns:
        # Amount Z-score
        card_amt_mean = df.groupby(card_col)[amount_col].transform('mean')
        card_amt_std = df.groupby(card_col)[amount_col].transform('std').fillna(1)
        features['amount_zscore'] = ((df[amount_col] - card_amt_mean) / (card_amt_std + 1e-10)).fillna(0).values
        features['amount_percentile'] = df.groupby(card_col)[amount_col].rank(pct=True).fillna(0).values

    # 图指标
    degrees = [len(tx_neighbors.get(i, set())) for i in range(n)]
    max_degree = max(degrees) if max(degrees) > 0 else 1
    features['degree_centrality'] = np.array(degrees) / max_degree

    # 清理
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    feature_names = list(features.keys())
    print(f"[INFO] Generated {len(feature_names)} features", flush=True)

    return features, feature_names


def generate_features_step(fraud_data_path, white_data_path, output_dir,
                         card_col='card_id', entity_cols=None, account_features=None,
                         transaction_features=None, train_ratio=0.7, seed=42,
                         mode='cpu', workers=4):
    """Step 1: 特征生成"""
    print("="*60, flush=True)
    print("GAR Feature Generation (Step 1/2)", flush=True)
    print("="*60, flush=True)

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    if entity_cols is None:
        entity_cols = DEFAULT_ENTITY_COLS
    if account_features is None:
        account_features = DEFAULT_ACCOUNT_FEATURES
    if transaction_features is None:
        transaction_features = DEFAULT_TRANSACTION_FEATURES

    # 加载数据
    fraud_df, card_col, entity_cols, account_features, transaction_features, label_col = \
        load_and_preprocess(fraud_data_path, card_col, entity_cols, account_features,
                          transaction_features, True)

    white_df, _, _, _, _, _ = \
        load_and_preprocess(white_data_path, card_col, entity_cols, account_features,
                           transaction_features, True)

    # 合并
    combined_df = merge_fraud_white(fraud_df, white_df, card_col, label_col)
    del fraud_df, white_df
    gc.collect()

    # 分割
    train_idx, test_idx = split_data(combined_df, train_ratio=train_ratio, seed=seed)
    print(f"[INFO] Train: {len(train_idx)}, Test: {len(test_idx)}", flush=True)

    # 构建图
    print("[INFO] Building graph...", flush=True)
    tx_neighbors = build_graph(combined_df, entity_cols)

    # 计算欺诈率
    train_df = combined_df.iloc[train_idx]
    entity_fraud_maps, pair_fraud_maps = compute_fraud_rates_from_train(train_df, entity_cols, label_col)

    # 构建特征
    print("[INFO] Building GAR features...", flush=True)
    features_dict, feature_names = build_gar_features(
        combined_df, train_idx, tx_neighbors, card_col,
        entity_cols, account_features, transaction_features, label_col,
        entity_fraud_maps, pair_fraud_maps
    )

    # 添加split和label
    split_arr = np.array(['train' if i in train_idx else 'test' for i in range(len(combined_df))])
    features_dict[label_col] = combined_df[label_col].values

    # 分块导出（处理大规模数据）
    print("[INFO] Exporting features...", flush=True)
    features_csv = os.path.join(output_dir, 'gar_features.csv')

    # 分批写入避免内存问题
    chunk_size = 100000
    n_total = len(combined_df)
    key_cols = [c for c in ['card_id', '卡号', 'timestamp', '时间戳'] if c in combined_df.columns]

    df_features = pd.DataFrame({name: features_dict[name] for name in feature_names})
    if key_cols:
        df_features = pd.concat([combined_df[key_cols], df_features], axis=1)
    df_features[label_col] = combined_df[label_col].values
    df_features['split'] = split_arr

    # 分块写入
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_total)
        chunk_df = df_features.iloc[start_idx:end_idx]

        if i == 0:
            chunk_df.to_csv(features_csv, index=False, mode='w')
        else:
            chunk_df.to_csv(features_csv, index=False, mode='a', header=False)

        print(f"[INFO]   Exported chunk {i+1}/{n_chunks} ({end_idx}/{n_total})", flush=True)

    # 保存元信息
    meta = {
        'fraud_data_path': fraud_data_path,
        'white_data_path': white_data_path,
        'total_count': len(combined_df),
        'fraud_count': int(combined_df[label_col].sum()),
        'white_count': int(len(combined_df) - combined_df[label_col].sum()),
        'train_count': len(train_idx),
        'test_count': len(test_idx),
        'imbalance_ratio': float((len(combined_df) - combined_df[label_col].sum()) / max(1, combined_df[label_col].sum())),
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'label_col': label_col,
        'split_col': 'split',
        'mode': mode,
        'workers': workers
    }

    meta_path = os.path.join(output_dir, 'experiment_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - start_time
    print(f"[INFO] Total time: {elapsed/60:.1f} minutes", flush=True)
    print(f"[INFO] Features saved to: {features_csv}", flush=True)

    return meta


def train_classifier_step(output_dir, class_weight=1.0, seed=42):
    """Step 2: 分类器训练"""
    print("="*60, flush=True)
    print("GAR Classifier Training (Step 2/2)", flush=True)
    print("="*60, flush=True)

    start_time = time.time()

    # 加载元信息
    meta_path = os.path.join(output_dir, 'experiment_meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    label_col = meta['label_col']
    feature_names = meta['feature_names']

    print(f"[INFO] Loading features from {os.path.join(output_dir, 'gar_features.csv')}...", flush=True)

    # 读取特征
    df_features = pd.read_csv(os.path.join(output_dir, 'gar_features.csv'))
    print(f"[INFO]   Records: {len(df_features)}, Features: {len(df_features.columns)}", flush=True)

    # 分割
    train_mask = df_features['split'] == 'train'
    test_mask = df_features['split'] == 'test'

    X_train = df_features.loc[train_mask, feature_names].values
    X_test = df_features.loc[test_mask, feature_names].values
    y_train = df_features.loc[train_mask, label_col].values
    y_test = df_features.loc[test_mask, label_col].values

    # 处理无穷值
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    print(f"[INFO] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}", flush=True)
    print(f"[INFO] Fraud in train: {y_train.sum()}/{len(y_train)} ({100*y_train.sum()/len(y_train):.2f}%)", flush=True)
    print(f"[INFO] Fraud in test: {y_test.sum()}/{len(y_test)} ({100*y_test.sum()/len(y_test):.2f}%)", flush=True)

    # 训练
    print("[INFO] Training GradientBoostingClassifier...", flush=True)
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    # 计算类别权重
    fraud_ratio = y_train.sum() / len(y_train)
    n_fraud = y_train.sum()
    n_white = len(y_train) - n_fraud

    print(f"[INFO] Class weight for fraud: {class_weight}x", flush=True)

    # 使用sample_weight处理类别不均衡
    sample_weights = np.ones(len(y_train))
    fraud_indices = y_train == 1
    sample_weights[fraud_indices] *= class_weight

    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed
    )
    gb.fit(X_train, y_train, sample_weight=sample_weights)

    # 预测
    train_proba = gb.predict_proba(X_train)[:, 1]
    test_proba = gb.predict_proba(X_test)[:, 1]

    # 评估
    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)

    test_pred = (test_proba > 0.5).astype(int)
    precision = precision_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)

    results = {
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'class_weight': class_weight,
        'feature_importance': list(zip(feature_names, gb.feature_importances_.tolist()))
    }

    # 保存结果
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"[INFO] Training time: {elapsed:.1f}s", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='GAR White+Fraud Experiment (Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: 生成特征
  python experiments/gar_white_fraud_experiment.py \\
      --fraud-data ./data/fraud.csv \\
      --white-data ./data/white.csv \\
      --output-dir ./outputs/exp \\
      --step generate

  # Step 2: 训练分类器（类别权重=20）
  python experiments/gar_white_fraud_experiment.py \\
      --fraud-data ./data/fraud.csv \\
      --white-data ./data/white.csv \\
      --output-dir ./outputs/exp \\
      --step train \\
      --class-weight 20

  # 使用NPU加速特征生成
  python experiments/gar_white_fraud_experiment.py \\
      --fraud-data ./data/fraud.csv \\
      --white-data ./data/white.csv \\
      --output-dir ./outputs/exp \\
      --step generate \\
      --mode npu
        """
    )

    parser.add_argument('--fraud-data', type=str, required=True,
                        help='欺诈数据文件路径')
    parser.add_argument('--white-data', type=str, required=True,
                        help='白样本数据文件路径')
    parser.add_argument('--output-dir', type=str, default='./outputs/exp',
                        help='输出目录')
    parser.add_argument('--step', type=str, choices=['generate', 'train', 'all'],
                        default='all', help='执行步骤')
    parser.add_argument('--card-col', type=str, default='card_id',
                        help='卡号列名')
    parser.add_argument('--entity-cols', type=str, default=None,
                        help='实体列名')
    parser.add_argument('--account-features', type=str, default=None,
                        help='账户级特征')
    parser.add_argument('--transaction-features', type=str, default=None,
                        help='交易级特征')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--class-weight', type=float, default=1.0,
                        help='欺诈样本权重（用于处理不均衡，默认1.0）')
    parser.add_argument('--mode', type=str, choices=['cpu', 'npu', 'dist'],
                        default='cpu', help='运行模式')
    parser.add_argument('--workers', type=int, default=4,
                        help='分布式worker数量')

    args = parser.parse_args()

    entity_cols = args.entity_cols.split(',') if args.entity_cols else None
    account_features = args.account_features.split(',') if args.account_features else None
    transaction_features = args.transaction_features.split(',') if args.transaction_features else None

    if args.step in ['generate', 'all']:
        generate_features_step(
            args.fraud_data, args.white_data, args.output_dir,
            args.card_col, entity_cols, account_features, transaction_features,
            args.train_ratio, args.seed, args.mode, args.workers
        )

    if args.step in ['train', 'all']:
        results = train_classifier_step(args.output_dir, args.class_weight, args.seed)

        print(f"\n{'='*60}", flush=True)
        print("Results", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Train AUC: {results['train_auc']:.4f}", flush=True)
        print(f"Test AUC: {results['test_auc']:.4f}", flush=True)
        print(f"Precision: {results['precision']:.4f}", flush=True)
        print(f"Recall: {results['recall']:.4f}", flush=True)
        print(f"F1: {results['f1']:.4f}", flush=True)

        print("\nTop 10 Features:", flush=True)
        for i, (name, imp) in enumerate(sorted(results['feature_importance'], key=lambda x: x[1], reverse=True)[:10]):
            print(f"  {i+1:2d}. {name:<40} {imp:.4f}", flush=True)


if __name__ == '__main__':
    main()