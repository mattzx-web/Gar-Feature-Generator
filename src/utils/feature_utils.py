"""
特征生成公共工具模块

提供GAR和KG Brute Force特征生成共用的工具函数。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import os


# 默认实体列配置
DEFAULT_ENTITY_COLS = ['card1', 'card2', 'addr1', 'P_emaildomain']
KG_ENTITY_COLS = ['card1', 'card2', 'card3', 'card4', 'addr1', 'addr2',
                  'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo']


def load_data(data_dir, entity_cols):
    """
    加载并预处理数据

    Args:
        data_dir: 数据目录路径
        entity_cols: 实体列名列表

    Returns:
        train_data, test_data, y_train, y_test, train_idx, test_idx, feature_cols
    """
    print(f"[INFO] Loading data from {data_dir}...", flush=True)

    # 加载交易数据
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

    # 划分训练/测试集 (70/30)
    n = len(train)
    n_train = int(0.7 * n)

    indices = np.arange(n)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_data = train.iloc[train_idx].copy()
    test_data = train.iloc[test_idx].copy()

    y_train = train_data['isFraud'].values
    y_test = test_data['isFraud'].values

    print(f"[INFO] Train: {len(train_data)}, Test: {len(test_data)}", flush=True)
    print(f"[INFO] Fraud rates: train={y_train.mean():.4f}, test={y_test.mean():.4f}", flush=True)

    return train_data, test_data, y_train, y_test, train_idx, test_idx


def build_graph(train_data, entity_cols, neighbor_threshold=300):
    """
    构建交易图结构

    Args:
        train_data: 训练数据DataFrame
        entity_cols: 实体列名列表
        neighbor_threshold: 每个实体组合的邻居数量上限

    Returns:
        tx_neighbors: 交易ID到邻居集合的映射
    """
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


def build_2hop_neighbors(tx_neighbors):
    """
    构建2跳邻居

    Args:
        tx_neighbors: 1跳邻居映射

    Returns:
        tx_2hop_neighbors: 2跳邻居映射
    """
    tx_2hop_neighbors = defaultdict(set)
    for tx, neighs in tx_neighbors.items():
        for n in neighs:
            tx_2hop_neighbors[tx].update(tx_neighbors.get(n, set()))
    tx_2hop_neighbors[tx].discard(tx)
    return tx_2hop_neighbors


def save_features_to_csv(features_dict, output_path, train_idx, test_idx, y_train, y_test):
    """
    将特征保存为CSV文件

    Args:
        features_dict: 特征名字典 {feature_name: feature_values}
        output_path: 输出CSV路径
        train_idx, test_idx: 训练/测试集索引
        y_train, y_test: 标签
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # 构建DataFrame
    train_features = {}
    test_features = {}

    for name, values in features_dict.items():
        if len(values) == len(train_idx) + len(test_idx):
            train_features[name] = values[:len(train_idx)]
            test_features[name] = values[len(train_idx):]
        else:
            # 如果特征长度不匹配，假设values是按原始数据顺序排列的
            train_features[name] = values[train_idx] if len(values) > len(test_idx) else values[:len(train_idx)]
            test_features[name] = values[test_idx] if len(values) > len(test_idx) else values[len(train_idx):]

    # 创建DataFrame
    df_train = pd.DataFrame(train_features)
    df_train['isFraud'] = y_train
    df_train['split'] = 'train'
    df_train['original_idx'] = train_idx

    df_test = pd.DataFrame(test_features)
    df_test['isFraud'] = y_test
    df_test['split'] = 'test'
    df_test['original_idx'] = test_idx

    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"[INFO] Features saved to {output_path}", flush=True)
    print(f"[INFO] Shape: {df.shape}", flush=True)

    return output_path


def load_features_from_csv(csv_path):
    """
    从CSV文件加载特征

    Args:
        csv_path: 特征CSV文件路径

    Returns:
        X_train, X_test, y_train, y_test, feature_names, split_info
    """
    print(f"[INFO] Loading features from {csv_path}...", flush=True)
    df = pd.read_csv(csv_path)

    split_info = df[['split', 'original_idx']].copy() if 'original_idx' in df.columns else None

    # 分离训练/测试集
    train_mask = df['split'] == 'train'
    test_mask = df['split'] == 'test'

    # 获取特征列（排除meta列）
    meta_cols = ['isFraud', 'split', 'original_idx']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X_train = df.loc[train_mask, feature_cols].values
    X_test = df.loc[test_mask, feature_cols].values
    y_train = df.loc[train_mask, 'isFraud'].values
    y_test = df.loc[test_mask, 'isFraud'].values

    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}", flush=True)
    print(f"[INFO] Features: {len(feature_cols)}", flush=True)

    return X_train, X_test, y_train, y_test, feature_cols, split_info