"""
GAR Feature Selector

基于高欺诈率相关的特征筛选模块。
使用点二列相关分析（point-biserial correlation）筛选与欺诈标签高度相关的特征。

用法:
    python -m src.gar.gar_feature_selector --features ./features/gar_features.csv --top-k 20 --output ./features/selected_features.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os
import sys
from typing import List, Tuple, Dict

sys.stdout.reconfigure(line_buffering=True)


def compute_point_biserial_correlation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    计算点二列相关系数

    Args:
        X: 特征矩阵 (n_samples, n_features)
        y: 二值标签 (n_samples,)

    Returns:
        correlations: 每个特征与标签的相关系数 (n_features,)
    """
    n_features = X.shape[1]
    correlations = np.zeros(n_features)

    for i in range(n_features):
        feature = X[:, i]
        # 过滤掉全为常量的特征
        if np.std(feature) < 1e-10:
            correlations[i] = 0
        else:
            corr, _ = stats.pointbiserialr(y, feature)
            correlations[i] = corr if not np.isnan(corr) else 0

    return correlations


def select_high_fraud_rate_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: int = 20,
    correlation_threshold: float = 0.05
) -> Tuple[List[str], pd.DataFrame]:
    """
    选择与欺诈率高度相关的特征

    Args:
        X: 特征矩阵
        y: 欺诈标签 (0/1)
        feature_names: 特征名列表
        top_k: 返回前k个特征
        correlation_threshold: 相关系数阈值，低于此值的特征被过滤

    Returns:
        selected_features: 选中的特征名列表
        correlation_df: 包含每个特征相关系数的DataFrame
    """
    print(f"[INFO] Computing point-biserial correlations for {len(feature_names)} features...", flush=True)

    # 计算相关系数
    correlations = compute_point_biserial_correlation(X, y)

    # 创建相关系数DataFrame
    corr_df = pd.DataFrame({
        'feature': feature_names,
        'correlation': correlations,
        'abs_correlation': np.abs(correlations)
    })
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)

    # 过滤低于阈值的特征
    filtered_df = corr_df[corr_df['abs_correlation'] >= correlation_threshold]

    # 取前k个
    selected_df = filtered_df.head(top_k)
    selected_features = selected_df['feature'].tolist()

    print(f"[INFO] Selected {len(selected_features)} features with correlation >= {correlation_threshold}", flush=True)
    print(f"[INFO] Top 10 features:", flush=True)
    for i, row in enumerate(selected_df.head(10).itertuples()):
        print(f"  {i+1:2d}. {row.feature:<40} r={row.correlation:.4f}", flush=True)

    return selected_features, corr_df


def filter_features_by_correlation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    correlation_threshold: float = 0.05
) -> List[str]:
    """
    过滤掉与欺诈标签相关性低的特征

    Args:
        X: 特征矩阵
        y: 欺诈标签
        feature_names: 特征名列表
        correlation_threshold: 阈值

    Returns:
        filtered_features: 通过阈值的特征名列表
    """
    correlations = compute_point_biserial_correlation(X, y)
    filtered = [
        name for name, corr in zip(feature_names, correlations)
        if abs(corr) >= correlation_threshold
    ]
    return filtered


def rank_features_by_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = 'correlation'
) -> pd.DataFrame:
    """
    对特征进行重要性排序

    Args:
        X: 特征矩阵
        y: 欺诈标签
        feature_names: 特征名列表
        method: 'correlation' 或 'mutual_info'

    Returns:
        DataFrame with feature ranking
    """
    if method == 'correlation':
        correlations = compute_point_biserial_correlation(X, y)
        ranking_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(correlations),
            'correlation': correlations
        })
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X, y, random_state=42)
        ranking_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mi_scores,
            'correlation': np.zeros(len(feature_names))
        })
    else:
        raise ValueError(f"Unknown method: {method}")

    ranking_df = ranking_df.sort_values('importance', ascending=False)
    return ranking_df


def select_features_by_group(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    groups: Dict[str, List[str]],
    top_k_per_group: int = 5
) -> List[str]:
    """
    按特征组选择，每个组选top-k个

    Args:
        X: 特征矩阵
        y: 欺诈标签
        feature_names: 特征名列表
        groups: 特征组字典 {'group_name': ['feature1', 'feature2', ...]}
        top_k_per_group: 每个组选几个

    Returns:
        选中的特征名列表
    """
    correlations = compute_point_biserial_correlation(X, y)
    name_to_corr = dict(zip(feature_names, correlations))

    selected = []
    for group_name, features in groups.items():
        group_corrs = [(f, name_to_corr.get(f, 0)) for f in features if f in feature_names]
        group_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        selected.extend([f for f, _ in group_corrs[:top_k_per_group]])
        print(f"[INFO] Group '{group_name}': selected {min(top_k_per_group, len(group_corrs))} features", flush=True)

    return selected


def export_selected_features(
    df: pd.DataFrame,
    selected_features: List[str],
    output_path: str
):
    """导出选中的特征"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # 保留关键列 + 选中的特征
    key_cols = []
    for col in ['card_id', 'timestamp', 'isFraud', 'split']:
        if col in df.columns:
            key_cols.append(col)

    output_cols = key_cols + selected_features
    available_cols = [c for c in output_cols if c in df.columns]

    df[available_cols].to_csv(output_path, index=False)
    print(f"[INFO] Selected features saved to {output_path}", flush=True)
    print(f"[INFO] Shape: {df[available_cols].shape}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='GAR Feature Selector - High Fraud Rate Feature Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基础用法：选择top-20特征
  python -m src.gar.gar_feature_selector \\
      --features ./features/gar_features.csv \\
      --top-k 20 \\
      --output ./features/selected_features.csv

  # 自定义相关系数阈值
  python -m src.gar.gar_feature_selector \\
      --features ./features/gar_features.csv \\
      --top-k 30 \\
      --threshold 0.03 \\
      --output ./features/selected_features.csv
        """
    )

    parser.add_argument('--features', type=str, required=True,
                        help='特征CSV文件路径')
    parser.add_argument('--top-k', type=int, default=20,
                        help='选择前k个特征（默认: 20）')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='相关系数阈值（默认: 0.05）')
    parser.add_argument('--output', type=str, default='./features/selected_features.csv',
                        help='输出CSV路径')
    parser.add_argument('--correlation-only', action='store_true',
                        help='只输出相关系数，不筛选特征')

    args = parser.parse_args()

    print("="*60, flush=True)
    print("GAR Feature Selector", flush=True)
    print("="*60, flush=True)

    # 加载特征
    print(f"[INFO] Loading features from {args.features}...", flush=True)
    df = pd.read_csv(args.features)

    # 获取特征列
    meta_cols = ['split', 'original_idx', 'isFraud', 'is_fraud', 'card_id', 'timestamp', '时间戳']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    feature_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"[INFO] Found {len(feature_cols)} features", flush=True)

    # 获取标签
    label_col = None
    for col in ['isFraud', 'fraud', 'label', 'is_fraud']:
        if col in df.columns:
            label_col = col
            break

    if not label_col:
        raise ValueError("No label column found in the feature file")

    y = df[label_col].values
    X = df[feature_cols].values

    # 相关系数分析
    selected_features, corr_df = select_high_fraud_rate_features(
        X, y, feature_cols,
        top_k=args.top_k,
        correlation_threshold=args.threshold
    )

    # 保存相关系数报告
    corr_output = args.output.replace('.csv', '_correlations.csv')
    corr_df.to_csv(corr_output, index=False)
    print(f"[INFO] Correlation report saved to {corr_output}", flush=True)

    if args.correlation_only:
        print("[INFO] --correlation-only mode: skipping feature filtering", flush=True)
    else:
        # 导出选中的特征
        export_selected_features(df, selected_features, args.output)

    print("\n" + "="*60, flush=True)
    print("Feature selection complete!", flush=True)
    print(f"Selected {len(selected_features)} features", flush=True)
    print("="*60, flush=True)


if __name__ == '__main__':
    main()