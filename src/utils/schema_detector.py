"""
Column Schema Auto-Detection and Mapping Utility

自动检测数据集的列类型并映射到标准列名。
支持用户自定义列名的数据集。

用法:
    from src.utils.schema_detector import detect_schema, map_columns

    schema = detect_schema(df)
    print(f"Detected: {schema}")

    # 映射后的DataFrame
    df_mapped = map_columns(df, schema)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re


# 标准列名定义
STANDARD_COLUMNS = {
    'card_id': ['card_id', 'card', 'card_no', '卡号', '银行卡号', 'customer_id', 'customer'],
    'timestamp': ['timestamp', 'time', 'datetime', 'trans_time', 'transaction_time', '时间戳', '交易时间', 'tx_datetime', 'trans_datetime'],
    'amount': ['amount', 'amt', 'transaction_amount', '交易金额', 'tx_amount', 'total'],
    'balance': ['balance', 'balance_after', '账户余额', '余额'],
    'merchant_id': ['merchant_id', 'merchant', 'mcc', '商户号', 'merchant_code'],
    'merchant_type': ['merchant_type', 'merchant_category', 'mcc_code', '商户类型'],
    'device': ['device', 'device_type', 'device_id', '设备', '交易设备'],
    'is_night': ['is_night', 'night_tx', '夜间交易'],
    'is_cross_border': ['is_cross_border', 'cross_border', '跨境', '境外交易'],
    'is_fraud': ['isFraud', 'fraud', 'label', 'is_fraud', 'fraud_label', '欺诈'],
    'card_level': ['card_level', 'level', '卡等级', '等级'],
    'card_location': ['card_location', 'location', 'card_region', '卡注册地', '地区'],
    'card_type': ['card_type', 'card_category', '卡类型', '类型'],
    'terminal_id': ['terminal_id', 'terminal', 'pos_id', '终端号', '终端ID'],
}

# 列类型检测的正则表达式
COLUMN_PATTERNS = {
    'id_pattern': re.compile(r'.*(id|no|号|码).*', re.IGNORECASE),
    'time_pattern': re.compile(r'.*(time|date|datetime|timestamp|时间|日).*', re.IGNORECASE),
    'amount_pattern': re.compile(r'.*(amount|amt|金额|总额|sum).*', re.IGNORECASE),
    'fraud_pattern': re.compile(r'.*(fraud|欺诈|风险).*', re.IGNORECASE),
    'device_pattern': re.compile(r'.*(device|设备|终端).*', re.IGNORECASE),
}


def detect_column_type(col_name: str) -> Optional[str]:
    """根据列名检测列类型"""
    col_lower = col_name.lower()

    # 检查是否匹配标准列名
    for col_type, aliases in STANDARD_COLUMNS.items():
        for alias in aliases:
            if alias in col_lower or col_lower in alias:
                return col_type

    # 检查正则模式
    if COLUMN_PATTERNS['fraud_pattern'].match(col_name):
        return 'is_fraud'
    if COLUMN_PATTERNS['time_pattern'].match(col_name):
        return 'timestamp'
    if COLUMN_PATTERNS['amount_pattern'].match(col_name):
        return 'amount'
    if COLUMN_PATTERNS['id_pattern'].match(col_name):
        if 'card' in col_lower:
            return 'card_id'
        if 'merchant' in col_lower:
            return 'merchant_id'
        if 'terminal' in col_lower:
            return 'terminal_id'
    if COLUMN_PATTERNS['device_pattern'].match(col_name):
        return 'device'

    return None


def analyze_dataframe(df: pd.DataFrame) -> Dict[str, any]:
    """分析DataFrame的数据特征"""
    analysis = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_pct': {},
        'unique_counts': {},
        'sample_values': {},
    }

    for col in df.columns:
        analysis['missing_pct'][col] = df[col].isna().sum() / len(df) * 100
        analysis['unique_counts'][col] = df[col].nunique()
        analysis['sample_values'][col] = df[col].dropna().head(3).tolist()

    return analysis


def detect_schema(df: pd.DataFrame, user_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    自动检测数据集的列类型

    Args:
        df: 输入的DataFrame
        user_mapping: 用户指定的列名映射，如 {'卡号': 'card_id', '交易金额': 'amount'}

    Returns:
        schema: 列名到标准类型的映射，如 {'card_id': 'card_no', 'amount': '交易金额', ...}
    """
    schema = {}
    detected_types = set()

    # 1. 首先应用用户指定的映射
    if user_mapping:
        for std_col, user_col in user_mapping.items():
            if user_col in df.columns:
                schema[std_col] = user_col
                detected_types.add(std_col)

    # 2. 自动检测剩余列
    for col in df.columns:
        if col in schema.values():
            continue  # 已被映射

        col_type = detect_column_type(col)

        if col_type and col_type not in detected_types:
            # 对于id类型的列，确保card_id只有一个
            if col_type in ['card_id', 'customer_id'] and 'card_id' in detected_types:
                continue
            if col_type in ['terminal_id', 'merchant_id'] and 'terminal_id' in detected_types:
                continue

            schema[col_type] = col
            detected_types.add(col_type)

    # 3. 验证必需列
    required_cols = ['card_id', 'amount']
    missing_required = [col for col in required_cols if col not in schema]

    return schema, missing_required


def auto_detect_from_data(df: pd.DataFrame) -> Dict[str, str]:
    """
    基于数据内容自动检测列类型

    检查列的数据分布来判断类型
    """
    schema = {}

    for col in df.columns:
        if df[col].dtype == 'object':
            # 尝试解析为时间
            try:
                pd.to_datetime(df[col], errors='raise')
                if 'time' not in schema and 'timestamp' not in schema:
                    schema['timestamp'] = col
                    continue
            except:
                pass

        # 检查数值列
        if df[col].dtype in ['int64', 'float64']:
            col_lower = col.lower()

            # 金额列：通常范围较大，有特定前缀/后缀
            if 'amount' in col_lower or 'amt' in col_lower or '金额' in col:
                if 'amount' not in schema:
                    schema['amount'] = col

            # 卡号列：通常是整数，unique数量中等
            elif ('card' in col_lower or '客户' in col) and ('id' in col_lower or '号' in col):
                if 'card_id' not in schema and df[col].nunique() > 10:
                    schema['card_id'] = col

            # 商户列
            elif 'merchant' in col_lower or '商户' in col:
                if 'merchant_id' not in schema:
                    schema['merchant_id'] = col

            # 终端列
            elif 'terminal' in col_lower or 'pos' in col_lower:
                if 'terminal_id' not in schema:
                    schema['terminal_id'] = col

            # 欺诈标签列：二值或比例
            if 'fraud' in col_lower or '欺诈' in col:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 2:
                    if 'is_fraud' not in schema:
                        schema['is_fraud'] = col

    return schema


def map_columns(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    """
    根据schema映射列名

    Args:
        df: 原始DataFrame
        schema: 标准列名到实际列名的映射

    Returns:
        映射后的DataFrame（保留原列名，同时添加标准列名）
    """
    df_mapped = df.copy()

    # 添加标准列名（如果不存在）
    for std_col, actual_col in schema.items():
        if std_col != actual_col and actual_col in df.columns:
            df_mapped[std_col] = df[actual_col]

    return df_mapped


def generate_config_from_schema(schema: Dict[str, str], entity_cols: List[str], account_features: List[str], transaction_features: List[str]) -> Dict:
    """
    根据检测到的schema生成命令行配置

    Returns:
        配置字典，可以用来调用GAR脚本
    """
    config = {
        'card_col': schema.get('card_id', 'card_id'),
        'entity_cols': entity_cols,
        'account_features': account_features,
        'transaction_features': transaction_features,
        'label_col': schema.get('is_fraud', None),
    }

    return config


def print_schema_report(schema: Dict[str, str], missing_required: List[str] = None):
    """打印schema检测报告"""
    print("\n" + "="*60, flush=True)
    print("Column Schema Detection Report", flush=True)
    print("="*60, flush=True)

    for std_col, actual_col in schema.items():
        print(f"  {std_col:<20} -> {actual_col}", flush=True)

    if missing_required:
        print("\n[WARNING] Missing required columns:", flush=True)
        for col in missing_required:
            print(f"  - {col}", flush=True)

    print("="*60, flush=True)


# CLI接口
def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Auto-detect column schema from CSV')
    parser.add_argument('--data', type=str, required=True, help='CSV file path')
    parser.add_argument('--mapping', type=str, default=None, help='User mapping JSON file')

    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"[INFO] Loaded {len(df)} records, {len(df.columns)} columns", flush=True)

    schema, missing = detect_schema(df)

    print_schema_report(schema, missing)

    # 输出映射配置
    entity_cols = [v for k, v in schema.items() if k in ['card_id', 'merchant_id', 'device', 'terminal_id']]
    account_features = [v for k, v in schema.items() if k in ['card_level', 'card_location', 'card_type']]
    transaction_features = [v for k, v in schema.items() if k in ['amount', 'balance', 'is_night', 'is_cross_border']]

    config = generate_config_from_schema(schema, entity_cols, account_features, transaction_features)

    print("\nRecommended CLI arguments:", flush=True)
    print(f"  --card-col {config['card_col']}", flush=True)
    if entity_cols:
        print(f"  --entity-cols {','.join(entity_cols)}", flush=True)
    if account_features:
        print(f"  --account-features {','.join(account_features)}", flush=True)
    if transaction_features:
        print(f"  --transaction-features {','.join(transaction_features)}", flush=True)
    if config['label_col']:
        print(f"  --label-col {config['label_col']}", flush=True)


if __name__ == '__main__':
    main()