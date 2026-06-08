"""
Fraud Detection Dataset Generator

基于 IEEE Fraud Detection Handbook 的模拟数据集生成器。
生成符合实际金融欺诈模式的交易数据。

用法:
    python -m src.data.fraud_dataset_generator --output ./data/simulated_fraud.csv

    # 自定义参数
    python -m src.data.fraud_dataset_generator \
        --n-customers 5000 \
        --n-terminals 10000 \
        --n-days 183 \
        --fraud-rate 0.008 \
        --output ./data/my_fraud_dataset.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json

sys.stdout.reconfigure(line_buffering=True)


@dataclass
class FraudDatasetConfig:
    """数据集生成配置"""
    n_customers: int = 5000
    n_terminals: int = 10000
    n_days: int = 183  # April 1 to September 30, 2018
    n_transactions: int = 1750000
    fraud_rate: float = 0.008  # Overall fraud rate ~0.8%
    seed: int = 42

    # Customer behavior parameters
    mean_amount_min: float = 5.0
    mean_amount_max: float = 100.0
    daily_tx_lambda_max: float = 4.0  # Poisson distribution parameter

    # Geographic parameters
    grid_size: int = 100  # 100x100 grid

    # Fraud scenario parameters
    amount_fraud_threshold: float = 220.0  # Scenario 1: amount-based
    terminal_compromise_days: int = 28  # Scenario 2: terminal compromise
    terminal_compromise_per_day: int = 2  # Compromised terminals per day
    burst_attack_days: int = 14  # Scenario 3: burst attacks
    burst_customers_per_day: int = 3  # Customers under attack per day

    # Card parameters
    card_levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    card_locations: List[str] = field(default_factory=lambda: [
        '北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都',
        '西安', '苏州', '天津', '重庆', '郑州', '长沙', '沈阳', '青岛'
    ])
    card_types: List[str] = field(default_factory=lambda: ['credit', 'debit', 'prepaid'])
    merchant_categories: List[str] = field(default_factory=lambda: [
        '餐饮', '超市', '服装', '电器', '娱乐', '旅游', '医疗', '教育',
        '加油', '酒店', '机票', '网购', '便利店', '电影院', '咖啡厅', '书店'
    ])
    device_types: List[str] = field(default_factory=lambda: [
        'POS', 'ATM', 'WEB', 'APP', 'MOB'
    ])


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)


def simulate_customer_profiles(config: FraudDatasetConfig) -> pd.DataFrame:
    """模拟客户档案（每个客户有个性化的交易参数）"""
    print(f"[INFO] Simulating {config.n_customers} customer profiles...", flush=True)

    customers = pd.DataFrame({
        'customer_id': range(config.n_customers),
        'mean_amount': np.random.uniform(config.mean_amount_min, config.mean_amount_max, config.n_customers),
        'std_amount_factor': 0.5,  # std = mean / 2
        'daily_tx_lambda': np.random.uniform(0.5, config.daily_tx_lambda_max, config.n_customers),
        'x': np.random.randint(0, config.grid_size, config.n_customers),
        'y': np.random.randint(0, config.grid_size, config.n_customers),
    })
    customers['std_amount'] = customers['mean_amount'] * customers['std_amount_factor']

    # 卡等级分布（正态分布，钟形）
    weights = [0.1, 0.25, 0.35, 0.2, 0.1]
    customers['card_level'] = np.random.choice(config.card_levels, config.n_customers, p=weights)

    # 卡注册地（随机）
    customers['card_location'] = np.random.choice(config.card_locations, config.n_customers)

    # 卡类型分布
    type_weights = [0.4, 0.4, 0.2]  # credit, debit, prepaid
    customers['card_type'] = np.random.choice(config.card_types, config.n_customers, p=type_weights)

    print(f"[INFO] Customer profiles generated", flush=True)
    return customers


def simulate_terminal_profiles(config: FraudDatasetConfig) -> pd.DataFrame:
    """模拟终端档案"""
    print(f"[INFO] Simulating {config.n_terminals} terminal profiles...", flush=True)

    terminals = pd.DataFrame({
        'terminal_id': range(config.n_terminals),
        'x': np.random.randint(0, config.grid_size, config.n_terminals),
        'y': np.random.randint(0, config.grid_size, config.n_terminals),
        'merchant_type': np.random.choice(config.merchant_categories, config.n_terminals),
    })

    print(f"[INFO] Terminal profiles generated", flush=True)
    return terminals


def get_nearby_terminals(customer_x: int, customer_y: int, terminals: pd.DataFrame, radius: int = 5) -> List[int]:
    """获取客户附近半径radius内的终端ID列表"""
    distances = np.sqrt((terminals['x'] - customer_x)**2 + (terminals['y'] - customer_y)**2)
    nearby = terminals[distances <= radius]['terminal_id'].tolist()
    return nearby if nearby else terminals['terminal_id'].tolist()[:10]


def generate_base_transactions(config: FraudDatasetConfig, customers: pd.DataFrame, terminals: pd.DataFrame) -> pd.DataFrame:
    """生成基础交易记录（无欺诈标签）"""
    print(f"[INFO] Generating ~{config.n_transactions:,} base transactions...", flush=True)

    start_date = datetime(2018, 4, 1)
    date_range = [start_date + timedelta(days=d) for d in range(config.n_days)]

    transactions = []

    # 每天生成约 config.n_transactions / config.n_days 笔交易
    daily_tx_target = config.n_transactions // config.n_days

    for day_idx, date in enumerate(date_range):
        daily_generated = 0

        # 每天每个客户根据 Poisson 分布生成交易
        for _, customer in customers.iterrows():
            n_tx_today = np.random.poisson(customer['daily_tx_lambda'])

            if daily_generated >= daily_tx_target * 1.2:  # 超过目标就停止
                break

            for _ in range(n_tx_today):
                if daily_generated >= daily_tx_target * 1.2:
                    break

                # 交易时间：高斯分布，中心在中午
                hour = int(np.random.normal(12, 6))
                hour = max(0, min(23, hour))
                minute = np.random.randint(0, 60)
                second = np.random.randint(0, 60)
                tx_datetime = date.replace(hour=hour, minute=minute, second=second)

                # 获取客户附近的终端
                nearby_terminals = get_nearby_terminals(customer['x'], customer['y'], terminals)

                # 交易金额：对数正态分布
                amount = np.random.lognormal(
                    mean=np.log(customer['mean_amount']),
                    sigma=0.5  # 对数标准差
                )
                amount = round(amount, 2)

                # 交易后余额（简化模拟）
                base_balance = np.random.uniform(1000, 50000)
                balance = round(base_balance + np.random.uniform(-amount, amount), 2)

                # 设备类型
                device = np.random.choice(config.device_types, p=[0.4, 0.1, 0.3, 0.15, 0.05])

                # 是否夜间交易
                is_night = 1 if hour >= 22 or hour <= 6 else 0

                # 是否跨境（基于卡注册地，5%概率）
                is_cross_border = 1 if np.random.random() < 0.05 else 0

                transactions.append({
                    'customer_id': customer['customer_id'],
                    'terminal_id': np.random.choice(nearby_terminals),
                    'tx_datetime': tx_datetime,
                    'amount': amount,
                    'balance': balance,
                    'card_level': customer['card_level'],
                    'card_location': customer['card_location'],
                    'card_type': customer['card_type'],
                    'device': device,
                    'is_night': is_night,
                    'is_cross_border': is_cross_border,
                    'isFraud': 0,  # 默认非欺诈
                })

                daily_generated += 1

    df = pd.DataFrame(transactions)
    print(f"[INFO] Generated {len(df):,} transactions", flush=True)

    return df


def inject_amount_fraud(df: pd.DataFrame, config: FraudDatasetConfig) -> pd.DataFrame:
    """
    Scenario 1: Amount-based fraud
    交易金额超过阈值时标记为欺诈
    """
    print("[INFO] Injecting amount-based fraud...", flush=True)

    # 高金额交易有更高欺诈概率
    fraud_mask = df['amount'] > config.amount_fraud_threshold

    # 对于金额 > threshold 的交易，随机标记部分为欺诈
    high_amount_txs = df[fraud_mask].copy()
    if len(high_amount_txs) > 0:
        # 约 50% 的高金额交易被标记为欺诈（提高比例）
        fraud_indices = high_amount_txs.sample(frac=0.5, random_state=config.seed).index
        df.loc[fraud_indices, 'isFraud'] = 1

    n_fraud = fraud_mask.sum()
    print(f"[INFO] Amount-based fraud: {n_fraud:,} transactions above threshold", flush=True)

    return df


def inject_terminal_fraud(df: pd.DataFrame, config: FraudDatasetConfig) -> pd.DataFrame:
    """
    Scenario 2: Terminal compromise fraud
    每天随机选择更多终端，连续28天被标记为欺诈
    """
    print("[INFO] Injecting terminal compromise fraud...", flush=True)

    start_date = datetime(2018, 4, 1)

    # 从第31天开始（5月1日之后）才有terminal compromise
    compromise_start_day = 31

    df = df.copy()
    df['tx_date'] = pd.to_datetime(df['tx_datetime']).dt.date

    # 每天选择更多终端（提高至10个）
    n_compromised_per_day = max(5, config.n_terminals // 200)  # 至少5个，约0.5%的终端

    for day_idx in range(compromise_start_day, compromise_start_day + config.terminal_compromise_days):
        date = start_date + timedelta(days=day_idx)
        date = date.date()

        # 随机选择 compromised terminals
        compromised_terminals = np.random.choice(
            range(config.n_terminals),
            size=n_compromised_per_day,
            replace=False
        )

        # 标记在该日期、与这些终端的交易为欺诈
        mask = (df['tx_date'] == date) & (df['terminal_id'].isin(compromised_terminals))
        df.loc[mask, 'isFraud'] = 1

    n_compromise_fraud = (df['isFraud'] == 1).sum() - len(df[df['amount'] > config.amount_fraud_threshold]) * 0.5
    print(f"[INFO] Terminal compromise fraud injected", flush=True)

    return df


def inject_burst_fraud(df: pd.DataFrame, config: FraudDatasetConfig) -> pd.DataFrame:
    """
    Scenario 3: Burst attack fraud
    每天随机选择更多客户，他们的交易金额乘以5（模拟攻击）
    """
    print("[INFO] Injecting burst attack fraud...", flush=True)

    start_date = datetime(2018, 4, 1)

    # 从第31天开始
    burst_start_day = 31

    df = df.copy()

    # 每天选择更多客户（提高至10个）
    n_attacked_per_day = max(10, config.n_customers // 100)  # 至少10个客户

    for day_idx in range(burst_start_day, burst_start_day + config.burst_attack_days):
        date = start_date + timedelta(days=day_idx)

        # 随机选择被攻击的客户
        attacked_customers = np.random.choice(
            range(config.n_customers),
            size=n_attacked_per_day,
            replace=False
        )

        # 标记这些客户在该日期的所有交易为欺诈
        mask = (df['tx_datetime'].dt.date == date.date()) & (df['customer_id'].isin(attacked_customers))
        df.loc[mask, 'isFraud'] = 1

    print(f"[INFO] Burst attack fraud injected", flush=True)

    return df


def add_card_id_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """将 customer_id 映射为 card_id（脱敏银行卡号）"""
    # card_id 就是 customer_id（脱敏后）
    df['card_id'] = df['customer_id'] + 100000  # 加上偏移量模拟真实卡号
    return df


def finalize_dataset(df: pd.DataFrame, config: FraudDatasetConfig) -> pd.DataFrame:
    """最终处理数据集"""
    print("[INFO] Finalizing dataset...", flush=True)

    # 添加 merchant_type（从 terminal 继承）
    # merchant_type 已经在 terminal 里了，rename terminal_id -> merchant_id
    df['merchant_id'] = df['terminal_id']

    # 时间戳列
    df['timestamp'] = df['tx_datetime']

    # 计算最终欺诈率
    fraud_count = df['isFraud'].sum()
    total_count = len(df)
    actual_fraud_rate = fraud_count / total_count

    print(f"[INFO] Dataset finalized:", flush=True)
    print(f"  Total transactions: {total_count:,}", flush=True)
    print(f"  Fraud count: {fraud_count:,}", flush=True)
    print(f"  Actual fraud rate: {actual_fraud_rate:.4f} ({actual_fraud_rate*100:.2f}%)", flush=True)

    # 按时间排序
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def generate_fraud_dataset(config: FraudDatasetConfig) -> pd.DataFrame:
    """
    主入口：生成完整的欺诈检测数据集

    Args:
        config: FraudDatasetConfig 配置对象

    Returns:
        pd.DataFrame: 包含所有交易记录和欺诈标签的 DataFrame
    """
    set_seed(config.seed)

    print("="*60, flush=True)
    print("Fraud Detection Dataset Generator", flush=True)
    print("="*60, flush=True)
    print(f"Configuration:", flush=True)
    print(f"  Customers: {config.n_customers:,}", flush=True)
    print(f"  Terminals: {config.n_terminals:,}", flush=True)
    print(f"  Days: {config.n_days}", flush=True)
    print(f"  Target transactions: {config.n_transactions:,}", flush=True)
    print(f"  Target fraud rate: {config.fraud_rate:.4f}", flush=True)
    print("="*60, flush=True)

    start_time = datetime.now()

    # Step 1: 模拟客户和终端档案
    customers = simulate_customer_profiles(config)
    terminals = simulate_terminal_profiles(config)

    # Step 2: 生成基础交易
    df = generate_base_transactions(config, customers, terminals)

    # Step 3: 注入欺诈场景
    df = inject_amount_fraud(df, config)
    df = inject_terminal_fraud(df, config)
    df = inject_burst_fraud(df, config)

    # Step 4: 添加 card_id 映射
    df = add_card_id_mapping(df)

    # Step 5: 最终处理
    df = finalize_dataset(df, config)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[INFO] Dataset generation completed in {elapsed:.1f} seconds", flush=True)

    return df


def save_dataset(df: pd.DataFrame, output_path: str, config: FraudDatasetConfig):
    """保存数据集到 CSV"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # 选择输出列
    output_cols = [
        'card_id', 'timestamp', 'amount', 'merchant_id', 'merchant_type',
        'balance', 'card_level', 'card_location', 'card_type',
        'device', 'is_night', 'is_cross_border', 'isFraud'
    ]

    # 检查哪些列存在
    available_cols = [c for c in output_cols if c in df.columns]

    df[available_cols].to_csv(output_path, index=False)
    print(f"[INFO] Dataset saved to {output_path}", flush=True)
    print(f"[INFO] Shape: {df[available_cols].shape}", flush=True)

    # 保存配置信息
    config_path = output_path.replace('.csv', '_config.json')
    config_dict = {
        'n_customers': config.n_customers,
        'n_terminals': config.n_terminals,
        'n_days': config.n_days,
        'n_transactions': len(df),
        'fraud_rate': float(df['isFraud'].sum() / len(df)),
        'seed': config.seed,
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"[INFO] Config saved to {config_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Simulated Fraud Detection Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 默认配置
  python -m src.data.fraud_dataset_generator --output ./data/fraud_dataset.csv

  # 自定义参数
  python -m src.data.fraud_dataset_generator \\
      --n-customers 5000 \\
      --n-terminals 10000 \\
      --n-days 183 \\
      --output ./data/my_fraud_dataset.csv
        """
    )

    parser.add_argument('--n-customers', type=int, default=5000,
                        help='客户数量（默认: 5000）')
    parser.add_argument('--n-terminals', type=int, default=10000,
                        help='终端数量（默认: 10000）')
    parser.add_argument('--n-days', type=int, default=183,
                        help='天数（默认: 183）')
    parser.add_argument('--n-transactions', type=int, default=1750000,
                        help='目标交易数量（默认: 1750000）')
    parser.add_argument('--fraud-rate', type=float, default=0.008,
                        help='目标欺诈率（默认: 0.008）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--output', type=str, default='./data/simulated_fraud_dataset.csv',
                        help='输出CSV路径（默认: ./data/simulated_fraud_dataset.csv）')

    args = parser.parse_args()

    # 创建配置
    config = FraudDatasetConfig(
        n_customers=args.n_customers,
        n_terminals=args.n_terminals,
        n_days=args.n_days,
        n_transactions=args.n_transactions,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )

    # 生成数据集
    df = generate_fraud_dataset(config)

    # 保存
    save_dataset(df, args.output, config)

    print("\n" + "="*60, flush=True)
    print("Dataset generation complete!", flush=True)
    print("="*60, flush=True)


if __name__ == '__main__':
    main()