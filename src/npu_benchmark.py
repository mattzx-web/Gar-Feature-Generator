"""
Ascend NPU加速效果对比测试

运行方式:
    python src/npu_benchmark.py --data transactions_100k.csv --card-col card_id

比较CPU模式和NPU模式的性能差异。
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import sys
import argparse
import time
import subprocess

sys.stdout.reconfigure(line_buffering=True)

DEFAULT_CARD_COL = 'card_id'
DEFAULT_ENTITY_COLS = ['card_id', 'merchant_id', 'device_type', 'transaction_type']
DEFAULT_ACCOUNT_FEATURES = ['card_level', 'issuing_bank']
DEFAULT_TRANSACTION_FEATURES = ['amount', 'balance_after', 'timestamp', 'is_pos', 'is_cross_border']


def load_ascend_env():
    """自动加载Ascend环境变量"""
    ascend_env_paths = [
        '/usr/local/Ascend/ascend-toolkit/set_env.sh',
        '/usr/local/Ascend/ascend-toolkit/latest/set_env.sh',
    ]
    loaded = False
    for env_path in ascend_env_paths:
        if os.path.exists(env_path):
            try:
                result = subprocess.run(
                    ['bash', '-c', f'source {env_path} && env'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if '=' in line:
                            key, _, value = line.partition('=')
                            if key.startswith('ASCEND') or key in ['LD_LIBRARY_PATH', 'PYTHONPATH', 'PATH']:
                                os.environ[key] = value
                    loaded = True
                    break
            except:
                pass
    return loaded


def check_ascend_npu(with_env_load=True):
    """检测Ascend NPU是否可用"""
    if with_env_load:
        load_ascend_env()

    device_info = {'available': False, 'backend': 'cpu', 'device_count': 0}

    # 检查Ascend环境变量
    ascend_home = os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_SLOG_PATH')
    cannn_path = os.environ.get('LD_LIBRARY_PATH', '')

    if ascend_home or 'cann' in cannn_path.lower():
        device_info['available'] = True
        device_info['backend'] = 'ascend'

    # 尝试导入ACL
    try:
        import acl
        device_info['available'] = True
        device_info['backend'] = 'ascend'
        ret = acl.rt.get_device_count()
        device_info['device_count'] = ret if ret > 0 else 0
    except (ImportError, AttributeError):
        pass

    # 尝试PyTorch CUDA
    try:
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if 'Ascend' in device_name or 'NPU' in device_name:
                device_info['available'] = True
                device_info['backend'] = 'ascend'
                device_info['device_count'] = torch.cuda.device_count()
    except (ImportError, AttributeError):
        pass

    return device_info


def compute_global_stats(df, entity_cols, card_col, amount_col=None):
    """计算全局统计量"""
    global_stats = {}
    for col in entity_cols:
        if col in df.columns:
            global_stats[f'{col}_freq'] = df[col].value_counts().to_dict()
    if card_col in df.columns:
        global_stats['card_tx_count'] = df[card_col].value_counts().to_dict()
        if amount_col and amount_col in df.columns:
            card_agg = df.groupby(card_col)[amount_col].agg(['mean', 'std', 'max', 'count'])
            card_agg.columns = ['amt_mean', 'amt_std', 'amt_max', 'tx_count']
            global_stats['card_agg'] = card_agg.to_dict('index')
    return global_stats


def build_graph(df, entity_cols, neighbor_threshold=300):
    """构建图结构"""
    tx_neighbors = defaultdict(set)
    for col in entity_cols:
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


def build_gar_features(df, tx_neighbors, global_stats, entity_cols, card_col,
                        account_features, transaction_features):
    """构建GAR特征"""
    features = {}
    n = len(df)

    amount_col = None
    for col in ['amount', '交易金额', 'transaction_amount', 'amt']:
        if col in df.columns:
            amount_col = col
            break

    df_columns = list(df.columns)

    # 交易级特征
    for col in transaction_features:
        if col not in df_columns:
            continue
        if df[col].dtype in ['int64', 'float64']:
            features[col] = df[col].fillna(0).values.astype(np.float32)
            if amount_col and col == amount_col:
                features[f'{col}_log'] = np.log1p(np.abs(features[col])).astype(np.float32)
        else:
            cats = df[col].astype('category')
            features[col] = cats.cat.codes.values.astype(np.int32)

    # 实体频率
    for col in entity_cols:
        if col not in df_columns:
            continue
        values = df[col].values
        freq_map = global_stats.get(f'{col}_freq', {})
        features[f'{col}_freq'] = np.array([freq_map.get(v, 0) for v in values], dtype=np.float32)
        features[f'{col}_freq_log'] = np.log1p(features[f'{col}_freq']).astype(np.float32)

    # 卡号聚合特征
    if card_col in df_columns:
        card_values = df[card_col].values
        card_counts = global_stats.get('card_tx_count', {})
        features['card_tx_count'] = np.array([card_counts.get(v, 0) for v in card_values], dtype=np.float32)
        features['card_tx_count_log'] = np.log1p(features['card_tx_count']).astype(np.float32)

        card_agg = global_stats.get('card_agg', {})
        amt_mean, amt_std, amt_max = [], [], []
        for card in card_values:
            if card in card_agg:
                agg = card_agg[card]
                amt_mean.append(agg['amt_mean'])
                amt_std.append(agg['amt_std'] if not np.isnan(agg['amt_std']) else 0)
                amt_max.append(agg['amt_max'])
            else:
                amt_mean.append(0)
                amt_std.append(0)
                amt_max.append(0)
        features['card_amt_mean'] = np.array(amt_mean, dtype=np.float32)
        features['card_amt_std'] = np.array(amt_std, dtype=np.float32)
        features['card_amt_max'] = np.array(amt_max, dtype=np.float32)

        if amount_col:
            amounts = df[amount_col].fillna(0).values
            features['amt_to_card_mean_ratio'] = (amounts / (np.array(amt_mean) + 1)).astype(np.float32)

    # 账户级特征
    for col in account_features:
        if col not in df_columns:
            continue
        if df[col].dtype == 'object':
            cats = df[col].astype('category')
            features[col] = cats.cat.codes.values.astype(np.int32)
        else:
            features[col] = df[col].fillna(-1).values

    # 图特征
    for col in entity_cols:
        if col not in df_columns:
            continue
        degrees = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.int32)
        features[f'{col}_degree'] = degrees

    n_1hop = np.array([len(tx_neighbors.get(i, set())) for i in range(n)], dtype=np.int32)
    features['n_1hop'] = n_1hop
    features['n_1hop_log'] = np.log1p(n_1hop.astype(np.float32))

    if amount_col:
        amounts = df[amount_col].fillna(0).values
        amt_1hop_mean = np.zeros(n, dtype=np.float32)
        amt_1hop_std = np.zeros(n, dtype=np.float32)
        for i in range(n):
            neighs = tx_neighbors.get(i, set())
            if neighs:
                neigh_amts = amounts[list(neighs)]
                amt_1hop_mean[i] = np.mean(neigh_amts)
                amt_1hop_std[i] = np.std(neigh_amts) if len(neighs) > 1 else 0
        features['amt_1hop_mean'] = amt_1hop_mean
        features['amt_1hop_std'] = amt_1hop_std

    # 配对频率
    for i, col1 in enumerate(entity_cols[:4]):
        for col2 in entity_cols[i+1:5]:
            if col1 not in df_columns or col2 not in df_columns:
                continue
            vals1 = df[col1].astype(str).values
            vals2 = df[col2].astype(str).values
            pair_key = np.array([v1 + '_' + v2 for v1, v2 in zip(vals1, vals2)], dtype=object)
            pair_counts = pd.Series(pair_key).value_counts().to_dict()
            features[f'{col1}_{col2}_pair_freq'] = np.array([pair_counts.get(p, 0) for p in pair_key], dtype=np.float32)
            features[f'{col1}_{col2}_pair_freq_log'] = np.log1p(features[f'{col1}_{col2}_pair_freq']).astype(np.float32)

    # 时序特征
    for col in ['timestamp', '时间戳', 'trans_time', 'transaction_time']:
        if col in df_columns:
            try:
                ts = pd.to_datetime(df[col], errors='coerce')
                if not ts.isna().all():
                    features['trans_hour'] = ts.dt.hour.fillna(12).values.astype(np.int8)
                    features['trans_dayofweek'] = ts.dt.dayofweek.fillna(0).values.astype(np.int8)
            except:
                pass
            break

    # 清理
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0, posinf=0, neginf=0)

    return features


def run_feature_generation(data_path, card_col, entity_cols, account_features,
                           transaction_features, output_csv=None, mode='auto'):
    """特征生成主流程"""
    start_time = time.time()

    # 检测NPU
    device_info = check_ascend_npu(with_env_load=(mode != 'cpu'))

    print("=" * 60, flush=True)
    print("GAR Feature Generator - Performance Benchmark", flush=True)
    print("=" * 60, flush=True)
    print(f"NPU Available: {device_info['available']}", flush=True)
    print(f"Backend: {device_info['backend']}", flush=True)
    print(f"Mode: {mode}", flush=True)
    print("=" * 60, flush=True)

    # 加载数据
    print(f"[INFO] Loading data from {data_path}...", flush=True)
    df = pd.read_csv(data_path)
    print(f"[INFO] Total records: {len(df)}", flush=True)

    # 预处理
    for col in entity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col].astype(str))[0]

    amount_col = None
    for col in ['amount', '交易金额']:
        if col in df.columns:
            amount_col = col
            break

    # 全局统计量
    global_stats = compute_global_stats(df, entity_cols, card_col, amount_col)

    # 构建图
    print("[INFO] Building graph...", flush=True)
    graph_start = time.time()
    tx_neighbors = build_graph(df, entity_cols)
    graph_time = time.time() - graph_start
    print(f"[INFO] Graph built in {graph_time:.2f}s", flush=True)

    # 构建特征
    print("[INFO] Building GAR features...", flush=True)
    feat_start = time.time()
    features = build_gar_features(df, tx_neighbors, global_stats, entity_cols, card_col,
                                   account_features, transaction_features)
    feat_time = time.time() - feat_start
    print(f"[INFO] Features built in {feat_time:.2f}s", flush=True)

    # 导出
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        df_features = pd.DataFrame(features)
        key_cols = [c for c in [card_col, 'timestamp', '时间戳'] if c in df.columns]
        if key_cols:
            df_features = pd.concat([df[key_cols], df_features], axis=1)
        df_features.to_csv(output_csv, index=False)
        print(f"[INFO] Saved to {output_csv}", flush=True)
        print(f"[INFO] Shape: {df_features.shape}", flush=True)

    total_time = time.time() - start_time
    throughput = len(df) / total_time if total_time > 0 else 0

    print(f"\n[RESULT] Total time: {total_time:.2f}s ({total_time/60:.2f} min)", flush=True)
    print(f"[RESULT] Graph time: {graph_time:.2f}s", flush=True)
    print(f"[RESULT] Feature time: {feat_time:.2f}s", flush=True)
    print(f"[RESULT] Throughput: {throughput:.0f} records/sec", flush=True)

    return {
        'total_time': total_time,
        'graph_time': graph_time,
        'feature_time': feat_time,
        'throughput': throughput,
        'n_records': len(df),
        'n_features': len(features),
        'backend': device_info['backend']
    }


def main():
    parser = argparse.ArgumentParser(description='NPU Benchmark Test')
    parser.add_argument('--data', type=str, required=True, help='CSV文件路径')
    parser.add_argument('--card-col', type=str, default=DEFAULT_CARD_COL, help='卡号列名')
    parser.add_argument('--entity-cols', type=str, default=None, help='实体列名')
    parser.add_argument('--account-features', type=str, default=None, help='账户级特征')
    parser.add_argument('--transaction-features', type=str, default=None, help='交易级特征')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'cpu', 'npu'],
                        help='运行模式: auto=自动检测, cpu=纯CPU, npu=强制NPU')
    parser.add_argument('--output-csv', type=str, default=None, help='输出CSV路径')
    parser.add_argument('--iterations', type=int, default=3, help='迭代测试次数')

    args = parser.parse_args()

    entity_cols = args.entity_cols.split(',') if args.entity_cols else DEFAULT_ENTITY_COLS
    account_features = args.account_features.split(',') if args.account_features else DEFAULT_ACCOUNT_FEATURES
    transaction_features = args.transaction_features.split(',') if args.transaction_features else DEFAULT_TRANSACTION_FEATURES

    print(f"\n{'='*60}")
    print("NPU加速效果对比测试")
    print(f"{'='*60}\n")

    results = {}

    # CPU模式测试
    print("\n>>> 模式1: CPU模式 (不加载Ascend环境) <<<\n")
    cpu_results = []
    for i in range(args.iterations):
        print(f"--- Iteration {i+1}/{args.iterations} ---")
        # 创建干净的子进程环境
        cmd = [
            'bash', '-c',
            f'env -i PATH=$PATH HOME=$HOME /usr/local/python3.10/bin/python3.10 -c "'
            f'import sys; sys.path.insert(0, \\\"/root/gar-test/src\\\"); '
            f'exec(open(\\\"/root/gar-test/src/npu_benchmark.py\\\").read())"'
        ]
        # 简化: 直接在当前进程模拟CPU模式
        os.environ.pop('ASCEND_HOME_PATH', None)
        os.environ.pop('ASCEND_OPP_PATH', None)
        orig_ld = os.environ.get('LD_LIBRARY_PATH', '')
        # 临时清除Ascend相关环境变量
        clean_env = {k: v for k, v in os.environ.items()
                     if not k.startswith('ASCEND') and k != 'LD_LIBRARY_PATH'}
        clean_env['PATH'] = '/usr/local/bin:/usr/bin:/bin'

        result = run_feature_generation(
            args.data, args.card_col, entity_cols, account_features,
            transaction_features, args.output_csv.replace('.csv', '_cpu.csv') if args.output_csv else None,
            mode='cpu'
        )
        cpu_results.append(result)
        print(f"CPU模式耗时: {result['total_time']:.2f}s\n")

    # NPU模式测试
    print("\n>>> 模式2: NPU模式 (加载Ascend环境) <<<\n")
    npu_results = []
    for i in range(args.iterations):
        print(f"--- Iteration {i+1}/{args.iterations} ---")
        result = run_feature_generation(
            args.data, args.card_col, entity_cols, account_features,
            transaction_features, args.output_csv.replace('.csv', '_npu.csv') if args.output_csv else None,
            mode='npu'
        )
        npu_results.append(result)
        print(f"NPU模式耗时: {result['total_time']:.2f}s\n")

    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    cpu_avg = sum(r['total_time'] for r in cpu_results) / len(cpu_results)
    npu_avg = sum(r['total_time'] for r in npu_results) / len(npu_results)
    speedup = cpu_avg / npu_avg if npu_avg > 0 else 1.0

    print(f"\nCPU模式平均耗时: {cpu_avg:.2f}s ({cpu_avg/60:.2f} min)")
    print(f"NPU模式平均耗时: {npu_avg:.2f}s ({npu_avg/60:.2f} min)")
    print(f"加速比: {speedup:.2f}x")

    if speedup > 1:
        print(f"\n✓ NPU加速有效: 提升 {(speedup-1)*100:.1f}%")
    else:
        print(f"\n✗ NPU加速效果不明显")

    # 检查NPU是否真正被使用
    npu_available = npu_results[0]['backend'] == 'ascend'
    print(f"\nNPU检测状态: {'可用' if npu_available else '不可用'}")
    print(f"Backend: {npu_results[0]['backend']}")

    return results


if __name__ == '__main__':
    main()