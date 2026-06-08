# GAR Feature Generator

基于图关联规则(Graph Association Rules)的金融反欺诈特征工程工具包。

## 项目简介

本项目实现GAR算法，将图结构中的关联规则与欺诈率结合，生成高质量欺诈检测特征。支持自定义数据集自动检测、扩展特征工程、无数据泄漏模式。

### 核心方法：GAR-Inspired

| 特征类型 | 维度 | Test AUC | 说明 |
|----------|------|----------|------|
| **Entity Fraud Rate** | 单实体欺诈率 | - | card=X的欺诈概率 |
| **Pair Fraud Rate** | 实体对欺诈率 | - | card=X 且 addr=Y的欺诈概率 |
| **Neighbor Fraud Rate** | 邻居欺诈率 | - | 1-hop邻居的平均欺诈率 |

**GAR-Inspired Full (22维)**: Test AUC = **0.8678**

**GAR-Inspired Expanded (59维)**: Test AUC = **0.8538** (+0.35% vs baseline)

---

## 快速开始

```bash
git clone https://github.com/mattzx-web/Gar-Feature-Generator.git
cd Gar-Feature-Generator

pip install pandas numpy scikit-learn scipy tqdm

# 生成GAR特征（自动检测列名 + 无泄漏模式）
python src/gar/gar_cpu.py \
    --data data/transactions.csv \
    --output-csv ./features.csv

# 指定列名
python src/gar/gar_cpu.py \
    --data data/transactions.csv \
    --card-col card_id \
    --entity-cols card_id,merchant_id,device \
    --account-features card_level,card_location,card_type \
    --transaction-features amount,balance,is_cross_border

# 分布式模式（多进程加速）
python src/gar/gar_dist.py \
    --data data/transactions.csv \
    --card-col card_id \
    --workers 4 \
    --output-csv ./features.csv
```

---

## 核心特性

### 1. 自动列名检测

支持自定义数据集，自动检测列名并映射到标准列名：

| 标准列名 | 支持的别名 |
|----------|------------|
| `card_id` | card_id, card, 卡号, 银行卡号, customer_id |
| `timestamp` | timestamp, datetime, 时间戳, 交易时间 |
| `amount` | amount, amt, 交易金额, tx_amount |
| `balance` | balance, balance_after, 账户余额 |
| `merchant_id` | merchant_id, merchant, 商户号 |
| `device` | device, device_type, 设备 |
| `is_fraud` | isFraud, fraud, 欺诈 |
| `card_level` | card_level, 卡等级 |
| `card_location` | card_location, 卡注册地 |
| `card_type` | card_type, 卡类型 |

### 2. 扩展特征工程（59维）

| 特征类别 | 特征名称 | 说明 |
|----------|----------|------|
| **基础特征** | amount, amount_log, balance | 交易级特征 |
| **实体频率** | card_id_freq, merchant_id_freq | 各实体出现频率 |
| **卡号聚合** | card_amt_mean, card_amt_std | 按卡号统计金额 |
| **配对频率** | card_merchant_pair_freq | 实体对共现次数 |
| **邻居统计** | amt_1hop_mean, n_1hop | 邻居金额/度统计 |
| **欺诈率** | card_id_fraud_rate | 单实体欺诈率 |
| **配对欺诈率** | card_merchant_fraud_rate | 实体对欺诈率 |
| **邻居欺诈率** | neigh_fraud_rate | 1-hop邻居平均欺诈率 |
| **时序特征** | trans_hour, trans_dayofweek | 时间维度 |
| **时序熵** | hour_entropy, day_entropy | 交易时间分布熵 |
| **金额统计** | amount_zscore, amount_percentile | 金额Z分数/百分位 |
| **交易速度** | tx_velocity_1h, tx_velocity_24h | 交易频率 |
| **风险评分** | terminal_risk_score, device_risk_score | 终端/设备风险 |
| **图指标** | degree_centrality, clustering_coeff | 图结构指标 |

### 3. 无数据泄漏模式

- 训练/测试集先划分，再计算特征
- 欺诈率仅从训练集统计
- 图结构从完整数据构建（获取所有邻居关系）
- 导出的CSV包含`split`列，标记每条记录属于train还是test

### 4. 进度条显示

- 所有版本支持`tqdm`进度条
- 未安装时自动跳过，不影响功能

---

## 使用方法

### 自动检测模式

```bash
# 自动检测列名（默认开启）
python src/gar/gar_cpu.py \
    --data ./data/my_custom_data.csv \
    --output-csv ./features.csv

# 关闭自动检测，手动指定列名
python src/gar/gar_cpu.py \
    --data ./data/my_custom_data.csv \
    --card-col 卡号 \
    --entity-cols 卡号,商户号,设备 \
    --account-features 卡等级,卡地区,卡类型 \
    --transaction-features 交易金额,余额,是否跨境
```

### 多版本选择

| 版本 | 命令 | 适用场景 |
|------|------|----------|
| **CPU** | `src/gar/gar_cpu.py` | 小规模数据（<10万），功能完整 |
| **Ascend NPU** | `src/gar/gar_ascend.py` | 中大规模（需要NPU），无泄漏模式 |
| **分布式** | `src/gar/gar_dist.py` | 大规模数据（多进程加速），无泄漏模式 |

```bash
# CPU模式（无泄漏，默认）
python src/gar/gar_cpu.py --data data.csv --card-col card_id --output-csv ./features.csv

# NPU加速（无泄漏模式，自动检测列名）
python src/gar/gar_ascend.py --data data.csv --card-col card_id --output-csv ./features.csv

# 分布式（4进程，无泄漏）
python src/gar/gar_dist.py --data data.csv --card-col card_id --workers 4 --output-csv ./features.csv

# 关闭无泄漏模式（不推荐）
python src/gar/gar_cpu.py --data data.csv --leakage --output-csv ./features.csv
```

**进度条支持**：所有版本支持`tqdm`进度条（自动检测，未安装时正常执行）

### 生成数据集

```bash
# 生成模拟欺诈数据集
python -m src.data.fraud_dataset_generator \
    --n-customers 5000 \
    --n-terminals 10000 \
    --n-days 183 \
    --n-transactions 100000 \
    --output ./data/fraud_dataset.csv
```

---

## 数据格式

### 标准CSV

```csv
card_id,timestamp,amount,merchant_id,balance,card_level,card_location,card_type,device,is_night,is_cross_border,isFraud
100206,2018-04-01 00:01:12,17.12,100042,8542.50,2,北京,credit,POS,1,0,0
100108,2018-04-01 00:02:46,10.40,100891,12893.20,3,上海,debit,APP,1,0,0
```

### 支持中文列名

```csv
卡号,时间戳,交易金额,商户号,余额,卡等级,卡地区,卡类型,设备,是否夜间,是否跨境,是否欺诈
100206,2018-04-01,17.12,100042,8542.50,2,北京,credit,POS,1,0,0
```

---

## 完整工作流

### 1. 生成GAR特征（自动导出split列）

```bash
# 生成扩展特征（59维），自动分割训练/测试集并导出
python src/gar/gar_cpu.py \
    --data ./data/fraud_dataset.csv \
    --output-csv ./features/gar_expanded_features.csv \
    --export-features-only

# 输出文件包含 split 列（train/test）和 isFraud 标签列
# 可直接用于后续模型训练
```

**输出文件格式：**
```csv
card_id,amount,amount_log,...,isFraud,split
100206,17.12,2.84,...,0,train
100108,10.40,2.44,...,1,test
```

### 2. 特征筛选（可选）

```bash
# 基于欺诈率相关性筛选top-20特征
python -m src.gar.gar_feature_selector \
    --features ./features/gar_expanded_features.csv \
    --top-k 20 \
    --threshold 0.03 \
    --output ./features/selected_features.csv
```

### 3. 模型训练（使用导出的split列）

生成的特征文件已包含 `split` 列，直接用于训练：

```bash
# 直接使用gar_cpu.py内置分类器（自动使用split列）
python src/gar/gar_cpu.py \
    --data ./data/fraud_dataset.csv

# 或使用独立训练脚本
python src/train_classifier.py \
    --features ./features/gar_expanded_features.csv \
    --split-col split
```

### 4. 运行对比实验

```bash
# GAR扩充前后对比
python experiments/gar_comparison_experiment.py \
    --data ./data/fraud_dataset.csv \
    --output-dir ./outputs/gar_comparison
```

---

## 实验结果

### GAR特征扩充对比（120k交易, 0.92%欺诈率）

| 模型 | 特征维度 | Test AUC | Precision | Recall | F1 |
|------|----------|----------|-----------|--------|-----|
| Baseline | 2 | 0.8140 | 0.3742 | 0.3422 | 0.3575 |
| Basic GAR | 45 | 0.8067 | 0.3395 | 0.2153 | 0.2635 |
| **Expanded GAR** | **59** | **0.8169** | **0.4074** | 0.0973 | 0.1571 |

**分析**：
- Expanded GAR 在 AUC 上最优 (+0.35% vs baseline)
- Precision 提升显著 (+8.86%)
- Recall 有所下降（欺诈样本稀疏）

### Top 10 重要特征

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | amount_log | 0.2100 |
| 2 | neigh_fraud_rate | 0.1960 |
| 3 | merchant_id_fraud_rate | 0.1038 |
| 4 | card_id_fraud_rate | 0.0529 |
| 5 | amt_1hop_std | 0.0469 |

### 消融实验

| 模型 | Test AUC | 贡献 |
|------|----------|------|
| Baseline | 0.6834 | - |
| + Entity Fraud Rates | ~0.80 | +11.66% |
| + Pair Fraud Rates | ~0.86 | +5.16% |
| + Neighbor Fraud Rate | 0.8678 | +0.84% |

**Pair Fraud Rate是最强信号**，贡献了主要的性能提升。

---

## 算法实现

### 图构建

```python
# 稀疏图结构：dict存储邻居列表
tx_neighbors = defaultdict(set)
for col in entity_cols:
    groups = df.groupby(col).indices
    for val, idx_list in groups.items():
        if 1 < len(idx_list) < threshold:
            for idx in idx_list:
                tx_neighbors[idx].update(idx_list)
```

### 欺诈率计算

```python
# 单实体欺诈率
fraud_map = df.groupby(entity_col)[label_col].mean().to_dict()
features[f'{entity_col}_fraud_rate'] = [fraud_map.get(v, 0) for v in df[entity_col]]

# 实体对欺诈率
pair_df = df[[col1, col2, label_col]].copy()
pair_df['_pair'] = pair_df[col1].astype(str) + '_' + pair_df[col2].astype(str)
fraud_map = pair_df.groupby('_pair')[label_col].mean().to_dict()
features[f'{col1}_{col2}_pair_fraud_rate'] = [fraud_map.get(p, 0) for p in pair_df['_pair']]
```

---

## 项目结构

```
Gar-Feature-Generator/
├── README.md
├── src/
│   ├── gar/
│   │   ├── gar_cpu.py              # 基础GAR实现（CPU模式）
│   │   ├── gar_ascend.py           # Ascend NPU加速版本
│   │   ├── gar_dist.py             # 分布式多进程版本
│   │   └── gar_feature_selector.py # 高欺诈率特征筛选
│   ├── kg/
│   │   ├── kg_cpu.py               # 基础KG实现
│   │   ├── kg_ascend.py            # Ascend NPU加速版本
│   │   ├── kg_dist.py              # 分布式版本
│   │   ├── kg_gpu.py               # CUDA GPU加速版本
│   │   └── kg_brute_force.py       # KG暴力枚举基线
│   ├── data/
│   │   └── fraud_dataset_generator.py # 模拟数据集生成器
│   ├── utils/
│   │   ├── feature_utils.py         # 公共工具函数
│   │   └── schema_detector.py       # 列名自动检测工具
│   ├── bench/
│   │   └── npu_benchmark.py         # NPU性能基准测试
│   └── train_classifier.py          # 模型训练脚本
├── experiments/
│   ├── run_fraud_detection_experiment.py # 完整实验流水线
│   └── gar_comparison_experiment.py       # GAR扩充对比实验
└── outputs/
```

---

## 加速选项

### 性能对比（36k交易）

| 版本 | 时间 | 特征数 | 并行度 |
|------|------|--------|--------|
| CPU | ~40s | 63 | 1 |
| Dist (4 workers) | ~2s | 65 | 4 |

### 适用场景

| 数据规模 | 推荐版本 | 原因 |
|----------|----------|------|
| <10万 | CPU | 简单易用 |
| 10万-100万 | Dist (多进程) | 并行加速 |
| >100万 | Ascend NPU | NPU加速 |

---

## 工具脚本

### 列名检测

```bash
# 检测数据集的列名类型
python -m src.utils.schema_detector --data ./data/my_custom.csv

# 输出示例:
# card_id -> 卡号
# amount -> 交易金额
# timestamp -> 时间戳
# isFraud -> 是否欺诈
```

### 数据集生成

```bash
# 生成标准欺诈数据集
python -m src.data.fraud_dataset_generator \
    --n-customers 5000 \
    --n-terminals 10000 \
    --n-days 183 \
    --output ./data/fraud_dataset.csv

# 生成小规模测试数据
python -m src.data.fraud_dataset_generator \
    --n-customers 100 \
    --n-terminals 200 \
    --n-days 30 \
    --n-transactions 5000 \
    --output ./data/test_dataset.csv
```

---

## 数据泄漏防护

1. 训练/测试集先划分，再计算特征
2. 欺诈率仅从训练集统计
3. 图结构从完整数据构建（获取所有邻居关系）

---

## License

MIT License
