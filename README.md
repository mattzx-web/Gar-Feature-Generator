# GAR Feature Generator

基于图关联规则(Graph Association Rules)的金融反欺诈特征工程工具包。

## 项目简介

本项目实现GAR算法，将图结构中的关联规则与欺诈率结合，生成高质量欺诈检测特征。

### 核心方法：GAR-Inspired

| 特征类型 | 维度 | Test AUC | 说明 |
|----------|------|----------|------|
| **Entity Fraud Rate** | 单实体欺诈率 | - | card=X的欺诈概率 |
| **Pair Fraud Rate** | 实体对欺诈率 | - | card=X 且 addr=Y的欺诈概率 |
| **Neighbor Fraud Rate** | 邻居欺诈率 | - | 1-hop邻居的平均欺诈率 |

**GAR-Inspired Full (18维)**: Test AUC = **0.8725±0.0014**

---

## 算法原理

### GAR形式化定义

```
GAR φ = Q[x̄](X → p0)

其中:
- Q[x̄]: graph pattern（图模式）
- X: precondition（前置条件，多个predicates）
- p0: consequence predicate（结果谓词）
```

### 本项目实现

将欺诈率视为图关联规则的输出：

```python
# 单实体欺诈率
fraud_rate(entity) = fraud_count(entity) / total_count(entity)

# 实体对欺诈率
pair_fraud_rate(e1, e2) = fraud_count(e1 ∧ e2) / total_count(e1 ∧ e2)

# 邻居欺诈率
neighbor_fraud_rate(node) = mean(fraud_rate(neighbor) for neighbor in neighbors)
```

---

## 快速开始

```bash
git clone https://github.com/mattzx-web/Gar-Feature-Generator.git
cd Gar-Feature-Generator

pip install pandas numpy scikit-learn

# 生成GAR特征
python src/gar/gar_cpu.py \
    --data data/transactions.csv \
    --card-col card_id \
    --output-csv ./features.csv
```

---

## 使用方法

### 白样本模式（无标签）

仅生成图结构特征（度、频率、邻居统计）：

```bash
python src/gar/gar_ascend.py \
    --data data/transactions.csv \
    --card-col card_id \
    --output-csv ./features.csv
```

### 有标签模式

生成完整GAR特征（含欺诈率）：

```bash
python src/gar/gar_cpu.py \
    --data /path/to/transactions.csv \
    --card-col card_id
```

### 输出特征

| 特征类型 | 示例 | 说明 |
|----------|------|------|
| 实体度 | `card_id_degree` | 图中邻居数量 |
| 实体频率 | `card_id_freq` | 出现次数 |
| 卡号聚合 | `card_amt_mean` | 按卡号统计 |
| 配对频率 | `card_merchant_pair_freq` | 实体对共现次数 |
| 邻居统计 | `amt_1hop_mean` | 邻居金额均值 |
| **欺诈率** | `card_id_fraud_rate` | 单实体欺诈率 |
| **配对欺诈率** | `card_merchant_fraud_rate` | 实体对欺诈率 |

---

## 数据格式

### 标准CSV

```csv
card_id,merchant_id,device_type,transaction_type,amount,timestamp,is_pos
123456,SHOP001,MOB010,POS,1500.00,2026-05-20,1
123456,SHOP002,WEB001,ONLINE,200.00,2026-05-20,0
```

### IEEE-CIS Data

下载链接: https://www.kaggle.com/competitions/ieee-fraud-detection/data

```
data_dir/
├── train_transaction.csv    # 交易记录（含isFraud标签）
└── train_identity.csv       # 身份信息
```

---

## 完整工作流

```bash
# 1. 生成GAR特征
python src/gar/gar_cpu.py \
    --data data/transactions.csv \
    --card-col card_id \
    --export-features-only \
    --output-csv ./features.csv

# 2. 训练分类器
python src/train_classifier.py \
    --features ./features.csv \
    --model gar
```

---

## 实验结果

### IEEE-CIS 590K数据集

| 方法 | 特征维度 | Test AUC | 提升 |
|------|---------|----------|------|
| Baseline | 1 | 0.7075 | - |
| KG Brute Force | 53 | 0.8421 | +13.46% |
| **GAR-Inspired** | 18 | **0.8725** | **+16.50%** |

### 消融实验

| 模型 | Test AUC | 贡献 |
|------|----------|------|
| Baseline | 0.7075 | - |
| + Entity Fraud Rates | 0.8125 | +10.50% |
| + Pair Fraud Rates | 0.8641 | +5.16% |
| + Neighbor Fraud Rate | 0.8725 | +0.84% |

**Pair Fraud Rate是最强信号**，贡献了主要的性能提升。

---

## 算法实现

### 图构建

```python
# 稀疏图结构：dict存储邻居列表
tx_neighbors = defaultdict(list)
for col in entity_cols:
    groups = df.groupby(col).indices
    for val, idx_list in groups.items():
        if 1 < len(idx_list) < threshold:
            for idx in idx_list:
                tx_neighbors[idx].extend(idx_list)
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
│   │   ├── gar_cpu.py          # 基础GAR实现（CPU模式）
│   │   ├── gar_ascend.py       # Ascend NPU加速版本
│   │   └── gar_dist.py         # 分布式多进程版本
│   ├── kg/
│   │   ├── kg_cpu.py          # 基础KG实现
│   │   ├── kg_ascend.py        # Ascend NPU加速版本
│   │   ├── kg_dist.py          # 分布式版本
│   │   ├── kg_gpu.py           # CUDA GPU加速版本
│   │   └── kg_brute_force.py   # KG暴力枚举基线
│   ├── utils/
│   │   └── feature_utils.py    # 公共工具函数
│   ├── bench/
│   │   └── npu_benchmark.py    # NPU性能基准测试
│   └── train_classifier.py     # 模型训练脚本
└── outputs/
```

---

## 加速选项

对于大规模数据，可使用NPU或分布式加速。

### 性能测试（Ascend 910B NPU服务器，100K记录）

| 实现 | 100K耗时 | 吞吐量 | 适用场景 |
|------|----------|--------|----------|
| **Sparse Dict** | 35秒 | 3724 rec/s | 生产环境（推荐） |
| CSR向量化 | 139秒 | 721 rec/s | 理论研究 |
| torch_npu | 556秒 | 185 rec/s | 超大规模(>1000万) |
| 分布式 | - | - | 亿级数据 |

**结论**：稀疏dict实现对于1000万以下数据最优。torch_npu版本因设备管理开销大，不适合中规模数据。

### 使用方式

```bash
# CPU模式
python src/gar/gar_cpu.py --data data.csv --card-col card_id --output-csv ./features.csv

# NPU加速版本（自动检测Ascend环境）
python src/gar/gar_ascend.py --data data.csv --card-col card_id --mode npu --output-csv ./features.csv

# 分布式（多进程）
python src/gar/gar_dist.py --data data.csv --card-col card_id --workers 16 --output-csv ./features.csv
```

---

## 数据泄漏防护

1. 训练/测试集先划分，再计算特征
2. 欺诈率仅从训练集统计
3. 图结构仅从训练集构建

---

## License

MIT License