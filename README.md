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
python src/gar_feature_generator.py \
    --data data/transactions.csv \
    --card-col card_id \
    --output-csv ./features.csv
```

---

## 使用方法

### 白样本模式（无标签）

仅生成图结构特征（度、频率、邻居统计）：

```bash
python src/gar_feature_generator_ascend.py \
    --data data/transactions.csv \
    --card-col card_id \
    --output-csv ./features.csv
```

### 有标签模式

生成完整GAR特征（含欺诈率）：

```bash
python src/gar_feature_generator.py \
    --data-dir /path/to/ieee-fraud-detection \
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

### IEEE-CIS

```
data_dir/
├── train_transaction.csv    # 交易记录（含isFraud标签）
└── train_identity.csv       # 身份信息
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
│   ├── gar_feature_generator.py          # 基础GAR实现
│   ├── gar_feature_generator_ascend.py  # 优化版本（含NPU支持）
│   ├── gar_feature_generator_dist.py    # 分布式版本
│   ├── kg_feature_generator.py          # KG基线
│   └── train_classifier.py              # 模型训练
└── outputs/
```

---

## 加速选项

对于大规模数据，可使用NPU或分布式加速：

```bash
# Ascend NPU
python src/gar_feature_generator_ascend.py \
    --data large_data.csv \
    --card-col card_id \
    --mode npu

# 分布式
python src/gar_feature_generator_dist.py \
    --data large_data.csv \
    --card-col card_id \
    --workers 16
```

---

## 数据泄漏防护

1. 训练/测试集先划分，再计算特征
2. 欺诈率仅从训练集统计
3. 图结构仅从训练集构建

---

## License

MIT License