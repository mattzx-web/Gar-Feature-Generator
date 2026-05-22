# GAR Feature Generator

基于图关联规则(Graph Association Rules)的金融反欺诈特征工程工具包。

## 项目简介

本项目提供两种知识图谱特征工程方案：

| 方法 | 特征维度 | Test AUC | 说明 |
|------|---------|----------|------|
| **GAR-Inspired** | 18维+ | 0.8725±0.0014 | 基于欺诈率的规则特征，配对欺诈率为核心信号 |
| **KG Brute Force** | 53维+ | 0.8421±0.0063 | 枚举式图特征(度/计数/VCD统计) |

**支持白样本（无标签）数据特征扩充**

---

## 核心算法

### GAR (Graph Association Rules)

将欺诈率视为图关联规则的输出：

| 特征类型 | 说明 |
|----------|------|
| **Entity Fraud Rate** | 单实体欺诈率（如 card1=X 的欺诈率） |
| **Pair Fraud Rate** | 实体对欺诈率（如 card1=X 且 addr1=Y 的欺诈率） |
| **Neighbor Fraud Rate** | 1跳邻居欺诈率均值 |

---

## 快速开始

```bash
git clone https://github.com/mattzx-web/Gar-Feature-Generator.git
cd Gar-Feature-Generator

pip install pandas numpy scikit-learn

# 生成特征
python src/gar_feature_generator_ascend.py --data data/transactions.csv --card-col card_id --output-csv ./features.csv
```

---

## 版本切换指南

本项目提供多种算法实现，适用于不同规模和硬件环境：

### 算法版本对比

| 版本 | 脚本 | 100K耗时 | 吞吐量 | 推荐场景 |
|------|------|----------|--------|----------|
| **Sparse Dict** | `gar_feature_generator_ascend.py` | 35秒 | 3724 rec/s | **生产环境（推荐）** |
| **CSR向量化** | `gar_feature_generator_fast.py` | 139秒 | 721 rec/s | 理论研究 |
| **torch_npu** | `gar_feature_generator_npu.py` | 556秒 | 185 rec/s | 超大规模数据(>1000万) |
| **分布式** | `gar_feature_generator_dist.py` | - | - | 亿级数据 |

### Sparse Dict版本（推荐）

使用dict+list存储稀疏图结构，逐节点遍历+向量化特征计算。

```bash
# Ascend NPU模式（自动检测）
python src/gar_feature_generator_ascend.py --data data.csv --card-col card_id --mode npu

# CPU模式（不加载Ascend环境）
python src/gar_feature_generator_ascend.py --data data.csv --card-col card_id --mode cpu

# 自动模式
python src/gar_feature_generator_ascend.py --data data.csv --card-col card_id --mode auto
```

### CSR向量化版本

使用CSR格式稀疏矩阵，向量化特征计算（理论更优但构建开销大）。

```bash
python src/gar_feature_generator_fast.py --data data.csv --card-col card_id --output-csv ./features.csv
```

### torch_npu版本

使用PyTorch NPU进行GPU加速（适合超大规模数据）。

```bash
# 需要安装torch-npu: pip install torch-npu --index-url https://download.pytorch.org/whl/npu
python src/gar_feature_generator_npu.py --data data.csv --card-col card_id --mode npu --output-csv ./features.csv
```

### 分布式版本

多进程并行处理，适合亿级数据。

```bash
python src/gar_feature_generator_dist.py --data data.csv --card-col card_id --workers 16 --output-csv ./features.csv
```

---

## 使用示例

### 白样本模式（无标签）

```bash
python src/gar_feature_generator_ascend.py \
    --data data/transactions.csv \
    --card-col card_id \
    --entity-cols card_id,merchant_id,device_type,transaction_type \
    --account-features card_level,issuing_bank \
    --transaction-features amount,balance_after,timestamp,is_pos,is_cross_border \
    --output-csv ./features.csv
```

### 检测NPU状态

```bash
python src/gar_feature_generator_ascend.py --check-npu
```

---

## 数据格式

### 标准CSV格式

```csv
card_id,merchant_id,device_type,transaction_type,amount,balance_after,timestamp,card_level,issuing_bank,is_pos,is_cross_border
123456,SHOP001,MOB010,POS,1500.00,8500.50,2026-05-20 10:30:00,1,BANK_A,1,0
123456,SHOP002,MOB010,CARD,200.00,8300.50,2026-05-20 11:00:00,1,BANK_A,0,0
```

### IEEE-CIS数据集（有标签）

```
data_dir/
├── train_transaction.csv    # 交易记录（TransactionID, TransactionAmt, card1, card2, addr1, isFraud）
└── train_identity.csv        # 身份信息（TransactionID, DeviceInfo, DeviceType）
```

---

## 算法核心实现

### Sparse Dict图构建

```python
# 图结构：dict[int, list] - 每行存储邻居索引
tx_neighbors = defaultdict(list)
for col in entity_cols:
    groups = df.groupby(col).indices
    for val, idx_list in groups.items():
        if 1 < len(idx_list) < neighbor_threshold:
            for idx in idx_list:
                tx_neighbors[idx].extend(idx_list)
```

### 邻居特征向量化计算

```python
# 预计算度
degrees = np.array([len(tx_neighbors.get(i, [])) for i in range(n)], dtype=np.float32)
features['n_1hop'] = degrees
features['n_1hop_log'] = np.log1p(degrees)

# 邻居金额统计（逐节点遍历）
for i in range(n):
    neighs = tx_neighbors.get(i, [])
    if neighs:
        neigh_amts = amounts[neighs]
        amt_1hop_mean[i] = neigh_amts.mean()
```

---

## 实验结果

### IEEE-CIS 590K全量数据集

| 方法 | 特征维度 | Test AUC | 提升 |
|------|---------|----------|------|
| Baseline | 1 | 0.7075±0.0055 | 0 |
| KG Brute Force | 53 | 0.8421±0.0063 | +13.46% |
| **GAR-Inspired Full** | 18 | **0.8725±0.0014** | **+16.50%** |

### GAR-Inspired消融实验

| 模型 | Test AUC | 说明 |
|------|----------|------|
| Baseline | 0.7075 | 仅TransactionAmt |
| Entity Fraud Rates (4维) | 0.8125 | 单实体欺诈率 |
| Pair Fraud Rates (6维) | 0.8641 | 实体对欺诈率 |
| Neighbor Fraud Rate (1维) | 0.5006 | 邻居欺诈率（单独无效） |
| **GAR-Inspired Full (18维)** | **0.8725** | 组合最优 |

---

## 数据泄漏防护

1. **训练/测试集严格划分**：先划分训练集(70%)和测试集(30%)，再进行特征计算
2. **欺诈率仅从训练集计算**：测试集的欺诈率特征使用训练集的统计量
3. **图结构仅从训练集构建**：邻居关系基于训练集构建

---

## 项目结构

```
Gar-Feature-Generator/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── src/
│   ├── feature_generator.py               # 统一入口（自动选择模式）
│   ├── gar_feature_generator.py            # GAR基础版本
│   ├── gar_feature_generator_dist.py      # 分布式GAR
│   ├── gar_feature_generator_ascend.py    # Sparse Dict版本（推荐）
│   ├── gar_feature_generator_fast.py       # CSR向量化版本
│   ├── gar_feature_generator_npu.py        # torch_npu版本
│   ├── kg_brute_force_generator.py        # KG Brute Force
│   ├── kg_feature_generator.py            # KG通用版本
│   ├── kg_feature_generator_dist.py       # 分布式KG
│   ├── kg_feature_generator_ascend.py     # Ascend NPU KG
│   ├── kg_feature_generator_gpu.py        # CUDA GPU KG
│   └── train_classifier.py                # 模型训练
└── outputs/                                # 实验结果输出
```

---

## 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| pandas | >=1.5 | 数据处理 |
| numpy | >=1.21 | 数值计算 |
| scikit-learn | >=1.0 | 模型训练 |

### 可选加速

| 硬件 | 安装 | 说明 |
|------|------|------|
| Ascend NPU | `pip install torch-npu` | 华为昇腾加速 |
| CUDA GPU | `pip install cupy` | NVIDIA加速 |

---

## License

MIT License