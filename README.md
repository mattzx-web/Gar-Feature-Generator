# GAR Feature Generator

基于图关联规则(Graph Association Rules)的金融反欺诈特征工程工具包。

## 项目简介

本项目提供两种知识图谱特征工程方案，用于IEEE-CIS Fraud Detection数据集：

| 方法 | 特征维度 | Test AUC | 说明 |
|------|---------|----------|------|
| **GAR-Inspired** | 18维 | 0.8725±0.0014 | 基于欺诈率的规则特征，配对欺诈率为核心信号 |
| **KG Brute Force** | 53维 | 0.8421±0.0063 | 枚举式图特征(度/计数/VCD统计) |

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/mattzx-web/Gar-Feature-Generator.git
cd Gar-Feature-Generator

# 创建Python环境 (推荐 Python 3.8+)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install pandas numpy scikit-learn
```

### 2. 数据准备

下载IEEE-CIS Fraud Detection数据集：

1. 访问 https://www.kaggle.com/competitions/ieee-fraud-detection/data
2. 下载以下文件：
   - `train_transaction.csv`
   - `train_identity.csv`
3. 解压到数据目录，结构如下：

```
YOUR_DATA_DIR/
├── train_transaction.csv
└── train_identity.csv
```

### 3. 运行实验

**GAR-Inspired (推荐)**

```bash
python src/gar_feature_generator.py --data-dir /path/to/your/data
```

**KG Brute Force**

```bash
python src/kg_brute_force_generator.py --data-dir /path/to/your/data
```

### 4. 多种子验证

```bash
# GAR-Inspired 多种子验证
python src/gar_feature_generator.py --data-dir /path/to/your/data --seeds 42 123 456

# KG Brute Force 多种子验证
python src/kg_brute_force_generator.py --data-dir /path/to/your/data --seeds 42 123 456
```

### 5. 查看结果

结果保存在 `outputs/` 目录下，包含JSON格式的实验结果和特征重要性。

---

## 项目结构

```
Gar-Feature-Generator/
├── README.md                    # 本文件
├── LICENSE                      # MIT License
├── src/
│   ├── gar_feature_generator.py     # GAR-Inspired特征生成器
│   └── kg_brute_force_generator.py  # KG Brute Force特征生成器
├── outputs/                     # 实验结果输出目录
└── docs/
    └── ALGORITHM_DETAILS.md     # 算法详细说明
```

---

## 方法详解

### GAR-Inspired (推荐)

**核心思想**：利用图结构计算实体和实体对的欺诈率作为特征。

**特征构成** (18维)：

| 组件 | 特征数 | 说明 |
|------|--------|------|
| TransactionAmt | 2 | 交易金额 + log |
| degree | 1 | 图度（邻居数量） |
| Entity Frequency | 4 | card1, card2, addr1, P_emaildomain 频率 |
| Entity Fraud Rates | 4 | 单实体欺诈率 |
| Pair Fraud Rates | 6 | 实体对欺诈率 |
| Neighbor Fraud Rate | 1 | 1跳邻居欺诈率均值 |

**为什么Pair Fraud Rates最有效？**

金融反诈场景中，特定实体组合是高风险信号：
- `card1=12345 + P_emaildomain=gmail.com` → 欺诈率40%
- `card1=12345 + addr1=NewYork` → 欺诈率35%

配对欺诈率直接捕捉了这种组合风险，比单独使用实体欺诈率更有效。

### KG Brute Force

**核心思想**：枚举所有可能的图结构特征（度、计数、统计量）。

**特征构成** (53维)：

| 组件 | 特征数 | 说明 |
|------|--------|------|
| Entity Degree | 5 | card1-card4, addr1 的度 |
| Entity Count | 10 | 实体计数 + log |
| 1-hop Features | 3 | 邻居数量、平均金额、标准差 |
| 2-hop Features | 2 | 2跳邻居数量、比率 |
| Pair Count | 20 | 实体对计数 + log |
| V/C/D Statistics | 11 | V/C/D列的统计特征 |
| TransactionAmt | 2 | 交易金额 + log |

---

## 命令行参数

### GAR-Inspired

```
--data-dir      数据目录（必需）
--output-dir    输出目录（默认: ./outputs）
--seed          随机种子（默认: 42）
--seeds         多种子模式（如: --seeds 42 123 456）
```

### KG Brute Force

```
--data-dir      数据目录（必需）
--output-dir    输出目录（默认: ./outputs）
--seed          随机种子（默认: 42）
--seeds         多种子模式（如: --seeds 42 123 456）
```

---

## 实验结果

### 590K全量数据集多种子验证

| 方法 | 特征维度 | Test AUC (mean±std) | 提升 vs Baseline |
|------|---------|---------------------|------------------|
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

本实现严格遵守数据泄漏防护原则：

1. **训练/测试集严格划分**：先划分训练集(70%)和测试集(30%)，再进行特征计算
2. **欺诈率仅从训练集计算**：测试集的欺诈率特征使用训练集的统计量
3. **图结构仅从训练集构建**：邻居关系基于训练集构建

---

## 扩展到其他数据集

如需在其他金融数据集上使用，请修改实体列配置：

```python
# 在 gar_feature_generator.py 中修改
DEFAULT_ENTITY_COLS = ['card1', 'card2', 'addr1', 'P_emaildomain']

# 替换为您的实体列
YOUR_ENTITY_COLS = ['entity_a', 'entity_b', 'entity_c']
```

CSV文件应包含：
- `TransactionID`: 交易ID
- `TransactionAmt`: 交易金额
- 实体列: 用于构建图的实体特征
- `isFraud`: 欺诈标签（仅训练集需要）

---

## 性能说明

- 590K数据集在单核CPU上约需15-20分钟
- 内存需求约4-8GB
- 可通过减少`--seeds`数量加速实验

---

## 引用

如果你在研究中使用了本代码，请引用：

```
Gar-Feature-Generator: Knowledge Graph Feature Engineering for Financial Anti-Fraud
Matt, 2026
```

---

## License

MIT License