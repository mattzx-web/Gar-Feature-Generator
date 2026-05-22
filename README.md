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

## 原始算法论文

### Graph Association Rules: A New Approach for Knowledge Discovery in Large Graphs

**论文来源**: IEEE Transactions on Knowledge and Data Engineering

**核心思想**: 利用图结构发现实体之间的关联规则，将欺诈率作为弱监督信号生成特征。

**核心概念**:

| 概念 | 说明 |
|------|------|
| **Graph Pattern** | 图中的节点和边模式，描述实体之间的关系结构 |
| **GAR Definition** | GAR φ = Q[x̄](X → p0)，表示"如果满足前置条件X，则结论p0成立" |
| **Support** | 规则在图中出现的频率 |
| **Confidence** | 规则的可靠性（条件成立时结论成立的比例） |

**GAR形式化定义**:
```
GAR φ = Q[x̄](X → p0)

其中:
- Q[x̄]: graph pattern（图模式，如 user(x) ∧ colleague(x, x') ∧ friend(x', y)）
- X: precondition（前置条件，多个predicates）
- p0: consequence predicate（结果谓词）

示例:
前置条件 X = {colleague(x, x'), friend(x', y)}
结论 p0 = follow(x, y)
含义: 如果x和x'是同事，且x'关注y，则x也关注y
```

**本项目实现**: 将GAR思想应用于金融反欺诈，将欺诈率视为图关联规则的输出：
- **Entity Fraud Rate**: 单实体欺诈率（如 card1=X 的欺诈率）
- **Pair Fraud Rate**: 实体对欺诈率（如 card1=X 且 addr1=Y 的欺诈率）
- **Neighbor Fraud Rate**: 1跳邻居欺诈率均值

---

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/mattzx-web/Gar-Feature-Generator.git
cd Gar-Feature-Generator

python -m venv venv
source venv/bin/activate

pip install pandas numpy scikit-learn
```

### 2. 数据格式

#### 格式一：IEEE-CIS数据集（已有标签）

```
data_dir/
├── train_transaction.csv    # 交易记录（TransactionID, TransactionAmt, card1, card2, addr1, P_emaildomain, isFraud）
└── train_identity.csv        # 身份信息（TransactionID, DeviceInfo, DeviceType）
```

#### 格式二：通用CSV（白样本或自定义）

```csv
card_id,merchant_id,device_type,transaction_type,amount,balance_after,timestamp,card_level,issuing_bank,is_pos,is_cross_border
123456,SHOP001,MOB010,POS,1500.00,8500.50,2026-05-20 10:30:00,1,BANK_A,1,0
123456,SHOP002,MOB010,CARD,200.00,8300.50,2026-05-20 11:00:00,1,BANK_A,0,0
123456,SHOP003,WEB001,ONLINE,5000.00,3300.50,2026-05-20 14:00:00,1,BANK_A,0,1
789012,SHOP001,MOB010,POS,800.00,9200.00,2026-05-20 10:45:00,2,BANK_B,1,0
```

**字段说明**:
- `card_id`: 卡号（账户标识）
- `merchant_id`: 商户ID
- `device_type`: 设备类型
- `transaction_type`: 交易类型
- `amount`: 交易金额
- `balance_after`: 交易后余额
- `timestamp`: 时间戳
- `card_level`: 卡等级（账户级特征，每个卡号重复）
- `issuing_bank`: 开户行（账户级特征）
- `is_pos`: 是否POS交易
- `is_cross_border`: 是否跨境交易

---

## 使用方法

### 模式一：IEEE-CIS数据集（有标签）

```bash
# GAR-Inspired
python src/gar_feature_generator.py --data-dir /path/to/ieee-fraud-detection

# KG Brute Force
python src/kg_brute_force_generator.py --data-dir /path/to/ieee-fraud-detection

# 多种子验证
python src/gar_feature_generator.py --data-dir /path/to/ieee-fraud-detection --seeds 42 123 456
```

### 模式二：白样本特征生成（无标签）

```bash
# KG特征生成
python src/kg_feature_generator.py --data /path/to/transactions.csv \
                                    --card-col card_id \
                                    --export-features-only \
                                    --output-csv ./features/kg_features.csv

# GAR特征生成
python src/gar_feature_generator.py --data /path/to/transactions.csv \
                                     --card-col card_id \
                                     --export-features-only \
                                     --output-csv ./features/gar_features.csv
```

### 模式三：自定义特征列

```bash
# 指定实体列、账户级特征、交易级特征
python src/kg_feature_generator.py --data /path/to/transactions.csv \
                                    --card-col card_id \
                                    --entity-cols card_id,merchant_id,device_type,transaction_type \
                                    --account-features card_level,issuing_bank \
                                    --transaction-features amount,balance_after,timestamp,is_pos,is_cross_border \
                                    --export-features-only \
                                    --output-csv ./features/custom_features.csv
```

### 模式四：从CSV加载特征训练模型

```bash
# 生成特征
python src/kg_feature_generator.py --data /path/to/transactions.csv \
                                    --card-col card_id \
                                    --export-features-only \
                                    --output-csv ./features/kg_features.csv

# 训练模型
python src/train_classifier.py --features ./features/kg_features.csv --model kg
```

---

## 命令行参数

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | CSV文件路径（白样本模式） | - |
| `--data-dir` | IEEE-CIS数据集目录 | - |
| `--card-col` | 卡号列名 | card_id |
| `--entity-cols` | 实体列名（逗号分隔） | card_id,merchant_id,device_type,transaction_type |
| `--account-features` | 账户级特征（逗号分隔） | card_level,issuing_bank |
| `--transaction-features` | 交易级特征（逗号分隔） | amount,balance_after,timestamp,is_pos,is_cross_border |
| `--export-features-only` | 仅生成特征，不训练模型 | False |
| `--output-csv` | 特征CSV输出路径 | - |
| `--output-dir` | 输出目录 | ./outputs |
| `--seed` | 随机种子 | 42 |
| `--seeds` | 多种子验证 | - |

---

## 生成的特征类型

### 白样本模式生成特征

| 特征类型 | 说明 | 示例 |
|----------|------|------|
| **实体度** | 交易在图中的邻居数量 | card_id_degree, merchant_id_degree |
| **实体频率** | 实体值出现次数 | card_id_freq, merchant_id_freq_log |
| **账户级特征** | 直接使用原值 | card_level, issuing_bank |
| **交易级特征** | 直接使用原值 | amount, balance_after, is_pos |
| **卡号聚合** | 按卡号聚合的统计量 | card_tx_count, card_amt_mean, card_amt_std |
| **配对频率** | 实体对组合出现次数 | card_id_merchant_id_pair_freq |
| **邻居特征** | 1-hop邻居的统计量 | n_1hop, amt_1hop_mean, amt_1hop_std |
| **时序特征** | 时间相关特征 | trans_hour, time_diff_prev |
| **欺诈率特征** | 仅当有标签时可用 | card_id_fraud_rate, pair_fraud_rate |

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
│   ├── gar_feature_generator.py      # GAR-Inspired特征生成器
│   ├── gar_feature_generator.py      # 通用GAR特征生成器（白样本模式）
│   ├── kg_brute_force_generator.py   # KG Brute Force特征生成器
│   ├── kg_feature_generator.py       # 通用KG特征生成器（白样本模式）
│   ├── train_classifier.py           # 独立模型训练器
│   └── feature_utils.py               # 公共工具
├── docs/
│   ├── ALGORITHM_DETAILS.md
│   └── PAPER_REFERENCES.md
└── outputs/                           # 实验结果输出
```

---

## 性能说明

- 590K数据集在单核CPU上约需15-20分钟
- 内存需求约4-8GB
- 可通过减少`--seeds`数量加速实验

---

## License

MIT License