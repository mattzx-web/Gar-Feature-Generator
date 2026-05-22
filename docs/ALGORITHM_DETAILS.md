# 算法详细说明

## 1. GAR-Inspired 特征生成算法

### 1.1 核心思想

GAR-Inspired (Graph Association Rules inspired) 是一种基于图关联规则的欺诈检测特征工程方法。核心思想是利用实体之间的图结构关系，计算欺诈率作为特征。

### 1.2 图构建过程

```
给定: 实体列 [card1, card2, addr1, P_emaildomain]
过程:
1. 对每个实体列，按实体值分组
2. 同一实体值的所有交易互为邻居
3. 构建 交易ID → 邻居集合 的映射

示例:
- card1=12345 的所有交易形成一组邻居
- addr1=NewYork 的所有交易形成一组邻居
- 一个交易可以属于多个组（多个邻居集合）
```

### 1.3 特征计算

#### 1.3.1 TransactionAmt特征 (2维)
```python
TransactionAmt        = 交易金额
TransactionAmt_log    = log(1 + 交易金额)
```

#### 1.3.2 degree特征 (1维)
```python
degree = 该交易的邻居数量（所有实体列的并集）
```

#### 1.3.3 Entity Frequency特征 (4维)
```python
对于每个实体列 (card1, card2, addr1, P_emaildomain):
    freq = 该实体值在训练集中出现的次数
```

#### 1.3.4 Entity Fraud Rates特征 (4维)
```python
对于每个实体列:
    fraud_rate = 该实体值的欺诈交易数 / 该实体值的总交易数

示例:
- card1=12345: 100笔交易中5笔欺诈 → fraud_rate = 0.05
- card1=67890: 50笔交易中20笔欺诈 → fraud_rate = 0.40 (高风险!)
```

#### 1.3.5 Pair Fraud Rates特征 (6维)
```python
对于实体对 (card1, card2), (card1, addr1), (card1, P_emaildomain),
           (card2, addr1), (card2, P_emaildomain), (addr1, P_emaildomain):

    pair_fraud = 该实体对的欺诈交易数 / 该实体对的总交易数

示例:
- card1=12345 + P_emaildomain=gmail.com: 50笔交易中20笔欺诈 → fraud_rate = 0.40
- 单独看 card1=12345 可能只有0.05，但组合后是0.40
```

#### 1.3.6 Neighbor Fraud Rate特征 (1维)
```python
对于每个交易:
    neigh_fraud_rate = 该交易所有1跳邻居的欺诈率均值

注: 测试时使用训练集的邻居欺诈率（避免数据泄漏）
```

### 1.4 数据泄漏防护

| 特征类型 | 计算方式 | 泄漏风险 |
|----------|----------|----------|
| Entity Frequency | 训练集统计 | 无 |
| Entity Fraud Rates | 训练集统计 | 无 |
| Pair Fraud Rates | 训练集统计 | 无 |
| Neighbor Fraud Rate | 训练集标签 | **测试时需特殊处理** |

**Neighbor Fraud Rate的测试处理**:
```python
# 对于测试集中的交易:
global_i = test_indices[i]  # 测试集交易的全局索引
neighs = tx_neighbors.get(global_i, set())  # 获取邻居（基于训练集构建的图）
train_neighs = [n for n in neighs if n in train_indices_set]  # 只取训练集中的邻居
neigh_fraud_rate = train_is_fraud[local_neighs].mean()  # 用训练集标签计算
```

---

## 2. KG Brute Force 特征生成算法

### 2.1 核心思想

KG Brute Force 是一种枚举式的图特征工程方法，穷举所有可能的图结构特征。

### 2.2 特征计算

#### 2.2.1 Entity Degree特征 (5维)
```python
对于 card1, card2, card3, card4, addr1:
    degree = 该实体列中，该实体值对应的交易数量
```

#### 2.2.2 Entity Count特征 (10维)
```python
对于 card1, card2, card3, card4, addr1:
    count = 该实体值在训练集中出现的次数
    count_log = log(1 + count)
```

#### 2.2.3 1-hop Neighbor特征 (3维)
```python
对于每个交易:
    n_1hop = 1跳邻居数量
    amt_1hop_mean = 1跳邻居交易的平均金额
    amt_1hop_std = 1跳邻居交易金额的标准差
```

#### 2.2.4 2-hop Neighbor特征 (2维)
```python
对于每个交易:
    n_2hop = 2跳邻居数量（邻居的邻居）
    2hop_1hop_ratio = n_2hop / (n_1hop + 1)
```

#### 2.2.5 Pair Count特征 (20维)
```python
对于实体对 (card1, card2), (card1, card3), ...:
    pair_count = 该实体对在数据集中出现的次数
    pair_count_log = log(1 + pair_count)
```

#### 2.2.6 V/C/D统计特征 (11维)
```python
V_cols = [V1, V2, ..., V339]
C_cols = [C1, C2, ..., C13]
D_cols = [D1, D2, ..., D14]

V_mean, V_std, V_sum, V_max, V_nan_count
C_mean, C_std, C_sum
D_mean, D_std
```

---

## 3. GAR vs KG Brute Force 对比

| 维度 | GAR-Inspired | KG Brute Force |
|------|--------------|----------------|
| 特征类型 | 规则式（欺诈率） | 枚举式（计数/统计） |
| 特征维度 | 18维 | 53维 |
| 最强特征 | Pair Fraud Rates | V/C/D统计 |
| 核心信号 | 组合风险 | 交易金额分布 |
| **Test AUC** | **0.8725** | 0.8421 |

---

## 4. 消融实验分析

### 4.1 GAR-Inspired消融

| 组件 | AUC | 贡献 |
|------|-----|------|
| Baseline (TransactionAmt) | 0.7075 | 0 |
| Entity Fraud Rates (4维) | 0.8125 | +10.5% |
| Pair Fraud Rates (6维) | 0.8641 | +15.7% |
| Neighbor Fraud Rate (1维) | 0.5006 | -20.7% (单独无效) |
| **Full (18维)** | **0.8725** | **+16.5%** |

### 4.2 关键发现

1. **Pair Fraud Rates是核心信号**：6维配对欺诈率贡献了+15.7%的提升
2. **Neighbor Fraud Rate单独使用无效**：单独只有0.5006（接近随机），但在组合中有增益
3. **组合优于任何单一组件**：Full > Pair > Entity

---

## 5. 理论基础

### 5.1 图关联规则 (GAR)

```
GAR φ = Q[x̄](X → p0)

其中:
- Q[x̄]: graph pattern（图模式）
- X: precondition（前置条件，多个predicates）
- p0: consequence predicate（结果谓词）

示例:
Q[x, x', y] = (user(x) ∧ colleague(x, x') ∧ friend(x', y))
X = {colleague(x, x'), friend(x', y)}
p0 = follow(x, y)

含义: 如果x和x'是同事，且x'关注y，则x也关注y
```

### 5.2 欺诈率作为弱监督信号

在金融反诈场景中，直接挖掘GAR规则可能计算代价高。我们的方法将欺诈率视为一种弱监督信号：
- 高欺诈率的实体组合 → 高风险交易
- 邻居的高欺诈率 → 交易可能被欺诈

这种方法在计算上高效，且能捕捉复杂的组合模式。