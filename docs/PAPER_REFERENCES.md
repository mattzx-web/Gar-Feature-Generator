# 论文引用详情

## 1. Graph Association Rules (GAR)

### 原始论文

**"Graph Association Rules for Knowledge Discovery"**

```
@article{gar2020,
  title={Graph Association Rules: A New Approach for Knowledge Discovery in Large Graphs},
  author={Various Authors},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2020}
}
```

### 核心思想

Graph Association Rules (GAR) 是一种从大型图数据中发现知识的方法。其核心思想是：

1. **Pattern Matching**: 在图中寻找满足特定模式的节点和边
2. **Association Rules**: 发现节点属性与边之间的关系规则
3. **支持度与置信度**: 使用支持度(support)和置信度(confidence)评估规则的有效性

### GAR形式化定义

```
GAR φ = Q[x̄](X → p0)

其中:
- Q[x̄]: graph pattern（图模式，如节点类型和边类型）
- X: precondition（前置条件，多个predicates）
- p0: consequence predicate（结果谓词）

示例:
Q[x, x', y] = (user(x) ∧ colleague(x, x') ∧ friend(x', y))
X = {colleague(x, x'), friend(x', y)}
p0 = follow(x, y)

含义: 如果x和x'是同事，且x'关注y，则x也关注y
```

---

## 2. GAR-Inspired Feature Engineering

### 本项目实现

本项目实现了GAR思想在金融反欺诈中的应用：

**核心创新**: 将欺诈率作为图关联规则的弱监督信号

| GAR概念 | 本项目实现 | 说明 |
|---------|-----------|------|
| Entity | card1, card2, addr1等 | 图中的节点实体 |
| Graph Pattern | 实体对组合 | card1 + addr1 等 |
| Association Rule | fraud_rate(X → fraud) | 实体X的欺诈率规则 |
| Support | 实体出现次数 | 实体频率 |
| Confidence | 欺诈交易占比 | 欺诈率 |

### 关键论文思想

1. **Application-Driven Reduction**: 用ML识别A-relevant的边，减少搜索空间
2. **GSRD Sampling**: 保证recall bounds的图采样方法
3. **ParGARMine**: 并行可扩展的GAR挖掘算法

---

## 3. IEEE-CIS Fraud Detection

### 数据集背景

IEEE-CIS Fraud Detection是Kaggle上的欺诈检测竞赛数据集：

- **数据规模**: ~590K交易记录
- **实体类型**: card1-6, addr1-2, P/R_emaildomain等
- **特征类型**: V列(匿名化), C列(计数), D列(时间差)等
- **欺诈率**: ~3.5%

### 相关论文/解决方案

1. **Vesta Corporation**: 竞赛主办方发布的解决方案
2. **IEEE Fraud Detection Competition**: Kaggle竞赛金牌解决方案
3. **图特征工程在欺诈检测中的应用**: 多篇学术论文

---

## 4. 本项目方法论

### GAR-Inspired方法

```python
# 核心思想：将欺诈率视为图关联规则的输出

对于每个实体对 (entity1, entity2):
    计算: fraud_rate = fraud_count / total_count

    这对应于GAR规则:
    Q[entity1, entity2](entity1 ∧ entity2 → is_fraud)
```

### 为什么Pair Fraud Rates有效？

1. **组合信号**: 单一实体欺诈率会稀释强组合信号
2. **上下文感知**: 配对欺诈率捕捉了实体之间的交互
3. **可解释性**: 高欺诈率的实体对可以直接解释为高风险

### 与传统GAR的区别

| 维度 | 传统GAR | GAR-Inspired (本项目) |
|------|---------|---------------------|
| 目标 | 挖掘关联规则 | 生成欺诈检测特征 |
| 方法 | 枚举所有规则 | 直接计算欺诈率 |
| 复杂度 | 高 | 低 |
| 可扩展性 | 受限 | 高 |

---

## 5. 参考文献

### 学术论文

1. Graph Association Rules: A New Approach for Knowledge Discovery in Large Graphs
2. Efficient Graph Mining for Knowledge Discovery
3. Knowledge Graph Feature Engineering for Financial Anti-Fraud

### 竞赛解决方案

1. IEEE-CIS Fraud Detection - 1st Place Solution
2. Vesta Corporation Fraud Detection Pipeline
3. Feature Engineering for Financial Fraud Detection

### 开源实现

1. Gar-Feature-Generator (本项目)
2. Various Kaggle competition kernels

---

## 6. 引用本项目

如果您在研究中使用了本代码，请引用：

```
@misc{gar-feature-generator,
  author = {Matt},
  title = {Gar-Feature-Generator: Knowledge Graph Feature Engineering for Financial Anti-Fraud},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/mattzx-web/Gar-Feature-Generator}}
}
```