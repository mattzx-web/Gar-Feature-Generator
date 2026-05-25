# GAR Feature Generator 实验报告

**日期**: 2026-05-25
**项目**: Gar-Feature-Generator
**状态**: 数据泄漏问题已修复

---

## 一、GAR核心原理

### 1.1 算法定义

GAR (Graph Association Rules) 将欺诈率视为图关联规则的输出：

```python
# 单实体欺诈率
fraud_rate(entity) = fraud_count(entity) / total_count(entity)

# 实体对欺诈率
pair_fraud_rate(e1, e2) = fraud_count(e1 ∧ e2) / total_count(e1 ∧ e2)

# 邻居欺诈率
neighbor_fraud_rate(node) = mean(fraud_rate(neighbor) for neighbor in neighbors)
```

### 1.2 GAR适用特征类型

**GAR支持账户级和交易级特征两者均可**：

| 类型 | 示例 | GAR中的作用 |
|------|------|-----------|
| **账户级** | card_id, issuing_bank, card_level | 实体身份标识 |
| **交易级** | merchant_id, device_type, ProductCD | 实体身份标识 |
| **金额类** | TransactionAmt, balance | 聚合统计特征 |

### 1.3 为什么GAR需要离散实体

| 特性 | GAR适用实体 (card_id等) | PCA匿名特征 (V1-V28) |
|------|-------------------------|---------------------|
| **类型** | 离散分类变量 | 连续数值 |
| **含义** | 可解释的身份标识 | 无物理含义的PCA投影 |
| **分组** | 同ID可聚合成组统计 | 每个值几乎都不同 |
| **统计量** | 组内样本足够，欺诈率可靠 | 组太小无法估计 |

---

## 二、数据泄漏问题与修复

### 2.1 问题描述

原始实现在完整数据上计算欺诈率，然后随机划分训练/测试集：
- 导致训练集和测试集的欺诈率完全相同（从同一数据计算）
- 结果虚高：AUC = 1.0

### 2.2 修复方案

**无泄漏模式**（默认启用）：
1. 先分割 train/test（70%/30%）
2. 仅在训练集上计算欺诈率
3. 将训练集欺诈率映射应用于测试集

**新增参数**：
```bash
--no-leakage    # 默认：防泄漏模式
--leakage       # 关闭防泄漏（不推荐）
```

### 2.3 代码修复

| 文件 | 修复内容 |
|------|----------|
| gar_cpu.py | 新增 `split_data()`, `compute_fraud_rates_from_train()`, `build_gar_features_no_leakage()` |
| gar_dist.py | 分布式版本同样支持无泄漏模式 |
| train_classifier.py | 修复 `is_fraud` 标签列识别，修复 split 列处理 |

---

## 三、实验结果

### 3.1 Synthetic Financial 数据集 (10K行)

| Method | Train AUC | Test AUC | Notes |
|--------|-----------|----------|-------|
| Baseline (amount) | 0.5100 | 0.5063 | 无泄漏 |
| **GAR-Inspired (No Leakage)** | 1.0000 | **0.9437** | ✅ 修复后 |

**Top 10 Feature Importance:**
```
 1. amt_to_card_mean_ratio                   0.6267
 2. neigh_fraud_rate                         0.2195
 3. user_id_fraud_rate                       0.0337
 4. card_amt_max                             0.0266
 5. transaction_type_country_pair_freq       0.0229
 6. n_1hop_log                               0.0214
 7. country_freq                             0.0141
 8. country_fraud_rate                       0.0090
 9. country_freq_log                         0.0070
10. n_1hop                                   0.0039
```

### 3.2 PaySim 数据集 (100K样本)

| Method | Train AUC | Test AUC | Notes |
|--------|-----------|----------|-------|
| Baseline (amount) | 0.8932 | 0.8826 | 无泄漏 |
| **GAR-Inspired (No Leakage)** | 1.0000 | 0.5000 | ❌ 不适合 |

**问题分析**：
- PaySim 每个用户平均 1.0002 次交易
- 几乎所有用户都是一次性交易
- 无法通过欺诈率区分用户

### 3.3 IEEE-CIS 数据集 (590K行) - 历史最佳

| Method | AUC | Precision | Recall | Features |
|--------|------|-----------|--------|----------|
| Baseline | 0.6834 | 0.6364 | 0.0011 | 1 |
| KG Brute Force | 0.7830 | 0.9453 | 0.0197 | 14 |
| **GAR-Inspired** | **0.8678** | **0.7220** | **0.2162** | 22 |

### 3.4 CARE-GNN 数据集对比

| Dataset | Nodes | Features | Fraud Rate | AUC |
|---------|-------|----------|------------|-----|
| **Amazon** | 11,944 | 25 | 6.9% | 0.9827 |
| **YelpChi** | 45,954 | 32 | 14.5% | 0.9089 |

---

## 四、数据集特性对比

| Dataset | Source | Rows | Fraud% | 实体重复率 | GAR适用性 | 结果 |
|---------|--------|------|--------|-----------|---------|------|
| **IEEE-CIS** | Kaggle | 590K | 3.5% | 高 | ✅ 最适合 | AUC 0.8678 |
| **Synthetic Financial** | Kaggle | 10K | 5.0% | 高 | ✅ 适合 | AUC 0.9437 |
| **Amazon** | CARE-GNN | 12K | 6.9% | 中 | ⚠️ 图特征 | AUC 0.9827 |
| **YelpChi** | CARE-GNN | 46K | 14.5% | 中 | ⚠️ 图特征 | AUC 0.9089 |
| **PaySim** | Kaggle | 6.3M | 0.13% | 极低 | ❌ 不适合 | AUC 0.5000 |
| **Credit Card** | Kaggle | 284K | 0.17% | N/A | ❌ PCA匿名 | - |

---

## 五、GAR适用性判断标准

### 5.1 适合GAR的数据集特征

1. **实体重复率高**：同一实体（card_id, user_id）有多条交易记录
2. **欺诈率有区分度**：不同实体的欺诈率差异明显
3. **实体有物理含义**：可解释的身份标识，不是匿名ID
4. **足够的样本量**：每个实体有足够多的交易记录

### 5.2 GAR不适用场景

1. **PCA匿名特征**：V1-V28等PCA投影特征，丢失实体语义
2. **单次交易用户**：每个用户只有一条记录，无法统计
3. **高基数低重复**：实体种类极多，重复率极低

---

## 六、命令使用

### 6.1 CPU模式（推荐小中型数据）

```bash
# 无泄漏模式（默认）
python src/gar/gar_cpu.py \
    --data data/transactions.csv \
    --card-col card_id \
    --entity-cols card_id,merchant_id,device_type \
    --export-features-only \
    --output-csv ./features.csv

# 泄漏模式（不推荐）
python src/gar/gar_cpu.py \
    --data data/transactions.csv \
    --card-col card_id \
    --leakage \
    --output-csv ./features.csv
```

### 6.2 分布式模式（推荐大数据）

```bash
# 8 workers
python src/gar/gar_dist.py \
    --data data/large.csv \
    --card-col card_id \
    --entity-cols card_id,merchant_id,device_type \
    --workers 8 \
    --output-csv ./features.csv
```

### 6.3 训练分类器

```bash
python src/train_classifier.py \
    --features ./features.csv \
    --model gar \
    --seed 42
```

---

## 七、结论

1. **数据泄漏已修复**：无泄漏模式通过先分割数据再计算欺诈率
2. **IEEE-CIS验证有效**：AUC从0.6834提升到0.8678
3. **Synthetic Financial效果优秀**：Test AUC达到0.9437
4. **PaySim不适合GAR**：实体重复率极低（一次性用户）

---

## 八、附录：下载的数据集

```bash
# PaySim
kaggle datasets download -d ealaxi/paysim1 -p ~/data/paysim --unzip

# Synthetic Financial Fraud
kaggle datasets download -d umitka/synthetic-financial-fraud-dataset -p ~/data/financial_fraud --unzip

# Online Payment Fraud
kaggle datasets download -d jainilcoder/online-payment-fraud-detection -p ~/data/online_payment_fraud --unzip
```