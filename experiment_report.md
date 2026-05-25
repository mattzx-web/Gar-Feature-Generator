# GAR Feature Generator 实验报告

**日期**: 2026-05-25
**数据集**: IEEE-CIS Fraud Detection (590,540条交易记录)

---

## 实验设置

- **Train/Test划分**: 70%/30% (seed=42, 确定性shuffle)
- **Train**: 413,378条 (欺诈率3.51%)
- **Test**: 177,162条 (欺诈率3.47%)
- **模型**: GradientBoostingClassifier (n_estimators=50, max_depth=5)
- **评估**: 特征在训练集/测试集上分别构建，避免数据泄漏

---

## 实验结果

### 主实验结果

| Method | AUC | Precision | Recall | Features |
|--------|------|-----------|--------|----------|
| Baseline | 0.6834 | 0.6364 | 0.0011 | 1 |
| KG Brute Force | 0.7830 | 0.9453 | 0.0197 | 14 |
| **GAR-Inspired** | **0.8678** | **0.7220** | **0.2162** | 22 |

### GAR特征详情 (22维)

1. **Entity Frequency (8维)**
   - card1_freq, card1_freq_log
   - ProductCD_freq, ProductCD_freq_log
   - addr1_freq, addr1_freq_log
   - P_emaildomain_freq, P_emaildomain_freq_log

2. **Card Aggregation (3维)**
   - card_count, card_count_log
   - card_amt_mean

3. **Pair Frequency (6维)**
   - card1_ProductCD_pair_freq, card1_ProductCD_pair_freq_log
   - card1_addr1_pair_freq, card1_addr1_pair_freq_log
   - card1_P_emaildomain_pair_freq, card1_P_emaildomain_pair_freq_log

4. **Entity Fraud Rate (4维)**
   - card1_fraud_rate, ProductCD_fraud_rate
   - addr1_fraud_rate, P_emaildomain_fraud_rate

5. **Pair Fraud Rate (6维)**
   - card1_ProductCD_pair_fraud_rate
   - card1_addr1_pair_fraud_rate
   - card1_P_emaildomain_pair_fraud_rate
   - ProductCD_addr1_pair_fraud_rate
   - ProductCD_P_emaildomain_pair_fraud_rate
   - addr1_P_emaildomain_pair_fraud_rate

---

## 分析

### GAR优势
- **AUC最高 (0.8678)**: 综合排序能力强
- **Recall最优 (21.6%)**: 能检测到更多欺诈案例
- **Precision良好 (72.2%)**: 误报率可控

### KG问题
- Precision极高 (94.5%) 但Recall极低 (2%)
- 模型过于保守，漏检大量欺诈

### Pair Fraud Rate贡献
根据消融实验，Pair Fraud Rate贡献了主要的性能提升(+5.16%)，是最强信号特征。

---

## 结论

1. GAR在综合性能上优于KG Brute Force
2. Pair Fraud Rate是关键特征
3. 特征必须基于训练集统计构建，避免数据泄漏
4. 推荐使用GAR作为欺诈检测特征工程方案

---

## 代码使用

```bash
# 生成特征
python src/gar/gar_cpu.py \
    --data ~/ieee-fraud-detection/train_transaction.csv \
    --card-col card1 \
    --entity-cols card1,ProductCD,addr1,P_emaildomain \
    --export-features-only \
    --output-csv ./features.csv

# 训练模型
python src/train_classifier.py --features ./features.csv --model gar
```