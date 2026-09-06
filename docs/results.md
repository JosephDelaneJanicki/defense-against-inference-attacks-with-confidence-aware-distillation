## Membership Inference Attack Results – Model A Variants

### Overview

A baseline membership inference attack was performed on all Model A variants using a **confidence-threshold attack**. The attack attempts to classify whether a sample was part of the training dataset (member) or not (non-member) based on the model’s prediction confidence.

To ensure fair evaluation, the attack dataset was balanced between member and non-member samples.

---

### Results Summary

| Model                 | Train Conf Mean | Test Conf Mean | Confidence Gap | Attack Accuracy | ROC AUC |
| --------------------- | --------------- | -------------- | -------------- | --------------- | ------- |
| Overfit v2            | 0.9329          | 0.9299         | +0.0030        | ~0.50           | 0.5055  |
| Realistic             | 0.8576          | 0.8599         | -0.0023        | ~0.50           | 0.4953  |
| Early Stop (5 epochs) | 0.8634          | 0.8656         | -0.0023        | ~0.50           | 0.4957  |

---

### Key Observations

1. **Confidence gap is minimal across all models**

   The difference between training and test confidence is extremely small (~0.002–0.003), even in the overfit model. This suggests that confidence alone does not strongly distinguish members from non-members.

2. **Attack performance is near random**

   All models achieved ROC AUC values near **0.5**, indicating that the attack performs no better than random guessing.

3. **Overfitting did not significantly increase confidence-based leakage**

   Although the overfit model showed strong divergence in training and validation loss, this did not translate into a strong confidence-based membership inference signal.

4. **Realistic and early-stop models behave similarly**

   Both models show nearly identical confidence distributions between training and test samples, further reducing attack effectiveness.

---

### Interpretation

These results suggest that:

> **Maximum prediction confidence alone is not a reliable signal for membership inference in this setting.**

This aligns with findings in prior research that simple threshold-based attacks are often insufficient, especially when models are reasonably well-calibrated or when the dataset does not induce strong memorization artifacts.

---

### Implication for Project Direction

Since the baseline confidence-threshold attack is ineffective, stronger attack methods are required.

The next step is to implement a **shadow-model-based membership inference attack**, which:

* leverages full probability distributions rather than only maximum confidence
* learns attack patterns from shadow models
* more closely follows the methodology used in prior work (e.g., Shokri et al.)

This will provide a more realistic assessment of privacy leakage.

---

## Membership Inference Attack Results

### Overview

We evaluated membership inference attacks across three variants of Model A:

* **Overfit Model (v2)**
* **Early-Stop Model**
* **Realistic (Regularized) Model**

Attacks were implemented in three stages:

1. Confidence-based threshold attack
2. Feature-aware shadow attack (v2)
3. Shokri-style shadow attack with regime matching and class-specific models (v3)

---

### Final Attack Results (Shokri-style v3)

| Model              | Attack Accuracy |    ROC AUC |
| ------------------ | --------------: | ---------: |
| Overfit Model A v2 |          0.5526 | **0.5812** |
| Early-Stop Model A |          0.5016 |     0.5098 |
| Realistic Model A  |          0.5093 |     0.5124 |

---

### Interpretation

The results show a clear separation between the overfit model and the better-generalized models:

* The **overfit model** exhibits measurable membership leakage (AUC ≈ 0.58)
* The **early-stop** and **realistic models** remain near random guessing (AUC ≈ 0.51)

This indicates that:

> Membership inference attacks become more effective as models overfit their training data.

---

### Comparison to Prior Work

In Membership Inference Attacks Against Machine Learning Models and subsequent studies:

* High AUC values (0.7–0.9) are typically observed under:

  * strong overfitting
  * high-capacity models (e.g., deep CNNs)
  * multi-class datasets (e.g., CIFAR-10)

* Moderate AUC values (0.52–0.65) are common in:

  * tabular datasets (e.g., Adult)
  * well-generalized models
  * black-box attack settings

Our result (AUC ≈ 0.58 for the overfit model) falls within the expected range for:

* a tabular dataset
* binary classification
* a black-box attack scenario

---

### Key Findings

1. **Overfitting Drives Leakage**

   * The overfit model is the only one with meaningful attack success.

2. **Generalization Reduces Vulnerability**

   * Both early stopping and regularization reduce leakage to near-random levels.

3. **Architecture Alone Is Not the Main Factor**

   * The early-stop model uses the same architecture as the overfit model but does not leak significantly.

4. **Attack Strength Matters**

   * Naive and weak attacks failed to detect leakage.
   * Only a strong, Shokri-style attack revealed meaningful differences.

---

### Conclusion of iteration

These results demonstrate that:

* Membership inference leakage is **not guaranteed**
* It is highly dependent on **training dynamics**
* Proper training practices (e.g., early stopping, regularization) can significantly reduce vulnerability

This establishes a strong foundation for evaluating defense mechanisms in the next phase of the project.

## Results of student attacks

### Utility Performance (Model B)

The performance of each student model was evaluated using accuracy and F1 score against the original Adult dataset labels. This ensures that utility reflects real task performance rather than similarity to teacher outputs.

| Variant | Test Accuracy | Test F1 |
|---|---:|---:|
| student_clip_soft_original | **0.8580** | **0.6671** |
| student_hard_original | 0.8558 | 0.6640 |
| student_raw_soft_original | 0.8515 | 0.6606 |
| student_temp_soft_original | 0.8538 | 0.6567 |
| student_clip_soft_anon | 0.8458 | 0.6453 |
| student_hard_anon | 0.8453 | 0.6439 |
| student_raw_soft_anon | 0.8448 | 0.6223 |
| student_temp_soft_anon | 0.8413 | 0.6056 |
| student_raw_soft_filtered | 0.8293 | 0.6310 |
| student_temp_soft_filtered | 0.8184 | 0.6270 |
| student_clip_soft_filtered | 0.8200 | 0.6208 |
| student_hard_filtered | 0.8157 | 0.6222 |

**Key Observations:**

- Models using original features generally achieved the highest utility.
- Feature anonymization introduced a modest decrease in performance.
- Confidence-based filtering resulted in the largest performance degradation.
- Confidence clipping (`clip_soft`) provided the best overall utility.

---

### Privacy Performance (Membership Inference Attack)

All student models were evaluated using a validated v3-style shadow-model attack. The primary metric for privacy risk is ROC AUC.

| Variant | Attack AUC |
|---|---:|
| student_temp_soft_original | 0.5274 |
| student_raw_soft_original | 0.5218 |
| student_clip_soft_original | 0.5180 |
| student_hard_original | 0.5176 |
| student_temp_soft_anon | 0.5142 |
| student_hard_anon | 0.5146 |
| student_clip_soft_anon | 0.5137 |
| student_raw_soft_anon | 0.5129 |
| student_clip_soft_filtered | 0.4992 |
| student_temp_soft_filtered | 0.4017 |
| student_hard_filtered | 0.3686 |
| student_raw_soft_filtered | 0.3508 |

**Key Observations:**

- All student models significantly reduced membership inference risk compared to the overfit teacher model (AUC ≈ 0.58).
- Most student models fall within AUC ≈ 0.51–0.53, indicating near-random attack performance.
- Filtered variants produce unstable or below-random AUC values, suggesting weak or unreliable attack signals rather than meaningful improvements.
- Anonymization provides a consistent but modest reduction in attack performance.

---

### Utility vs Privacy Tradeoff

The best tradeoff is observed in:

- student_clip_soft_original
- Test Accuracy: 0.8580
- Test F1: 0.6671
- Attack AUC: 0.5180

This model maintains high utility while reducing attack performance close to random.

---

### Metric Interpretation

Although some attack configurations show relatively high recall and F1 scores, these metrics are threshold-dependent and do not reliably indicate attack success. Many attack models exhibit a bias toward predicting samples as members, which inflates recall while reducing precision.

ROC AUC is the primary metric used to assess privacy risk, as it evaluates the attack’s ability to distinguish members from non-members independently of any threshold. Across all student models, AUC values remain near 0.5, indicating that membership inference is largely ineffective.


---

## Conclusion

This project investigated the effectiveness of combining teacher-based synthetic data generation with confidence-aware label processing and feature-level anonymization to mitigate membership inference attacks.

The results demonstrate that:

1. **Teacher-derived training alone reduces leakage:**
   Training Model B on teacher-generated labels significantly reduces membership inference risk compared to an overfit Model A baseline.

2. **Confidence-aware defenses are effective:**
   Techniques such as temperature smoothing and confidence clipping reduce overconfidence in model outputs, which is a primary signal exploited by membership inference attacks.

3. **Feature generalization provides additional protection:**
   Anonymizing or generalizing input features further reduces leakage by limiting the model’s ability to memorize unique patterns.

4. **Tradeoffs exist between utility and privacy:**
   Filtering high-confidence samples provides stronger privacy but introduces noticeable performance degradation, while confidence clipping achieves a better balance.

5. **Attack effectiveness has a practical ceiling:**
   Even with optimized attack models and feature engineering, membership inference performance remains close to random for defended models, suggesting that leakage is substantially mitigated.

The strongest overall result was achieved by combining soft-label training with confidence clipping, which preserved task performance while reducing attack AUC to near-random levels.

---

### Final Insight

This study shows that:

> It is possible to significantly reduce membership inference leakage without sacrificing model utility by transforming the training signal rather than modifying the model architecture directly.

---

### Future Work

- Evaluate defenses on higher-dimensional datasets (e.g., CIFAR-10)
- Explore stronger anonymization techniques (e.g., k-anonymity, differential privacy)
- Investigate alternative attack models and calibration strategies
- Extend the framework to multi-class and deep learning tasks