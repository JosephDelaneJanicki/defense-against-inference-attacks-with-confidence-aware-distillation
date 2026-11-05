# Regarding Metrics: Accuracy vs. ROC-AUC

## Why accuracy alone can be misleading

In membership inference attacks (MIAs), accuracy depends on a particular decision threshold and the balance of member and nonmember examples. An attacker can achieve apparently high accuracy by predicting the majority class without learning a useful membership signal.

**Guiding question for ROC-AUC:** *How often does the attack assign a higher membership score to a randomly chosen member than to a randomly chosen nonmember?*

An ROC-AUC of 0.5 represents chance-level ranking. A value above 0.5 suggests that members tend to receive higher scores; a value below 0.5 can indicate an inverted score direction rather than stronger privacy. Small deviations from 0.5 may be sampling noise, so **AUC > 0.5 does not automatically prove real leakage**. Repeated seeds and uncertainty estimates matter.

## What accuracy measures

Accuracy answers: *How many predictions were correct at the selected threshold?*

For example, if the score is a calibrated membership probability:

```python
if attack_score >= 0.5:
    predict = "member"
else:
    predict = "nonmember"
```

The threshold of 0.5 is not universally appropriate: a score need not be a calibrated probability, and the best threshold depends on the operating objective. Precision, recall, and F1 are also threshold-dependent. **ROC-AUC uses the ranking over all possible thresholds**, so it answers a different question.

## Examples from this project

### Example 1: Misleading accuracy from imbalance — initial confidence-threshold attack

- Accuracy: approximately **0.7619**
- Recall: **1.0**
- ROC-AUC: approximately **0.50**

The evaluation set contained more members than nonmembers, and the attacker predicted essentially every sample as a member. Its high accuracy reflected class prevalence, not useful separation. The near-0.5 ROC-AUC indicated little detectable ranking signal under that attack.

### Example 2: Weak attack signal — early v1 and v2

- Accuracy: approximately **0.50**
- ROC-AUC: approximately **0.50**

Both metrics were consistent with little detectable membership discrimination in these evaluations. This does **not** prove that the underlying model reveals no membership information to any possible attacker.

### Example 3: Feature-aware attack — v2

| Target | ROC-AUC |
|---|---:|
| Overfit Model A | ~0.5677 |
| Early-stop / realistic | ~0.51 |

Richer attack features exposed a modest signal for the overfit target, while the other targets remained close to chance under this attacker.

### Example 4: Shokri-style attack — v3 (historical run)

| Target | Accuracy | ROC-AUC |
|---|---:|---:|
| Overfit Model A v2 | 0.5526 | 0.5812 |
| Early-stop | 0.5016 | 0.5098 |
| Realistic | 0.5093 | 0.5124 |

These are **historical v3 results**, not interchangeable with later v5 or Gemma results. The overfit model had a clearer ranking signal despite relatively modest accuracy. The early-stop and realistic results were near chance.

## Gemma 3 4B PT: Membership inference and utility

### Experimental setting

The Gemma experiments evaluate **fine-tuning membership** on WikiText-103, not whether examples appeared in Gemma's original pretraining corpus. The realistic targets use `google/gemma-3-4b-pt` with LoRA (rank 8, alpha 16, dropout 0.05), two epochs, learning rate `2e-4`, effective batch size eight, and maximum sequence length 256. Four training seeds are planned: **42, 123, 456, 789**. Target evaluation uses 5,000 known members and 5,000 nonmembers. The target corpus is fixed across seeds, so the multi-seed study tests variation due to training randomness rather than independent dataset draws.

Attack features include negative log-likelihood (NLL), token-loss statistics, and Min-K variants. The learned attackers use logistic regression and random forest. ROC-AUC is the primary ranking metric; threshold-based metrics remain useful for diagnosing classifier behavior.

### Realistic teacher: Seed 42

| Evaluation | ROC-AUC |
|---|---:|
| Direct NLL score | 0.532789 |
| Direct Min-K 10% | 0.545011 |
| Direct token-loss standard deviation | 0.547462 |
| Shokri-style logistic regression | 0.545891 |
| Shokri-style random forest | 0.544778 |

These are **different attack methods** and should not be treated as repeated measurements of the same estimator. The Shokri-style classifiers are trained on separate shadow-model membership examples. Direct metrics score the target without training a learned classifier.

### Undefended teacher: Multi-seed learned-attack baseline

| Attacker | Mean ROC-AUC | Standard deviation | Seeds |
|---|---:|---:|---|
| Logistic regression | 0.560067 | 0.015483 | 42, 123, 456, 789 |
| Random forest | 0.557471 | 0.014146 | 42, 123, 456, 789 |

The standard deviations describe variation across four runs, **not** confidence intervals. They are not sufficient by themselves to establish statistical significance.

### Raw-soft defended student: Seed 42

| Attack type | Logistic ROC-AUC | Random forest ROC-AUC |
|---|---:|---:|
| Undefended teacher baseline (Shokri-style) | 0.545891 | 0.544778 |
| Adaptive attack on raw-soft student | 0.523833 | 0.526652 |
| Black-box hard-label extraction attack on raw-soft student | **0.495678** | **0.483501** |

**Threat-model distinction matters:**

- **Transfer:** train the attack on ordinary, undefended shadows and evaluate it on the defended target.
- **Black-box hard-label extraction:** query the released defended student on attacker-owned shadow data, train fresh ordinary surrogate models on hard next-token pseudo-labels, then train the MIA on those surrogates. This is one specific black-box threat model, not every possible black-box attack.
- **Adaptive:** reproduce the defense on attacker-owned shadow teachers/students, then train an attack against those defended shadows. It does not require access to the target's private teacher.

The available Seed 42 results suggest reduced membership discrimination under the tested adaptive and black-box attacks. **They do not establish that the defense eliminates leakage**, and the attack methods are not identical. Transfer results should be included in the final comparison after verifying the recorded values.

### Gemma black-box example: 50% accuracy but different ROC-AUCs

The raw-soft student produced the following Seed 42 black-box results:

| Metric | Logistic regression | Random forest |
|---|---:|---:|
| Accuracy | 0.5000 | 0.5000 |
| Precision | 0.5000 | 0.0000 |
| Recall | 1.0000 | 0.0000 |
| F1 | 0.6667 | 0.0000 |
| **ROC-AUC** | **0.49567836** | **0.48350118** |

On this balanced 5,000-member / 5,000-nonmember test set, the logistic classifier appears to label **every example as a member** at the chosen threshold, while the random forest appears to label **every example as a nonmember**. This explains their identical 50% accuracy and very different precision/recall/F1 values.

ROC-AUC remains calculable from their continuous scores, even when thresholded predictions collapse to one class. Both AUCs are near chance. Because the forest's AUC is below 0.5, check score orientation and report the original AUC rather than describing the lower number as automatically better privacy. An attacker able to reverse the score would obtain `1 - 0.48350118 = 0.51649882` on this evaluation; whether that is meaningful requires uncertainty analysis and an appropriate, non-leaky rule for choosing direction.

**Diagnostic follow-up:** inspect predicted score distributions and the fractions predicted as members at threshold 0.5. Confirm member/nonmember labels, attack-score direction, and absence of overlap between target-private data and attacker-owned shadow data.

## Utility is a separate axis from attack effectiveness

A defense that pushes ROC-AUC toward 0.5 by destroying model performance is not necessarily useful. We therefore track language-modeling utility alongside privacy.

| Gemma model (Seed 42) | Validation NLL | Validation perplexity (approx.) |
|---|---:|---:|
| Realistic teacher | 2.245350 | 9.44 |
| Raw-soft student | 2.246518 | 9.45 |

The student NLL is **0.001168 higher** (about **0.052%** relative to the teacher's NLL), indicating very similar validation language-modeling performance on this split. Perplexity is `exp(NLL)`; lower is better for both measures.

These are **validation-set** results. The reserved `data/gemma/utility_test.jsonl` should be evaluated independently before claiming held-out test-set utility. NLL/perplexity measure next-token prediction, not every aspect of downstream language-model quality.

## Reporting rules for the final paper

1. **Primary privacy comparison:** report ROC-AUC for each attacker, defense, and seed; retain the threat-model label. Include uncertainty (e.g., seed-level variability and, where suitable, bootstrap confidence intervals over evaluation examples).
2. **Do not overinterpret near-chance AUC:** an AUC of 0.50 means this tested attack did not reliably rank members above nonmembers; it is not a privacy guarantee.
3. **Check below-chance AUC:** a reversed scoring direction can contain usable signal. Do not cherry-pick whichever orientation looks best after examining target labels.
4. **Retain accuracy, precision, recall, and F1 as diagnostics:** they explain operating-point behavior but do not replace ROC-AUC.
5. **Compare utility at the same time:** report validation and independent test NLL/perplexity for teacher and student, plus training cost where available.
6. **Distinguish versions:** Adult v2/v3/v5 results, Gemma direct scores, Gemma transfer, black-box, and adaptive learned attacks are different experiments and must remain clearly labeled.
7. **Avoid significance claims from point estimates alone:** small improvements above 0.5 can arise from sampling variation or experimental instability.

## Key lesson

**Accuracy answers whether the classifier is correct at one threshold. ROC-AUC measures whether its continuous scores rank members above nonmembers. Neither metric alone proves privacy or leakage.** The strongest defense evaluation combines correctly specified attack threat models, repeated-seed ROC-AUC, calibrated interpretation of below-chance values, and an explicit utility comparison.
