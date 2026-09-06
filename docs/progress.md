# Project Progress Log

**Project:** Membership Inference Resistance via Confidence-Aware Distillation and Anonymization
**Author:** Joseph.D.Janicki

---

## Overview

This document tracks progress on the project, including what has been implemented so far, the reasoning behind design decisions, and the planned next steps.

---

## Work Completed

### 1. Environment Setup

* Created a Python virtual environment (`venv`) within the project directory
* Installed required libraries:

  * PyTorch
  * scikit-learn
  * NumPy / Pandas
  * matplotlib
  * ucimlrepo

**Reasoning:**
Using a project-specific virtual environment ensures clean dependency isolation and avoids conflicts with other projects (e.g., SLLT). 

---

### 2. Project Structure Initialization

Created the following directory structure:

```
MIA_Project/
├── data/
├── mialab/
├── notebooks/
├── proposal/
├── results/
└── src/
```

**Reasoning:**
Separating code, data, results, and documentation improves maintainability and aligns with standard machine learning project practices.

---

### 3. Dataset Selection

Selected the **Adult Income dataset** (UCI repository).

**Reasoning:**
While many membership inference papers use image datasets (e.g., CIFAR-10), this project requires:

* meaningful anonymization
* interpretable features
* quasi-identifiers

Tabular datasets such as Adult are better suited for:

* k-anonymity
* feature generalization
* privacy-focused transformations

---

### 4. Teacher Model (Model A) Implementation

Implemented a baseline neural network using PyTorch:

* Fully connected (MLP) architecture
* Trained on preprocessed dataset
* Includes:

  * numeric scaling
  * categorical encoding (one-hot)
  * missing value handling

**Reasoning:**
This model serves as the **target model** in membership inference attacks and as the **knowledge source** for the student model. It is expected to:

* achieve strong predictive performance
* exhibit measurable membership leakage

---

### 5. Confidence Extraction

Extracted prediction probabilities and computed confidence:

```
confidence = max(softmax(output))
```

for both:

* training samples (members)
* test samples (non-members)

**Reasoning:**
Membership inference attacks rely heavily on confidence differences between:

* training data (typically higher confidence)
* unseen data (lower confidence)

This step establishes the core signal that the project will attempt to suppress.

---
### 6. Teacher Model Baseline Run

Successfully trained the teacher model on the Adult dataset after adding preprocessing for mixed numeric and categorical features. The model achieved approximately 83.95% accuracy on both the training and test sets.

Sample confidence values were also extracted for both member (training) and non-member (test) samples.

**Why this matters:**  
This confirms that the first baseline model is working and that the project can now move into membership inference evaluation. Because the model shows similar train and test accuracy, the next step is to determine whether confidence-based membership leakage still exists even without obvious overfitting.
---

### 7. Improved Teacher Model (Model A v2)

An improved version of the teacher model was implemented (`train_teacher_v2.py`) with the following enhancements:

* deeper neural network architecture
* mini-batch training using DataLoader
* increased number of epochs (60)
* introduction of validation set
* tracking of training vs validation loss and accuracy

Training and validation curves were saved for analysis.

**Observed Results:**

* Training accuracy increased to approximately 94%
* Validation accuracy stabilized around 83–85%
* Training loss decreased steadily
* Validation loss increased significantly over time

This indicates **clear overfitting behavior**, where the model continues to improve on the training data while performance on unseen data degrades.

**Reasoning:**

This version of the model was intentionally allowed to overfit in order to:

* amplify memorization effects
* increase the likelihood of membership inference leakage
* create a strong baseline for evaluating attack performance

Membership inference attacks rely on differences between training and non-training data behavior, and overfitting is a primary driver of such differences.

**Key Insight:**

The divergence between training and validation performance suggests that the model is learning patterns specific to the training data rather than generalizable structure. This makes it an ideal candidate for evaluating membership inference attacks.

---

### 8. Realistic Teacher Model

A new teacher model script was created as `train_teacher_realistic.py`. Unlike the overfit teacher model, this version was designed to represent a more realistic deployed machine learning model.

This version added:

- dropout regularization
- batch normalization
- AdamW optimizer
- weight decay
- learning rate scheduling
- early stopping
- accuracy, F1, and loss tracking

**Observed behavior:**

- Training accuracy stabilized around 87%
- Validation accuracy stabilized around 85–86%
- Validation loss stayed near 0.31 rather than exploding upward
- Training and validation curves stayed relatively close compared with the overfit model

**Reasoning:**

The previous overfit model is useful for exaggerating memorization and membership inference leakage. However, real models are usually tuned to perform well on unseen data. The realistic model provides a stronger practical baseline because it performs better than the initial baseline while avoiding the severe overfitting seen in the long-trained teacher model.

**Key takeaway:**

This model gives the project a realistic test case for evaluating whether membership inference attacks still succeed against a better-generalized model.

---
### 9. Early-Stop / Matching-Fit Teacher Model

A third teacher model was implemented as `model_a_early_stop.py`. This model uses the same architecture as the overfit teacher model, but training is stopped after only 5 epochs.

**Observed Results:**

- Train accuracy: 0.8643
- Validation accuracy: 0.8612
- Test accuracy: 0.8584
- Train F1: 0.6831
- Validation F1: 0.6797
- Test F1: 0.6703
- Generalization gap: 0.0031

**Reasoning:**

This model was created to test whether membership inference leakage is primarily caused by the architecture’s capacity or by extended overtraining. Because this model uses the same architecture as the overfit model but stops early, it acts as a controlled comparison.

i noticed at early epochs the overfit model had good accuracy with a low gap in traning and validation loss, seems like a good test case for how much that gap matters as well as architecture

**Key Takeaway:**

The early-stop model achieves strong performance while maintaining a very small train-validation gap. This makes it a strong “matching-fit” candidate for realistic membership inference testing.
---
### 10. Baseline Membership Inference Attack (Confidence-Based)

A baseline membership inference attack was implemented using model confidence scores. The attack attempts to classify whether a sample was part of the training set (member) or not (non-member) based on prediction confidence.

The following steps were performed:

* Loaded confidence outputs for:

  * Overfit Model A
  * Realistic Model A
  * Early-Stop Model A
* Constructed attack datasets using:

  * training samples → labeled as members (1)
  * test samples → labeled as non-members (0)
* Implemented:

  * threshold-based attack
  * ROC curve evaluation
  * AUC scoring

**Initial Observation (Before Fix):**

The attack reported an accuracy of ~0.7619 across all models. However, this was determined to be misleading due to dataset imbalance:

* Training samples significantly outnumber test samples
* The attack defaulted to predicting all samples as “member”
* This resulted in artificially inflated accuracy

**Correction Applied:**

The attack dataset was rebalanced by sampling equal numbers of training and test examples before evaluation.

**Corrected Results:**

* Overfit Model A ROC AUC: ~0.505
* Realistic Model A ROC AUC: ~0.495
* Early-Stop Model A ROC AUC: ~0.496

These values are effectively random (≈0.5), indicating:

> A simple confidence-threshold attack is insufficient to reliably infer membership for these models.

**Key Insight:**

Even in the overfit model, the difference in mean confidence between training and test samples is extremely small (~0.003). This suggests that:

* confidence alone is a weak signal for membership inference in this setup
* more advanced attacks are required to exploit subtle behavioral differences

ultimately confidence and overfitting are the big leverage with inference attacks, this test was just to see how far that by itself can get an attack, results definately indicate structure mimmickery are still a big piece of the puzzle
---

### Implication for Project Direction

This result motivates the need for a stronger attack model.

The next step is to implement a **shadow-model-based membership inference attack**, which uses:

* full probability vectors (not just max confidence)
* learned decision boundaries via an attack model

This aligns with prior work (e.g., Shokri et al.) and is expected to reveal more nuanced leakage patterns.

---

### Next Step

Implement shadow-model attack pipeline:

* train shadow models on similar data distributions
* generate labeled attack dataset
* train attack classifier
* evaluate against Model A variants

This will provide a more realistic measure of privacy leakage.

---
### 11. Shadow-Model Membership Inference Attack v1

A shadow-model-based membership inference attack was implemented following the general structure of Shokri et al. The attack trains multiple shadow models, uses their outputs on member and non-member samples to create an attack dataset, then trains an attack classifier to predict membership.

**Initial Shadow Attack Results:**

- Overfit Model A ROC AUC: ~0.501
- Realistic Model A ROC AUC: ~0.507
- Early-Stop Model A ROC AUC: ~0.506

The attack accuracy remained around 0.50, with recall near 1.0, indicating that the attack model effectively learned to predict all samples as members rather than learning a meaningful membership boundary.

**Interpretation:**

This result suggests that using only the raw two-class probability vector is not enough to expose membership leakage in this setup. The attack model needs richer behavioral features from the target model.

**Key Insight:**

Membership inference is not only about maximum confidence. It can also rely on:

- prediction entropy
- prediction loss
- correctness
- confidence margin
- full output distribution shape

Because the simple shadow attack did not reveal meaningful leakage, the next step is to implement a feature-aware shadow attack.

---

### Next Step

Implement `shadow_attack_v2.py`, which extends the attack input from only class probabilities to a richer feature vector:

- class 0 probability
- class 1 probability
- maximum confidence
- entropy
- confidence margin
- loss on true label
- correctness flag

This should better capture the behavioral signals that membership inference attacks exploit.

---

### 12. Feature-Aware Shadow Attack v2

A second shadow-model attack was implemented using a richer feature representation instead of only raw class probabilities. The attack input now includes:

- class probability values
- maximum confidence
- entropy
- confidence margin
- true-label loss
- correctness flag

**Results:**

| Model | Attack Accuracy | ROC AUC |
|---|---:|---:|
| Overfit Model A v2 | 0.5516 | 0.5677 |
| Realistic Model A | 0.5049 | 0.5076 |
| Early-Stop Model A | 0.5060 | 0.5102 |

**Interpretation:**

The feature-aware attack successfully detected membership leakage in the overfit model (that is to say more then random on a heavily overfit model), while the realistic and early-stop models remained close to random guessing. This suggests that the overfit model contains stronger membership signals, while models with smaller train-validation gaps leak less information under this attack.

good indicator for the overfitting focus

**Key Insight:**

The early-stop model is particularly useful because it uses the same architecture as the overfit model but stops training before the generalization gap grows. Its near-random attack result suggests that training behavior, not just model capacity, plays a major role in membership inference vulnerability.

**Next Step:**

The next step is not to begin the defense pipeline yet. Because the feature-aware attack only produced moderate leakage on the overfit model and near-random results on the other models, the attack should be strengthened first. The next implementation will move closer to the Shokri et al. design by using class-specific attack models and shadow models trained to better match each target model’s training regime.

---

### 13. Shokri-Style Shadow Attack v3 Results

A third version of the membership inference attack was implemented to closely follow the methodology of Shokri et al. This version introduced:

- regime-matched shadow models (overfit, early-stop, realistic)
- class-specific attack models (one per predicted class)
- enhanced feature representation
- balanced evaluation

**Results:**

| Model | Attack Accuracy | ROC AUC |
|---|---:|---:|
| Overfit Model A v2 | 0.5526 | 0.5812 |
| Early-Stop Model A | 0.5016 | 0.5098 |
| Realistic Model A | 0.5093 | 0.5124 |

**Interpretation:**

The overfit model shows clear membership inference leakage under a strong attack, while the early-stop and realistic models remain near random. This confirms that overfitting significantly increases vulnerability, while models with low generalization gaps are much more resistant.

**Additional Insight:**

Class-specific attack models revealed that leakage is not uniform across predictions. Certain predicted classes exhibited stronger membership signals, consistent with prior research.

**Conclusion:**

This experiment demonstrates that:

- weak attacks may fail to detect leakage
- strong, well-aligned attacks reveal leakage in overfit models
- reducing generalization gap significantly reduces vulnerability

These results validate the attack pipeline and establish a strong baseline for evaluating defense mechanisms in the next phase of the project.

---

### 14. Shadow Attack v4 (Expanded Feature Set and Scaling)

A fourth version of the shadow-model attack was implemented to test whether increasing feature complexity and training scale would further improve attack performance. This version included:

- expanded feature set (~25 features)
- interaction terms between confidence, loss, and correctness
- increased shadow models (16)
- longer shadow training for overfit regime
- threshold tuning per class-specific attack model

**Results:**

| Model | ROC AUC |
|---|---:|
| Overfit Model A v2 | 0.5601 |
| Early-Stop Model A | 0.5076 |
| Realistic Model A | 0.5159 |

**Interpretation:**

Despite significantly increasing model complexity and data scale, attack performance did not improve over the v3 attack. The overfit model remains distinguishable, but only moderately, while the early-stop and realistic models remain near random.

**Key Insight:**

This suggests diminishing returns in attack complexity. Once key behavioral features are captured, further feature expansion does not substantially improve leakage detection.

**Conclusion:**

The v3 attack appears to capture the majority of available membership signal in this setting. Additional complexity does not yield meaningful gains, indicating that the observed leakage is bounded by the underlying model and dataset characteristics.

---

### 15. Random Forest Shadow Attack Check

A Random Forest version of the shadow attack was tested to determine whether the MLP attack model was limiting membership inference performance. An initial forest attempt incorrectly trained directly on target model outputs, producing artificially high AUC values. This was identified as invalid because the attacker was learning from the victim model’s known member/non-member labels.

A corrected version was then implemented using the proper shadow-model pipeline:

1. Train shadow models
2. Generate shadow member/non-member outputs
3. Train class-specific Random Forest attack models on shadow outputs
4. Evaluate the trained forest attacks on target model outputs

**Corrected Results:**

| Model | Random Forest ROC AUC |
|---|---:|
| Overfit Model A v2 | 0.5624 |
| Early-Stop Model A | 0.4940 |
| Realistic Model A | 0.4994 |

**Interpretation:**

The corrected forest attack restores the expected pattern: the overfit model shows measurable leakage, while the early-stop and realistic models remain near random. However, the forest attack does not outperform the v3 MLP attack, which achieved an AUC of approximately 0.5812 on the overfit model.

**Key Insight:**

The attack does not appear to be limited by the neural attack classifier. Instead, the available membership signal in this dataset/model setup appears to be modest. This supports the conclusion that the v3 MLP attack captures most of the usable leakage signal.

---

### 16. Student Dataset Generation

Generated a full grid of teacher-derived datasets for Model B training using the overfit teacher model as the source. The generated dataset variants combine label-processing defenses and feature-processing defenses.

**Label modes:**

- raw soft labels
- temperature-smoothed soft labels
- confidence-clipped soft labels
- hard labels

**Feature modes:**

- original features
- anonymized/generalized features
- high-confidence filtered features

This produced 12 Model B dataset variants:

| Label Mode | Feature Mode |
|---|---|
| raw_soft | original / anon / filtered |
| temp_soft | original / anon / filtered |
| clip_soft | original / anon / filtered |
| hard | original / anon / filtered |

**Observed Output:**

- Original/filtered feature dimension: 107
- Anonymized feature dimension: 76
- Manifest saved to `data/student_datasets/manifest.csv`

**Reasoning:**

This dataset grid allows the project to evaluate which defense combinations best preserve utility while reducing membership inference leakage. Instead of testing one student model, the project now supports an ablation study across label-level and feature-level privacy transformations.

**Next Step:**

Train one Model B student model for each generated dataset variant and compare:

- accuracy
- F1 score
- membership inference attack AUC

The dataset generation script combines teacher-output generation, confidence-aware label modification, and feature-level generalization into a single pipeline. The anonymization layer currently uses feature generalization rather than strict k-anonymity.

---

### 17. Model B Student Grid Training

Trained 12 Model B student models using the generated teacher-derived datasets. Each student variant combines one label-processing method with one feature-processing method.

**Best-performing student model:**

- `student_clip_soft_original`
- Test accuracy: ~0.8580
- Test F1: ~0.6671

**Observed Patterns:**

- Original feature variants generally preserved the most utility.
- Anonymized variants showed a small to moderate utility decrease.
- Filtered variants showed the largest performance drop.
- Confidence clipping preserved utility best among the tested label defenses.

**Interpretation:**

The initial utility results suggest that confidence clipping may reduce extreme teacher confidence while preserving enough predictive structure for the student model to perform well. However, final conclusions require membership inference evaluation on each student model.

**Next Step:**

Attack the Model B student grid using the validated v3 membership inference attack and compare privacy leakage against utility.

**Clarification on Evaluation Metrics:**

All reported accuracy and F1 scores for Model B are evaluated against the original ground truth labels from the Adult dataset, not the synthetic (teacher-generated) labels used during training.

Although Model B is trained on teacher-derived outputs (soft or hard labels), its performance is measured against the true labels to assess how well it preserves the underlying task signal.

This distinction is important because:

- Evaluating against synthetic labels would only measure how well Model B replicates Model A (distillation quality).
- Evaluating against true labels measures actual task utility and generalization.

**Interpretation:**

The reported metrics reflect how effectively each defense preserves real-world predictive performance, rather than how closely the student model mimics the teacher model.

This ensures that any observed privacy gains are not achieved at the cost of completely losing meaningful predictive capability.

---

### 18. Model B Student Attack Grid

All 12 Model B student variants were evaluated using the validated v3-style shadow-model attack. The attack was trained separately for each student configuration using matching shadow student models.

**Key Results:**

- Original feature student models generally produced attack AUC values around 0.518–0.527.
- Anonymized feature models produced attack AUC values around 0.513–0.515.
- Filtered models produced unstable or below-random AUC values, suggesting weak or unreliable attack signal.

Compared to the overfit teacher model baseline (AUC ≈ 0.5812), the student models showed substantially reduced membership inference leakage.

**Best Utility/Privacy Candidate:**

`student_clip_soft_original`

- Test accuracy: ~0.8580
- Test F1: ~0.6671
- Attack AUC: ~0.5180

**Interpretation:**

The student training pipeline appears to reduce leakage while preserving utility. Confidence clipping in particular provides a strong privacy/utility tradeoff, maintaining the best observed task performance while reducing membership inference AUC close to random.

**Caution:**

AUC values below 0.5 should not be interpreted as “better than random privacy.” They usually indicate that the attack score is unstable or inverted. These should be described as near-random or unreliable attack signals.

---

### CIFAR-10 Teacher Attack Validation

The v3-style shadow attack was applied to two CIFAR-10 teacher models:

| Target | Attack Accuracy | Attack AUC |
|---|---:|---:|
| Overfit teacher | 0.7317 | 0.7908 |
| Realistic teacher | 0.5291 | 0.5372 |

**Interpretation:**

The overfit CIFAR teacher shows strong membership inference vulnerability, while the realistic CIFAR teacher shows much weaker leakage. This confirms that the attack pipeline is capable of detecting strong membership leakage when it exists.

**Key Insight:**

The lower attack AUC values observed on the Adult dataset are likely due to the dataset/model setting having weaker available leakage, not because the attack implementation is ineffective.

**Conclusion:**

CIFAR-10 provides a useful validation case: overfitting greatly increases attack success, while more realistic training reduces membership leakage.

---

### CIFAR-10 Student Dataset Generation

Generated student training label variants for both CIFAR-10 teacher models:

- overfit teacher
- realistic teacher

Each teacher produced 8 student variants:

- raw soft labels
- temperature-smoothed soft labels
- confidence-clipped soft labels
- hard labels
- each with and without high-confidence filtering

This produced 16 total CIFAR student variants.

**Purpose:**

The CIFAR experiment tests whether the teacher-to-student defense pipeline reduces membership inference leakage in both:

1. a high-leakage regime, using the overfit teacher
2. a lower-leakage but still vulnerable regime, using the realistic teacher

**Next Step:**

Train one CIFAR student CNN for each generated variant and evaluate each student on CIFAR-10 test accuracy, F1, and membership inference vulnerability.

---
### CIFAR-10 Overfit Teacher v2

The original CIFAR overfit teacher reached near-total memorization, with training confidence almost entirely saturated near 1.0. This made confidence filtering unusable because an absolute threshold removed all training samples.

A new overfit teacher, `overfit_teacher_v2`, was trained with a less extreme setup:

- Training size: 15,000 samples
- Epochs: 10
- Train accuracy: 0.9223
- Test accuracy: 0.6923
- Train confidence mean: 0.9016
- Train confidence median: 0.9835
- Test confidence mean: 0.8511

**Interpretation:**

This model is still clearly overfit, but it is no longer completely confidence-saturated. It provides a better target for the defense pipeline because high-confidence filtering can now remove only the most suspicious samples rather than eliminating the entire dataset.
### CIFAR-10 Student Attack Results

All 16 CIFAR-10 student variants were attacked using a v3-style class-specific shadow-model attack.

**Key Results:**

For students trained from the overfit teacher v2:

| Variant | Test Accuracy | Attack AUC |
|---|---:|---:|
| raw_soft_original | 0.7751 | 0.5140 |
| temp_soft_original | 0.7569 | 0.5141 |
| clip_soft_original | 0.7744 | 0.5207 |
| hard_original | 0.7447 | 0.5063 |

The overfit teacher v2 had test accuracy around 0.6923, so several student models improved task accuracy while reducing attack AUC close to random guessing.

For students trained from the realistic teacher:

| Variant | Test Accuracy | Attack AUC |
|---|---:|---:|
| raw_soft_original | 0.8681 | 0.5153 |
| temp_soft_original | 0.8596 | 0.5062 |
| clip_soft_original | 0.8766 | 0.5173 |
| hard_original | 0.8717 | 0.5092 |
| hard_filtered | 0.8667 | 0.5007 |

**Interpretation:**

The CIFAR-10 student pipeline substantially reduced membership inference leakage. Across most variants, attack AUC stayed near 0.49 - 0.50 - 0.52, indicating that the attack was close to random guessing.

**Main Finding:**

The strongest result is observed in the overfit-teacher setting. Student models trained on teacher-derived labels improved test accuracy from approximately 0.6923 to around 0.77 while reducing membership inference leakage to near-random levels.

**Caution:**

Filtered variants sometimes produced AUC values below 0.5. These are best interpreted as unstable or inverted attack signals rather than meaningful evidence of stronger-than-random privacy.