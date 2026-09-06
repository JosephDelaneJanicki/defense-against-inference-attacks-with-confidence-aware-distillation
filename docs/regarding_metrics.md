# Regarding Metrics: Accuracy vs ROC AUC

## Why Accuracy Alone Can Be Misleading

In membership inference attacks, raw accuracy can be misleading because it depends heavily on the chosen decision threshold and the balance between member and non-member samples.

For example, if the attack dataset contains more member samples than non-member samples, an attack model could predict every sample as a member and still achieve a deceptively high accuracy. This happened in the initial confidence-threshold attack, where accuracy appeared high even though the attack had not learned a meaningful distinction.

In that case, high recall occurred because the model predicted nearly everything as a member.

think about it as “How well are members ranked above non-members?”

## What Accuracy Measures

Accuracy measures:

> How many predictions were correct at one specific threshold?

For example:

if attack_score >= 0.5:
    predict member
else:
    predict non-member
    ## Examples from This Project

### Example 1: Misleading Accuracy from Imbalance

In the initial confidence-based attack, the model reported:

- Accuracy ≈ 0.7619  
- Recall = 1.0  

At first glance, this looks like a strong attack. However, this occurred because:

- The dataset was imbalanced (more members than non-members)
- The attack predicted **every sample as a member**

So effectively:

Prediction: member for all samples

This results in high accuracy simply because most samples are members.
However, the ROC AUC for this same attack was:
    AUC ≈ 0.50

This shows that:
    The attack had no real ability to distinguish members from non-members.


## Example 2: Weak Attack Signal (v1 and v2)

In early attack versions:
    Accuracy ≈ 0.50
    ROC AUC ≈ 0.50

This indicates:
    No meaningful separation


Predictions are essentially random



## Example 3: Feature-Aware Attack (v2)

After adding richer features, the attack improved:
    Overfit Model A:
        AUC ≈ 0.5677

Early-stop / realistic:
    AUC ≈ 0.51




This shows:
    The attack begins to detect a signal, but only in the overfit model.


## Example 4: Shokri-Style Attack (v3)

# Final attack results:

Model                   Accuracy    ROC AUC
Overfit Model A v2      0.5526      0.5812
Early-Stop Model        0.5016      0.5098
Realistic Model         0.5093      0.5124
# Interpretation:
Overfit model:
    Accuracy ≈ 0.55 → modest improvement
    AUC ≈ 0.58 → real separation exists
Early-stop / realistic:
    Accuracy ≈ 0.50
    AUC ≈ 0.51 → essentially random

# This demonstrates:
    Even when accuracy appears similar across models, ROC AUC reveals meaningful differences in attack effectiveness.


## Key Lesson from These Examples
Across all experiments:
    - Accuracy alone failed to reveal whether the attack was meaningful
    - ROC AUC consistently reflected the true presence (or absence) of a membership signal


High accuracy does not guarantee a good attack.But AUC > 0.5 always indicates real leakage.