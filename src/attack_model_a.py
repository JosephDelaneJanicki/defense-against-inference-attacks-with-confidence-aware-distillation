import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


# ================================
# Config
# ================================
RESULTS_DIR = "results"

MODELS = {
    "Overfit v2": {
        "train": "model_a_train_confidence.npy",
        "test": "model_a_test_confidence.npy",
    },
    "Realistic": {
        "train": "model_a_realistic_train_confidence.npy",
        "test": "model_a_realistic_test_confidence.npy",
    },
    "Early Stop 5ep": {
        "train": "model_a_early_stop_5ep_train_confidence.npy",
        "test": "model_a_early_stop_5ep_test_confidence.npy",
    },
}


# ================================
# Attack Helpers
# ================================
def load_confidences(train_file, test_file):
    train_conf = np.load(os.path.join(RESULTS_DIR, train_file))
    test_conf = np.load(os.path.join(RESULTS_DIR, test_file))
    return train_conf, test_conf


def build_attack_data(train_conf, test_conf):
    scores = np.concatenate([train_conf, test_conf])

    # train samples are members = 1
    # test samples are non-members = 0
    labels = np.concatenate([
        np.ones_like(train_conf),
        np.zeros_like(test_conf),
    ])

    return scores, labels


def best_threshold_attack(scores, labels):
    thresholds = np.linspace(scores.min(), scores.max(), 500)

    best = {
        "threshold": None,
        "accuracy": -1,
        "precision": None,
        "recall": None,
        "f1": None,
    }

    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)

        acc = accuracy_score(labels, preds)

        if acc > best["accuracy"]:
            best["threshold"] = threshold
            best["accuracy"] = acc
            best["precision"] = precision_score(labels, preds, zero_division=0)
            best["recall"] = recall_score(labels, preds, zero_division=0)
            best["f1"] = f1_score(labels, preds, zero_division=0)

    return best


def plot_confidence_histogram(model_name, train_conf, test_conf):
    plt.figure()
    plt.hist(train_conf, bins=50, alpha=0.5, label="Train / Members")
    plt.hist(test_conf, bins=50, alpha=0.5, label="Test / Non-Members")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title(f"Confidence Distribution - {model_name}")
    plt.legend()

    out_path = os.path.join(
        RESULTS_DIR,
        f"attack_confidence_hist_{model_name.lower().replace(' ', '_')}.png"
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path


def plot_roc(model_name, scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Membership Inference ROC - {model_name}")
    plt.legend()

    out_path = os.path.join(
        RESULTS_DIR,
        f"attack_roc_{model_name.lower().replace(' ', '_')}.png"
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path, auc


# ================================
# Main
# ================================
summary_rows = []

for model_name, files in MODELS.items():
    print("\n" + "=" * 60)
    print(f"Attacking: {model_name}")
    print("=" * 60)

    train_conf, test_conf = load_confidences(files["train"], files["test"])
    scores, labels = build_attack_data(train_conf, test_conf)

    attack_result = best_threshold_attack(scores, labels)
    roc_path, auc = plot_roc(model_name, scores, labels)
    hist_path = plot_confidence_histogram(model_name, train_conf, test_conf)

    train_mean = train_conf.mean()
    test_mean = test_conf.mean()
    confidence_gap = train_mean - test_mean

    print(f"Train confidence mean: {train_mean:.4f}")
    print(f"Test confidence mean:  {test_mean:.4f}")
    print(f"Confidence gap:        {confidence_gap:.4f}")
    print()
    print(f"Best threshold:        {attack_result['threshold']:.4f}")
    print(f"Attack accuracy:       {attack_result['accuracy']:.4f}")
    print(f"Attack precision:      {attack_result['precision']:.4f}")
    print(f"Attack recall:         {attack_result['recall']:.4f}")
    print(f"Attack F1:             {attack_result['f1']:.4f}")
    print(f"ROC AUC:               {auc:.4f}")
    print()
    print(f"Saved histogram:       {hist_path}")
    print(f"Saved ROC:             {roc_path}")

    summary_rows.append([
        model_name,
        train_mean,
        test_mean,
        confidence_gap,
        attack_result["threshold"],
        attack_result["accuracy"],
        attack_result["precision"],
        attack_result["recall"],
        attack_result["f1"],
        auc,
    ])


# ================================
# Save Summary CSV
# ================================
summary_path = os.path.join(RESULTS_DIR, "model_a_attack_summary.csv")

header = (
    "model,train_conf_mean,test_conf_mean,confidence_gap,"
    "best_threshold,attack_accuracy,attack_precision,attack_recall,attack_f1,roc_auc"
)

np.savetxt(
    summary_path,
    summary_rows,
    delimiter=",",
    fmt="%s",
    header=header,
    comments="",
)

print("\n" + "=" * 60)
print("Attack summary saved:")
print(summary_path)
print("=" * 60)