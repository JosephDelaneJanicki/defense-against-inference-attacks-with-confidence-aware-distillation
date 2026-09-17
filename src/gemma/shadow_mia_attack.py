import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "nll",
    "median_token_loss",
    "min_token_loss",
    "max_token_loss",
    "std_token_loss",
    "min_k_10",
    "min_k_20",
    "min_k_30",
    "min_k_40",
]


def evaluate(name, model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    result = {
        "attacker": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "f1": f1_score(
            y, pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y, prob),
    }

    print(f"\n{name}")
    print(f"  Accuracy : {result['accuracy']:.6f}")
    print(f"  Precision: {result['precision']:.6f}")
    print(f"  Recall   : {result['recall']:.6f}")
    print(f"  F1       : {result['f1']:.6f}")
    print(f"  ROC-AUC  : {result['roc_auc']:.6f}")

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shadow-scores",
        default=(
            "results/gemma/shadows/"
            "all_shadow_membership_scores.csv"
        ),
    )

    parser.add_argument(
        "--target-scores",
        default=(
            "results/gemma/"
            "realistic_lora_membership_scores.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/gemma/"
            "realistic_lora_shadow_mia.csv"
        ),
    )

    parser.add_argument(
        "--importance-output",
        default=(
            "results/gemma/"
            "realistic_lora_shadow_mia_feature_importance.csv"
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SHOKRI-STYLE SHADOW-MODEL MIA")
    print("=" * 70)

    print("\nLoading shadow attack population...")
    shadow = pd.read_csv(args.shadow_scores)

    print("Loading target attack population...")
    target = pd.read_csv(args.target_scores)

    # --------------------------------------------------
    # Integrity checks
    # --------------------------------------------------

    missing_shadow = [
        f for f in FEATURES
        if f not in shadow.columns
    ]

    missing_target = [
        f for f in FEATURES
        if f not in target.columns
    ]

    if missing_shadow:
        raise RuntimeError(
            f"Shadow scores missing features: "
            f"{missing_shadow}"
        )

    if missing_target:
        raise RuntimeError(
            f"Target scores missing features: "
            f"{missing_target}"
        )

    if "membership" not in shadow.columns:
        raise RuntimeError(
            "Shadow scores missing membership labels."
        )

    if "membership" not in target.columns:
        raise RuntimeError(
            "Target scores missing membership labels."
        )

    print("\nShadow population:")
    print("  Total     :", len(shadow))
    print(
        "  Members   :",
        int(shadow["membership"].sum()),
    )
    print(
        "  Nonmembers:",
        int((shadow["membership"] == 0).sum()),
    )

    print("\nTarget population:")
    print("  Total     :", len(target))
    print(
        "  Members   :",
        int(target["membership"].sum()),
    )
    print(
        "  Nonmembers:",
        int((target["membership"] == 0).sum()),
    )

    # --------------------------------------------------
    # IMPORTANT:
    #
    # Attacker FITTING uses ONLY shadow observations.
    #
    # Target labels below are used ONLY for final
    # evaluation metrics.
    # --------------------------------------------------

    X_shadow = shadow[FEATURES]
    y_shadow = shadow["membership"].astype(int)

    X_target = target[FEATURES]
    y_target = target["membership"].astype(int)

    # --------------------------------------------------
    # Logistic regression
    # --------------------------------------------------

    logistic = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])

    print(
        "\nTraining logistic attacker "
        "on shadow models ONLY..."
    )

    logistic.fit(
        X_shadow,
        y_shadow,
    )

    logistic_result = evaluate(
        "shadow_logistic_regression",
        logistic,
        X_target,
        y_target,
    )

    # --------------------------------------------------
    # Random forest
    # --------------------------------------------------

    forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print(
        "\nTraining random-forest attacker "
        "on shadow models ONLY..."
    )

    forest.fit(
        X_shadow,
        y_shadow,
    )

    forest_result = evaluate(
        "shadow_random_forest",
        forest,
        X_target,
        y_target,
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame([
        logistic_result,
        forest_result,
    ])

    results.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------
    # RF feature importance
    # --------------------------------------------------

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance":
            forest.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance_path = Path(
        args.importance_output
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    print("\nRandom-forest feature importance:")
    print(
        importance.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("SHADOW ATTACK COMPLETE")
    print("=" * 70)

    print("\nSaved:")
    print(" ", output_path)
    print(" ", importance_path)

    print(
        "\nMethodology:"
        "\n  Attacker training = shadow models only"
        "\n  Target labels      = evaluation only"
        "\n  Target examples    = never used to fit attacker"
    )


if __name__ == "__main__":
    main()