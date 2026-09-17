import argparse
from pathlib import Path

import numpy as np
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


SEED = 42

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help="Membership score CSV produced by the Gemma scorer.",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for this target.",
    )

    return parser.parse_args()


def evaluate(name, model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    return {
        "attack_model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("GEMMA LEARNED MEMBERSHIP ATTACK")
    print("=" * 60)

    print("\nTarget:", args.name)
    print("Scores:", args.scores)

    df = pd.read_csv(args.scores)

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing attack features: {missing}"
        )

    X = df[FEATURES].values
    y = df["membership"].values

    print("\nExamples:", len(df))
    print("Members:", int(y.sum()))
    print("Nonmembers:", int((y == 0).sum()))
    print("Features:", len(FEATURES))

    # --------------------------------------------------
    # Fixed attack train/test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.50,
        random_state=SEED,
        stratify=y,
    )

    print("\nAttack training examples:", len(y_train))
    print("Attack test examples:", len(y_test))

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
                random_state=SEED,
            ),
        ),
    ])

    print("\nTraining logistic-regression attacker...")
    logistic.fit(X_train, y_train)

    logistic_results = evaluate(
        "logistic_regression",
        logistic,
        X_test,
        y_test,
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
        random_state=SEED,
        n_jobs=-1,
    )

    print("Training random-forest attacker...")
    forest.fit(X_train, y_train)

    forest_results = evaluate(
        "random_forest",
        forest,
        X_test,
        y_test,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    results = pd.DataFrame([
        logistic_results,
        forest_results,
    ])

    results.insert(
        0,
        "target",
        args.name,
    )

    print("\n=== LEARNED MIA RESULTS ===")

    for _, row in results.iterrows():
        print(
            f"\n{row['attack_model']}"
            f"\n  Accuracy : {row['accuracy']:.6f}"
            f"\n  Precision: {row['precision']:.6f}"
            f"\n  Recall   : {row['recall']:.6f}"
            f"\n  F1       : {row['f1']:.6f}"
            f"\n  ROC-AUC  : {row['auc']:.6f}"
        )

    # --------------------------------------------------
    # RF feature importance
    # --------------------------------------------------

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": forest.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")

    for _, row in importance.iterrows():
        print(
            f"{row['feature']:22s} "
            f"{row['importance']:.6f}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_dir = Path("results/gemma")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        output_dir
        / f"{args.name}_learned_mia.csv"
    )

    importance_path = (
        output_dir
        / f"{args.name}_learned_mia_feature_importance.csv"
    )

    results.to_csv(
        result_path,
        index=False,
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    print("\nSaved:", result_path)
    print("Saved:", importance_path)


if __name__ == "__main__":
    main()