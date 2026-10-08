import argparse
import json
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


def metrics(y_true, probability):

    prediction = (
        probability >= 0.5
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true,
            prediction,
        ),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probability,
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shadow-scores",
        required=True,
    )

    parser.add_argument(
        "--target-scores",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--defense",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    shadow_path = Path(
        args.shadow_scores
    )

    target_path = Path(
        args.target_scores
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("BLACK-BOX MIA")
    print("=" * 80)

    print("Defense:", args.defense)
    print("Target seed:", args.seed)

    print(
        "Shadow scores:",
        shadow_path,
    )

    print(
        "Target scores:",
        target_path,
    )

    shadow = pd.read_csv(
        shadow_path
    )

    target = pd.read_csv(
        target_path
    )

    # --------------------------------------------------------
    # Validate schema
    # --------------------------------------------------------

    required = set(
        FEATURES + ["membership"]
    )

    for name, df in [
        ("shadow", shadow),
        ("target", target),
    ]:

        missing = (
            required - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{name} data missing columns: "
                f"{sorted(missing)}"
            )

    X_shadow = shadow[
        FEATURES
    ].to_numpy()

    y_shadow = shadow[
        "membership"
    ].astype(int).to_numpy()

    X_target = target[
        FEATURES
    ].to_numpy()

    y_target = target[
        "membership"
    ].astype(int).to_numpy()

    print("\nAttack training population:")
    print("Total:", len(shadow))
    print(
        "Members:",
        int((y_shadow == 1).sum()),
    )
    print(
        "Nonmembers:",
        int((y_shadow == 0).sum()),
    )

    print("\nFrozen target population:")
    print("Total:", len(target))
    print(
        "Members:",
        int((y_target == 1).sum()),
    )
    print(
        "Nonmembers:",
        int((y_target == 0).sum()),
    )

    # ========================================================
    # Logistic regression
    # ========================================================

    print("\nTraining Logistic attacker...")

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

    logistic.fit(
        X_shadow,
        y_shadow,
    )

    logistic_probability = (
        logistic.predict_proba(
            X_target
        )[:, 1]
    )

    logistic_metrics = metrics(
        y_target,
        logistic_probability,
    )

    # ========================================================
    # Random forest
    # ========================================================

    print("Training Random Forest attacker...")

    forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    forest.fit(
        X_shadow,
        y_shadow,
    )

    forest_probability = (
        forest.predict_proba(
            X_target
        )[:, 1]
    )

    forest_metrics = metrics(
        y_target,
        forest_probability,
    )

    # ========================================================
    # Summary
    # ========================================================

    rows = []

    for attacker, result in [
        (
            "logistic",
            logistic_metrics,
        ),
        (
            "random_forest",
            forest_metrics,
        ),
    ]:

        rows.append({
            "seed": args.seed,
            "defense": args.defense,
            "attack_type": "black_box",
            "attacker": attacker,
            **result,
        })

    summary = pd.DataFrame(rows)

    prefix = (
        f"black_box_mia_"
        f"{args.defense}_"
        f"seed_{args.seed}"
    )

    summary_path = (
        output_dir /
        f"{prefix}_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # Predictions
    # ========================================================

    logistic_predictions = (
        target.copy()
    )

    logistic_predictions[
        "attack_probability"
    ] = logistic_probability

    logistic_predictions[
        "attack_prediction"
    ] = (
        logistic_probability >= 0.5
    ).astype(int)

    logistic_predictions.to_csv(
        output_dir /
        f"{prefix}_logistic_predictions.csv",
        index=False,
    )

    forest_predictions = (
        target.copy()
    )

    forest_predictions[
        "attack_probability"
    ] = forest_probability

    forest_predictions[
        "attack_prediction"
    ] = (
        forest_probability >= 0.5
    ).astype(int)

    forest_predictions.to_csv(
        output_dir /
        f"{prefix}_random_forest_predictions.csv",
        index=False,
    )

    # ========================================================
    # RF feature importance
    # ========================================================

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance":
            forest.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        output_dir /
        f"{prefix}_random_forest_feature_importance.csv",
        index=False,
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "experiment":
            "black_box_student_surrogate_mia",

        "seed":
            args.seed,

        "defense":
            args.defense,

        "attack_type":
            "black_box",

        "threat_model": {
            "target_teacher_access":
                False,

            "target_private_membership_access":
                False,

            "defense_reproduction":
                False,

            "target_student_black_box_access":
                True,

            "shadow_training_labels":
                "hard_argmax_predictions_from_released_student",
        },

        "features":
            FEATURES,

        "shadow_training_rows":
            len(shadow),

        "shadow_members":
            int(
                (y_shadow == 1).sum()
            ),

        "shadow_nonmembers":
            int(
                (y_shadow == 0).sum()
            ),

        "target_rows":
            len(target),

        "target_members":
            int(
                (y_target == 1).sum()
            ),

        "target_nonmembers":
            int(
                (y_target == 0).sum()
            ),

        "shadow_scores":
            str(shadow_path),

        "target_scores":
            str(target_path),
    }

    with (
        output_dir /
        f"{prefix}_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # Console results
    # ========================================================

    print("\n" + "=" * 80)
    print("BLACK-BOX MIA RESULTS")
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nRF feature importance:")
    print(
        importance.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(summary_path)


if __name__ == "__main__":
    main()