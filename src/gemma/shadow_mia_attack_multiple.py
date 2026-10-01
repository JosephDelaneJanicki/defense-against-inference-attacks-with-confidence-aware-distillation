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


# ============================================================
# Configuration
# ============================================================

MULTISEED_ROOT = Path(
    "results/gemma/multiseed"
)

AGGREGATE_DIR = (
    MULTISEED_ROOT / "aggregate"
)

SEEDS = [
    42,
    123,
    456,
    789,
]

# Keep attacker randomness fixed.
#
# The experimental seeds above represent variation in
# target/shadow LoRA training. We intentionally do not
# introduce attack-model RNG as an additional source
# of variation.
ATTACK_SEED = 42


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


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    name,
    model,
    X,
    y,
    seed,
):
    pred = model.predict(X)

    prob = (
        model
        .predict_proba(X)[:, 1]
    )

    result = {
        "seed":
            seed,

        "attacker":
            name,

        "accuracy":
            accuracy_score(
                y,
                pred,
            ),

        "precision":
            precision_score(
                y,
                pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y,
                pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y,
                pred,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y,
                prob,
            ),
    }

    print(f"\n{name}")

    print(
        f"  Accuracy : "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"  Precision: "
        f"{result['precision']:.6f}"
    )

    print(
        f"  Recall   : "
        f"{result['recall']:.6f}"
    )

    print(
        f"  F1       : "
        f"{result['f1']:.6f}"
    )

    print(
        f"  ROC-AUC  : "
        f"{result['roc_auc']:.6f}"
    )

    return result


# ============================================================
# Integrity checks
# ============================================================

def validate_population(
    df,
    population_name,
):
    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{population_name} missing features: "
            f"{missing}"
        )

    if "membership" not in df.columns:
        raise RuntimeError(
            f"{population_name} missing "
            f"membership labels."
        )


# ============================================================
# Run one seed
# ============================================================

def run_seed(seed):

    seed_root = (
        MULTISEED_ROOT
        / f"seed_{seed}"
    )

    shadow_scores = (
        seed_root
        / "shadows"
        / "all_shadow_membership_scores.csv"
    )

    target_scores = (
        seed_root
        / "realistic_lora"
        / "membership_scores.csv"
    )

    output_path = (
        seed_root
        / "realistic_lora"
        / "shadow_mia.csv"
    )

    importance_path = (
        seed_root
        / "realistic_lora"
        / "shadow_mia_feature_importance.csv"
    )

    # --------------------------------------------------------
    # Verify inputs
    # --------------------------------------------------------

    if not shadow_scores.exists():
        raise FileNotFoundError(
            f"Missing shadow scores for "
            f"seed {seed}: "
            f"{shadow_scores}"
        )

    if not target_scores.exists():
        raise FileNotFoundError(
            f"Missing target scores for "
            f"seed {seed}: "
            f"{target_scores}"
        )

    print("\n\n")
    print("=" * 70)

    print(
        f"SHOKRI-STYLE SHADOW-MODEL MIA "
        f"— SEED {seed}"
    )

    print("=" * 70)

    print(
        "\nShadow scores:",
        shadow_scores,
    )

    print(
        "Target scores:",
        target_scores,
    )

    # --------------------------------------------------------
    # Load populations
    # --------------------------------------------------------

    print(
        "\nLoading shadow attack population..."
    )

    shadow = pd.read_csv(
        shadow_scores
    )

    print(
        "Loading target attack population..."
    )

    target = pd.read_csv(
        target_scores
    )

    validate_population(
        shadow,
        "Shadow scores",
    )

    validate_population(
        target,
        "Target scores",
    )

    # --------------------------------------------------------
    # Population information
    # --------------------------------------------------------

    print("\nShadow population:")

    print(
        "  Total     :",
        len(shadow),
    )

    print(
        "  Members   :",
        int(
            shadow[
                "membership"
            ].sum()
        ),
    )

    print(
        "  Nonmembers:",
        int(
            (
                shadow["membership"]
                == 0
            ).sum()
        ),
    )

    print("\nTarget population:")

    print(
        "  Total     :",
        len(target),
    )

    print(
        "  Members   :",
        int(
            target[
                "membership"
            ].sum()
        ),
    )

    print(
        "  Nonmembers:",
        int(
            (
                target["membership"]
                == 0
            ).sum()
        ),
    )

    # --------------------------------------------------------
    # IMPORTANT METHODOLOGY
    #
    # The attacker is fitted ONLY on shadow observations.
    #
    # Target membership labels are used ONLY to calculate
    # final evaluation metrics.
    # --------------------------------------------------------

    X_shadow = shadow[
        FEATURES
    ]

    y_shadow = (
        shadow["membership"]
        .astype(int)
    )

    X_target = target[
        FEATURES
    ]

    y_target = (
        target["membership"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Logistic regression
    # --------------------------------------------------------

    logistic = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=
                    ATTACK_SEED,
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
        seed,
    )

    # --------------------------------------------------------
    # Random forest
    # --------------------------------------------------------

    forest = (
        RandomForestClassifier(
            n_estimators=500,

            max_depth=8,

            min_samples_leaf=10,

            max_features="sqrt",

            class_weight="balanced",

            random_state=
                ATTACK_SEED,

            n_jobs=-1,
        )
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
        seed,
    )

    # --------------------------------------------------------
    # Save per-seed attack results
    # --------------------------------------------------------

    results = pd.DataFrame([
        logistic_result,
        forest_result,
    ])

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Random-forest feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "seed":
            seed,

        "feature":
            FEATURES,

        "importance":
            forest.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    print(
        "\nRandom-forest feature importance:"
    )

    print(
        importance[
            [
                "feature",
                "importance",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        " ",
        output_path,
    )

    print(
        " ",
        importance_path,
    )

    return (
        results,
        importance,
    )


# ============================================================
# Aggregate all seeds
# ============================================================

def aggregate_results(
    all_results,
    all_importance,
):

    AGGREGATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Raw results from all four seeds
    # --------------------------------------------------------

    results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    all_seed_path = (
        AGGREGATE_DIR
        / "shadow_mia_all_seeds.csv"
    )

    results_df.to_csv(
        all_seed_path,
        index=False,
    )

    # --------------------------------------------------------
    # Mean / SD attack metrics
    # --------------------------------------------------------

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    summary_rows = []

    for attacker, group in (
        results_df.groupby(
            "attacker"
        )
    ):

        row = {
            "attacker":
                attacker,

            "n_seeds":
                len(group),
        }

        for metric in metrics:

            row[
                f"{metric}_mean"
            ] = (
                group[metric]
                .mean()
            )

            # Sample standard deviation.
            row[
                f"{metric}_std"
            ] = (
                group[metric]
                .std(ddof=1)
            )

            row[
                f"{metric}_min"
            ] = (
                group[metric]
                .min()
            )

            row[
                f"{metric}_max"
            ] = (
                group[metric]
                .max()
            )

        summary_rows.append(
            row
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        AGGREGATE_DIR
        / "shadow_mia_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Feature importance across seeds
    # --------------------------------------------------------

    importance_df = pd.concat(
        all_importance,
        ignore_index=True,
    )

    importance_all_path = (
        AGGREGATE_DIR
        / "shadow_mia_feature_importance_all_seeds.csv"
    )

    importance_df.to_csv(
        importance_all_path,
        index=False,
    )

    importance_summary = (
        importance_df
        .groupby("feature")[
            "importance"
        ]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        .reset_index()
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    importance_summary_path = (
        AGGREGATE_DIR
        / "shadow_mia_feature_importance_summary.csv"
    )

    importance_summary.to_csv(
        importance_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 70)
    print("MULTI-SEED SHADOW-MIA SUMMARY")
    print("=" * 70)

    for _, row in (
        summary_df.iterrows()
    ):

        print(
            f"\n{row['attacker']}"
        )

        print(
            f"  ROC-AUC : "
            f"{row['roc_auc_mean']:.6f} "
            f"± "
            f"{row['roc_auc_std']:.6f}"
        )

        print(
            f"  Accuracy: "
            f"{row['accuracy_mean']:.6f} "
            f"± "
            f"{row['accuracy_std']:.6f}"
        )

        print(
            f"  Precision: "
            f"{row['precision_mean']:.6f} "
            f"± "
            f"{row['precision_std']:.6f}"
        )

        print(
            f"  Recall   : "
            f"{row['recall_mean']:.6f} "
            f"± "
            f"{row['recall_std']:.6f}"
        )

        print(
            f"  F1       : "
            f"{row['f1_mean']:.6f} "
            f"± "
            f"{row['f1_std']:.6f}"
        )

        print(
            f"  AUC range: "
            f"{row['roc_auc_min']:.6f}"
            f"–"
            f"{row['roc_auc_max']:.6f}"
        )

    print(
        "\nMean RF feature importance:"
    )

    print(
        importance_summary[
            [
                "feature",
                "mean",
                "std",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nAggregate files:"
    )

    print(
        " ",
        all_seed_path,
    )

    print(
        " ",
        summary_path,
    )

    print(
        " ",
        importance_all_path,
    )

    print(
        " ",
        importance_summary_path,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "GEMMA MULTI-SEED "
        "SHOKRI-STYLE SHADOW-MODEL MIA"
    )
    print("=" * 70)

    print(
        "\nLoRA training seeds:",
        SEEDS,
    )

    print(
        "Fixed attack-model seed:",
        ATTACK_SEED,
    )

    print(
        "\nMethodology:"
        "\n  Attacker training = shadow models only"
        "\n  Target labels      = evaluation only"
        "\n  Target examples    = never used to fit attacker"
    )

    all_results = []
    all_importance = []

    # --------------------------------------------------------
    # Four independent LoRA-training seeds
    # --------------------------------------------------------

    for seed in SEEDS:

        results, importance = (
            run_seed(seed)
        )

        all_results.append(
            results
        )

        all_importance.append(
            importance
        )

    # --------------------------------------------------------
    # Aggregate robustness statistics
    # --------------------------------------------------------

    aggregate_results(
        all_results,
        all_importance,
    )

    print("\n" + "=" * 70)
    print(
        "MULTI-SEED SHADOW ATTACK COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()