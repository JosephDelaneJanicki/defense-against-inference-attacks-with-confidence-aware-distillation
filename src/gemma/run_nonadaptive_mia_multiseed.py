from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# FIXED EXPERIMENT CONFIGURATION
# ============================================================

SEEDS = [42, 123, 456, 789]

# Keep attacker randomness fixed across every defense and seed.
ATTACKER_SEED = 42

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
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Run the fixed non-adaptive Shokri-style membership "
        "inference attack against a defended Gemma student "
        "across all available seeds."
    )
)

parser.add_argument(
    "defense",
    type=str,
    help=(
        "Defense/student directory name under "
        "results/gemma/defense/seed_<seed>/. "
        "Examples: raw_soft, temp_soft, clip_soft, "
        "confidence_filtered"
    ),
)

args = parser.parse_args()

DEFENSE = args.defense.strip()

if not DEFENSE:
    raise ValueError("Defense name cannot be empty.")


# ============================================================
# PATH HELPERS
# ============================================================

def shadow_scores_path(seed):
    """
    Original UNDEFENDED shadow-model observations.

    These remain fixed for every non-adaptive defense attack.
    """
    return Path(
        f"results/gemma/multiseed/seed_{seed}/shadows/"
        "all_shadow_membership_scores.csv"
    )


def target_scores_path(seed):
    """
    Membership features produced by the defended target student.
    """
    return Path(
        f"results/gemma/defense/seed_{seed}/{DEFENSE}/"
        "membership_scores.csv"
    )


def output_dir(seed):
    """
    Store the attack alongside the defended student while
    retaining descriptive filenames.
    """
    return Path(
        f"results/gemma/defense/seed_{seed}/{DEFENSE}/"
        "nonadaptive_mia"
    )


def aggregate_dir():
    return Path(
        f"results/gemma/defense/{DEFENSE}/"
        "nonadaptive_mia_multiseed"
    )


def file_prefix(seed):
    """
    Every generated result explicitly identifies:
      - attack type
      - defense
      - seed
      - target model type
    """
    return (
        f"nonadaptive_mia_{DEFENSE}_"
        f"seed_{seed}_defended_student"
    )


# ============================================================
# HELPERS
# ============================================================

def find_membership_column(df):
    candidates = [
        "membership",
        "member",
        "is_member",
        "label",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find membership column. "
        f"Available columns: {list(df.columns)}"
    )


def validate_features(df, dataset_name):
    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing features: {missing}"
        )


def evaluate(
    name,
    model,
    X_shadow,
    y_shadow,
    X_target,
    y_target,
    out_dir,
    prefix,
):
    """
    Train on ordinary undefended shadow-model observations and
    evaluate on the defended target student.
    """

    model.fit(
        X_shadow,
        y_shadow,
    )

    predictions = model.predict(
        X_target
    )

    probabilities = model.predict_proba(
        X_target
    )[:, 1]

    metrics = {
        "attacker": name,
        "accuracy": accuracy_score(
            y_target,
            predictions,
        ),
        "precision": precision_score(
            y_target,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_target,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_target,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_target,
            probabilities,
        ),
    }

    prediction_df = pd.DataFrame({
        "membership": y_target,
        "prediction": predictions,
        "membership_probability": probabilities,
    })

    prediction_df.to_csv(
        out_dir
        / f"{prefix}_{name}_predictions.csv",
        index=False,
    )

    return metrics, model


# ============================================================
# SINGLE-SEED EXPERIMENT
# ============================================================

def run_seed(seed):

    shadow_path = shadow_scores_path(seed)
    target_path = target_scores_path(seed)

    out_dir = output_dir(seed)
    prefix = file_prefix(seed)

    print("\n" + "=" * 80)
    print(
        f"SEED {seed} | DEFENSE: {DEFENSE}"
    )
    print("=" * 80)

    print("\nAttack training population:")
    print(
        f"  Undefended shadows:"
        f"\n  {shadow_path}"
    )

    print("\nAttack target:")
    print(
        f"  Defended student:"
        f"\n  {target_path}"
    )

    # --------------------------------------------------------
    # Availability checks
    # --------------------------------------------------------

    if not shadow_path.exists():
        print("\n[SKIP]")
        print(
            "Original undefended shadow scores "
            "do not exist."
        )
        return None

    if not target_path.exists():
        print("\n[SKIP]")
        print(
            f"{DEFENSE} defended target scores "
            "do not exist yet."
        )
        return None

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    shadow_df = pd.read_csv(
        shadow_path
    )

    target_df = pd.read_csv(
        target_path
    )

    shadow_label = find_membership_column(
        shadow_df
    )

    target_label = find_membership_column(
        target_df
    )

    validate_features(
        shadow_df,
        "shadow dataset",
    )

    validate_features(
        target_df,
        "target dataset",
    )

    print("\nDataset sizes:")
    print(
        f"  Shadow observations: "
        f"{len(shadow_df)}"
    )
    print(
        f"  Target observations: "
        f"{len(target_df)}"
    )

    print("\nShadow membership distribution:")
    print(
        shadow_df[shadow_label]
        .value_counts()
        .sort_index()
    )

    print("\nTarget membership distribution:")
    print(
        target_df[target_label]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Matrices
    # --------------------------------------------------------

    X_shadow = shadow_df[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )

    y_shadow = shadow_df[
        shadow_label
    ].to_numpy(
        dtype=np.int64
    )

    X_target = target_df[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )

    y_target = target_df[
        target_label
    ].to_numpy(
        dtype=np.int64
    )

    if not np.isfinite(
        X_shadow
    ).all():
        raise ValueError(
            f"Seed {seed}: shadow features "
            "contain NaN or Inf."
        )

    if not np.isfinite(
        X_target
    ).all():
        raise ValueError(
            f"Seed {seed}: target features "
            "contain NaN or Inf."
        )

    # --------------------------------------------------------
    # Logistic Regression
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
                random_state=ATTACKER_SEED,
            ),
        ),
    ])

    logistic_metrics, _ = evaluate(
        name="logistic",
        model=logistic,
        X_shadow=X_shadow,
        y_shadow=y_shadow,
        X_target=X_target,
        y_target=y_target,
        out_dir=out_dir,
        prefix=prefix,
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=ATTACKER_SEED,
        n_jobs=-1,
    )

    rf_metrics, rf_model = evaluate(
        name="random_forest",
        model=random_forest,
        X_shadow=X_shadow,
        y_shadow=y_shadow,
        X_target=X_target,
        y_target=y_target,
        out_dir=out_dir,
        prefix=prefix,
    )

    # --------------------------------------------------------
    # Random Forest feature importance
    # --------------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance":
            rf_model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        out_dir
        / f"{prefix}_random_forest_feature_importance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Per-seed summary
    # --------------------------------------------------------

    results = pd.DataFrame([
        logistic_metrics,
        rf_metrics,
    ])

    results.insert(
        0,
        "seed",
        seed,
    )

    results.insert(
        1,
        "defense",
        DEFENSE,
    )

    results.insert(
        2,
        "attack_type",
        "nonadaptive",
    )

    results.insert(
        3,
        "shadow_model_type",
        "undefended_lora",
    )

    results.insert(
        4,
        "target_model_type",
        f"{DEFENSE}_defended_student",
    )

    summary_path = (
        out_dir
        / f"{prefix}_summary.csv"
    )

    results.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # JSON metadata
    # --------------------------------------------------------

    metadata = {
        "seed": seed,
        "defense": DEFENSE,
        "attack_type": "nonadaptive",
        "threat_model": (
            "Attacker trained on undefended LoRA "
            "shadow outputs and evaluated on the "
            "defended target student."
        ),
        "shadow_model_type":
            "undefended_lora",
        "target_model_type":
            f"{DEFENSE}_defended_student",
        "attacker_seed":
            ATTACKER_SEED,
        "shadow_scores":
            str(shadow_path),
        "target_scores":
            str(target_path),
        "shadow_observations":
            int(len(shadow_df)),
        "target_observations":
            int(len(target_df)),
        "features":
            FEATURES,
        "results": {
            "logistic":
                logistic_metrics,
            "random_forest":
                rf_metrics,
        },
    }

    with open(
        out_dir
        / f"{prefix}_metadata.json",
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nNON-ADAPTIVE RESULTS")

    print(
        f"  Logistic ROC-AUC : "
        f"{logistic_metrics['roc_auc']:.6f}"
    )

    print(
        f"  RF ROC-AUC       : "
        f"{rf_metrics['roc_auc']:.6f}"
    )

    print("\nSaved:")
    print(
        f"  {summary_path}"
    )

    return results


# ============================================================
# MULTI-SEED PIPELINE
# ============================================================

print("=" * 80)
print(
    "DEFENSE-AGNOSTIC NON-ADAPTIVE "
    "MIA — MULTI-SEED"
)
print("=" * 80)

print(
    f"\nDefense under attack: {DEFENSE}"
)

print(
    "\nThreat model:\n"
    "  Train attacker on UNDEFENDED shadow LoRAs.\n"
    f"  Evaluate attacker on {DEFENSE} defended students.\n"
)

print(
    "Fixed attacker seed: "
    f"{ATTACKER_SEED}"
)

all_results = []


for seed in SEEDS:

    result = run_seed(
        seed
    )

    if result is not None:
        all_results.append(
            result
        )


# ============================================================
# AGGREGATE AVAILABLE SEEDS
# ============================================================

if not all_results:
    raise RuntimeError(
        "No seed currently has both the "
        "undefended shadow scores and the "
        f"{DEFENSE} defended target scores."
    )


combined = pd.concat(
    all_results,
    ignore_index=True,
)


agg_dir = aggregate_dir()

agg_dir.mkdir(
    parents=True,
    exist_ok=True,
)


completed_seeds = sorted(
    combined["seed"]
    .unique()
    .tolist()
)


seed_tag = "-".join(
    str(seed)
    for seed in completed_seeds
)


aggregate_prefix = (
    f"nonadaptive_mia_{DEFENSE}_"
    f"seeds_{seed_tag}_defended_students"
)


# ============================================================
# SAVE COMBINED RESULTS
# ============================================================

combined_path = (
    agg_dir
    / f"{aggregate_prefix}_all_results.csv"
)

combined.to_csv(
    combined_path,
    index=False,
)


# ============================================================
# AGGREGATE STATISTICS
# ============================================================

summary_rows = []


for attacker in [
    "logistic",
    "random_forest",
]:

    subset = combined[
        combined["attacker"]
        == attacker
    ]

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:

        values = subset[
            metric
        ]

        summary_rows.append({
            "defense":
                DEFENSE,
            "attack_type":
                "nonadaptive",
            "target_model_type":
                f"{DEFENSE}_defended_student",
            "attacker":
                attacker,
            "metric":
                metric,
            "n_seeds":
                len(values),
            "seeds":
                seed_tag,
            "mean":
                values.mean(),
            "std":
                (
                    values.std(ddof=1)
                    if len(values) > 1
                    else np.nan
                ),
            "min":
                values.min(),
            "max":
                values.max(),
        })


aggregate_summary = pd.DataFrame(
    summary_rows
)


aggregate_summary_path = (
    agg_dir
    / f"{aggregate_prefix}_aggregate_summary.csv"
)


aggregate_summary.to_csv(
    aggregate_summary_path,
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print(
    "NON-ADAPTIVE MULTI-SEED "
    "PIPELINE COMPLETE"
)
print("=" * 80)


print(
    f"\nDefense: {DEFENSE}"
)


print("\nCompleted seeds:")

print(
    "  "
    + ", ".join(
        str(seed)
        for seed in completed_seeds
    )
)


print("\nROC-AUC BY SEED")


auc_table = combined[
    [
        "seed",
        "defense",
        "attacker",
        "roc_auc",
    ]
]


print(
    auc_table.to_string(
        index=False
    )
)


print("\nAGGREGATE ROC-AUC")


auc_summary = aggregate_summary[
    aggregate_summary["metric"]
    == "roc_auc"
]


for _, row in auc_summary.iterrows():

    if row["n_seeds"] > 1:

        print(
            f"  "
            f"{row['attacker']:15s}: "
            f"{row['mean']:.6f} "
            f"+/- {row['std']:.6f} "
            f"(n={int(row['n_seeds'])})"
        )

    else:

        print(
            f"  "
            f"{row['attacker']:15s}: "
            f"{row['mean']:.6f} "
            f"(n=1; no cross-seed std yet)"
        )


print("\nSaved aggregate results:")

print(
    f"  {combined_path}"
)

print(
    f"  {aggregate_summary_path}"
)