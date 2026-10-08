from pathlib import Path
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
# CONFIGURATION
# ============================================================

SEED = 42

SHADOW_SCORES = Path(
    "results/gemma/defense/seed_42/raw_soft/shadows/"
    "all_shadow_membership_scores.csv"
)

TARGET_SCORES = Path(
    "results/gemma/defense/seed_42/raw_soft/"
    "membership_scores.csv"
)

OUTPUT_DIR = Path(
    "results/gemma/defense/seed_42/raw_soft/adaptive_mia"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
# HELPERS
# ============================================================

def find_membership_column(df: pd.DataFrame) -> str:
    """
    Find the membership-label column while tolerating the naming
    conventions used by the earlier scoring scripts.
    """

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
        "Could not find membership column.\n"
        f"Available columns: {list(df.columns)}"
    )


def evaluate_model(name, model, X_train, y_train, X_target, y_target):
    print("\n" + "=" * 72)
    print(name.upper())
    print("=" * 72)

    model.fit(X_train, y_train)

    pred = model.predict(X_target)
    prob = model.predict_proba(X_target)[:, 1]

    metrics = {
        "attacker": name,
        "accuracy": accuracy_score(y_target, pred),
        "precision": precision_score(
            y_target,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_target,
            pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_target,
            pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_target, prob),
    }

    for key, value in metrics.items():
        if key != "attacker":
            print(f"{key:10s}: {value:.6f}")

    predictions = pd.DataFrame({
        "membership": y_target,
        "prediction": pred,
        "membership_probability": prob,
    })

    predictions.to_csv(
        OUTPUT_DIR / f"{name}_target_predictions.csv",
        index=False,
    )

    return metrics, model


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("RAW-SOFT DEFENSE-AWARE / ADAPTIVE SHOKRI MIA")
print("=" * 72)

print("\nShadow attack-training data:")
print(f"  {SHADOW_SCORES}")

print("\nDefended target evaluation data:")
print(f"  {TARGET_SCORES}")


shadow_df = pd.read_csv(SHADOW_SCORES)
target_df = pd.read_csv(TARGET_SCORES)


shadow_label_col = find_membership_column(shadow_df)
target_label_col = find_membership_column(target_df)


print("\nDetected membership columns:")
print(f"  Shadow: {shadow_label_col}")
print(f"  Target: {target_label_col}")


# ============================================================
# SANITY CHECKS
# ============================================================

missing_shadow = [
    feature
    for feature in FEATURES
    if feature not in shadow_df.columns
]

missing_target = [
    feature
    for feature in FEATURES
    if feature not in target_df.columns
]

if missing_shadow:
    raise ValueError(
        f"Shadow dataset missing features: {missing_shadow}"
    )

if missing_target:
    raise ValueError(
        f"Target dataset missing features: {missing_target}"
    )


print("\nDataset sizes:")
print(f"  Shadow observations: {len(shadow_df)}")
print(f"  Target observations: {len(target_df)}")

print("\nShadow membership distribution:")
print(shadow_df[shadow_label_col].value_counts().sort_index())

print("\nTarget membership distribution:")
print(target_df[target_label_col].value_counts().sort_index())


# ============================================================
# BUILD MATRICES
# ============================================================

X_shadow = shadow_df[FEATURES].to_numpy(dtype=np.float64)
y_shadow = shadow_df[shadow_label_col].to_numpy(dtype=np.int64)

X_target = target_df[FEATURES].to_numpy(dtype=np.float64)
y_target = target_df[target_label_col].to_numpy(dtype=np.int64)


if not np.isfinite(X_shadow).all():
    raise ValueError(
        "Shadow feature matrix contains NaN or infinite values."
    )

if not np.isfinite(X_target).all():
    raise ValueError(
        "Target feature matrix contains NaN or infinite values."
    )


# ============================================================
# ATTACK 1: LOGISTIC REGRESSION
# ============================================================

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


logistic_metrics, logistic_model = evaluate_model(
    name="logistic",
    model=logistic,
    X_train=X_shadow,
    y_train=y_shadow,
    X_target=X_target,
    y_target=y_target,
)


# ============================================================
# ATTACK 2: RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)


rf_metrics, rf_model = evaluate_model(
    name="random_forest",
    model=random_forest,
    X_train=X_shadow,
    y_train=y_shadow,
    X_target=X_target,
    y_target=y_target,
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf_model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
)

importance_df.to_csv(
    OUTPUT_DIR / "random_forest_feature_importance.csv",
    index=False,
)


print("\n" + "=" * 72)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 72)

for _, row in importance_df.iterrows():
    print(
        f"{row['feature']:20s}: "
        f"{row['importance']:.6f}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

results_df = pd.DataFrame([
    logistic_metrics,
    rf_metrics,
])

results_df.to_csv(
    OUTPUT_DIR / "adaptive_mia_summary.csv",
    index=False,
)


summary_json = {
    "seed": SEED,
    "attack_type": "defense-aware adaptive shadow-model MIA",
    "defense": "raw_soft_distillation",
    "shadow_scores": str(SHADOW_SCORES),
    "target_scores": str(TARGET_SCORES),
    "shadow_observations": int(len(shadow_df)),
    "target_observations": int(len(target_df)),
    "features": FEATURES,
    "results": {
        row["attacker"]: {
            key: float(value)
            for key, value in row.items()
            if key != "attacker"
        }
        for row in [
            logistic_metrics,
            rf_metrics,
        ]
    },
}

with open(
    OUTPUT_DIR / "adaptive_mia_summary.json",
    "w",
) as f:
    json.dump(
        summary_json,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 72)
print("ADAPTIVE RAW-SOFT MIA COMPLETE")
print("=" * 72)

print("\nResults:")

print(
    f"  Logistic ROC-AUC : "
    f"{logistic_metrics['roc_auc']:.6f}"
)

print(
    f"  RF ROC-AUC       : "
    f"{rf_metrics['roc_auc']:.6f}"
)

print("\nOriginal Seed-42 undefended reference:")
print("  Logistic ROC-AUC : 0.545891")
print("  RF ROC-AUC       : 0.544778")

print("\nSaved:")
print(f"  {OUTPUT_DIR / 'adaptive_mia_summary.csv'}")
print(f"  {OUTPUT_DIR / 'adaptive_mia_summary.json'}")
print(
    f"  "
    f"{OUTPUT_DIR / 'random_forest_feature_importance.csv'}"
)