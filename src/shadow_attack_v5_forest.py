import os
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# ================================
# Config
# ================================
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
NUM_SHADOW_MODELS = 12

np.random.seed(SEED)


# ================================
# Target Regimes (same as v3/v4)
# ================================
TARGET_REGIMES = {
    "Overfit v2": {
        "shadow_epochs": 80,
        "member_probs": "model_a_train_probs.npy",
        "nonmember_probs": "model_a_test_probs.npy",
    },
    "Early Stop 5ep": {
        "shadow_epochs": 5,
        "member_probs": "model_a_early_stop_5ep_train_probs.npy",
        "nonmember_probs": "model_a_early_stop_5ep_test_probs.npy",
    },
    "Realistic": {
        "shadow_epochs": 120,
        "member_probs": "model_a_realistic_train_probs.npy",
        "nonmember_probs": "model_a_realistic_test_probs.npy",
    },
}


# ================================
# Load Adult Dataset (same pipeline)
# ================================
adult = fetch_ucirepo(id=2)
X = adult.data.features.copy()
y = adult.data.targets.copy()

if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

y = y.astype(str).str.strip().str.replace(".", "", regex=False)
y = (y == ">50K").astype(int)
y_np = y.to_numpy()

categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "category", "string"]).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

X_processed = preprocessor.fit_transform(X)
X_processed = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed


# ================================
# Rebuild splits
# ================================
X_temp, X_test_df, y_temp, y_test_split = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_df, X_val_df, y_train_split, y_val_split = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

target_y_train_np = y_train_split.to_numpy()
target_y_test_np = y_test_split.to_numpy()


# ================================
# Feature extraction (reuse v3)
# ================================
def extract_features(probs, true_labels):
    eps = 1e-10

    probs = np.asarray(probs)
    true_labels = np.asarray(true_labels).astype(int)

    # Base probabilities
    p0 = probs[:, 0]
    p1 = probs[:, 1]

    # Sorted probabilities (Shokri-style signal)
    sorted_desc = np.sort(probs, axis=1)[:, ::-1]
    top1 = sorted_desc[:, 0]
    top2 = sorted_desc[:, 1]

    # Core behavioral features
    max_conf = np.max(probs, axis=1)
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    margin = top1 - top2

    # Prediction behavior
    pred_labels = np.argmax(probs, axis=1)
    correct = (pred_labels == true_labels).astype(float)

    # Loss signal (VERY important for MIA)
    true_label_probs = probs[np.arange(len(probs)), true_labels]
    loss = -np.log(true_label_probs + eps)

    # Final feature vector (v3)
    features = np.column_stack([
        p0,         
        p1,
        top1,          
        top2,
        max_conf,
        entropy,
        margin,
        loss,
        correct,
        pred_labels   
    ])

    return features, pred_labels

# ================================
# Train RF attack model
# ================================
def train_rf_attack(X, y):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def evaluate_rf(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "auc": roc_auc_score(y, probs),
    }


# ================================
# Main
# ================================
results = []

for name, cfg in TARGET_REGIMES.items():
    print("\n" + "="*60)
    print(f"Random Forest Attack on {name}")
    print("="*60)

    member_probs = np.load(os.path.join(RESULTS_DIR, cfg["member_probs"]))
    nonmember_probs = np.load(os.path.join(RESULTS_DIR, cfg["nonmember_probs"]))

    member_features, _ = extract_features(member_probs, target_y_train_np)
    nonmember_features, _ = extract_features(nonmember_probs, target_y_test_np)

    X_attack = np.vstack([member_features, nonmember_features])
    y_attack = np.concatenate([
        np.ones(len(member_features)),
        np.zeros(len(nonmember_features))
    ])

    # balance
    n = min(len(member_features), len(nonmember_features))
    rng = np.random.default_rng(SEED)

    idx_m = rng.choice(len(member_features), n, replace=False)
    idx_n = rng.choice(len(nonmember_features), n, replace=False)

    X_bal = np.vstack([member_features[idx_m], nonmember_features[idx_n]])
    y_bal = np.concatenate([np.ones(n), np.zeros(n)])

    model = train_rf_attack(X_bal, y_bal)

    metrics = evaluate_rf(model, X_bal, y_bal)

    print(metrics)

    metrics["model"] = name
    results.append(metrics)


df = pd.DataFrame(results)
out_path = os.path.join(RESULTS_DIR, "shadow_attack_v5_forest.csv")
df.to_csv(out_path, index=False)

print("\nSaved:", out_path)
print(df)