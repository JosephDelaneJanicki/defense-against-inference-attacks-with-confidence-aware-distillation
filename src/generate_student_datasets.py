import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ================================
# Config
# ================================
RESULTS_DIR = "results"
OUTPUT_DIR = "data/student_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEACHER_MODEL_PATH = "results/model_a_teacher.pt"

TEMPERATURE = 2.0
CLIP_MAX = 0.85
FILTER_CONFIDENCE_THRESHOLD = 0.95

SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================
# Teacher Architecture
# Same as overfit Model A v2
# ================================
class TeacherMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


# ================================
# Load Adult Dataset
# ================================
adult = fetch_ucirepo(id=2)
X = adult.data.features.copy()
y = adult.data.targets.copy()

if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

y = y.astype(str).str.strip().str.replace(".", "", regex=False)
y = (y == ">50K").astype(int)


# ================================
# Raw Feature Anonymization / Generalization
# ================================
def anonymize_adult_features(X_df):
    X_anon = X_df.copy()

    # Age buckets
    X_anon["age"] = pd.cut(
        X_anon["age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["0-25", "26-35", "36-45", "46-55", "56-65", "66+"],
        include_lowest=True,
    ).astype(str)

    # Hours per week buckets
    X_anon["hours-per-week"] = pd.cut(
        X_anon["hours-per-week"],
        bins=[0, 20, 35, 40, 50, 60, 100],
        labels=["0-20", "21-35", "36-40", "41-50", "51-60", "61+"],
        include_lowest=True,
    ).astype(str)

    # Capital gain/loss generalized into present/none
    X_anon["capital-gain"] = (X_anon["capital-gain"] > 0).map({
        True: "gain_present",
        False: "no_gain",
    })

    X_anon["capital-loss"] = (X_anon["capital-loss"] > 0).map({
        True: "loss_present",
        False: "no_loss",
    })

    # fnlwgt is a census weighting feature, not directly useful for interpretability
    # Generalize into quantile buckets
    X_anon["fnlwgt"] = pd.qcut(
        X_anon["fnlwgt"],
        q=5,
        duplicates="drop",
        labels=False,
    ).astype(str)

    # Education num generalized
    X_anon["education-num"] = pd.cut(
        X_anon["education-num"],
        bins=[0, 8, 12, 16],
        labels=["low", "middle", "high"],
        include_lowest=True,
    ).astype(str)

    # Native country generalized
    X_anon["native-country"] = X_anon["native-country"].astype(str).apply(
        lambda v: "United-States" if v == "United-States" else "Other"
    )

    # Workclass generalized
    def simplify_workclass(v):
        v = str(v)
        if "Private" in v:
            return "Private"
        if "gov" in v or "Gov" in v:
            return "Government"
        if "Self" in v:
            return "Self-employed"
        return "Other"

    X_anon["workclass"] = X_anon["workclass"].apply(simplify_workclass)

    # Marital status generalized
    X_anon["marital-status"] = X_anon["marital-status"].astype(str).apply(
        lambda v: "Married" if "Married" in v else "Not-married"
    )

    return X_anon


# ================================
# Preprocessing Helper
# ================================
def build_preprocessor(X_df):
    categorical_cols = X_df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_cols = X_df.select_dtypes(
        exclude=["object", "category", "string"]
    ).columns.tolist()

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

    return preprocessor


def to_dense_array(X_processed):
    return X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed


# ================================
# Split Data
# Same split pattern as Model A
# ================================
X_temp, X_test_df, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train_df, X_val_df, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.2,
    random_state=42,
    stratify=y_temp,
)


# ================================
# Original Feature Processing
# ================================
original_preprocessor = build_preprocessor(X_train_df)

X_train_original = original_preprocessor.fit_transform(X_train_df)
X_val_original = original_preprocessor.transform(X_val_df)
X_test_original = original_preprocessor.transform(X_test_df)

X_train_original = to_dense_array(X_train_original)
X_val_original = to_dense_array(X_val_original)
X_test_original = to_dense_array(X_test_original)


# ================================
# Anonymized Feature Processing
# ================================
X_train_anon_df = anonymize_adult_features(X_train_df)
X_val_anon_df = anonymize_adult_features(X_val_df)
X_test_anon_df = anonymize_adult_features(X_test_df)

anon_preprocessor = build_preprocessor(X_train_anon_df)

X_train_anon = anon_preprocessor.fit_transform(X_train_anon_df)
X_val_anon = anon_preprocessor.transform(X_val_anon_df)
X_test_anon = anon_preprocessor.transform(X_test_anon_df)

X_train_anon = to_dense_array(X_train_anon)
X_val_anon = to_dense_array(X_val_anon)
X_test_anon = to_dense_array(X_test_anon)


# ================================
# Load Teacher Model
# ================================
input_dim = X_train_original.shape[1]
teacher = TeacherMLP(input_dim).to(device)
teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=device))
teacher.eval()

print("Loaded teacher:", TEACHER_MODEL_PATH)
print("Teacher input dim:", input_dim)


# ================================
# Teacher Outputs
# ================================
def get_teacher_outputs(X_array):
    X_tensor = torch.tensor(X_array, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = teacher(X_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, preds = torch.max(probs, dim=1)

    return (
        logits.cpu().numpy(),
        probs.cpu().numpy(),
        confidence.cpu().numpy(),
        preds.cpu().numpy(),
    )


train_logits, train_probs, train_conf, train_preds = get_teacher_outputs(X_train_original)
val_logits, val_probs, val_conf, val_preds = get_teacher_outputs(X_val_original)
test_logits, test_probs, test_conf, test_preds = get_teacher_outputs(X_test_original)


# ================================
# Label Processing
# ================================
def temperature_smooth_logits(logits, temperature):
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    probs = torch.softmax(logits_tensor / temperature, dim=1)
    return probs.numpy()


def clip_probabilities(probs, clip_max):
    clipped = np.clip(probs, 1.0 - clip_max, clip_max)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return clipped


label_variants = {
    "raw_soft": {
        "type": "soft",
        "train": train_probs,
        "val": val_probs,
        "test": test_probs,
    },
    "temp_soft": {
        "type": "soft",
        "train": temperature_smooth_logits(train_logits, TEMPERATURE),
        "val": temperature_smooth_logits(val_logits, TEMPERATURE),
        "test": temperature_smooth_logits(test_logits, TEMPERATURE),
    },
    "clip_soft": {
        "type": "soft",
        "train": clip_probabilities(train_probs, CLIP_MAX),
        "val": clip_probabilities(val_probs, CLIP_MAX),
        "test": clip_probabilities(test_probs, CLIP_MAX),
    },
    "hard": {
        "type": "hard",
        "train": train_preds,
        "val": val_preds,
        "test": test_preds,
    },
}


# ================================
# Feature Variants
# ================================
feature_variants = {
    "original": {
        "train": X_train_original,
        "val": X_val_original,
        "test": X_test_original,
        "filter": None,
    },
    "anon": {
        "train": X_train_anon,
        "val": X_val_anon,
        "test": X_test_anon,
        "filter": None,
    },
    "filtered": {
        "train": X_train_original,
        "val": X_val_original,
        "test": X_test_original,
        "filter": train_conf <= FILTER_CONFIDENCE_THRESHOLD,
    },
}


# ================================
# Save Global Reference Data
# ================================
np.save(os.path.join(OUTPUT_DIR, "y_train_true.npy"), y_train.to_numpy())
np.save(os.path.join(OUTPUT_DIR, "y_val_true.npy"), y_val.to_numpy())
np.save(os.path.join(OUTPUT_DIR, "y_test_true.npy"), y_test.to_numpy())

np.save(os.path.join(OUTPUT_DIR, "teacher_train_probs.npy"), train_probs)
np.save(os.path.join(OUTPUT_DIR, "teacher_val_probs.npy"), val_probs)
np.save(os.path.join(OUTPUT_DIR, "teacher_test_probs.npy"), test_probs)

np.save(os.path.join(OUTPUT_DIR, "teacher_train_confidence.npy"), train_conf)
np.save(os.path.join(OUTPUT_DIR, "teacher_val_confidence.npy"), val_conf)
np.save(os.path.join(OUTPUT_DIR, "teacher_test_confidence.npy"), test_conf)

np.save(os.path.join(OUTPUT_DIR, "teacher_train_preds.npy"), train_preds)
np.save(os.path.join(OUTPUT_DIR, "teacher_val_preds.npy"), val_preds)
np.save(os.path.join(OUTPUT_DIR, "teacher_test_preds.npy"), test_preds)


# ================================
# Generate Student Dataset Grid
# ================================
manifest = []

for label_name, label_data in label_variants.items():
    for feature_name, feature_data in feature_variants.items():

        variant_name = f"student_{label_name}_{feature_name}"
        variant_dir = os.path.join(OUTPUT_DIR, variant_name)
        os.makedirs(variant_dir, exist_ok=True)

        train_filter = feature_data["filter"]

        X_train_variant = feature_data["train"]
        y_train_variant = label_data["train"]
        y_true_train_variant = y_train.to_numpy()
        teacher_train_conf_variant = train_conf

        if train_filter is not None:
            X_train_variant = X_train_variant[train_filter]
            y_train_variant = y_train_variant[train_filter]
            y_true_train_variant = y_true_train_variant[train_filter]
            teacher_train_conf_variant = teacher_train_conf_variant[train_filter]

        np.save(os.path.join(variant_dir, "X_train.npy"), X_train_variant)
        np.save(os.path.join(variant_dir, "X_val.npy"), feature_data["val"])
        np.save(os.path.join(variant_dir, "X_test.npy"), feature_data["test"])

        np.save(os.path.join(variant_dir, "y_train_student.npy"), y_train_variant)
        np.save(os.path.join(variant_dir, "y_val_student.npy"), label_data["val"])
        np.save(os.path.join(variant_dir, "y_test_student.npy"), label_data["test"])

        np.save(os.path.join(variant_dir, "y_train_true.npy"), y_true_train_variant)
        np.save(os.path.join(variant_dir, "y_val_true.npy"), y_val.to_numpy())
        np.save(os.path.join(variant_dir, "y_test_true.npy"), y_test.to_numpy())

        np.save(
            os.path.join(variant_dir, "teacher_train_confidence.npy"),
            teacher_train_conf_variant,
        )

        metadata = {
            "variant_name": variant_name,
            "label_mode": label_name,
            "label_type": label_data["type"],
            "feature_mode": feature_name,
            "num_train_samples": int(len(X_train_variant)),
            "num_val_samples": int(len(feature_data["val"])),
            "num_test_samples": int(len(feature_data["test"])),
            "input_dim": int(X_train_variant.shape[1]),
            "temperature": TEMPERATURE if label_name == "temp_soft" else None,
            "clip_max": CLIP_MAX if label_name == "clip_soft" else None,
            "filter_confidence_threshold": (
                FILTER_CONFIDENCE_THRESHOLD if feature_name == "filtered" else None
            ),
        }

        with open(os.path.join(variant_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        manifest.append(metadata)

        print(
            f"Saved {variant_name} | "
            f"X_train: {X_train_variant.shape} | "
            f"labels: {label_data['type']}"
        )


# ================================
# Save Manifest
# ================================
manifest_df = pd.DataFrame(manifest)
manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
manifest_df.to_csv(manifest_path, index=False)

print("\nDone generating student datasets.")
print("Manifest:", manifest_path)
print(manifest_df)