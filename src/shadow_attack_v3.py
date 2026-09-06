import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader


# ================================
# Config
# ================================
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

SEED = 42
NUM_SHADOW_MODELS = 8
BATCH_SIZE = 256
ATTACK_EPOCHS = 50

torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================
# Target Regimes
# ================================
TARGET_REGIMES = {
    "Overfit v2": {
        "shadow_epochs": 60,
        "architecture": "plain",
        "member_probs": "model_a_train_probs.npy",
        "nonmember_probs": "model_a_test_probs.npy",
    },
    "Early Stop 5ep": {
        "shadow_epochs": 5,
        "architecture": "plain",
        "member_probs": "model_a_early_stop_5ep_train_probs.npy",
        "nonmember_probs": "model_a_early_stop_5ep_test_probs.npy",
    },
    "Realistic": {
        "shadow_epochs": 120,
        "architecture": "realistic",
        "member_probs": "model_a_realistic_train_probs.npy",
        "nonmember_probs": "model_a_realistic_test_probs.npy",
    },
}


# ================================
# Load + Preprocess Adult Dataset
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

X_tensor = torch.tensor(X_processed, dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.long)

input_dim = X_tensor.shape[1]

print("Input dim:", input_dim)
print("Dataset size:", len(X_tensor))


# ================================
# Rebuild Target Splits
# ================================
X_temp, X_test_df, y_temp, y_test_split = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train_df, X_val_df, y_train_split, y_val_split = train_test_split(
    X_temp,
    y_temp,
    test_size=0.2,
    random_state=42,
    stratify=y_temp,
)

target_y_train_np = y_train_split.to_numpy()
target_y_test_np = y_test_split.to_numpy()


# ================================
# Models
# ================================
class PlainTeacherMLP(nn.Module):
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


class RealisticTeacherMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class AttackModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def build_shadow_model(architecture):
    if architecture == "plain":
        return PlainTeacherMLP(input_dim).to(DEVICE)
    if architecture == "realistic":
        return RealisticTeacherMLP(input_dim).to(DEVICE)

    raise ValueError(f"Unknown architecture: {architecture}")


# ================================
# Feature Extraction
# ================================
def extract_attack_features(probs, true_labels):
    eps = 1e-10

    probs = np.asarray(probs)
    true_labels = np.asarray(true_labels).astype(int)

    max_conf = np.max(probs, axis=1)
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)

    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    pred_labels = np.argmax(probs, axis=1)
    correct = (pred_labels == true_labels).astype(float)

    true_label_probs = probs[np.arange(len(probs)), true_labels]
    loss = -np.log(true_label_probs + eps)

    # Include both original and sorted probability view.
    sorted_desc = np.sort(probs, axis=1)[:, ::-1]

    features = np.column_stack([
        probs[:, 0],
        probs[:, 1],
        sorted_desc[:, 0],
        sorted_desc[:, 1],
        max_conf,
        entropy,
        margin,
        loss,
        correct,
        pred_labels,
    ])

    return features, pred_labels


# ================================
# Training Helpers
# ================================
def train_shadow_model(model, train_loader, epochs, architecture):
    criterion = nn.CrossEntropyLoss()

    if architecture == "realistic":
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epochs):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model


def get_probabilities(model, X_data):
    model.eval()

    loader = DataLoader(
        TensorDataset(X_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_probs = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def balance_binary_data(member_features, nonmember_features):
    n = min(len(member_features), len(nonmember_features))

    if n == 0:
        return None, None

    rng = np.random.default_rng(SEED)

    member_idx = rng.choice(len(member_features), size=n, replace=False)
    nonmember_idx = rng.choice(len(nonmember_features), size=n, replace=False)

    member_features = member_features[member_idx]
    nonmember_features = nonmember_features[nonmember_idx]

    X_attack = np.vstack([member_features, nonmember_features])
    y_attack = np.concatenate([
        np.ones(len(member_features)),
        np.zeros(len(nonmember_features)),
    ])

    return X_attack, y_attack


def train_single_attack_model(X_attack, y_attack, class_id, regime_name):
    X_train, X_val, y_train, y_val = train_test_split(
        X_attack,
        y_attack,
        test_size=0.2,
        random_state=SEED,
        stratify=y_attack,
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = AttackModel(input_dim=X_attack.shape[1]).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_auc = -1
    best_state = None

    for epoch in range(ATTACK_EPOCHS):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(DEVICE))
            val_scores = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()

        val_auc = roc_auc_score(y_val.numpy(), val_scores)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"Class-specific attack model | Regime: {regime_name} | "
        f"Predicted class: {class_id} | Best Val AUC: {best_auc:.4f}"
    )

    return model


def train_class_specific_attacks(features, pred_classes, membership_labels, regime_name):
    attack_models = {}

    for class_id in [0, 1]:
        class_mask = pred_classes == class_id

        class_features = features[class_mask]
        class_membership = membership_labels[class_mask]

        member_features = class_features[class_membership == 1]
        nonmember_features = class_features[class_membership == 0]

        X_attack, y_attack = balance_binary_data(member_features, nonmember_features)

        if X_attack is None:
            print(f"Skipping class {class_id}: insufficient data.")
            continue

        print(
            f"\nTraining attack model for predicted class {class_id} "
            f"with {len(X_attack)} samples."
        )

        attack_models[class_id] = train_single_attack_model(
            X_attack,
            y_attack,
            class_id,
            regime_name,
        )

    return attack_models


def evaluate_class_specific_attack(attack_models, features, pred_classes, membership_labels):
    all_scores = np.zeros(len(features), dtype=np.float32)

    for class_id, attack_model in attack_models.items():
        mask = pred_classes == class_id

        if mask.sum() == 0:
            continue

        X_class = torch.tensor(features[mask], dtype=torch.float32).to(DEVICE)

        attack_model.eval()
        with torch.no_grad():
            logits = attack_model(X_class)
            scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        all_scores[mask] = scores

    preds = (all_scores >= 0.5).astype(int)

    acc = accuracy_score(membership_labels, preds)
    precision = precision_score(membership_labels, preds, zero_division=0)
    recall = recall_score(membership_labels, preds, zero_division=0)
    f1 = f1_score(membership_labels, preds, zero_division=0)
    auc = roc_auc_score(membership_labels, all_scores)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


# ================================
# Main Per-Regime Attack
# ================================
all_results = []

indices = np.arange(len(X_tensor))

for regime_name, config in TARGET_REGIMES.items():
    print("\n" + "#" * 80)
    print(f"Building Shokri-style v3 attack for target regime: {regime_name}")
    print("#" * 80)

    shadow_features_all = []
    shadow_pred_classes_all = []
    shadow_membership_all = []

    for i in range(NUM_SHADOW_MODELS):
        print(f"\nShadow model {i + 1}/{NUM_SHADOW_MODELS} for {regime_name}")

        shadow_train_idx, shadow_test_idx = train_test_split(
            indices,
            test_size=0.5,
            random_state=SEED + i,
            stratify=y_np,
        )

        X_shadow_train = X_tensor[shadow_train_idx]
        y_shadow_train = y_tensor[shadow_train_idx]
        y_shadow_train_np = y_np[shadow_train_idx]

        X_shadow_test = X_tensor[shadow_test_idx]
        y_shadow_test_np = y_np[shadow_test_idx]

        shadow_loader = DataLoader(
            TensorDataset(X_shadow_train, y_shadow_train),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        shadow_model = build_shadow_model(config["architecture"])
        shadow_model = train_shadow_model(
            shadow_model,
            shadow_loader,
            config["shadow_epochs"],
            config["architecture"],
        )

        member_probs = get_probabilities(shadow_model, X_shadow_train)
        nonmember_probs = get_probabilities(shadow_model, X_shadow_test)

        member_features, member_pred_classes = extract_attack_features(
            member_probs,
            y_shadow_train_np,
        )

        nonmember_features, nonmember_pred_classes = extract_attack_features(
            nonmember_probs,
            y_shadow_test_np,
        )

        features = np.vstack([member_features, nonmember_features])
        pred_classes = np.concatenate([member_pred_classes, nonmember_pred_classes])
        membership = np.concatenate([
            np.ones(len(member_features), dtype=int),
            np.zeros(len(nonmember_features), dtype=int),
        ])

        shadow_features_all.append(features)
        shadow_pred_classes_all.append(pred_classes)
        shadow_membership_all.append(membership)

    shadow_features_all = np.vstack(shadow_features_all)
    shadow_pred_classes_all = np.concatenate(shadow_pred_classes_all)
    shadow_membership_all = np.concatenate(shadow_membership_all)

    print("\nShadow attack dataset:")
    print("Features:", shadow_features_all.shape)
    print("Membership ratio:", shadow_membership_all.mean())
    print("Predicted class 0 count:", (shadow_pred_classes_all == 0).sum())
    print("Predicted class 1 count:", (shadow_pred_classes_all == 1).sum())

    attack_models = train_class_specific_attacks(
        shadow_features_all,
        shadow_pred_classes_all,
        shadow_membership_all,
        regime_name,
    )

    # Load target model outputs
    member_probs = np.load(os.path.join(RESULTS_DIR, config["member_probs"]))
    nonmember_probs = np.load(os.path.join(RESULTS_DIR, config["nonmember_probs"]))

    member_features, member_pred_classes = extract_attack_features(
        member_probs,
        target_y_train_np,
    )

    nonmember_features, nonmember_pred_classes = extract_attack_features(
        nonmember_probs,
        target_y_test_np,
    )

    target_features = np.vstack([member_features, nonmember_features])
    target_pred_classes = np.concatenate([member_pred_classes, nonmember_pred_classes])
    target_membership = np.concatenate([
        np.ones(len(member_features), dtype=int),
        np.zeros(len(nonmember_features), dtype=int),
    ])

    # Balance final evaluation so accuracy is meaningful.
    member_mask = target_membership == 1
    nonmember_mask = target_membership == 0

    X_eval, y_eval = balance_binary_data(
        target_features[member_mask],
        target_features[nonmember_mask],
    )

    # Need matching pred classes after balancing.
    rng = np.random.default_rng(SEED)
    n = min(member_mask.sum(), nonmember_mask.sum())

    member_indices = np.where(member_mask)[0]
    nonmember_indices = np.where(nonmember_mask)[0]

    selected_member = rng.choice(member_indices, size=n, replace=False)
    selected_nonmember = rng.choice(nonmember_indices, size=n, replace=False)
    selected_indices = np.concatenate([selected_member, selected_nonmember])

    X_eval = target_features[selected_indices]
    y_eval = target_membership[selected_indices]
    pred_classes_eval = target_pred_classes[selected_indices]

    result = evaluate_class_specific_attack(
        attack_models,
        X_eval,
        pred_classes_eval,
        y_eval,
    )

    print("\n" + "=" * 60)
    print(f"Shokri-style v3 Attack on {regime_name}")
    print("=" * 60)
    print(f"Attack Accuracy:  {result['accuracy']:.4f}")
    print(f"Attack Precision: {result['precision']:.4f}")
    print(f"Attack Recall:    {result['recall']:.4f}")
    print(f"Attack F1:        {result['f1']:.4f}")
    print(f"Attack ROC AUC:   {result['auc']:.4f}")

    result["model"] = regime_name
    result["num_shadow_models"] = NUM_SHADOW_MODELS
    result["shadow_epochs"] = config["shadow_epochs"]
    result["architecture"] = config["architecture"]

    all_results.append(result)


# ================================
# Save Results
# ================================
df = pd.DataFrame(all_results)

summary_path = os.path.join(RESULTS_DIR, "shadow_attack_v3_summary.csv")
df.to_csv(summary_path, index=False)

print("\nSaved v3 shadow attack summary:")
print(summary_path)
print(df)