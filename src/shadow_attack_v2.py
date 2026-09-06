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

NUM_SHADOW_MODELS = 5
SHADOW_EPOCHS = 8
ATTACK_EPOCHS = 40
BATCH_SIZE = 256
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


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
y_np = y.to_numpy()
y_tensor = torch.tensor(y_np, dtype=torch.long)

input_dim = X_tensor.shape[1]
print("Input dim:", input_dim)
print("Dataset size:", len(X_tensor))


# ================================
# Models
# ================================
class ShadowMLP(nn.Module):
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


class AttackModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# ================================
# Feature Extraction
# ================================
def extract_attack_features(probs, true_labels):
    """
    Converts model outputs into richer attack features.

    Features:
    1. class 0 probability
    2. class 1 probability
    3. max confidence
    4. entropy
    5. confidence margin
    6. true-label loss
    7. correctness flag
    """
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

    features = np.column_stack([
        probs[:, 0],
        probs[:, 1],
        max_conf,
        entropy,
        margin,
        loss,
        correct,
    ])

    return features


# ================================
# Helper Functions
# ================================
def train_shadow_model(model, train_loader, epochs):
    criterion = nn.CrossEntropyLoss()
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


def balance_attack_data(member_features, nonmember_features):
    n = min(len(member_features), len(nonmember_features))

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


def train_attack_model(X_attack, y_attack):
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

    attack_model = AttackModel(input_dim=X_attack.shape[1]).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

    for epoch in range(ATTACK_EPOCHS):
        attack_model.train()

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = attack_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            attack_model.eval()

            with torch.no_grad():
                val_logits = attack_model(X_val.to(DEVICE))
                val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
                val_preds = (val_probs >= 0.5).astype(int)

            val_acc = accuracy_score(y_val.numpy(), val_preds)
            val_auc = roc_auc_score(y_val.numpy(), val_probs)

            print(
                f"Attack epoch {epoch + 1:03d}/{ATTACK_EPOCHS} | "
                f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
            )

    return attack_model


def evaluate_attack_model(attack_model, member_features, nonmember_features, model_name):
    X_eval, y_eval = balance_attack_data(member_features, nonmember_features)

    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(DEVICE)

    attack_model.eval()

    with torch.no_grad():
        logits = attack_model(X_eval_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_eval, preds)
    precision = precision_score(y_eval, preds, zero_division=0)
    recall = recall_score(y_eval, preds, zero_division=0)
    f1 = f1_score(y_eval, preds, zero_division=0)
    auc = roc_auc_score(y_eval, probs)

    print("\n" + "=" * 60)
    print(f"Feature-Aware Shadow Attack on {model_name}")
    print("=" * 60)
    print(f"Attack Accuracy:  {acc:.4f}")
    print(f"Attack Precision: {precision:.4f}")
    print(f"Attack Recall:    {recall:.4f}")
    print(f"Attack F1:        {f1:.4f}")
    print(f"Attack ROC AUC:   {auc:.4f}")

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


# ================================
# 1. Train Shadow Models + Build Attack Dataset
# ================================
shadow_member_features = []
shadow_nonmember_features = []

indices = np.arange(len(X_tensor))

print("\nTraining shadow models...")

for i in range(NUM_SHADOW_MODELS):
    print(f"\nShadow model {i + 1}/{NUM_SHADOW_MODELS}")

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

    shadow_model = ShadowMLP(input_dim).to(DEVICE)
    shadow_model = train_shadow_model(shadow_model, shadow_loader, SHADOW_EPOCHS)

    member_probs = get_probabilities(shadow_model, X_shadow_train)
    nonmember_probs = get_probabilities(shadow_model, X_shadow_test)

    member_features = extract_attack_features(member_probs, y_shadow_train_np)
    nonmember_features = extract_attack_features(nonmember_probs, y_shadow_test_np)

    shadow_member_features.append(member_features)
    shadow_nonmember_features.append(nonmember_features)

shadow_member_features = np.vstack(shadow_member_features)
shadow_nonmember_features = np.vstack(shadow_nonmember_features)

X_attack, y_attack = balance_attack_data(
    shadow_member_features,
    shadow_nonmember_features,
)

print("\nFeature-aware shadow attack dataset:")
print("X_attack:", X_attack.shape)
print("y_attack:", y_attack.shape)
print("Member ratio:", y_attack.mean())


# ================================
# 2. Train Attack Model
# ================================
print("\nTraining feature-aware attack model...")
attack_model = train_attack_model(X_attack, y_attack)

torch.save(
    attack_model.state_dict(),
    os.path.join(RESULTS_DIR, "shadow_attack_v2_model.pt"),
)


# ================================
# 3. Attack Target Model A Variants
# ================================
TARGETS = {
    "Overfit v2": {
        "member_probs": "model_a_train_probs.npy",
        "nonmember_probs": "model_a_test_probs.npy",
    },
    "Realistic": {
        "member_probs": "model_a_realistic_train_probs.npy",
        "nonmember_probs": "model_a_realistic_test_probs.npy",
    },
    "Early Stop 5ep": {
        "member_probs": "model_a_early_stop_5ep_train_probs.npy",
        "nonmember_probs": "model_a_early_stop_5ep_test_probs.npy",
    },
}

results = []

for model_name, files in TARGETS.items():
    member_probs = np.load(os.path.join(RESULTS_DIR, files["member_probs"]))
    nonmember_probs = np.load(os.path.join(RESULTS_DIR, files["nonmember_probs"]))

    # Target model labels use the same split proportions as the training scripts.
    # Rebuild the same target split labels so loss/correctness can be computed.
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

    y_train_np = y_train_split.to_numpy()
    y_test_np = y_test_split.to_numpy()

    member_features = extract_attack_features(member_probs, y_train_np)
    nonmember_features = extract_attack_features(nonmember_probs, y_test_np)

    result = evaluate_attack_model(
        attack_model,
        member_features,
        nonmember_features,
        model_name,
    )

    results.append(result)


# ================================
# 4. Save Results
# ================================
summary_path = os.path.join(RESULTS_DIR, "shadow_attack_v2_summary.csv")

df = pd.DataFrame(results)
df.to_csv(summary_path, index=False)

print("\nSaved feature-aware shadow attack summary:")
print(summary_path)