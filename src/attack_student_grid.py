import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader


# ================================
# Config
# ================================
DATASET_DIR = "data/student_datasets"
STUDENT_RESULTS_DIR = "results/student_models"
ATTACK_RESULTS_DIR = "results/student_attacks"
os.makedirs(ATTACK_RESULTS_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

SEED = 42
NUM_SHADOW_MODELS = 6
SHADOW_EPOCHS = 12
ATTACK_EPOCHS = 40
BATCH_SIZE = 256
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================
# Models
# ================================
class StudentMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),

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


# ================================
# Losses
# ================================
def soft_label_loss(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def hard_label_loss(logits, hard_targets):
    return nn.CrossEntropyLoss()(logits, hard_targets)


# ================================
# Feature Extraction - v3 Style
# ================================
def extract_attack_features(probs, true_labels):
    eps = 1e-10

    probs = np.asarray(probs)
    true_labels = np.asarray(true_labels).astype(int)

    p0 = probs[:, 0]
    p1 = probs[:, 1]

    sorted_desc = np.sort(probs, axis=1)[:, ::-1]
    top1 = sorted_desc[:, 0]
    top2 = sorted_desc[:, 1]

    max_conf = np.max(probs, axis=1)
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    margin = top1 - top2

    pred_labels = np.argmax(probs, axis=1)
    correct = (pred_labels == true_labels).astype(float)

    true_label_probs = probs[np.arange(len(probs)), true_labels]
    loss = -np.log(true_label_probs + eps)

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
        pred_labels,
    ])

    return features, pred_labels


# ================================
# Helpers
# ================================
def to_label_tensor(y, label_type):
    if label_type == "hard":
        return torch.tensor(y, dtype=torch.long)
    return torch.tensor(y, dtype=torch.float32)


def train_shadow_student(X_train, y_train_student, input_dim, label_type):
    model = StudentMLP(input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = to_label_tensor(y_train_student, label_type)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    for _ in range(SHADOW_EPOCHS):
        model.train()

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)

            if label_type == "hard":
                loss = hard_label_loss(logits, yb)
            else:
                loss = soft_label_loss(logits, yb)

            loss.backward()
            optimizer.step()

    return model


def get_probabilities(model, X_array):
    model.eval()

    X_t = torch.tensor(X_array, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_probs = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
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
        np.ones(len(member_features), dtype=int),
        np.zeros(len(nonmember_features), dtype=int),
    ])

    return X_attack, y_attack


def train_single_attack_model(X_attack, y_attack, class_id):
    X_train, X_val, y_train, y_val = train_test_split(
        X_attack,
        y_attack,
        test_size=0.2,
        random_state=SEED,
        stratify=y_attack,
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = AttackModel(input_dim=X_attack.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_auc = -1
    best_state = None

    for _ in range(ATTACK_EPOCHS):
        model.train()

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_scores = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()

        val_auc = roc_auc_score(y_val, val_scores)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Attack model class {class_id} | best shadow val AUC: {best_auc:.4f}")

    return model


def train_class_specific_attacks(features, pred_classes, membership_labels):
    attack_models = {}

    for class_id in [0, 1]:
        mask = pred_classes == class_id

        class_features = features[mask]
        class_membership = membership_labels[mask]

        member_features = class_features[class_membership == 1]
        nonmember_features = class_features[class_membership == 0]

        X_attack, y_attack = balance_binary_data(member_features, nonmember_features)

        if X_attack is None:
            print(f"Skipping class {class_id}: insufficient shadow data.")
            continue

        print(
            f"Training attack model for predicted class {class_id} "
            f"with {len(X_attack)} samples."
        )

        attack_models[class_id] = train_single_attack_model(
            X_attack,
            y_attack,
            class_id,
        )

    return attack_models


def evaluate_class_specific_attack(attack_models, features, pred_classes, membership_labels):
    all_scores = np.zeros(len(features), dtype=np.float32)

    for class_id, attack_model in attack_models.items():
        mask = pred_classes == class_id

        if mask.sum() == 0:
            continue

        X_class = torch.tensor(features[mask], dtype=torch.float32).to(device)

        attack_model.eval()

        with torch.no_grad():
            logits = attack_model(X_class)
            scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        all_scores[mask] = scores

    preds = (all_scores >= 0.5).astype(int)

    return {
        "attack_accuracy": accuracy_score(membership_labels, preds),
        "attack_precision": precision_score(membership_labels, preds, zero_division=0),
        "attack_recall": recall_score(membership_labels, preds, zero_division=0),
        "attack_f1": f1_score(membership_labels, preds, zero_division=0),
        "attack_auc": roc_auc_score(membership_labels, all_scores),
    }


def build_balanced_target_eval(train_features, test_features, train_pred, test_pred):
    n = min(len(train_features), len(test_features))

    rng = np.random.default_rng(SEED)

    train_idx = rng.choice(len(train_features), size=n, replace=False)
    test_idx = rng.choice(len(test_features), size=n, replace=False)

    features = np.vstack([
        train_features[train_idx],
        test_features[test_idx],
    ])

    pred_classes = np.concatenate([
        train_pred[train_idx],
        test_pred[test_idx],
    ])

    membership = np.concatenate([
        np.ones(n, dtype=int),
        np.zeros(n, dtype=int),
    ])

    return features, pred_classes, membership


# ================================
# Attack One Student Variant
# ================================
def attack_student_variant(row):
    variant_name = row["variant_name"]
    label_type = row["label_type"]

    variant_dir = os.path.join(DATASET_DIR, variant_name)
    student_dir = os.path.join(STUDENT_RESULTS_DIR, variant_name)

    print("\n" + "=" * 80)
    print(f"Attacking student variant: {variant_name}")
    print("=" * 80)

    # Load full student dataset
    X_train = np.load(os.path.join(variant_dir, "X_train.npy"))
    X_val = np.load(os.path.join(variant_dir, "X_val.npy"))
    X_test = np.load(os.path.join(variant_dir, "X_test.npy"))

    y_train_student = np.load(os.path.join(variant_dir, "y_train_student.npy"))
    y_val_student = np.load(os.path.join(variant_dir, "y_val_student.npy"))
    y_test_student = np.load(os.path.join(variant_dir, "y_test_student.npy"))

    y_train_true = np.load(os.path.join(variant_dir, "y_train_true.npy"))
    y_val_true = np.load(os.path.join(variant_dir, "y_val_true.npy"))
    y_test_true = np.load(os.path.join(variant_dir, "y_test_true.npy"))

    # Combine all available data for shadow sampling.
    X_all = np.vstack([X_train, X_val, X_test])
    y_true_all = np.concatenate([y_train_true, y_val_true, y_test_true])

    if label_type == "hard":
        y_student_all = np.concatenate([y_train_student, y_val_student, y_test_student])
    else:
        y_student_all = np.vstack([y_train_student, y_val_student, y_test_student])

    input_dim = X_all.shape[1]
    indices = np.arange(len(X_all))

    shadow_features_all = []
    shadow_pred_classes_all = []
    shadow_membership_all = []

    for i in range(NUM_SHADOW_MODELS):
        print(f"Shadow model {i + 1}/{NUM_SHADOW_MODELS}")

        shadow_train_idx, shadow_test_idx = train_test_split(
            indices,
            test_size=0.5,
            random_state=SEED + i,
            stratify=y_true_all,
        )

        X_shadow_train = X_all[shadow_train_idx]
        X_shadow_test = X_all[shadow_test_idx]

        y_shadow_train_student = y_student_all[shadow_train_idx]
        y_shadow_train_true = y_true_all[shadow_train_idx]
        y_shadow_test_true = y_true_all[shadow_test_idx]

        shadow_model = train_shadow_student(
            X_shadow_train,
            y_shadow_train_student,
            input_dim,
            label_type,
        )

        member_probs = get_probabilities(shadow_model, X_shadow_train)
        nonmember_probs = get_probabilities(shadow_model, X_shadow_test)

        member_features, member_pred_classes = extract_attack_features(
            member_probs,
            y_shadow_train_true,
        )

        nonmember_features, nonmember_pred_classes = extract_attack_features(
            nonmember_probs,
            y_shadow_test_true,
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

    print("Shadow attack dataset:", shadow_features_all.shape)

    attack_models = train_class_specific_attacks(
        shadow_features_all,
        shadow_pred_classes_all,
        shadow_membership_all,
    )

    # Load target student outputs.
    train_probs = np.load(os.path.join(student_dir, "train_probs.npy"))
    test_probs = np.load(os.path.join(student_dir, "test_probs.npy"))

    train_features, train_pred_classes = extract_attack_features(
        train_probs,
        y_train_true,
    )

    test_features, test_pred_classes = extract_attack_features(
        test_probs,
        y_test_true,
    )

    X_eval, pred_eval, membership_eval = build_balanced_target_eval(
        train_features,
        test_features,
        train_pred_classes,
        test_pred_classes,
    )

    attack_metrics = evaluate_class_specific_attack(
        attack_models,
        X_eval,
        pred_eval,
        membership_eval,
    )

    with open(os.path.join(student_dir, "summary.json"), "r") as f:
        utility_summary = json.load(f)

    result = {
        "variant_name": variant_name,
        "label_mode": row["label_mode"],
        "label_type": label_type,
        "feature_mode": row["feature_mode"],
        "test_acc": utility_summary["test_acc"],
        "test_f1": utility_summary["test_f1"],
        "num_train_samples": utility_summary["num_train_samples"],
        **attack_metrics,
    }

    print("\nAttack Result")
    print(result)

    return result


# ================================
# Main
# ================================
manifest = pd.read_csv(MANIFEST_PATH)

results = []

for _, row in manifest.iterrows():
    result = attack_student_variant(row)
    results.append(result)

results_df = pd.DataFrame(results)

out_path = os.path.join(
    ATTACK_RESULTS_DIR,
    f"student_attack_grid_v3style_ns{NUM_SHADOW_MODELS}_seed{SEED}.csv",
)

results_df.to_csv(out_path, index=False)

print("\nDone attacking student grid.")
print("Saved:", out_path)
print(results_df)