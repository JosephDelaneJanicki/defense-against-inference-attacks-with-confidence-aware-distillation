import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset, TensorDataset


# ================================
# Config
# ================================
RESULTS_DIR = "results/cifar"
ATTACK_DIR = "results/cifar/attacks"
os.makedirs(ATTACK_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_SHADOW_MODELS = 6
ATTACK_EPOCHS = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)


TARGETS = {
    "overfit_teacher": {
        "target_dir": os.path.join(RESULTS_DIR, "overfit_teacher"),
        "shadow_train_size": 8000,
        "shadow_epochs": 80,
        "dropout": False,
        "use_aug": False,
    },
    "realistic_teacher": {
        "target_dir": os.path.join(RESULTS_DIR, "realistic_teacher"),
        "shadow_train_size": 25000,
        "shadow_epochs": 30,
        "dropout": True,
        "use_aug": True,
    },
}


# ================================
# Transforms
# ================================
basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

aug_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


# ================================
# Load CIFAR
# ================================
train_basic = torchvision.datasets.CIFAR10(
    root="data/cifar10",
    train=True,
    download=True,
    transform=basic_transform,
)

train_aug = torchvision.datasets.CIFAR10(
    root="data/cifar10",
    train=True,
    download=True,
    transform=aug_transform,
)

test_set = torchvision.datasets.CIFAR10(
    root="data/cifar10",
    train=False,
    download=True,
    transform=basic_transform,
)

all_indices = np.arange(len(train_basic))
labels_np = np.array(train_basic.targets)


# ================================
# CNN Model - same as teacher script
# ================================
class CIFARSmallCNN(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()

        d1 = 0.25 if dropout else 0.0
        d2 = 0.35 if dropout else 0.0

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64) if dropout else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64) if dropout else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(d1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128) if dropout else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128) if dropout else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(d2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256) if dropout else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(d2),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class AttackModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


# ================================
# Feature Extraction - CIFAR v3 style
# ================================
def extract_attack_features(probs, true_labels):
    eps = 1e-10

    probs = np.asarray(probs)
    true_labels = np.asarray(true_labels).astype(int)

    sorted_desc = np.sort(probs, axis=1)[:, ::-1]

    top1 = sorted_desc[:, 0]
    top2 = sorted_desc[:, 1]
    top3 = sorted_desc[:, 2]

    max_conf = top1
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    margin_12 = top1 - top2
    margin_23 = top2 - top3

    pred_labels = np.argmax(probs, axis=1)
    correct = (pred_labels == true_labels).astype(float)

    true_label_probs = probs[np.arange(len(probs)), true_labels]
    loss = -np.log(true_label_probs + eps)

    # Feature set:
    # full prob vector + sorted distribution + behavioral features
    features = np.column_stack([
        probs,
        sorted_desc,
        max_conf,
        entropy,
        margin_12,
        margin_23,
        loss,
        correct,
        pred_labels,
    ])

    return features, pred_labels


# ================================
# Helpers
# ================================
def train_shadow_model(model, loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epochs):
        model.train()

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model


def get_probs_from_loader(model, loader):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.extend(yb.numpy())

    return np.vstack(all_probs), np.array(all_labels)


def balance_binary(member_features, nonmember_features):
    n = min(len(member_features), len(nonmember_features))

    if n == 0:
        return None, None

    rng = np.random.default_rng(SEED)

    idx_m = rng.choice(len(member_features), n, replace=False)
    idx_n = rng.choice(len(nonmember_features), n, replace=False)

    X = np.vstack([member_features[idx_m], nonmember_features[idx_n]])
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    return X, y


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

    model = AttackModel(X_attack.shape[1]).to(device)

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
            logits = model(X_val_t.to(device))
            scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        auc = roc_auc_score(y_val, scores)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Class {class_id} attack model best val AUC: {best_auc:.4f}")

    return model


def train_class_specific_attacks(features, pred_classes, membership):
    attack_models = {}

    for class_id in range(NUM_CLASSES):
        mask = pred_classes == class_id

        class_features = features[mask]
        class_membership = membership[mask]

        member_features = class_features[class_membership == 1]
        nonmember_features = class_features[class_membership == 0]

        X_attack, y_attack = balance_binary(member_features, nonmember_features)

        if X_attack is None or len(X_attack) < 100:
            print(f"Skipping class {class_id}: insufficient data.")
            continue

        print(f"Training class-specific attack model for class {class_id} with {len(X_attack)} samples.")
        attack_models[class_id] = train_single_attack_model(X_attack, y_attack, class_id)

    return attack_models


def evaluate_class_specific_attack(attack_models, features, pred_classes, membership):
    scores_all = np.zeros(len(features), dtype=np.float32)

    for class_id, attack_model in attack_models.items():
        mask = pred_classes == class_id

        if mask.sum() == 0:
            continue

        X_t = torch.tensor(features[mask], dtype=torch.float32).to(device)

        attack_model.eval()

        with torch.no_grad():
            logits = attack_model(X_t)
            scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        scores_all[mask] = scores

    preds = (scores_all >= 0.5).astype(int)

    return {
        "attack_accuracy": accuracy_score(membership, preds),
        "attack_precision": precision_score(membership, preds, zero_division=0),
        "attack_recall": recall_score(membership, preds, zero_division=0),
        "attack_f1": f1_score(membership, preds, zero_division=0),
        "attack_auc": roc_auc_score(membership, scores_all),
    }


def build_balanced_target_eval(train_features, test_features, train_pred, test_pred):
    n = min(len(train_features), len(test_features))

    rng = np.random.default_rng(SEED)

    idx_train = rng.choice(len(train_features), n, replace=False)
    idx_test = rng.choice(len(test_features), n, replace=False)

    features = np.vstack([train_features[idx_train], test_features[idx_test]])
    pred_classes = np.concatenate([train_pred[idx_train], test_pred[idx_test]])
    membership = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    return features, pred_classes, membership


# ================================
# Attack One Target
# ================================
def attack_target(target_name, config):
    print("\n" + "#" * 80)
    print(f"CIFAR v3 shadow attack on {target_name}")
    print("#" * 80)

    dataset_for_training = train_aug if config["use_aug"] else train_basic
    dataset_for_eval = train_basic

    shadow_features_all = []
    shadow_pred_all = []
    shadow_membership_all = []

    for i in range(NUM_SHADOW_MODELS):
        print(f"\nShadow model {i + 1}/{NUM_SHADOW_MODELS}")

        shadow_train_idx, shadow_test_idx = train_test_split(
            all_indices,
            train_size=config["shadow_train_size"],
            test_size=config["shadow_train_size"],
            random_state=SEED + i,
            stratify=labels_np,
        )

        shadow_train_set = Subset(dataset_for_training, shadow_train_idx)
        shadow_train_eval_set = Subset(dataset_for_eval, shadow_train_idx)
        shadow_test_eval_set = Subset(dataset_for_eval, shadow_test_idx)

        shadow_train_loader = DataLoader(
            shadow_train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        shadow_train_eval_loader = DataLoader(
            shadow_train_eval_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        shadow_test_eval_loader = DataLoader(
            shadow_test_eval_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        shadow_model = CIFARSmallCNN(dropout=config["dropout"]).to(device)
        shadow_model = train_shadow_model(
            shadow_model,
            shadow_train_loader,
            config["shadow_epochs"],
        )

        member_probs, member_labels = get_probs_from_loader(shadow_model, shadow_train_eval_loader)
        nonmember_probs, nonmember_labels = get_probs_from_loader(shadow_model, shadow_test_eval_loader)

        member_features, member_pred = extract_attack_features(member_probs, member_labels)
        nonmember_features, nonmember_pred = extract_attack_features(nonmember_probs, nonmember_labels)

        features = np.vstack([member_features, nonmember_features])
        pred_classes = np.concatenate([member_pred, nonmember_pred])
        membership = np.concatenate([
            np.ones(len(member_features), dtype=int),
            np.zeros(len(nonmember_features), dtype=int),
        ])

        shadow_features_all.append(features)
        shadow_pred_all.append(pred_classes)
        shadow_membership_all.append(membership)

    shadow_features_all = np.vstack(shadow_features_all)
    shadow_pred_all = np.concatenate(shadow_pred_all)
    shadow_membership_all = np.concatenate(shadow_membership_all)

    print("\nShadow attack dataset:")
    print("Features:", shadow_features_all.shape)
    print("Membership ratio:", shadow_membership_all.mean())

    attack_models = train_class_specific_attacks(
        shadow_features_all,
        shadow_pred_all,
        shadow_membership_all,
    )

    target_dir = config["target_dir"]

    train_probs = np.load(os.path.join(target_dir, "train_probs.npy"))
    test_probs = np.load(os.path.join(target_dir, "test_probs.npy"))
    y_train = np.load(os.path.join(target_dir, "y_train_true.npy"))
    y_test = np.load(os.path.join(target_dir, "y_test_true.npy"))

    train_features, train_pred = extract_attack_features(train_probs, y_train)
    test_features, test_pred = extract_attack_features(test_probs, y_test)

    features_eval, pred_eval, membership_eval = build_balanced_target_eval(
        train_features,
        test_features,
        train_pred,
        test_pred,
    )

    metrics = evaluate_class_specific_attack(
        attack_models,
        features_eval,
        pred_eval,
        membership_eval,
    )

    print("\nAttack Result")
    print(metrics)

    metrics["target"] = target_name
    metrics["num_shadow_models"] = NUM_SHADOW_MODELS
    metrics["shadow_train_size"] = config["shadow_train_size"]
    metrics["shadow_epochs"] = config["shadow_epochs"]

    return metrics


# ================================
# Main
# ================================
results = []

for target_name, config in TARGETS.items():
    metrics = attack_target(target_name, config)
    results.append(metrics)

df = pd.DataFrame(results)

out_path = os.path.join(
    ATTACK_DIR,
    f"cifar_teacher_attack_v3_ns{NUM_SHADOW_MODELS}_seed{SEED}.csv",
)

df.to_csv(out_path, index=False)

print("\nDone CIFAR teacher attacks.")
print("Saved:", out_path)
print(df)