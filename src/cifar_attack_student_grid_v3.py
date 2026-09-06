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
from torch.utils.data import DataLoader, Dataset


# ================================
# Config
# ================================
DATASET_DIR = "data/cifar_student_datasets"
STUDENT_RESULTS_DIR = "results/cifar/student_models"
ATTACK_RESULTS_DIR = "results/cifar/student_attacks"
os.makedirs(ATTACK_RESULTS_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

SEED = 42
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_SHADOW_MODELS = 6
SHADOW_EPOCHS = 12
ATTACK_EPOCHS = 35
LR = 0.001

OVERFIT_TRAIN_SIZE = 15000
REALISTIC_TRAIN_SIZE = 45000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================
# Transforms
# ================================
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


# ================================
# CIFAR Data
# ================================
train_aug = torchvision.datasets.CIFAR10(
    root="data/cifar10",
    train=True,
    download=True,
    transform=train_transform,
)

train_eval = torchvision.datasets.CIFAR10(
    root="data/cifar10",
    train=True,
    download=True,
    transform=eval_transform,
)

test_set = torchvision.datasets.CIFAR10(
    root="data/cifar10",
    train=False,
    download=True,
    transform=eval_transform,
)

all_indices = np.arange(len(train_eval))
rng = np.random.default_rng(SEED)
rng.shuffle(all_indices)

teacher_train_indices = {
    "overfit_teacher_v2": all_indices[:OVERFIT_TRAIN_SIZE],
    "realistic_teacher": all_indices[:REALISTIC_TRAIN_SIZE],
}


# ================================
# Dataset Wrapper
# ================================
class CIFARStudentDataset(Dataset):
    def __init__(self, base_dataset, indices, student_labels, true_labels, label_type):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices)
        self.student_labels = student_labels
        self.true_labels = true_labels
        self.label_type = label_type

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[int(self.indices[idx])]

        if self.label_type == "hard":
            student_label = torch.tensor(self.student_labels[idx], dtype=torch.long)
        else:
            student_label = torch.tensor(self.student_labels[idx], dtype=torch.float32)

        true_label = torch.tensor(self.true_labels[idx], dtype=torch.long)

        return image, student_label, true_label


# ================================
# Models
# ================================
class CIFARStudentCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.30),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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
# Losses
# ================================
def soft_label_loss(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def hard_label_loss(logits, hard_targets):
    return nn.CrossEntropyLoss()(logits, hard_targets)


# ================================
# Attack Feature Extraction
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
def get_variant_indices(row, variant_dir):
    teacher_name = row["teacher_name"]
    filter_mode = row["filter_mode"]

    base_indices = teacher_train_indices[teacher_name]

    if filter_mode == "filtered":
        teacher_dir = f"results/cifar/{teacher_name}"
        full_probs = np.load(os.path.join(teacher_dir, "train_probs.npy"))
        full_conf = np.max(full_probs, axis=1)

        filter_top_percent = 0.20
        cutoff = np.quantile(full_conf, 1.0 - filter_top_percent)
        mask = full_conf <= cutoff

        return base_indices[mask]

    return base_indices


def train_shadow_student(X_indices, y_student, y_true, label_type):
    model = CIFARStudentCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    dataset = CIFARStudentDataset(
        train_aug,
        X_indices,
        y_student,
        y_true,
        label_type,
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(SHADOW_EPOCHS):
        model.train()

        for xb, yb_student, _ in loader:
            xb = xb.to(device)
            yb_student = yb_student.to(device)

            optimizer.zero_grad()
            logits = model(xb)

            if label_type == "hard":
                loss = hard_label_loss(logits, yb_student)
            else:
                loss = soft_label_loss(logits, yb_student)

            loss.backward()
            optimizer.step()

    return model


def get_probs(model, base_dataset, indices, y_student, y_true, label_type):
    dataset = CIFARStudentDataset(
        base_dataset,
        indices,
        y_student,
        y_true,
        label_type,
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()

    all_probs = []
    all_true = []

    with torch.no_grad():
        for xb, _, yb_true in loader:
            xb = xb.to(device)

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_true.extend(yb_true.numpy())

    return np.vstack(all_probs), np.array(all_true)


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

    loader = DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
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

    print(f"Attack class {class_id} best shadow val AUC: {best_auc:.4f}")

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

        print(f"Training attack model for class {class_id} with {len(X_attack)} samples")
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


def build_balanced_eval(train_features, test_features, train_pred, test_pred):
    n = min(len(train_features), len(test_features))

    rng = np.random.default_rng(SEED)

    idx_train = rng.choice(len(train_features), n, replace=False)
    idx_test = rng.choice(len(test_features), n, replace=False)

    features = np.vstack([train_features[idx_train], test_features[idx_test]])
    pred_classes = np.concatenate([train_pred[idx_train], test_pred[idx_test]])
    membership = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    return features, pred_classes, membership


# ================================
# Attack Variant
# ================================
def attack_variant(row):
    variant_name = row["variant_name"]
    teacher_name = row["teacher_name"]
    label_type = row["label_type"]

    variant_dir = os.path.join(DATASET_DIR, variant_name)
    student_dir = os.path.join(STUDENT_RESULTS_DIR, variant_name)

    print("\n" + "=" * 90)
    print(f"Attacking CIFAR student: {variant_name}")
    print("=" * 90)

    y_train_student = np.load(os.path.join(variant_dir, "y_train_student.npy"))
    y_train_true = np.load(os.path.join(variant_dir, "y_train_true.npy"))

    y_test_student = np.load(os.path.join(variant_dir, "y_test_student.npy"))
    y_test_true = np.load(os.path.join(variant_dir, "y_test_true.npy"))

    train_indices = get_variant_indices(row, variant_dir)

    # Shadow pool = split the student training set into shadow members/nonmembers.
    # This avoids sampling more nonmembers than CIFAR test contains.
    shadow_features_all = []
    shadow_pred_all = []
    shadow_membership_all = []

    train_pool_local = np.arange(len(train_indices))

    for i in range(NUM_SHADOW_MODELS):
        print(f"Shadow model {i + 1}/{NUM_SHADOW_MODELS}")

        shadow_train_local, shadow_holdout_local = train_test_split(
            train_pool_local,
            test_size=0.5,
            random_state=SEED + i,
            stratify=y_train_true,
        )

        shadow_train_indices = train_indices[shadow_train_local]
        shadow_nonmember_indices = train_indices[shadow_holdout_local]

        y_shadow_train_student = y_train_student[shadow_train_local]
        y_shadow_train_true = y_train_true[shadow_train_local]

        y_shadow_nonmember_student = y_train_student[shadow_holdout_local]
        y_shadow_nonmember_true = y_train_true[shadow_holdout_local]

        shadow_model = train_shadow_student(
            shadow_train_indices,
            y_shadow_train_student,
            y_shadow_train_true,
            label_type,
        )

        member_probs, member_true = get_probs(
            shadow_model,
            train_eval,
            shadow_train_indices,
            y_shadow_train_student,
            y_shadow_train_true,
            label_type,
        )

        nonmember_probs, nonmember_true = get_probs(
            shadow_model,
            train_eval,
            shadow_nonmember_indices,
            y_shadow_nonmember_student,
            y_shadow_nonmember_true,
            label_type,
        )

        member_features, member_pred = extract_attack_features(member_probs, member_true)
        nonmember_features, nonmember_pred = extract_attack_features(nonmember_probs, nonmember_true)

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

    attack_models = train_class_specific_attacks(
        shadow_features_all,
        shadow_pred_all,
        shadow_membership_all,
    )

    # Load target student outputs.
    train_probs = np.load(os.path.join(student_dir, "train_probs.npy"))
    test_probs = np.load(os.path.join(student_dir, "test_probs.npy"))

    target_y_train = np.load(os.path.join(student_dir, "y_train_true.npy"))
    target_y_test = np.load(os.path.join(student_dir, "y_test_true.npy"))

    train_features, train_pred = extract_attack_features(train_probs, target_y_train)
    test_features, test_pred = extract_attack_features(test_probs, target_y_test)

    features_eval, pred_eval, membership_eval = build_balanced_eval(
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

    with open(os.path.join(student_dir, "summary.json"), "r") as f:
        summary = json.load(f)

    result = {
        "variant_name": variant_name,
        "teacher_name": teacher_name,
        "label_mode": row["label_mode"],
        "label_type": row["label_type"],
        "filter_mode": row["filter_mode"],
        "test_acc": summary["test_acc"],
        "test_f1": summary["test_f1"],
        **metrics,
    }

    print("Result:", result)

    return result


# ================================
# Main
# ================================
manifest = pd.read_csv(MANIFEST_PATH)

results = []

for _, row in manifest.iterrows():
    results.append(attack_variant(row))

df = pd.DataFrame(results)

out_path = os.path.join(
    ATTACK_RESULTS_DIR,
    f"cifar_student_attack_v3_ns{NUM_SHADOW_MODELS}_seed{SEED}.csv",
)

df.to_csv(out_path, index=False)

print("\nDone attacking CIFAR student grid.")
print("Saved:", out_path)
print(df)