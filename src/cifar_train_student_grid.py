import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, Subset


# ================================
# Config
# ================================
DATASET_DIR = "data/cifar_student_datasets"
RESULTS_DIR = "results/cifar/student_models"
os.makedirs(RESULTS_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

SEED = 42
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 40
PATIENCE = 6
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
# Datasets
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
# Custom Student Dataset
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
# Model
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


# ================================
# Losses
# ================================
def soft_label_loss(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def hard_label_loss(logits, hard_targets):
    return nn.CrossEntropyLoss()(logits, hard_targets)


# ================================
# Evaluation
# ================================
def evaluate(model, loader):
    model.eval()

    all_probs = []
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, _, y_true in loader:
            xb = xb.to(device)

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y_true.numpy())

    probs = np.vstack(all_probs)
    preds = np.array(all_preds)
    true = np.array(all_true)

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="macro")

    return acc, f1, probs, preds, true


def evaluate_loss(model, loader, label_type):
    model.eval()

    total_loss = 0
    total_n = 0

    with torch.no_grad():
        for xb, y_student, _ in loader:
            xb = xb.to(device)
            y_student = y_student.to(device)

            logits = model(xb)

            if label_type == "hard":
                loss = hard_label_loss(logits, y_student)
            else:
                loss = soft_label_loss(logits, y_student)

            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)

    return total_loss / total_n


# ================================
# Reconstruct Train Indices
# ================================
def get_variant_indices(row, variant_dir):
    teacher_name = row["teacher_name"]
    filter_mode = row["filter_mode"]

    if teacher_name not in teacher_train_indices:
        raise ValueError(
            f"Unknown teacher_name '{teacher_name}'. "
            f"Expected one of: {list(teacher_train_indices.keys())}"
        )

    base_indices = teacher_train_indices[teacher_name]

    if filter_mode == "filtered":
        teacher_dir = f"results/cifar/{teacher_name}"
        full_probs = np.load(os.path.join(teacher_dir, "train_probs.npy"))
        full_conf = np.max(full_probs, axis=1)

        # Match the filtered labels generated by cifar_generate_student_datasets.py.
        # Removes the top 20% highest-confidence samples.
        FILTER_TOP_PERCENT = 0.20
        cutoff = np.quantile(full_conf, 1.0 - FILTER_TOP_PERCENT)
        mask = full_conf <= cutoff

        print(
            f"Filtered {teacher_name}: keeping {mask.sum()} / {len(mask)} "
            f"samples with confidence cutoff {cutoff:.4f}"
        )

        return base_indices[mask]

    return base_indices


# ================================
# Train One Student
# ================================
def train_variant(row):
    variant_name = row["variant_name"]
    label_type = row["label_type"]

    variant_dir = os.path.join(DATASET_DIR, variant_name)
    out_dir = os.path.join(RESULTS_DIR, variant_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Training CIFAR student: {variant_name}")
    print("=" * 80)

    y_train_student = np.load(os.path.join(variant_dir, "y_train_student.npy"))
    y_train_true = np.load(os.path.join(variant_dir, "y_train_true.npy"))

    y_test_student = np.load(os.path.join(variant_dir, "y_test_student.npy"))
    y_test_true = np.load(os.path.join(variant_dir, "y_test_true.npy"))

    train_indices = get_variant_indices(row, variant_dir)

    # split student train into train/val
    train_idx_local, val_idx_local = train_test_split(
        np.arange(len(train_indices)),
        test_size=0.15,
        random_state=SEED,
        stratify=y_train_true,
    )

    student_train_indices = train_indices[train_idx_local]
    student_val_indices = train_indices[val_idx_local]

    y_student_train = y_train_student[train_idx_local]
    y_student_val = y_train_student[val_idx_local]

    y_true_train = y_train_true[train_idx_local]
    y_true_val = y_train_true[val_idx_local]

    train_dataset = CIFARStudentDataset(
        train_aug,
        student_train_indices,
        y_student_train,
        y_true_train,
        label_type,
    )

    val_dataset = CIFARStudentDataset(
        train_eval,
        student_val_indices,
        y_student_val,
        y_true_val,
        label_type,
    )

    test_indices = np.arange(len(test_set))

    test_dataset = CIFARStudentDataset(
        test_set,
        test_indices,
        y_test_student,
        y_test_true,
        label_type,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CIFARStudentCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_n = 0

        for xb, y_student, _ in train_loader:
            xb = xb.to(device)
            y_student = y_student.to(device)

            optimizer.zero_grad()
            logits = model(xb)

            if label_type == "hard":
                loss = hard_label_loss(logits, y_student)
            else:
                loss = soft_label_loss(logits, y_student)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)

        train_loss = total_loss / total_n
        val_loss = evaluate_loss(model, val_loader, label_type)

        train_acc, train_f1, _, _, _ = evaluate(model, train_loader)
        val_acc, val_f1, _, _, _ = evaluate(model, val_loader)
        test_acc, test_f1, _, _, _ = evaluate(model, test_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "test_f1": test_f1,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}/{EPOCHS} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                f"Test Acc: {test_acc:.4f} | Val Loss: {val_loss:.4f}"
            )

        if stale >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(best_state)
    model.to(device)

    train_acc, train_f1, train_probs, train_preds, train_true = evaluate(model, train_loader)
    val_acc, val_f1, val_probs, val_preds, val_true = evaluate(model, val_loader)
    test_acc, test_f1, test_probs, test_preds, test_true = evaluate(model, test_loader)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    np.save(os.path.join(out_dir, "train_probs.npy"), train_probs)
    np.save(os.path.join(out_dir, "val_probs.npy"), val_probs)
    np.save(os.path.join(out_dir, "test_probs.npy"), test_probs)

    np.save(os.path.join(out_dir, "train_preds.npy"), train_preds)
    np.save(os.path.join(out_dir, "val_preds.npy"), val_preds)
    np.save(os.path.join(out_dir, "test_preds.npy"), test_preds)

    np.save(os.path.join(out_dir, "y_train_true.npy"), train_true)
    np.save(os.path.join(out_dir, "y_val_true.npy"), val_true)
    np.save(os.path.join(out_dir, "y_test_true.npy"), test_true)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    summary = {
        "variant_name": variant_name,
        "teacher_name": row["teacher_name"],
        "label_mode": row["label_mode"],
        "label_type": label_type,
        "filter_mode": row["filter_mode"],
        "num_train_samples": int(len(train_dataset)),
        "num_val_samples": int(len(val_dataset)),
        "num_test_samples": int(len(test_dataset)),
        "best_epoch": int(best_epoch),
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "train_f1": float(train_f1),
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("Final:", summary)

    return summary


# ================================
# Main
# ================================
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    manifest = pd.read_csv(MANIFEST_PATH)

    summaries = []

    for _, row in manifest.iterrows():
        summaries.append(train_variant(row))

    summary_df = pd.DataFrame(summaries)
    out_path = os.path.join(RESULTS_DIR, "cifar_student_grid_summary.csv")
    summary_df.to_csv(out_path, index=False)

    print("\nDone training CIFAR student grid.")
    print("Summary:", out_path)
    print(summary_df)