import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader


# ================================
# Config
# ================================
DATASET_DIR = "data/student_datasets"
RESULTS_DIR = "results/student_models"
os.makedirs(RESULTS_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

SEED = 42
BATCH_SIZE = 256
EPOCHS = 40
LR = 0.001
PATIENCE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================
# Student Model
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


# ================================
# Loss Functions
# ================================
def soft_label_loss(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def hard_label_loss(logits, hard_targets):
    return nn.CrossEntropyLoss()(logits, hard_targets)


# ================================
# Evaluation
# ================================
def evaluate(model, X, y_true):
    model.eval()

    loader = DataLoader(
        TensorDataset(X, y_true),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return acc, f1, all_probs, all_preds


def evaluate_loss(model, X, y_student, label_type):
    model.eval()

    if label_type == "hard":
        y_student_tensor = y_student.long()
    else:
        y_student_tensor = y_student.float()

    loader = DataLoader(
        TensorDataset(X, y_student_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    total_loss = 0
    n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)

            if label_type == "hard":
                loss = hard_label_loss(logits, yb)
            else:
                loss = soft_label_loss(logits, yb)

            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)

    return total_loss / n


# ================================
# Train One Student
# ================================
def train_student(variant_row):
    variant_name = variant_row["variant_name"]
    label_type = variant_row["label_type"]

    variant_dir = os.path.join(DATASET_DIR, variant_name)
    output_dir = os.path.join(RESULTS_DIR, variant_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Training {variant_name}")
    print("=" * 70)

    X_train = np.load(os.path.join(variant_dir, "X_train.npy"))
    X_val = np.load(os.path.join(variant_dir, "X_val.npy"))
    X_test = np.load(os.path.join(variant_dir, "X_test.npy"))

    y_train_student = np.load(os.path.join(variant_dir, "y_train_student.npy"))
    y_val_student = np.load(os.path.join(variant_dir, "y_val_student.npy"))

    y_train_true = np.load(os.path.join(variant_dir, "y_train_true.npy"))
    y_val_true = np.load(os.path.join(variant_dir, "y_val_true.npy"))
    y_test_true = np.load(os.path.join(variant_dir, "y_test_true.npy"))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    if label_type == "hard":
        y_train_student_t = torch.tensor(y_train_student, dtype=torch.long)
        y_val_student_t = torch.tensor(y_val_student, dtype=torch.long)
    else:
        y_train_student_t = torch.tensor(y_train_student, dtype=torch.float32)
        y_val_student_t = torch.tensor(y_val_student, dtype=torch.float32)

    y_train_true_t = torch.tensor(y_train_true, dtype=torch.long)
    y_val_true_t = torch.tensor(y_val_true, dtype=torch.long)
    y_test_true_t = torch.tensor(y_test_true, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_student_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = StudentMLP(X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    history = []

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0
        total_n = 0

        for xb, yb in train_loader:
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

            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)

        train_loss = total_loss / total_n
        val_loss = evaluate_loss(model, X_val_t, y_val_student_t, label_type)

        train_acc, train_f1, _, _ = evaluate(model, X_train_t, y_train_true_t)
        val_acc, val_f1, _, _ = evaluate(model, X_val_t, y_val_true_t)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc_true": train_acc,
            "val_acc_true": val_acc,
            "train_f1_true": train_f1,
            "val_f1_true": val_f1,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Train F1: {train_f1:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

        if stale_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(best_state)
    model.to(device)

    train_acc, train_f1, train_probs, train_preds = evaluate(model, X_train_t, y_train_true_t)
    val_acc, val_f1, val_probs, val_preds = evaluate(model, X_val_t, y_val_true_t)
    test_acc, test_f1, test_probs, test_preds = evaluate(model, X_test_t, y_test_true_t)

    print("\nFinal Results")
    print(f"Best Epoch: {best_epoch}")
    print(f"Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
    print(f"Test Acc:  {test_acc:.4f} | Test F1:  {test_f1:.4f}")

    torch.save(model.state_dict(), os.path.join(output_dir, "student_model.pt"))

    np.save(os.path.join(output_dir, "train_probs.npy"), train_probs)
    np.save(os.path.join(output_dir, "val_probs.npy"), val_probs)
    np.save(os.path.join(output_dir, "test_probs.npy"), test_probs)

    np.save(os.path.join(output_dir, "train_preds.npy"), train_preds)
    np.save(os.path.join(output_dir, "val_preds.npy"), val_preds)
    np.save(os.path.join(output_dir, "test_preds.npy"), test_preds)

    np.save(os.path.join(output_dir, "y_train_true.npy"), y_train_true)
    np.save(os.path.join(output_dir, "y_val_true.npy"), y_val_true)
    np.save(os.path.join(output_dir, "y_test_true.npy"), y_test_true)

    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    summary = {
        "variant_name": variant_name,
        "label_mode": variant_row["label_mode"],
        "label_type": label_type,
        "feature_mode": variant_row["feature_mode"],
        "input_dim": int(X_train.shape[1]),
        "num_train_samples": int(len(X_train)),
        "best_epoch": int(best_epoch),
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "train_f1": float(train_f1),
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return summary


# ================================
# Main
# ================================
manifest = pd.read_csv(MANIFEST_PATH)

summaries = []

for _, row in manifest.iterrows():
    summary = train_student(row)
    summaries.append(summary)

summary_df = pd.DataFrame(summaries)
summary_path = os.path.join(RESULTS_DIR, "student_grid_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\nDone training student grid.")
print("Summary:", summary_path)
print(summary_df)