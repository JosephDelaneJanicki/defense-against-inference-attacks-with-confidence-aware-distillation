import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split


# ================================
# Config
# ================================
RESULTS_DIR = "results/cifar"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
NUM_CLASSES = 10

OVERFIT_TRAIN_SIZE = 8000
OVERFIT_EPOCHS = 80

REALISTIC_TRAIN_SIZE = 45000
REALISTIC_VAL_SIZE = 5000
REALISTIC_EPOCHS = 50
PATIENCE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)


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
# Dataset
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
rng = np.random.default_rng(SEED)
rng.shuffle(all_indices)

overfit_indices = all_indices[:OVERFIT_TRAIN_SIZE]

realistic_train_indices = all_indices[:REALISTIC_TRAIN_SIZE]
realistic_val_indices = all_indices[REALISTIC_TRAIN_SIZE:REALISTIC_TRAIN_SIZE + REALISTIC_VAL_SIZE]

overfit_train_set = Subset(train_basic, overfit_indices)
realistic_train_set = Subset(train_aug, realistic_train_indices)
realistic_val_set = Subset(train_basic, realistic_val_indices)

# For output extraction, use non-augmented versions of the same splits.
overfit_train_eval_set = Subset(train_basic, overfit_indices)
realistic_train_eval_set = Subset(train_basic, realistic_train_indices)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

overfit_train_loader = DataLoader(overfit_train_set, batch_size=BATCH_SIZE, shuffle=True)
overfit_train_eval_loader = DataLoader(overfit_train_eval_set, batch_size=BATCH_SIZE, shuffle=False)

realistic_train_loader = DataLoader(realistic_train_set, batch_size=BATCH_SIZE, shuffle=True)
realistic_train_eval_loader = DataLoader(realistic_train_eval_set, batch_size=BATCH_SIZE, shuffle=False)
realistic_val_loader = DataLoader(realistic_val_set, batch_size=BATCH_SIZE, shuffle=False)


# ================================
# CNN Models
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


# ================================
# Helpers
# ================================
def evaluate(model, loader):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1


def get_outputs(model, loader):
    model.eval()

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

    return np.vstack(all_probs), np.array(all_preds), np.array(all_labels)


def train_overfit_teacher():
    print("\n" + "=" * 70)
    print("Training CIFAR overfit teacher")
    print("=" * 70)

    model = CIFARSmallCNN(dropout=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = []

    for epoch in range(OVERFIT_EPOCHS):
        model.train()

        for xb, yb in overfit_train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        train_loss, train_acc, train_f1 = evaluate(model, overfit_train_eval_loader)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_f1": train_f1,
            "test_f1": test_f1,
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}/{OVERFIT_EPOCHS} | "
                f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
                f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
            )

    out_dir = os.path.join(RESULTS_DIR, "overfit_teacher")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    train_probs, train_preds, train_labels = get_outputs(model, overfit_train_eval_loader)
    test_probs, test_preds, test_labels = get_outputs(model, test_loader)

    np.save(os.path.join(out_dir, "train_probs.npy"), train_probs)
    np.save(os.path.join(out_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(out_dir, "train_preds.npy"), train_preds)
    np.save(os.path.join(out_dir, "test_preds.npy"), test_preds)
    np.save(os.path.join(out_dir, "y_train_true.npy"), train_labels)
    np.save(os.path.join(out_dir, "y_test_true.npy"), test_labels)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    summary = {
        "model": "cifar_overfit_teacher",
        "train_size": OVERFIT_TRAIN_SIZE,
        "epochs": OVERFIT_EPOCHS,
        "train_acc": float(history[-1]["train_acc"]),
        "test_acc": float(history[-1]["test_acc"]),
        "train_f1": float(history[-1]["train_f1"]),
        "test_f1": float(history[-1]["test_f1"]),
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved overfit teacher:", out_dir)
    print(summary)


def train_realistic_teacher():
    print("\n" + "=" * 70)
    print("Training CIFAR realistic teacher")
    print("=" * 70)

    model = CIFARSmallCNN(dropout=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(REALISTIC_EPOCHS):
        model.train()

        for xb, yb in realistic_train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        train_loss, train_acc, train_f1 = evaluate(model, realistic_train_eval_loader)
        val_loss, val_acc, val_f1 = evaluate(model, realistic_val_loader)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader)

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
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
            stale_epochs = 0
        else:
            stale_epochs += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}/{REALISTIC_EPOCHS} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                f"Test Acc: {test_acc:.4f} | Val Loss: {val_loss:.4f}"
            )

        if stale_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(best_state)
    model.to(device)

    train_loss, train_acc, train_f1 = evaluate(model, realistic_train_eval_loader)
    val_loss, val_acc, val_f1 = evaluate(model, realistic_val_loader)
    test_loss, test_acc, test_f1 = evaluate(model, test_loader)

    out_dir = os.path.join(RESULTS_DIR, "realistic_teacher")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    train_probs, train_preds, train_labels = get_outputs(model, realistic_train_eval_loader)
    test_probs, test_preds, test_labels = get_outputs(model, test_loader)

    np.save(os.path.join(out_dir, "train_probs.npy"), train_probs)
    np.save(os.path.join(out_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(out_dir, "train_preds.npy"), train_preds)
    np.save(os.path.join(out_dir, "test_preds.npy"), test_preds)
    np.save(os.path.join(out_dir, "y_train_true.npy"), train_labels)
    np.save(os.path.join(out_dir, "y_test_true.npy"), test_labels)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    summary = {
        "model": "cifar_realistic_teacher",
        "train_size": REALISTIC_TRAIN_SIZE,
        "best_epoch": best_epoch,
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "train_f1": float(train_f1),
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved realistic teacher:", out_dir)
    print(summary)


# ================================
# Main
# ================================
if __name__ == "__main__":
    train_overfit_teacher()
    train_realistic_teacher()