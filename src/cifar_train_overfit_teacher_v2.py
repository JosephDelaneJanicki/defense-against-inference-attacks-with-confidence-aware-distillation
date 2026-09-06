import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset


# ================================
# Config
# ================================
RESULTS_DIR = "results/cifar/overfit_teacher_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
NUM_CLASSES = 10

TRAIN_SIZE = 15000
EPOCHS = 10
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================
# Transform
# ================================
basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


# ================================
# Dataset
# ================================
train_set = torchvision.datasets.CIFAR10(
    root="data/cifar10",
    train=True,
    download=True,
    transform=basic_transform,
)

test_set = torchvision.datasets.CIFAR10(
    root="data/cifar10",
    train=False,
    download=True,
    transform=basic_transform,
)

all_indices = np.arange(len(train_set))
rng = np.random.default_rng(SEED)
rng.shuffle(all_indices)

train_indices = all_indices[:TRAIN_SIZE]

train_subset = Subset(train_set, train_indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
train_eval_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


# ================================
# Model
# ================================
class CIFARSmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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

    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    all_conf = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())
            all_conf.extend(conf.cpu().numpy())

    return (
        np.vstack(all_logits),
        np.vstack(all_probs),
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_conf),
    )


# ================================
# Train
# ================================
model = CIFARSmallCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

history = []

for epoch in range(EPOCHS):
    model.train()

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    train_loss, train_acc, train_f1 = evaluate(model, train_eval_loader)
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

    print(
        f"Epoch {epoch+1:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
    )


# ================================
# Save Outputs
# ================================
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model.pt"))

train_logits, train_probs, train_preds, train_labels, train_conf = get_outputs(
    model,
    train_eval_loader,
)

test_logits, test_probs, test_preds, test_labels, test_conf = get_outputs(
    model,
    test_loader,
)

np.save(os.path.join(RESULTS_DIR, "train_logits.npy"), train_logits)
np.save(os.path.join(RESULTS_DIR, "test_logits.npy"), test_logits)

np.save(os.path.join(RESULTS_DIR, "train_probs.npy"), train_probs)
np.save(os.path.join(RESULTS_DIR, "test_probs.npy"), test_probs)

np.save(os.path.join(RESULTS_DIR, "train_preds.npy"), train_preds)
np.save(os.path.join(RESULTS_DIR, "test_preds.npy"), test_preds)

np.save(os.path.join(RESULTS_DIR, "y_train_true.npy"), train_labels)
np.save(os.path.join(RESULTS_DIR, "y_test_true.npy"), test_labels)

np.save(os.path.join(RESULTS_DIR, "train_confidence.npy"), train_conf)
np.save(os.path.join(RESULTS_DIR, "test_confidence.npy"), test_conf)

np.save(os.path.join(RESULTS_DIR, "train_indices.npy"), train_indices)

with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
    json.dump(history, f, indent=4)

summary = {
    "model": "cifar_overfit_teacher_v2",
    "train_size": TRAIN_SIZE,
    "epochs": EPOCHS,
    "train_acc": float(history[-1]["train_acc"]),
    "test_acc": float(history[-1]["test_acc"]),
    "train_f1": float(history[-1]["train_f1"]),
    "test_f1": float(history[-1]["test_f1"]),
    "train_conf_min": float(train_conf.min()),
    "train_conf_mean": float(train_conf.mean()),
    "train_conf_median": float(np.median(train_conf)),
    "train_conf_max": float(train_conf.max()),
    "test_conf_mean": float(test_conf.mean()),
}

with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print("\nSaved CIFAR overfit teacher v2:")
print(RESULTS_DIR)
print(summary)