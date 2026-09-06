import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

# ================================
# Config
# ================================
SEED = 42
BATCH_SIZE = 128
NUM_CLASSES = 10

NUM_SHADOW_MODELS = 6
SHADOW_EPOCHS = 30
ATTACK_EPOCHS = 40

RESULTS_DIR = "results/cifar/attacks"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ================================
# Data
# ================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

train_set = torchvision.datasets.CIFAR10(
    root="data/cifar10", train=True, download=True, transform=transform
)

test_set = torchvision.datasets.CIFAR10(
    root="data/cifar10", train=False, download=True, transform=transform
)

# ================================
# Load teacher outputs
# ================================
teacher_dir = "results/cifar/overfit_teacher_v2"

train_probs = np.load(os.path.join(teacher_dir, "train_probs.npy"))
test_probs = np.load(os.path.join(teacher_dir, "test_probs.npy"))

y_train = np.load(os.path.join(teacher_dir, "y_train_true.npy"))
y_test = np.load(os.path.join(teacher_dir, "y_test_true.npy"))

train_indices = np.load(os.path.join(teacher_dir, "train_indices.npy"))

# ================================
# Feature extraction
# ================================
def extract_features(probs, true_labels):
    eps = 1e-10

    sorted_desc = np.sort(probs, axis=1)[:, ::-1]

    top1 = sorted_desc[:, 0]
    top2 = sorted_desc[:, 1]
    top3 = sorted_desc[:, 2]

    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    margin = top1 - top2

    pred = np.argmax(probs, axis=1)
    correct = (pred == true_labels).astype(float)

    loss = -np.log(probs[np.arange(len(probs)), true_labels] + eps)

    features = np.column_stack([
        probs,
        sorted_desc,
        entropy,
        margin,
        loss,
        correct
    ])

    return features, pred

# ================================
# Attack model
# ================================
class AttackModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# ================================
# Build dataset
# ================================
train_features, train_pred = extract_features(train_probs, y_train)
test_features, test_pred = extract_features(test_probs, y_test)

n = min(len(train_features), len(test_features))

idx_train = np.random.choice(len(train_features), n, replace=False)
idx_test = np.random.choice(len(test_features), n, replace=False)

X = np.vstack([train_features[idx_train], test_features[idx_test]])
y = np.concatenate([np.ones(n), np.zeros(n)])

# ================================
# Train attack model
# ================================
X_train, X_val, y_train_split, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_split, dtype=torch.long)

X_val_t = torch.tensor(X_val, dtype=torch.float32)

loader = DataLoader(
    torch.utils.data.TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = AttackModel(X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(ATTACK_EPOCHS):
    model.train()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

# ================================
# Evaluate
# ================================
model.eval()

with torch.no_grad():
    logits = model(X_val_t.to(device))
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

preds = (probs >= 0.5).astype(int)

metrics = {
    "attack_accuracy": accuracy_score(y_val, preds),
    "attack_precision": precision_score(y_val, preds),
    "attack_recall": recall_score(y_val, preds),
    "attack_f1": f1_score(y_val, preds),
    "attack_auc": roc_auc_score(y_val, probs),
}

print("\nOverfit Teacher v2 Attack Result:")
print(metrics)

# Save
out_path = os.path.join(
    RESULTS_DIR,
    "cifar_overfit_v2_attack.csv"
)

import pandas as pd
pd.DataFrame([metrics]).to_csv(out_path, index=False)

print("Saved:", out_path)