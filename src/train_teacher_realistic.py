import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader


# ================================
# Setup
# ================================
os.makedirs("results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)


# ================================
# Load Dataset
# ================================
adult = fetch_ucirepo(id=2)
X = adult.data.features.copy()
y = adult.data.targets.copy()

if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

y = y.astype(str).str.strip().str.replace(".", "", regex=False)
y = (y == ">50K").astype(int)


# ================================
# Preprocessing
# ================================
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


# ================================
# Train / Validation / Test Split
# ================================
X_temp, X_test_df, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_df, X_val_df, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

X_train = preprocessor.fit_transform(X_train_df)
X_val = preprocessor.transform(X_val_df)
X_test = preprocessor.transform(X_test_df)

X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_val = X_val.toarray() if hasattr(X_val, "toarray") else X_val
X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.long)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)


# ================================
# DataLoaders
# ================================
batch_size = 512

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
)


# ================================
# Model A: Realistic Teacher
# ================================
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


model = RealisticTeacherMLP(X_train.shape[1]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
)


# ================================
# Evaluation Function
# ================================
def evaluate(model, loader):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, acc, f1


# ================================
# Training Loop with Early Stopping
# ================================
epochs = 120
patience = 12
best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

train_losses = []
val_losses = []
train_accs = []
val_accs = []
train_f1s = []
val_f1s = []

best_model_path = "results/model_a_realistic_teacher.pt"

for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * xb.size(0)

    train_loss = total_train_loss / len(train_loader.dataset)
    val_loss, val_acc, val_f1 = evaluate(model, val_loader)
    train_eval_loss, train_acc, train_f1 = evaluate(model, train_loader)

    scheduler.step(val_loss)

    train_losses.append(train_eval_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 5 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train Loss: {train_eval_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

    if epochs_without_improvement >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1}. Best epoch: {best_epoch}")
        break


# ================================
# Load Best Model
# ================================
model.load_state_dict(torch.load(best_model_path, map_location=device))


# ================================
# Final Evaluation
# ================================
train_final_loss, train_final_acc, train_final_f1 = evaluate(model, train_loader)
val_final_loss, val_final_acc, val_final_f1 = evaluate(model, val_loader)
test_final_loss, test_final_acc, test_final_f1 = evaluate(model, test_loader)

print("\nFinal Results Using Best Validation Model")
print(f"Best Epoch:      {best_epoch}")
print(f"Train Accuracy:  {train_final_acc:.4f}")
print(f"Val Accuracy:    {val_final_acc:.4f}")
print(f"Test Accuracy:   {test_final_acc:.4f}")
print(f"Train F1:        {train_final_f1:.4f}")
print(f"Val F1:          {val_final_f1:.4f}")
print(f"Test F1:         {test_final_f1:.4f}")


# ================================
# Save Curves
# ================================
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Realistic Model A Training vs Validation Loss")
plt.legend()
plt.savefig("results/model_a_realistic_loss_curve.png", dpi=300)
plt.close()

plt.figure()
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Realistic Model A Training vs Validation Accuracy")
plt.legend()
plt.savefig("results/model_a_realistic_accuracy_curve.png", dpi=300)
plt.close()

plt.figure()
plt.plot(train_f1s, label="Train F1")
plt.plot(val_f1s, label="Validation F1")
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.title("Realistic Model A Training vs Validation F1")
plt.legend()
plt.savefig("results/model_a_realistic_f1_curve.png", dpi=300)
plt.close()


# ================================
# Save Confidence Scores / Probs
# ================================
def get_outputs(model, X_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        confidence, preds = torch.max(probs, dim=1)

    return confidence.cpu().numpy(), probs.cpu().numpy(), preds.cpu().numpy()


train_conf, train_probs, train_preds = get_outputs(model, X_train)
val_conf, val_probs, val_preds = get_outputs(model, X_val)
test_conf, test_probs, test_preds = get_outputs(model, X_test)

np.save("results/model_a_realistic_train_confidence.npy", train_conf)
np.save("results/model_a_realistic_val_confidence.npy", val_conf)
np.save("results/model_a_realistic_test_confidence.npy", test_conf)

np.save("results/model_a_realistic_train_probs.npy", train_probs)
np.save("results/model_a_realistic_val_probs.npy", val_probs)
np.save("results/model_a_realistic_test_probs.npy", test_probs)

np.save("results/model_a_realistic_train_preds.npy", train_preds)
np.save("results/model_a_realistic_val_preds.npy", val_preds)
np.save("results/model_a_realistic_test_preds.npy", test_preds)

print("\nSaved realistic teacher artifacts.")