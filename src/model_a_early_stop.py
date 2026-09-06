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
# Config
# ================================
EARLY_EPOCHS = 5
MODEL_NAME = f"model_a_early_stop_{EARLY_EPOCHS}ep"

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
batch_size = 256

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=batch_size,
    shuffle=False,
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=batch_size,
    shuffle=False,
)


# ================================
# Model: Same Architecture as Overfit v2
# ================================
class TeacherMLP(nn.Module):
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


model = TeacherMLP(X_train.shape[1]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ================================
# Evaluation
# ================================
def evaluate(model, loader):
    model.eval()

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
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, acc, f1


# ================================
# Train Early-Stop Model
# ================================
train_losses = []
val_losses = []
train_accs = []
val_accs = []
train_f1s = []
val_f1s = []

for epoch in range(EARLY_EPOCHS):
    model.train()

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    train_loss, train_acc, train_f1 = evaluate(model, train_loader)
    val_loss, val_acc, val_f1 = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)

    print(
        f"Epoch {epoch + 1:03d}/{EARLY_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Train F1: {train_f1:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )


# ================================
# Final Evaluation
# ================================
train_loss, train_acc, train_f1 = evaluate(model, train_loader)
val_loss, val_acc, val_f1 = evaluate(model, val_loader)
test_loss, test_acc, test_f1 = evaluate(model, test_loader)

print("\nFinal Early-Stop Model Results")
print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f} | Test F1:  {test_f1:.4f}")
print(f"Generalization Gap: {train_acc - val_acc:.4f}")


# ================================
# Save Curves
# ================================
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Early-Stop Model A Loss ({EARLY_EPOCHS} Epochs)")
plt.legend()
plt.savefig(f"results/{MODEL_NAME}_loss_curve.png", dpi=300)
plt.close()

plt.figure()
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Early-Stop Model A Accuracy ({EARLY_EPOCHS} Epochs)")
plt.legend()
plt.savefig(f"results/{MODEL_NAME}_accuracy_curve.png", dpi=300)
plt.close()

plt.figure()
plt.plot(train_f1s, label="Train F1")
plt.plot(val_f1s, label="Validation F1")
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.title(f"Early-Stop Model A F1 ({EARLY_EPOCHS} Epochs)")
plt.legend()
plt.savefig(f"results/{MODEL_NAME}_f1_curve.png", dpi=300)
plt.close()


# ================================
# Save Confidence Scores / Outputs
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

np.save(f"results/{MODEL_NAME}_train_confidence.npy", train_conf)
np.save(f"results/{MODEL_NAME}_val_confidence.npy", val_conf)
np.save(f"results/{MODEL_NAME}_test_confidence.npy", test_conf)

np.save(f"results/{MODEL_NAME}_train_probs.npy", train_probs)
np.save(f"results/{MODEL_NAME}_val_probs.npy", val_probs)
np.save(f"results/{MODEL_NAME}_test_probs.npy", test_probs)

np.save(f"results/{MODEL_NAME}_train_preds.npy", train_preds)
np.save(f"results/{MODEL_NAME}_val_preds.npy", val_preds)
np.save(f"results/{MODEL_NAME}_test_preds.npy", test_preds)

torch.save(model.state_dict(), f"results/{MODEL_NAME}.pt")

print("\nSaved early-stop artifacts:")
print(f"- results/{MODEL_NAME}.pt")
print(f"- results/{MODEL_NAME}_loss_curve.png")
print(f"- results/{MODEL_NAME}_accuracy_curve.png")
print(f"- results/{MODEL_NAME}_f1_curve.png")
print(f"- results/{MODEL_NAME}_train_confidence.npy")
print(f"- results/{MODEL_NAME}_test_confidence.npy")