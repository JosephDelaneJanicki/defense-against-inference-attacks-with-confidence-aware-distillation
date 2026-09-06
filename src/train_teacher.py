import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


# ============================================
#  LOAD DATA
# ============================================
adult = fetch_ucirepo(id=2)
X = adult.data.features.copy()
y = adult.data.targets.copy()

# Make sure y is a 1D binary vector
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

y = y.astype(str).str.strip()
y = (y == ">50K").astype(int)


# ============================================
#  IDENTIFY COLUMN TYPES
# ============================================
categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "category","string"]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)


# ============================================
#  PREPROCESSING
# ============================================
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)


# ============================================
#  TRAIN / TEST SPLIT
# ============================================
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = preprocessor.fit_transform(X_train_df)
X_test = preprocessor.transform(X_test_df)

# Convert sparse matrix to dense for PyTorch
X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)


# ============================================
#  MODEL
# ============================================
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ============================================
#  TRAIN
# ============================================
epochs = 20

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


# ============================================
#  EVALUATE
# ============================================
model.eval()
with torch.no_grad():
    train_logits = model(X_train)
    test_logits = model(X_test)

    train_probs = torch.softmax(train_logits, dim=1)
    test_probs = torch.softmax(test_logits, dim=1)

    train_confidence, train_preds = torch.max(train_probs, dim=1)
    test_confidence, test_preds = torch.max(test_probs, dim=1)

train_acc = accuracy_score(y_train.numpy(), train_preds.numpy())
test_acc = accuracy_score(y_test.numpy(), test_preds.numpy())

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")

print("\nSample train confidences:", train_confidence[:10].numpy())
print("Sample test confidences: ", test_confidence[:10].numpy())

#=============================================
# SAVE RESULTS
#==============================================

import numpy as np

np.save("results/train_confidence.npy", train_confidence.numpy())
np.save("results/test_confidence.npy", test_confidence.numpy())