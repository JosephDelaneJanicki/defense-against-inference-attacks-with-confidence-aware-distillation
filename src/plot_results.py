import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================
# Load CSVs
# ================================
adult_path = "results/student_attacks/student_attack_grid_v3style_ns6_seed42.csv"
cifar_path = "results/cifar/student_attacks/cifar_student_attack_v3_ns6_seed42.csv"

adult_df = pd.read_csv(adult_path)
cifar_df = pd.read_csv(cifar_path)

# Combine for unified plotting
adult_df["dataset"] = "Adult"
cifar_df["dataset"] = "CIFAR-10"

df = pd.concat([adult_df, cifar_df], ignore_index=True)

# ================================
# Load teacher results manually
# ================================

teacher_rows = [
    {
        "model": "Adult Overfit Teacher",
        "dataset": "Adult",
        "test_acc": 0.8395,
        "attack_auc": 0.5812,
    },
    {
        "model": "CIFAR Overfit Teacher v2",
        "dataset": "CIFAR-10",
        "test_acc": 0.6923,
        "attack_auc": 0.6746,
    },
    {
        "model": "CIFAR Realistic Teacher",
        "dataset": "CIFAR-10",
        "test_acc": 0.8680,
        "attack_auc": 0.5372,
    },
]

teacher_df = pd.DataFrame(teacher_rows)

# ================================
# Plot 1: Accuracy vs AUC
# ================================
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="test_acc",
    y="attack_auc",
    hue="dataset",
    style="label_mode",
    s=100
)

plt.axhline(0.5, linestyle="--", color="red", label="Random Attack")

plt.title("Model Accuracy vs Membership Leakage (AUC)")
plt.xlabel("Test Accuracy")
plt.ylabel("Attack AUC")

plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_auc.png"))
plt.close()


# ================================
# Plot 2: Variant Comparison
# ================================
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="label_mode",
    y="attack_auc",
    hue="dataset"
)

plt.axhline(0.5, linestyle="--", color="red")

plt.title("Attack AUC by Label Mode")
plt.xlabel("Label Mode")
plt.ylabel("Attack AUC")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "auc_by_label_mode.png"))
plt.close()


# ================================
# Plot 3: Filter vs Original
# ================================
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="filter_mode",
    y="attack_auc",
    hue="dataset"
)

plt.axhline(0.5, linestyle="--", color="red")

plt.title("Effect of Filtering on Attack AUC")
plt.xlabel("Filter Mode")
plt.ylabel("Attack AUC")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "filter_effect.png"))
plt.close()


# ================================
# Plot 4: Accuracy comparison
# ================================
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="label_mode",
    y="test_acc",
    hue="dataset"
)

plt.title("Accuracy by Label Mode")
plt.xlabel("Label Mode")
plt.ylabel("Test Accuracy")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_by_label_mode.png"))
plt.close()

# ======================================
# Plot 5: teacher vs student scatter
# ======================================

plt.figure(figsize=(10, 6))

# Students
sns.scatterplot(
    data=df,
    x="test_acc",
    y="attack_auc",
    hue="dataset",
    style="label_mode",
    alpha=0.7,
    s=80,
)

# Teachers
sns.scatterplot(
    data=teacher_df,
    x="test_acc",
    y="attack_auc",
    color="black",
    s=150,
    marker="X",
)
# Label teacher points
for _, teacher in teacher_df.iterrows():
    plt.annotate(
        teacher["model"],
        (teacher["test_acc"], teacher["attack_auc"]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        fontweight="bold"
    )
plt.axhline(0.5, linestyle="--", color="red")

plt.title("Teacher vs Student: Accuracy vs Leakage")
plt.xlabel("Test Accuracy")
plt.ylabel("Attack AUC")

from matplotlib.ticker import MultipleLocator

ax = plt.gca()

# Enable minor ticks 
ax.minorticks_on()

# Major ticks
ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.05))

# Minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(0.01))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))

# Grid styling
ax.grid(which='major', linestyle='-', linewidth= 1.2, alpha=1)
ax.grid(which='minor', linestyle='--', linewidth=1, alpha=0.8)

# NOW do layout + save
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "teacher_vs_student_scatter.png"))
plt.close()

# =====================================
# Plot 6: Arrow Plot
# =====================================

plt.figure(figsize=(10, 6))

for _, teacher in teacher_df.iterrows():
    dataset = teacher["dataset"]

    # match students from same dataset
    if teacher["model"] == "CIFAR Overfit v2":
        students = df[df["teacher_name"] == "overfit_teacher_v2"]
    elif teacher["model"] == "CIFAR Realistic":
        students = df[df["teacher_name"] == "realistic_teacher"]
    else:
        students = df[df["dataset"] == dataset]

    for _, student in students.iterrows():
        plt.arrow(
            teacher["test_acc"],
            teacher["attack_auc"],
            student["test_acc"] - teacher["test_acc"],
            student["attack_auc"] - teacher["attack_auc"],
            alpha=0.3,
            head_width=0.005,
            length_includes_head=True
        )

# Plot teacher points
plt.scatter(
    teacher_df["test_acc"],
    teacher_df["attack_auc"],
    color="black",
    s=150,
    label="Teacher",
    marker="X"
)
# Label teacher points
for _, teacher in teacher_df.iterrows():
    plt.annotate(
        teacher["model"],
        (teacher["test_acc"], teacher["attack_auc"]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        fontweight="bold"
    )
# Plot student points
plt.scatter(
    df["test_acc"],
    df["attack_auc"],
    alpha=0.6,
    label="Students"
)

plt.axhline(0.5, linestyle="--", color="red")

plt.title("Teacher → Student Transformation (Accuracy vs Leakage)")
plt.xlabel("Test Accuracy")
plt.ylabel("Attack AUC")

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "teacher_to_student_arrows.png"))
plt.close()



print("Plots saved to:", OUTPUT_DIR)