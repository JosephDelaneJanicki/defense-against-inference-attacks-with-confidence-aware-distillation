import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUTPUT_DIR = "results/final_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPER_CSV = "results/super_results.csv"


# ================================
# Helpers
# ================================
def add_row(rows, **kwargs):
    base = {
        "dataset": None,
        "role": None,              # teacher or student
        "teacher_group": None,     # adult_overfit, cifar_overfit_v2, cifar_realistic
        "model_name": None,
        "label_mode": None,
        "label_type": None,
        "feature_mode": None,
        "filter_mode": None,
        "test_acc": None,
        "test_f1": None,
        "attack_auc": None,
        "attack_accuracy": None,
        "attack_precision": None,
        "attack_recall": None,
        "attack_f1": None,
    }
    base.update(kwargs)
    rows.append(base)


rows = []


# ================================
# Teacher rows
# ================================
# Adult overfit teacher from earlier Model A attack/results.
add_row(
    rows,
    dataset="Adult",
    role="teacher",
    teacher_group="adult_overfit",
    model_name="Adult Overfit Teacher",
    test_acc=0.8395,
    test_f1=None,
    attack_auc=0.5812,
)

# CIFAR overfit teacher v2.
add_row(
    rows,
    dataset="CIFAR-10",
    role="teacher",
    teacher_group="cifar_overfit_v2",
    model_name="CIFAR Overfit Teacher v2",
    test_acc=0.6923,
    test_f1=0.6908,
    attack_auc=0.674572,
    attack_accuracy=0.635,
    attack_precision=0.593167701863354,
    attack_recall=0.8595,
    attack_f1=0.7019191506737444,
)

# CIFAR realistic teacher.
add_row(
    rows,
    dataset="CIFAR-10",
    role="teacher",
    teacher_group="cifar_realistic",
    model_name="CIFAR Realistic Teacher",
    test_acc=0.8680,
    test_f1=None,
    attack_auc=0.537200395,
    attack_accuracy=0.5291,
    attack_precision=0.5157689389834182,
    attack_recall=0.9518,
    attack_f1=0.6690096295775638,
)


# ================================
# Adult student rows
# ================================
adult_students_path = "results/student_attacks/student_attack_grid_v3style_ns6_seed42.csv"

adult_students = pd.read_csv(adult_students_path)

for _, r in adult_students.iterrows():
    add_row(
        rows,
        dataset="Adult",
        role="student",
        teacher_group="adult_overfit",
        model_name=r["variant_name"],
        label_mode=r.get("label_mode"),
        label_type=r.get("label_type"),
        feature_mode=r.get("feature_mode"),
        filter_mode=r.get("feature_mode"),
        test_acc=r.get("test_acc"),
        test_f1=r.get("test_f1"),
        attack_auc=r.get("attack_auc"),
        attack_accuracy=r.get("attack_accuracy"),
        attack_precision=r.get("attack_precision"),
        attack_recall=r.get("attack_recall"),
        attack_f1=r.get("attack_f1"),
    )


# ================================
# CIFAR student rows
# ================================
cifar_students_path = "results/cifar/student_attacks/cifar_student_attack_v3_ns6_seed42.csv"

cifar_students = pd.read_csv(cifar_students_path)

def map_cifar_teacher_group(teacher_name):
    if teacher_name == "overfit_teacher_v2":
        return "cifar_overfit_v2"
    if teacher_name == "realistic_teacher":
        return "cifar_realistic"
    return teacher_name

for _, r in cifar_students.iterrows():
    add_row(
        rows,
        dataset="CIFAR-10",
        role="student",
        teacher_group=map_cifar_teacher_group(r["teacher_name"]),
        model_name=r["variant_name"],
        label_mode=r.get("label_mode"),
        label_type=r.get("label_type"),
        feature_mode=None,
        filter_mode=r.get("filter_mode"),
        test_acc=r.get("test_acc"),
        test_f1=r.get("test_f1"),
        attack_auc=r.get("attack_auc"),
        attack_accuracy=r.get("attack_accuracy"),
        attack_precision=r.get("attack_precision"),
        attack_recall=r.get("attack_recall"),
        attack_f1=r.get("attack_f1"),
    )


# ================================
# Save super CSV
# ================================
super_df = pd.DataFrame(rows)
super_df.to_csv(SUPER_CSV, index=False)

print(f"Saved super CSV: {SUPER_CSV}")
print(super_df)


# ================================
# Combined scatter + arrow plot
# ================================
plt.figure(figsize=(12, 7))
ax = plt.gca()

teacher_df = super_df[super_df["role"] == "teacher"]
student_df = super_df[super_df["role"] == "student"]

colors = {
    "adult_overfit": "tab:blue",
    "cifar_overfit_v2": "tab:orange",
    "cifar_realistic": "tab:green",
}

markers = {
    "raw_soft": "o",
    "temp_soft": "s",
    "clip_soft": "D",
    "hard": "^",
}

# Draw arrows first
for _, teacher in teacher_df.iterrows():
    students = student_df[student_df["teacher_group"] == teacher["teacher_group"]]

    for _, student in students.iterrows():
        ax.arrow(
            teacher["test_acc"],
            teacher["attack_auc"],
            student["test_acc"] - teacher["test_acc"],
            student["attack_auc"] - teacher["attack_auc"],
            color=colors.get(teacher["teacher_group"], "gray"),
            alpha=0.22,
            linewidth=0.8,
            head_width=0.004,
            head_length=0.006,
            length_includes_head=True,
        )

# Plot students
for _, student in student_df.iterrows():
    ax.scatter(
        student["test_acc"],
        student["attack_auc"],
        color=colors.get(student["teacher_group"], "gray"),
        marker=markers.get(student["label_mode"], "o"),
        s=55,
        alpha=0.75,
    )

# ================================
# Draw filtered vs original arrows
# ================================
# Small black arrows:
# original -> filtered for each teacher_group + label_mode

for teacher_group in student_df["teacher_group"].dropna().unique():
    group_df = student_df[student_df["teacher_group"] == teacher_group]

    for label_mode in group_df["label_mode"].dropna().unique():
        pair = group_df[group_df["label_mode"] == label_mode]

        original = pair[pair["filter_mode"].isin(["original", None])]
        filtered = pair[pair["filter_mode"] == "filtered"]

        if len(original) == 0 or len(filtered) == 0:
            continue

        original = original.iloc[0]
        filtered = filtered.iloc[0]

        ax.annotate(
            "",
            xy=(filtered["test_acc"], filtered["attack_auc"]),
            xytext=(original["test_acc"], original["attack_auc"]),
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                linewidth=1.1,
                alpha=0.75,
                shrinkA=4,
                shrinkB=4,
            ),
        )

# Plot teachers
for _, teacher in teacher_df.iterrows():
    ax.scatter(
        teacher["test_acc"],
        teacher["attack_auc"],
        color="black",
        marker="X",
        s=180,
        zorder=5,
    )
    ax.annotate(
        teacher["model_name"],
        (teacher["test_acc"], teacher["attack_auc"]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        fontweight="bold",
    )

# Random attack reference
ax.axhline(0.5, linestyle="--", color="red", linewidth=1.2, label="Random Attack AUC")

# Ticks / grid
ax.minorticks_on()
ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.xaxis.set_minor_locator(MultipleLocator(0.01))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))
ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.65)
ax.grid(which="minor", linestyle="--", linewidth=0.45, alpha=0.35)

ax.set_title("Teacher → Student: Accuracy vs Membership Leakage")
ax.set_xlabel("Test Accuracy")
ax.set_ylabel("Attack ROC AUC")

# Manual legend
for group, color in colors.items():
    ax.scatter([], [], color=color, marker="o", label=group)

for label, marker in markers.items():
    ax.scatter([], [], color="gray", marker=marker, label=label)

ax.scatter([], [], color="black", marker="X", s=120, label="Teacher")

ax.plot([], [], color="black", linewidth=1.1, label="Original → Filtered")

ax.legend(loc="best", fontsize=8)

from matplotlib.ticker import MultipleLocator

ax.minorticks_on()

ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.05))

ax.xaxis.set_minor_locator(MultipleLocator(0.01))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))

ax.grid(which="major", linestyle="-", linewidth=1, alpha=1)
ax.grid(which="minor", linestyle="--", linewidth=0.7, alpha=0.7)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "teacher_student_super_plot.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved plot: {out_path}")


# ================================
# Optional: grouped bar/point table CSV
# ================================
summary_cols = [
    "dataset",
    "role",
    "teacher_group",
    "model_name",
    "label_mode",
    "filter_mode",
    "test_acc",
    "test_f1",
    "attack_auc",
]

summary_path = "results/super_results_summary.csv"
super_df[summary_cols].to_csv(summary_path, index=False)
print(f"Saved compact summary: {summary_path}")