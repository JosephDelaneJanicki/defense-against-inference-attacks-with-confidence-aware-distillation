import os
import json
import numpy as np
import pandas as pd

# ================================
# Config
# ================================
CIFAR_RESULTS_DIR = "results/cifar"
OUTPUT_DIR = "data/cifar_student_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMPERATURE = 2.0
CLIP_MAX = 0.85
FILTER_CONFIDENCE_THRESHOLD = 0.95

TEACHERS = {
    "overfit_teacher_v2": "results/cifar/overfit_teacher_v2",
    "realistic_teacher": "results/cifar/realistic_teacher",
}


# ================================
# Label Processing
# ================================
def temperature_smooth_probs(probs, temperature):
    eps = 1e-10
    logits = np.log(probs + eps)
    scaled = logits / temperature
    exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def clip_probabilities(probs, clip_max):
    num_classes = probs.shape[1]
    min_val = (1.0 - clip_max) / (num_classes - 1)

    clipped = np.clip(probs, min_val, clip_max)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)

    return clipped


manifest = []

for teacher_name, teacher_dir in TEACHERS.items():
    print("\n" + "=" * 80)
    print(f"Generating CIFAR student datasets from {teacher_name}")
    print("=" * 80)

    train_probs = np.load(os.path.join(teacher_dir, "train_probs.npy"))
    test_probs = np.load(os.path.join(teacher_dir, "test_probs.npy"))

    train_preds = np.load(os.path.join(teacher_dir, "train_preds.npy"))
    test_preds = np.load(os.path.join(teacher_dir, "test_preds.npy"))

    y_train_true = np.load(os.path.join(teacher_dir, "y_train_true.npy"))
    y_test_true = np.load(os.path.join(teacher_dir, "y_test_true.npy"))

    train_conf = np.max(train_probs, axis=1)

    label_variants = {
        "raw_soft": {
            "type": "soft",
            "train": train_probs,
            "test": test_probs,
        },
        "temp_soft": {
            "type": "soft",
            "train": temperature_smooth_probs(train_probs, TEMPERATURE),
            "test": temperature_smooth_probs(test_probs, TEMPERATURE),
        },
        "clip_soft": {
            "type": "soft",
            "train": clip_probabilities(train_probs, CLIP_MAX),
            "test": clip_probabilities(test_probs, CLIP_MAX),
        },
        "hard": {
            "type": "hard",
            "train": train_preds,
            "test": test_preds,
        },
    }
    FILTER_TOP_PERCENT = 0.20
    cutoff = np.quantile(train_conf, 1.0 - FILTER_TOP_PERCENT)

    filter_modes = {
        "original": None,
        "filtered": train_conf <= cutoff,
    }

    for label_name, label_data in label_variants.items():
        for filter_name, train_filter in filter_modes.items():
            variant_name = f"{teacher_name}_{label_name}_{filter_name}"
            variant_dir = os.path.join(OUTPUT_DIR, variant_name)
            os.makedirs(variant_dir, exist_ok=True)

            y_train_student = label_data["train"]
            y_test_student = label_data["test"]
            y_train_true_variant = y_train_true
            train_conf_variant = train_conf

            if train_filter is not None:
                y_train_student = y_train_student[train_filter]
                y_train_true_variant = y_train_true_variant[train_filter]
                train_conf_variant = train_conf_variant[train_filter]

            np.save(os.path.join(variant_dir, "y_train_student.npy"), y_train_student)
            np.save(os.path.join(variant_dir, "y_test_student.npy"), y_test_student)

            np.save(os.path.join(variant_dir, "y_train_true.npy"), y_train_true_variant)
            np.save(os.path.join(variant_dir, "y_test_true.npy"), y_test_true)

            np.save(os.path.join(variant_dir, "teacher_train_confidence.npy"), train_conf_variant)

            metadata = {
                "variant_name": variant_name,
                "teacher_name": teacher_name,
                "label_mode": label_name,
                "label_type": label_data["type"],
                "filter_mode": filter_name,
                "num_train_samples": int(len(y_train_true_variant)),
                "num_test_samples": int(len(y_test_true)),
                "temperature": TEMPERATURE if label_name == "temp_soft" else None,
                "clip_max": CLIP_MAX if label_name == "clip_soft" else None,
                "filter_confidence_cutoff": float(cutoff) if filter_name == "filtered" else None,
                "filter_removed_top_percent": FILTER_TOP_PERCENT if filter_name == "filtered" else None,
            }

            with open(os.path.join(variant_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)

            manifest.append(metadata)

            print(
                f"Saved {variant_name} | "
                f"train samples: {len(y_train_true_variant)} | "
                f"label type: {label_data['type']}"
            )


manifest_df = pd.DataFrame(manifest)
manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
manifest_df.to_csv(manifest_path, index=False)

print("\nDone generating CIFAR student dataset labels.")
print("Manifest:", manifest_path)
print(manifest_df)