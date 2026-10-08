import csv
import json
from pathlib import Path

import pandas as pd


RESULTS_ROOT = Path("results/gemma")
DEFENSE_ROOT = RESULTS_ROOT / "defense"
ANALYSIS_ROOT = RESULTS_ROOT / "analysis"
MASTER_ROOT = ANALYSIS_ROOT / "master"

SEEDS = [42, 123, 456, 789]

ATTACK_TYPES = [
    "transfer",
    "black_box",
    "adaptive",
]


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def safe_read_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def find_first_csv(root, patterns):
    if not root.exists():
        return None

    for pattern in patterns:
        matches = sorted(root.rglob(pattern))

        if matches:
            return matches[0]

    return None


def find_training_summary(defense_dir):
    candidates = [
        defense_dir / "training_summary.json",
        defense_dir / "adapter" / "training_summary.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(defense_dir.glob("*training_summary*.json"))

    return matches[0] if matches else None


def normalize_utility(seed, defense, defense_dir):
    summary_path = find_training_summary(defense_dir)

    if summary_path is None:
        return None

    summary = safe_read_json(summary_path)

    if summary is None:
        return None

    rows = []

    mappings = {
        "initial_validation_nll": [
            "initial_eval_loss",
            "initial_validation_loss",
        ],
        "final_validation_nll": [
            "final_eval_loss",
            "final_validation_loss",
            "validation_nll",
        ],
        "train_loss": [
            "train_loss",
        ],
        "training_seconds": [
            "training_seconds",
        ],
    }

    for metric, keys in mappings.items():
        value = None

        for key in keys:
            if key in summary:
                value = summary[key]
                break

        if value is not None:
            rows.append({
                "seed": seed,
                "defense": defense,
                "metric": metric,
                "value": value,
                "source": str(summary_path),
            })

    # Handle either our older nested GPU format or newer flat format.
    peak_vram = summary.get("peak_vram_gb")

    if peak_vram is None:
        gpu_memory = summary.get("gpu_memory", {})

        if isinstance(gpu_memory, dict):
            peak_vram = gpu_memory.get("peak_allocated_gb")

    if peak_vram is not None:
        rows.append({
            "seed": seed,
            "defense": defense,
            "metric": "peak_vram_gb",
            "value": peak_vram,
            "source": str(summary_path),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


def normalize_direct_mia(seed, defense, defense_dir):
    path = find_first_csv(
        defense_dir,
        [
            "membership_auc.csv",
            "*membership_auc*.csv",
        ],
    )

    if path is None:
        return None

    df = safe_read_csv(path)

    if df is None or df.empty:
        return None

    rows = []

    # Support common layouts:
    # signal, roc_auc
    # feature, auc
    # metric, auc
    signal_column = None
    auc_column = None

    for candidate in ["signal", "feature", "metric"]:
        if candidate in df.columns:
            signal_column = candidate
            break

    for candidate in ["roc_auc", "auc", "AUC"]:
        if candidate in df.columns:
            auc_column = candidate
            break

    if signal_column is not None and auc_column is not None:
        for _, row in df.iterrows():
            rows.append({
                "seed": seed,
                "defense": defense,
                "signal": row[signal_column],
                "roc_auc": row[auc_column],
                "source": str(path),
            })

    # Also support one-row wide files.
    elif len(df) == 1:
        for column in df.columns:
            if "auc" in column.lower():
                rows.append({
                    "seed": seed,
                    "defense": defense,
                    "signal": column,
                    "roc_auc": df.iloc[0][column],
                    "source": str(path),
                })

    if not rows:
        print(
            f"[WARN] Could not normalize direct MIA layout: {path}"
        )
        return None

    return pd.DataFrame(rows)


def normalize_attack_csv(
    seed,
    defense,
    attack_type,
    path,
):
    df = safe_read_csv(path)

    if df is None or df.empty:
        return None

    # We only normalize summary-level CSVs.
    # Prediction files can contain thousands of rows and should
    # never be copied into the analysis summary tree.
    if len(df) > 20:
        return None

    attacker_column = None

    for candidate in [
        "attacker",
        "model",
        "attack_model",
        "classifier",
    ]:
        if candidate in df.columns:
            attacker_column = candidate
            break

    metric_columns = {
        "accuracy": ["accuracy", "acc"],
        "precision": ["precision"],
        "recall": ["recall"],
        "f1": ["f1", "f1_score"],
        "roc_auc": ["roc_auc", "auc"],
    }

    rows = []

    for index, source_row in df.iterrows():

        if attacker_column:
            attacker = str(
                source_row[attacker_column]
            )
        elif len(df) == 2:
            attacker = (
                "logistic"
                if index == df.index[0]
                else "random_forest"
            )
        else:
            attacker = "unknown"

        output = {
            "seed": seed,
            "defense": defense,
            "attack_type": attack_type,
            "attacker": attacker,
            "source": str(path),
        }

        found_metric = False

        for normalized, candidates in metric_columns.items():

            value = None

            for candidate in candidates:
                if candidate in df.columns:
                    value = source_row[candidate]
                    break

            output[normalized] = value

            if value is not None:
                found_metric = True

        if found_metric:
            rows.append(output)

    if not rows:
        return None

    return pd.DataFrame(rows)


def find_attack_summary(defense_dir, attack_type):
    patterns = {
        "transfer": [
            "*nonadaptive*summary*.csv",
            "*transfer*summary*.csv",
        ],
        "black_box": [
            "*black_box*summary*.csv",
            "*black*box*summary*.csv",
        ],
        "adaptive": [
            "*adaptive_mia_summary*.csv",
            "*adaptive*summary*.csv",
        ],
    }

    return find_first_csv(
        defense_dir,
        patterns[attack_type],
    )


def discover_defenses():
    discovered = {}

    if not DEFENSE_ROOT.exists():
        return discovered

    for seed_dir in sorted(DEFENSE_ROOT.glob("seed_*")):

        try:
            seed = int(
                seed_dir.name.split("_", 1)[1]
            )
        except ValueError:
            continue

        discovered[seed] = {}

        for child in sorted(seed_dir.iterdir()):

            if not child.is_dir():
                continue

            # A defense is considered a target model directory
            # only if it has an adapter.
            if not (child / "adapter").exists():
                continue

            discovered[seed][child.name] = child

    return discovered


def write_manifest(discovered, status_rows):
    manifest = pd.DataFrame(status_rows)

    path = ANALYSIS_ROOT / "manifest.csv"

    manifest.to_csv(path, index=False)

    print(f"[WRITE] {path}")


def aggregate_attacks(all_attack_frames):
    if not all_attack_frames:
        return

    master = pd.concat(
        all_attack_frames,
        ignore_index=True,
    )

    MASTER_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_path = (
        MASTER_ROOT /
        "all_attack_results.csv"
    )

    master.to_csv(
        master_path,
        index=False,
    )

    print(f"[WRITE] {master_path}")

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    records = []

    group_columns = [
        "defense",
        "attack_type",
        "attacker",
    ]

    for keys, group in master.groupby(group_columns):

        defense, attack_type, attacker = keys

        for metric in metrics:

            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            records.append({
                "defense": defense,
                "attack_type": attack_type,
                "attacker": attacker,
                "metric": metric,
                "n_seeds": int(values.count()),
                "mean": values.mean(),
                "std": (
                    values.std(ddof=1)
                    if len(values) > 1
                    else float("nan")
                ),
                "min": values.min(),
                "max": values.max(),
            })

    aggregate = pd.DataFrame(records)

    aggregate_path = (
        MASTER_ROOT /
        "attack_summary.csv"
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    print(f"[WRITE] {aggregate_path}")


def aggregate_utility(all_utility_frames):
    if not all_utility_frames:
        return

    master = pd.concat(
        all_utility_frames,
        ignore_index=True,
    )

    path = MASTER_ROOT / "all_utility_results.csv"

    master.to_csv(path, index=False)

    print(f"[WRITE] {path}")

    aggregate = (
        master
        .groupby(["defense", "metric"])["value"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_seeds"})
    )

    aggregate_path = (
        MASTER_ROOT /
        "utility_summary.csv"
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    print(f"[WRITE] {aggregate_path}")


def aggregate_direct(all_direct_frames):
    if not all_direct_frames:
        return

    master = pd.concat(
        all_direct_frames,
        ignore_index=True,
    )

    path = MASTER_ROOT / "all_direct_mia.csv"

    master.to_csv(path, index=False)

    print(f"[WRITE] {path}")

    aggregate = (
        master
        .groupby(["defense", "signal"])["roc_auc"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_seeds"})
    )

    aggregate_path = (
        MASTER_ROOT /
        "direct_mia_summary.csv"
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    print(f"[WRITE] {aggregate_path}")


def main():

    print("=" * 80)
    print("GEMMA ANALYSIS TREE BUILDER")
    print("=" * 80)

    ANALYSIS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    MASTER_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    discovered = discover_defenses()

    all_attacks = []
    all_utility = []
    all_direct = []
    status_rows = []

    for seed in sorted(discovered):

        for defense, defense_dir in sorted(
            discovered[seed].items()
        ):

            print(
                f"\nseed={seed} defense={defense}"
            )

            output_dir = (
                ANALYSIS_ROOT /
                defense /
                f"seed_{seed}"
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ---------------------------------------------
            # Utility
            # ---------------------------------------------

            utility = normalize_utility(
                seed,
                defense,
                defense_dir,
            )

            utility_status = "missing"

            if utility is not None:
                utility.to_csv(
                    output_dir / "utility.csv",
                    index=False,
                )

                all_utility.append(utility)
                utility_status = "complete"

            # ---------------------------------------------
            # Direct MIA
            # ---------------------------------------------

            direct = normalize_direct_mia(
                seed,
                defense,
                defense_dir,
            )

            direct_status = "missing"

            if direct is not None:
                direct.to_csv(
                    output_dir / "direct_mia.csv",
                    index=False,
                )

                all_direct.append(direct)
                direct_status = "complete"

            # ---------------------------------------------
            # Learned attacks
            # ---------------------------------------------

            attack_status = {}

            for attack_type in ATTACK_TYPES:

                source = find_attack_summary(
                    defense_dir,
                    attack_type,
                )

                if source is None:
                    attack_status[attack_type] = "missing"
                    continue

                normalized = normalize_attack_csv(
                    seed,
                    defense,
                    attack_type,
                    source,
                )

                if normalized is None:
                    attack_status[attack_type] = "unreadable"
                    continue

                normalized.to_csv(
                    output_dir /
                    f"{attack_type}_mia.csv",
                    index=False,
                )

                all_attacks.append(normalized)

                attack_status[attack_type] = "complete"

            status_rows.append({
                "seed": seed,
                "defense": defense,
                "utility": utility_status,
                "direct": direct_status,
                "transfer": attack_status.get(
                    "transfer",
                    "missing",
                ),
                "black_box": attack_status.get(
                    "black_box",
                    "missing",
                ),
                "adaptive": attack_status.get(
                    "adaptive",
                    "missing",
                ),
                "target_adapter": str(
                    defense_dir / "adapter"
                ),
            })

    write_manifest(
        discovered,
        status_rows,
    )

    aggregate_attacks(all_attacks)
    aggregate_utility(all_utility)
    aggregate_direct(all_direct)

    print("\n" + "=" * 80)
    print("ANALYSIS TREE COMPLETE")
    print("=" * 80)

    print(
        "\nManifest:",
        ANALYSIS_ROOT / "manifest.csv",
    )

    print(
        "Master:",
        MASTER_ROOT,
    )


if __name__ == "__main__":
    main()