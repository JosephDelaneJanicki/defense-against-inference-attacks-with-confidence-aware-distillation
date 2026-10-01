import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "google/gemma-3-4b-pt"

DATA_DIR = Path("data/gemma")
CONTROL_DIR = DATA_DIR / "neighbor_control"
MULTISEED_ROOT = Path("results/gemma/multiseed")
OUTPUT_DIR = MULTISEED_ROOT / "neighbor_control"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789]

MAX_LENGTH = 256
BATCH_SIZE = 4

FEATURES = [
    "nll",
    "median_token_loss",
    "min_token_loss",
    "max_token_loss",
    "std_token_loss",
    "min_k_10",
    "min_k_20",
    "min_k_30",
    "min_k_40",
]


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_batch(model, tokenizer, texts):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    )

    input_ids = encoded["input_ids"].cuda()
    attention_mask = encoded["attention_mask"].cuda()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        shift_logits = outputs.logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        token_loss = F.cross_entropy(
            shift_logits.transpose(1, 2),
            shift_labels,
            reduction="none",
        )

    results = []

    for i in range(len(texts)):
        valid = shift_mask[i].bool()
        losses = token_loss[i][valid]

        if losses.numel() == 0:
            raise RuntimeError(
                "Encountered sequence with no valid prediction tokens."
            )

        sorted_losses, _ = torch.sort(
            losses,
            descending=True,
        )

        def min_k_score(frac):
            k = max(
                1,
                int(np.ceil(losses.numel() * frac)),
            )
            # Higher score = more member-like, matching the existing
            # Gemma Min-K convention: negate the worst-token loss.
            return -float(
                sorted_losses[:k].mean().item()
            )

        nll = float(losses.mean().item())

        results.append({
            "nll": nll,
            "perplexity": float(np.exp(min(nll, 50.0))),
            "median_token_loss": float(losses.median().item()),
            "min_token_loss": float(losses.min().item()),
            "max_token_loss": float(losses.max().item()),
            "std_token_loss": float(
                losses.std(unbiased=False).item()
            ),
            "min_k_10": min_k_score(0.10),
            "min_k_20": min_k_score(0.20),
            "min_k_30": min_k_score(0.30),
            "min_k_40": min_k_score(0.40),
            "num_prediction_tokens": int(losses.numel()),
        })

    return results


def score_neighbors_for_seed(seed, tokenizer, neighbors):
    seed_root = MULTISEED_ROOT / f"seed_{seed}"
    adapter_dir = seed_root / "realistic_lora" / "adapter"
    output_path = (
        OUTPUT_DIR / f"seed_{seed}_adjacent_scores.csv"
    )

    if output_path.exists():
        print(
            f"\nSeed {seed} adjacent scores already exist — loading."
        )
        return pd.read_csv(output_path)

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Missing target adapter: {adapter_dir}"
        )

    cleanup()

    print("\n" + "=" * 72)
    print(f"SCORING ADJACENT CONTROLS — SEED {seed}")
    print("=" * 72)

    print("Loading base Gemma...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )

    print("Loading adapter:", adapter_dir)
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
    )
    model = model.cuda()
    model.eval()

    output_rows = []

    for start in range(0, len(neighbors), BATCH_SIZE):
        batch = neighbors[start:start + BATCH_SIZE]
        scores = score_batch(
            model,
            tokenizer,
            [row["text"] for row in batch],
        )

        for source, score in zip(batch, scores):
            output_rows.append({
                "pair_id": source["pair_id"],
                "sha256": source["sha256"],
                "control_group": "adjacent_nonmember",
                "membership": 0,
                "offset_chunks": source.get("offset_chunks"),
                "absolute_offset_chunks":
                    source.get("absolute_offset_chunks"),
                **score,
            })

        if start % 100 == 0:
            print(
                f"Seed {seed}: scored "
                f"{min(start + BATCH_SIZE, len(neighbors))}"
                f"/{len(neighbors)}"
            )

    df = pd.DataFrame(output_rows)
    df.to_csv(output_path, index=False)
    print("Saved:", output_path)

    del model
    cleanup()

    return df


def directional_auc(df_a, df_b, feature, a_positive=True):
    """
    Population AUC using a single feature.

    For loss-like features, smaller values are generally more member-like.
    For Min-K features, larger values are already more member-like.
    """
    a = df_a[feature].to_numpy()
    b = df_b[feature].to_numpy()

    y = np.concatenate([
        np.ones(len(a), dtype=int),
        np.zeros(len(b), dtype=int),
    ])

    values = np.concatenate([a, b])

    if feature in {
        "nll",
        "median_token_loss",
        "min_token_loss",
        "max_token_loss",
        "std_token_loss",
    }:
        scores = -values
    else:
        scores = values

    auc = roc_auc_score(y, scores)

    if not a_positive:
        auc = 1.0 - auc

    return auc


def analyze_seed(seed, adjacent):
    seed_root = MULTISEED_ROOT / f"seed_{seed}"
    target_path = (
        seed_root
        / "realistic_lora"
        / "membership_scores.csv"
    )

    if not target_path.exists():
        raise FileNotFoundError(
            f"Missing target membership scores: {target_path}"
        )

    target = pd.read_csv(target_path)

    members = target[
        target["membership"].astype(int) == 1
    ].copy()

    random_nonmembers = target[
        target["membership"].astype(int) == 0
    ].copy()

    print("\n" + "-" * 72)
    print(f"CONTROL ANALYSIS — SEED {seed}")
    print("-" * 72)
    print("Exact members:", len(members))
    print("Adjacent nonmembers:", len(adjacent))
    print("Random nonmembers:", len(random_nonmembers))

    # --------------------------------------------------
    # Single-feature diagnostic AUCs
    # --------------------------------------------------

    comparisons = [
        (
            "exact_vs_random",
            members,
            random_nonmembers,
        ),
        (
            "exact_vs_adjacent",
            members,
            adjacent,
        ),
        (
            "adjacent_vs_random",
            adjacent,
            random_nonmembers,
        ),
    ]

    auc_rows = []

    for comparison_name, positive, negative in comparisons:
        print(f"\n{comparison_name}")

        for feature in FEATURES:
            auc = directional_auc(
                positive,
                negative,
                feature,
            )

            auc_rows.append({
                "seed": seed,
                "comparison": comparison_name,
                "feature": feature,
                "roc_auc": auc,
            })

            print(
                f"  {feature:20s}: "
                f"{auc:.6f}"
            )

    auc_df = pd.DataFrame(auc_rows)

    auc_path = (
        OUTPUT_DIR
        / f"seed_{seed}_single_feature_control_auc.csv"
    )
    auc_df.to_csv(auc_path, index=False)

    # --------------------------------------------------
    # Matched-pair analysis
    # --------------------------------------------------

    exact_control = load_jsonl(
        CONTROL_DIR / "exact_members.jsonl"
    )

    pair_map = {
        row["sha256"]: row["pair_id"]
        for row in exact_control
    }

    members["pair_id"] = members["sha256"].map(pair_map)

    if members["pair_id"].isna().any():
        raise RuntimeError(
            "Could not map all scored members to control pair IDs."
        )

    paired = members[
        ["pair_id", "sha256"] + FEATURES
    ].merge(
        adjacent[
            ["pair_id", "sha256"] + FEATURES
        ],
        on="pair_id",
        suffixes=("_member", "_adjacent"),
        validate="one_to_one",
    )

    pair_rows = []

    for feature in FEATURES:
        m = paired[f"{feature}_member"].to_numpy()
        a = paired[f"{feature}_adjacent"].to_numpy()

        if feature in {
            "nll",
            "median_token_loss",
            "min_token_loss",
            "max_token_loss",
            "std_token_loss",
        }:
            member_more_memberlike = m < a
            signed_margin = a - m
        else:
            member_more_memberlike = m > a
            signed_margin = m - a

        ties = np.isclose(
            m,
            a,
            rtol=1e-12,
            atol=1e-12,
        )

        wins = int(
            np.sum(member_more_memberlike & ~ties)
        )
        tie_count = int(np.sum(ties))
        losses = len(paired) - wins - tie_count

        pair_rows.append({
            "seed": seed,
            "feature": feature,
            "pairs": len(paired),
            "member_more_memberlike": wins,
            "adjacent_more_memberlike": losses,
            "ties": tie_count,
            "member_win_fraction": (
                wins + 0.5 * tie_count
            ) / len(paired),
            "mean_signed_margin":
                float(np.mean(signed_margin)),
            "median_signed_margin":
                float(np.median(signed_margin)),
        })

    pair_df = pd.DataFrame(pair_rows)

    pair_path = (
        OUTPUT_DIR
        / f"seed_{seed}_matched_pair_control.csv"
    )
    pair_df.to_csv(pair_path, index=False)

    print("\nMatched exact-member vs adjacent control:")
    print(
        pair_df[
            [
                "feature",
                "member_win_fraction",
                "mean_signed_margin",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(" ", auc_path)
    print(" ", pair_path)

    return auc_df, pair_df


def aggregate(all_auc, all_pairs):
    auc = pd.concat(
        all_auc,
        ignore_index=True,
    )

    pairs = pd.concat(
        all_pairs,
        ignore_index=True,
    )

    auc_all_path = OUTPUT_DIR / "control_auc_all_seeds.csv"
    pair_all_path = OUTPUT_DIR / "matched_pair_all_seeds.csv"

    auc.to_csv(auc_all_path, index=False)
    pairs.to_csv(pair_all_path, index=False)

    auc_summary = (
        auc.groupby(
            ["comparison", "feature"]
        )["roc_auc"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    pair_summary = (
        pairs.groupby("feature")
        .agg(
            member_win_fraction_mean=(
                "member_win_fraction",
                "mean",
            ),
            member_win_fraction_std=(
                "member_win_fraction",
                "std",
            ),
            mean_signed_margin_mean=(
                "mean_signed_margin",
                "mean",
            ),
            mean_signed_margin_std=(
                "mean_signed_margin",
                "std",
            ),
        )
        .reset_index()
    )

    auc_summary_path = (
        OUTPUT_DIR / "control_auc_summary.csv"
    )
    pair_summary_path = (
        OUTPUT_DIR / "matched_pair_summary.csv"
    )

    auc_summary.to_csv(
        auc_summary_path,
        index=False,
    )
    pair_summary.to_csv(
        pair_summary_path,
        index=False,
    )

    print("\n\n" + "=" * 72)
    print("NEIGHBOR CONTROL — FOUR-SEED SUMMARY")
    print("=" * 72)

    print("\nSingle-feature population AUC:")
    print(
        auc_summary.to_string(index=False)
    )

    print("\nMatched-pair summary:")
    print(
        pair_summary.to_string(index=False)
    )

    print("\nSaved aggregate files:")
    print(" ", auc_all_path)
    print(" ", auc_summary_path)
    print(" ", pair_all_path)
    print(" ", pair_summary_path)


def main():
    print("=" * 72)
    print("GEMMA MULTI-SEED ADJACENT-CONTEXT CONTROL SCORER")
    print("=" * 72)

    adjacent_path = (
        CONTROL_DIR / "adjacent_nonmembers.jsonl"
    )

    if not adjacent_path.exists():
        raise FileNotFoundError(
            f"Run prepare_neighbor_control.py first: "
            f"{adjacent_path} not found."
        )

    neighbors = load_jsonl(adjacent_path)

    print("\nAdjacent controls:", len(neighbors))
    print("Seeds:", SEEDS)
    print("New GPU evaluations:", len(neighbors) * len(SEEDS))

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_auc = []
    all_pairs = []

    for seed in SEEDS:
        adjacent = score_neighbors_for_seed(
            seed,
            tokenizer,
            neighbors,
        )

        auc_df, pair_df = analyze_seed(
            seed,
            adjacent,
        )

        all_auc.append(auc_df)
        all_pairs.append(pair_df)

    aggregate(
        all_auc,
        all_pairs,
    )

    print("\n" + "=" * 72)
    print("ADJACENT-CONTEXT CONTROL COMPLETE")
    print("=" * 72)

    print(
        "\nInterpretation guide:"
        "\n  exact ≠ adjacent; adjacent ≈ random"
        "\n      -> membership-specific evidence"
        "\n  exact ≈ adjacent; adjacent ≠ random"
        "\n      -> contextual/distribution confound"
        "\n  exact ≠ adjacent; adjacent ≠ random"
        "\n      -> both effects may contribute"
        "\n  all ≈ chance"
        "\n      -> original signal requires re-examination"
    )


if __name__ == "__main__":
    main()
