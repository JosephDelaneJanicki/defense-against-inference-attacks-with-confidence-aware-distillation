import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-4b-pt"
MEMBERS_PATH = Path("data/gemma/members.jsonl")
NONMEMBERS_PATH = Path("data/gemma/nonmembers.jsonl")
DEFAULT_ADAPTER = Path("results/gemma/defense/seed_42/raw_soft/adapter")
DEFAULT_OUTPUT_DIR = Path("results/gemma/defense/seed_42/raw_soft")
MAX_LENGTH = 256
BATCH_SIZE_SCORE = 4
MIN_K_VALUES = [0.10, 0.20, 0.30, 0.40]


def load_raw_jsonl(path, membership):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["membership"] = membership
            rows.append(row)
    return rows


def score_batch(model, tokenizer, texts):
    # Kept aligned with the established shadow scoring implementation.
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
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
    shift_mask = attention_mask[:, 1:].bool()

    log_probs = F.log_softmax(shift_logits, dim=-1)

    true_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    token_losses = -true_log_probs
    results = []

    for i in range(len(texts)):
        mask = shift_mask[i]
        losses = token_losses[i][mask]
        observed_log_probs = true_log_probs[i][mask]
        mean_loss = losses.mean().item()

        row = {
            "nll": mean_loss,
            "perplexity": float(np.exp(min(mean_loss, 20.0))),
            "median_token_loss": losses.median().item(),
            "min_token_loss": losses.min().item(),
            "max_token_loss": losses.max().item(),
            "std_token_loss": (
                losses.std().item()
                if losses.numel() > 1
                else 0.0
            ),
            "num_scored_tokens": int(losses.numel()),
        }

        sorted_lp, _ = torch.sort(
            observed_log_probs,
            descending=False,
        )
        n_tokens = sorted_lp.numel()

        for k in MIN_K_VALUES:
            count = max(
                1,
                int(np.ceil(n_tokens * k)),
            )
            label = int(k * 100)
            row[f"min_k_{label}"] = (
                sorted_lp[:count].mean().item()
            )

        results.append(row)

    return results


def direct_auc(df):
    # Direction is frozen from the original scoring methodology.
    specs = [
        ("-NLL", "nll", -1.0),
        ("-Perplexity", "perplexity", -1.0),
        ("Median token loss", "median_token_loss", -1.0),
        ("Min token loss", "min_token_loss", -1.0),
        ("Max token loss", "max_token_loss", -1.0),
        ("Std token loss", "std_token_loss", -1.0),
        ("Min-K 10%", "min_k_10", 1.0),
        ("Min-K 20%", "min_k_20", 1.0),
        ("Min-K 30%", "min_k_30", 1.0),
        ("Min-K 40%", "min_k_40", 1.0),
    ]

    y = df["membership"].astype(int).to_numpy()
    rows = []

    for metric, feature, direction in specs:
        score = direction * df[feature].to_numpy()
        rows.append({
            "metric": metric,
            "feature": feature,
            "roc_auc": roc_auc_score(y, score),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        type=Path,
        default=DEFAULT_ADAPTER,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--members",
        type=Path,
        default=MEMBERS_PATH,
    )
    parser.add_argument(
        "--nonmembers",
        type=Path,
        default=NONMEMBERS_PATH,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "membership_scores.csv"
    auc_path = args.output_dir / "membership_auc.csv"

    print("=" * 72)
    print("GEMMA DEFENSE STUDENT MEMBERSHIP SCORING")
    print("=" * 72)
    print("\nAdapter:", args.adapter)
    print("Members:", args.members)
    print("Nonmembers:", args.nonmembers)

    for path in (args.adapter, args.members, args.nonmembers):
        if not path.exists():
            raise FileNotFoundError(path)

    members = load_raw_jsonl(args.members, 1)
    nonmembers = load_raw_jsonl(args.nonmembers, 0)
    rows = members + nonmembers

    print("\nFrozen attack population:")
    print("Members:", len(members))
    print("Nonmembers:", len(nonmembers))
    print("Total:", len(rows))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading fresh Gemma base in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )

    print("Loading defended student adapter...")
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
    )
    model = model.cuda()
    model.eval()

    print("\nScoring...")

    output_rows = []

    for start in range(0, len(rows), BATCH_SIZE_SCORE):
        batch = rows[start:start + BATCH_SIZE_SCORE]
        texts = [row["text"] for row in batch]

        scores = score_batch(
            model,
            tokenizer,
            texts,
        )

        for source, score in zip(batch, scores):
            output_rows.append({
                "sha256": source["sha256"],
                "membership": source["membership"],
                **score,
            })

        done = min(
            start + BATCH_SIZE_SCORE,
            len(rows),
        )

        if (
            start == 0
            or done % 500 == 0
            or done == len(rows)
        ):
            print(f"Scored {done}/{len(rows)}")

    df = pd.DataFrame(output_rows)

    if len(df) != len(rows):
        raise RuntimeError("Scoring row-count mismatch.")
    if int(df["membership"].sum()) != len(members):
        raise RuntimeError("Membership-label integrity check failed.")
    if df["sha256"].duplicated().any():
        raise RuntimeError("Duplicate hashes in attack population.")

    df.to_csv(score_path, index=False)

    auc_df = direct_auc(df)
    auc_df.to_csv(auc_path, index=False)

    print("\n=== RAW-SOFT STUDENT DIRECT MIA ===")
    for _, row in auc_df.iterrows():
        print(
            f"{row['metric']:22s}: "
            f"AUC = {row['roc_auc']:.6f}"
        )

    member_nll = df.loc[df["membership"] == 1, "nll"]
    nonmember_nll = df.loc[df["membership"] == 0, "nll"]

    print("\n=== NLL GROUP STATISTICS ===")
    print(
        f"Members     : mean={member_nll.mean():.6f}, "
        f"median={member_nll.median():.6f}"
    )
    print(
        f"Nonmembers  : mean={nonmember_nll.mean():.6f}, "
        f"median={nonmember_nll.median():.6f}"
    )

    print("\n" + "=" * 72)
    print("DEFENSE STUDENT SCORING COMPLETE")
    print("=" * 72)
    print("\nSaved:")
    print(" ", score_path)
    print(" ", auc_path)



if __name__ == "__main__":
    main()
