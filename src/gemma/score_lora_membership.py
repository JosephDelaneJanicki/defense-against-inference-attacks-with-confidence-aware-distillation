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

DATA_DIR = Path("data/gemma")
OUTPUT_DIR = Path("results/gemma")

MAX_LENGTH = 256
BATCH_SIZE = 4

# Min-K% values commonly useful for examining low-probability tokens.
MIN_K_VALUES = [0.10, 0.20, 0.30, 0.40]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a Gemma LoRA target for membership inference."
    )

    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to the PEFT LoRA adapter.",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name used for result files.",
    )

    return parser.parse_args()


def load_jsonl(path, membership):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            item["membership"] = membership
            rows.append(item)

    return rows


def score_batch(model, tokenizer, texts):
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

    # Position t predicts token t+1.
    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].bool()

    # Log probability for every vocabulary token.
    log_probs = F.log_softmax(
        shift_logits,
        dim=-1,
    )

    # Extract log probability assigned to the actual next token.
    true_token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    # Negative log likelihood for each actual token.
    token_losses = -true_token_log_probs

    results = []

    for i in range(len(texts)):
        mask = shift_mask[i]

        losses = token_losses[i][mask]
        token_log_probs = true_token_log_probs[i][mask]

        mean_loss = losses.mean().item()
        median_loss = losses.median().item()
        min_loss = losses.min().item()
        max_loss = losses.max().item()

        if losses.numel() > 1:
            std_loss = losses.std().item()
        else:
            std_loss = 0.0

        perplexity = float(
            np.exp(min(mean_loss, 20.0))
        )

        row = {
            "nll": mean_loss,
            "perplexity": perplexity,
            "median_token_loss": median_loss,
            "min_token_loss": min_loss,
            "max_token_loss": max_loss,
            "std_token_loss": std_loss,
            "num_scored_tokens": int(losses.numel()),
        }

        # --------------------------------------------------
        # Min-K%
        #
        # Select the K% least-likely actual tokens and take
        # their mean log probability.
        #
        # Higher score = target assigned those difficult
        # tokens greater probability.
        # --------------------------------------------------

        sorted_log_probs, _ = torch.sort(
            token_log_probs,
            descending=False,
        )

        n_tokens = sorted_log_probs.numel()

        for k in MIN_K_VALUES:
            count = max(
                1,
                int(np.ceil(n_tokens * k)),
            )

            min_k_score = (
                sorted_log_probs[:count]
                .mean()
                .item()
            )

            label = int(k * 100)

            row[f"min_k_{label}"] = min_k_score

        results.append(row)

    # Explicitly release the enormous vocabulary tensors.
    del outputs
    del shift_logits
    del log_probs
    del true_token_log_probs
    del token_losses

    return results


def main():
    args = parse_args()

    if not args.adapter.exists():
        raise FileNotFoundError(
            f"Adapter does not exist: {args.adapter}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("GEMMA LoRA MEMBERSHIP SCORER")
    print("=" * 60)

    print("\nTarget name:", args.name)
    print("Base model:", MODEL_ID)
    print("Adapter:", args.adapter)
    print("Precision: BF16")
    print("Quantization: NONE")

    # --------------------------------------------------
    # Membership population
    # --------------------------------------------------

    print("\nLoading controlled membership population...")

    members = load_jsonl(
        DATA_DIR / "members.jsonl",
        membership=1,
    )

    nonmembers = load_jsonl(
        DATA_DIR / "nonmembers.jsonl",
        membership=0,
    )

    rows = members + nonmembers

    print("Members:", len(members))
    print("Nonmembers:", len(nonmembers))
    print("Total:", len(rows))

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base Gemma...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        args.adapter,
    )

    model = model.cuda()
    model.eval()

    print("Model ready.")

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    print(
        f"VRAM after load: {allocated:.2f} GB"
    )

    # --------------------------------------------------
    # Score
    # --------------------------------------------------

    print("\nScoring membership population...")

    output_rows = []

    for start in range(
        0,
        len(rows),
        BATCH_SIZE,
    ):
        batch = rows[
            start:start + BATCH_SIZE
        ]

        texts = [
            item["text"]
            for item in batch
        ]

        scores = score_batch(
            model,
            tokenizer,
            texts,
        )

        for source, score in zip(
            batch,
            scores,
        ):
            output_rows.append({
                "sha256": source["sha256"],
                "membership":
                    source["membership"],
                **score,
            })

        if start % 100 == 0:
            print(
                f"Scored "
                f"{min(start + BATCH_SIZE, len(rows))}"
                f"/{len(rows)}"
            )

    df = pd.DataFrame(output_rows)

    # --------------------------------------------------
    # Save per-example scores
    # --------------------------------------------------

    score_path = (
        OUTPUT_DIR
        / f"{args.name}_membership_scores.csv"
    )

    df.to_csv(
        score_path,
        index=False,
    )

    print("\nSaved:", score_path)

    # --------------------------------------------------
    # Membership AUC
    # --------------------------------------------------

    y = df["membership"].values

    metrics = {
        "-NLL":
            -df["nll"].values,

        "-Perplexity":
            -df["perplexity"].values,

        "-Median token loss":
            -df["median_token_loss"].values,

        "-Max token loss":
            -df["max_token_loss"].values,

        "-Std token loss":
            -df["std_token_loss"].values,

        "Min-K 10%":
            df["min_k_10"].values,

        "Min-K 20%":
            df["min_k_20"].values,

        "Min-K 30%":
            df["min_k_30"].values,

        "Min-K 40%":
            df["min_k_40"].values,
    }

    print("\n=== MEMBERSHIP INFERENCE RESULTS ===")

    summary = []

    for metric_name, scores in metrics.items():
        auc = roc_auc_score(
            y,
            scores,
        )

        effective_auc = max(
            auc,
            1.0 - auc,
        )

        summary.append({
            "target": args.name,
            "metric": metric_name,
            "roc_auc": auc,
            "effective_auc":
                effective_auc,
        })

        print(
            f"{metric_name:22s}: "
            f"AUC = {auc:.6f}  "
            f"(effective {effective_auc:.6f})"
        )

    summary_df = pd.DataFrame(summary)

    auc_path = (
        OUTPUT_DIR
        / f"{args.name}_membership_auc.csv"
    )

    summary_df.to_csv(
        auc_path,
        index=False,
    )

    print("\nSaved:", auc_path)

    # --------------------------------------------------
    # Basic group statistics
    # --------------------------------------------------

    print("\n=== NLL GROUP STATISTICS ===")

    for membership, label in [
        (1, "Members"),
        (0, "Nonmembers"),
    ]:
        subset = df[
            df["membership"] == membership
        ]

        print(
            f"{label:12s}: "
            f"mean={subset['nll'].mean():.6f}, "
            f"median={subset['nll'].median():.6f}"
        )

    print("\nSCORING COMPLETE")


if __name__ == "__main__":
    main()