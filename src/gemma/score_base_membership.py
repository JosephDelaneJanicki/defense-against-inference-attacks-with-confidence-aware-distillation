import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "google/gemma-3-4b-pt"

DATA_DIR = Path("data/gemma")
OUTPUT_DIR = Path("results/gemma")

MAX_LENGTH = 256
BATCH_SIZE = 4


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

    # For causal LM:
    # logits at position t predict token t+1.
    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].bool()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    token_losses = loss_fn(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    ).reshape(shift_labels.shape)

    results = []

    for i in range(len(texts)):
        losses = token_losses[i][shift_mask[i]]

        mean_loss = losses.mean().item()
        median_loss = losses.median().item()
        min_loss = losses.min().item()
        max_loss = losses.max().item()
        std_loss = losses.std().item()

        # Avoid numerical overflow from exp on unusual examples.
        perplexity = float(np.exp(min(mean_loss, 20.0)))

        results.append({
            "nll": mean_loss,
            "perplexity": perplexity,
            "median_token_loss": median_loss,
            "min_token_loss": min_loss,
            "max_token_loss": max_loss,
            "std_token_loss": std_loss,
            "num_scored_tokens": int(losses.numel()),
        })

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading controlled membership sets...")

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

    print("\nLoading Gemma 3 4B PT in BF16...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Gemma tokenizer may not define a padding token for this use.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    ).cuda()

    model.eval()

    print("Model loaded.")
    print("Scoring examples...")

    output_rows = []

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        texts = [x["text"] for x in batch]

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

        if start % 100 == 0:
            print(
                f"Scored {min(start + BATCH_SIZE, len(rows))}"
                f"/{len(rows)}"
            )

    df = pd.DataFrame(output_rows)

    output_path = OUTPUT_DIR / "base_membership_scores.csv"
    df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)

    # --------------------------------------------------
    # Negative-control AUCs
    #
    # Lower NLL/PPL implies stronger member prediction,
    # hence negative signs.
    # --------------------------------------------------

    y = df["membership"].values

    metrics = {
        "-NLL": -df["nll"].values,
        "-Perplexity": -df["perplexity"].values,
        "-Median token loss": -df["median_token_loss"].values,
        "-Max token loss": -df["max_token_loss"].values,
        "-Std token loss": -df["std_token_loss"].values,
    }

    print("\n=== UNTOUCHED GEMMA NEGATIVE CONTROL ===")

    summary = []

    for name, score in metrics.items():
        auc = roc_auc_score(y, score)

        summary.append({
            "metric": name,
            "roc_auc": auc,
        })

        print(f"{name:20s}: AUC = {auc:.6f}")

    summary_df = pd.DataFrame(summary)

    summary_path = OUTPUT_DIR / "base_membership_auc.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved:", summary_path)


if __name__ == "__main__":
    main()