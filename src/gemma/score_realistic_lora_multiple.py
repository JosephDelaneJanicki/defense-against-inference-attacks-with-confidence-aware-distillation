import gc
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

MULTISEED_ROOT = Path("results/gemma/multiseed")

SEEDS = [
    42,
    123,
    456,
    789,
]



MAX_LENGTH = 256

BATCH_SIZE = 4



# Min-K% values commonly useful for examining low-probability tokens.

MIN_K_VALUES = [0.10, 0.20, 0.30, 0.40]





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





def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    MULTISEED_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GEMMA REALISTIC LoRA — MULTI-SEED MEMBERSHIP SCORER")
    print("=" * 70)
    print("\nBase model:", MODEL_ID)
    print("Seeds:", SEEDS)
    print("Precision: BF16")
    print("Quantization: NONE")

    # Same fixed target population for every seed.
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

    print("\nLoading tokenizer once...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for seed_index, seed in enumerate(SEEDS, start=1):
        target_dir = (
            MULTISEED_ROOT
            / f"seed_{seed}"
            / "realistic_lora"
        )
        adapter_dir = target_dir / "adapter"
        score_path = target_dir / "membership_scores.csv"
        auc_path = target_dir / "membership_auc.csv"

        print("\n\n" + "=" * 70)
        print(f"TARGET SEED {seed} ({seed_index}/{len(SEEDS)})")
        print("=" * 70)

        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"Adapter does not exist: {adapter_dir}"
            )

        # Resume support.
        if score_path.exists() and auc_path.exists():
            print(f"Seed {seed} already scored — SKIPPING")
            print("Scores:", score_path)
            print("AUC:", auc_path)
            continue

        cleanup()

        print("\nLoading fresh base Gemma...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
        )

        print("Loading LoRA adapter:")
        print(adapter_dir)

        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
        )
        model = model.cuda()
        model.eval()

        allocated = torch.cuda.memory_allocated() / 1024**3
        print("Model ready.")
        print(f"VRAM after load: {allocated:.2f} GB")

        print("\nScoring membership population...")
        output_rows = []

        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            texts = [item["text"] for item in batch]

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
                    f"Seed {seed}: scored "
                    f"{min(start + BATCH_SIZE, len(rows))}"
                    f"/{len(rows)}"
                )

        df = pd.DataFrame(output_rows)
        df.to_csv(score_path, index=False)
        print("\nSaved:", score_path)

        y = df["membership"].values

        # Same metrics/directions as the original target scorer.
        metrics = {
            "-NLL": -df["nll"].values,
            "-Perplexity": -df["perplexity"].values,
            "-Median token loss": -df["median_token_loss"].values,
            "-Max token loss": -df["max_token_loss"].values,
            "-Std token loss": -df["std_token_loss"].values,
            "Min-K 10%": df["min_k_10"].values,
            "Min-K 20%": df["min_k_20"].values,
            "Min-K 30%": df["min_k_30"].values,
            "Min-K 40%": df["min_k_40"].values,
        }

        print("\n=== MEMBERSHIP INFERENCE RESULTS ===")
        summary = []

        for metric_name, metric_scores in metrics.items():
            auc = roc_auc_score(y, metric_scores)
            effective_auc = max(auc, 1.0 - auc)

            summary.append({
                "seed": seed,
                "target": f"realistic_lora_seed_{seed}",
                "metric": metric_name,
                "roc_auc": auc,
                "effective_auc": effective_auc,
            })

            print(
                f"{metric_name:22s}: "
                f"AUC = {auc:.6f}  "
                f"(effective {effective_auc:.6f})"
            )

        pd.DataFrame(summary).to_csv(
            auc_path,
            index=False,
        )
        print("\nSaved:", auc_path)

        print("\n=== NLL GROUP STATISTICS ===")
        for membership, label in [
            (1, "Members"),
            (0, "Nonmembers"),
        ]:
            subset = df[df["membership"] == membership]
            print(
                f"{label:12s}: "
                f"mean={subset['nll'].mean():.6f}, "
                f"median={subset['nll'].median():.6f}"
            )

        del model
        del df
        del output_rows
        cleanup()

        print(f"\nSEED {seed} SCORING COMPLETE")

    print("\n" + "=" * 70)
    print("MULTI-SEED TARGET SCORING COMPLETE")
    print("=" * 70)

    for seed in SEEDS:
        target_dir = (
            MULTISEED_ROOT
            / f"seed_{seed}"
            / "realistic_lora"
        )
        print(
            f"Seed {seed}: "
            f"{target_dir / 'membership_scores.csv'}"
        )
        print(
            f"          "
            f"{target_dir / 'membership_auc.csv'}"
        )

    print(
        "\nNext step:\n"
        "Train each seed's Shokri-style attack classifier "
        "on that seed's 60,000 shadow observations and "
        "evaluate it on these 10,000 target observations."
    )


if __name__ == "__main__":

    main()