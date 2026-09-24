import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
)


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "google/gemma-3-4b-pt"

DATA_ROOT = Path("data/gemma/shadows")
MULTISEED_ROOT = Path("results/gemma/multiseed")

N_SHADOWS = 6

MAX_LENGTH = 256
BATCH_SIZE_SCORE = 4

SEEDS = [
    42,
    123,
    456,
    789,
]

# Must match realistic target.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4
NUM_EPOCHS = 2

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM = 8

MIN_K_VALUES = [
    0.10,
    0.20,
    0.30,
    0.40,
]


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):

    def __init__(self, path, tokenizer):

        self.rows = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                row = json.loads(line)

                encoded = tokenizer(
                    row["text"],
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=False,
                    add_special_tokens=True,
                )

                self.rows.append({
                    "input_ids":
                        encoded["input_ids"],

                    "attention_mask":
                        encoded["attention_mask"],
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def load_raw_jsonl(path, membership):

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            row = json.loads(line)

            row["membership"] = membership

            rows.append(row)

    return rows


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Cleanup
# ============================================================

def cleanup():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# Train shadow
# ============================================================

def train_shadow(
    shadow_name,
    data_dir,
    result_dir,
    tokenizer,
    seed,
):

    adapter_dir = (
        result_dir / "adapter"
    )

    complete_file = (
        result_dir / "training_complete.json"
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if (
        adapter_dir.exists()
        and complete_file.exists()
    ):

        print(
            f"\n[SKIP TRAIN] {shadow_name} "
            f"already complete."
        )

        return

    print("\n" + "=" * 70)
    print(f"TRAINING {shadow_name}")
    print("=" * 70)

    cleanup()
    set_seed(seed)

    train_dataset = TextDataset(
        data_dir / "members.jsonl",
        tokenizer,
    )

    eval_dataset = TextDataset(
        data_dir / "validation.jsonl",
        tokenizer,
    )

    print(
        "Train:",
        len(train_dataset),
    )

    print(
        "Validation:",
        len(eval_dataset),
    )

    # --------------------------------------------------------
    # Fresh untouched base model
    # --------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # --------------------------------------------------------
    # LoRA — identical to realistic target
    # --------------------------------------------------------

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(
        model,
        config,
    )

    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    checkpoint_dir = (
        result_dir / "checkpoints"
    )

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=
            TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=
            EVAL_BATCH_SIZE,

        gradient_accumulation_steps=
            GRAD_ACCUM,

        learning_rate=LEARNING_RATE,

        bf16=True,
        fp16=False,

        logging_strategy="steps",
        logging_steps=50,

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2,

        # Match realistic target.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        gradient_checkpointing=True,

        optim="adamw_torch",

        seed=seed,
        data_seed=seed,

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print(
        "\nEvaluating before training..."
    )

    initial_eval = trainer.evaluate()

    print(
        "Initial validation loss:",
        initial_eval["eval_loss"],
    )

    print("\nTraining...")

    start = time.time()

    train_result = trainer.train()

    elapsed = time.time() - start

    print("\nFinal evaluation...")

    final_eval = trainer.evaluate()

    # --------------------------------------------------------
    # Save selected adapter
    # --------------------------------------------------------

    trainer.model.save_pretrained(
        adapter_dir
    )

    tokenizer.save_pretrained(
        adapter_dir
    )

    summary = {
        "shadow": shadow_name,
        "seed": seed,

        "initial_eval_loss":
            initial_eval["eval_loss"],

        "final_eval_loss":
            final_eval["eval_loss"],

        "train_loss":
            train_result.metrics.get(
                "train_loss"
            ),

        "training_seconds":
            elapsed,

        "peak_vram_gb":
            torch.cuda.max_memory_allocated()
            / 1024**3,

        "epochs":
            NUM_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "lora_r":
            LORA_R,

        "lora_alpha":
            LORA_ALPHA,

        "lora_dropout":
            LORA_DROPOUT,

        "quantization":
            None,

        "precision":
            "bfloat16",
    }

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with complete_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"\n[TRAIN COMPLETE] {shadow_name}"
    )

    # Explicitly destroy model before scoring reload.
    del trainer
    del model
    del train_dataset
    del eval_dataset

    cleanup()


# ============================================================
# Score one batch
# ============================================================

def score_batch(
    model,
    tokenizer,
    texts,
):

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = (
        encoded["input_ids"].cuda()
    )

    attention_mask = (
        encoded["attention_mask"].cuda()
    )

    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    shift_logits = (
        outputs.logits[:, :-1, :]
        .float()
    )

    shift_labels = (
        input_ids[:, 1:]
    )

    shift_mask = (
        attention_mask[:, 1:]
        .bool()
    )

    log_probs = F.log_softmax(
        shift_logits,
        dim=-1,
    )

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

        observed_log_probs = (
            true_log_probs[i][mask]
        )

        mean_loss = (
            losses.mean().item()
        )

        row = {
            "nll":
                mean_loss,

            "perplexity":
                float(
                    np.exp(
                        min(mean_loss, 20.0)
                    )
                ),

            "median_token_loss":
                losses.median().item(),

            "min_token_loss":
                losses.min().item(),

            "max_token_loss":
                losses.max().item(),

            "std_token_loss":
                (
                    losses.std().item()
                    if losses.numel() > 1
                    else 0.0
                ),

            "num_scored_tokens":
                int(losses.numel()),
        }

        sorted_lp, _ = torch.sort(
            observed_log_probs,
            descending=False,
        )

        n_tokens = (
            sorted_lp.numel()
        )

        for k in MIN_K_VALUES:

            count = max(
                1,
                int(
                    np.ceil(
                        n_tokens * k
                    )
                ),
            )

            label = int(k * 100)

            row[
                f"min_k_{label}"
            ] = (
                sorted_lp[:count]
                .mean()
                .item()
            )

        results.append(row)

    del outputs
    del shift_logits
    del log_probs
    del true_log_probs
    del token_losses

    return results


# ============================================================
# Score shadow
# ============================================================

def score_shadow(
    shadow_name,
    data_dir,
    result_dir,
    tokenizer,
):

    score_file = (
        result_dir
        / "membership_scores.csv"
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if score_file.exists():

        print(
            f"[SKIP SCORE] {shadow_name} "
            f"already scored."
        )

        return

    print("\n" + "=" * 70)
    print(f"SCORING {shadow_name}")
    print("=" * 70)

    cleanup()

    members = load_raw_jsonl(
        data_dir / "members.jsonl",
        membership=1,
    )

    nonmembers = load_raw_jsonl(
        data_dir / "nonmembers.jsonl",
        membership=0,
    )

    rows = members + nonmembers

    print(
        "Scoring:",
        len(rows),
        "examples",
    )

    # --------------------------------------------------------
    # Reload fresh base + trained adapter
    # --------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )

    from peft import PeftModel

    model = PeftModel.from_pretrained(
        model,
        result_dir / "adapter",
    )

    model = model.cuda()
    model.eval()

    output_rows = []

    for start in range(
        0,
        len(rows),
        BATCH_SIZE_SCORE,
    ):

        batch = rows[
            start:
            start + BATCH_SIZE_SCORE
        ]

        texts = [
            row["text"]
            for row in batch
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
                "shadow":
                    shadow_name,

                "sha256":
                    source["sha256"],

                "membership":
                    source["membership"],

                **score,
            })

        if start % 500 == 0:

            print(
                f"{shadow_name}: "
                f"{min(start + BATCH_SIZE_SCORE, len(rows))}"
                f"/{len(rows)}"
            )

    df = pd.DataFrame(
        output_rows
    )

    df.to_csv(
        score_file,
        index=False,
    )

    print(
        f"\n[SCORE COMPLETE] "
        f"{shadow_name}: {score_file}"
    )

    del model

    cleanup()


# ============================================================
# Combine all shadow attack data
# ============================================================

def combine_shadow_scores(result_root):

    print("\n" + "=" * 70)
    print("COMBINING SHADOW ATTACK DATA")
    print("=" * 70)

    frames = []

    for shadow_id in range(
        N_SHADOWS
    ):

        name = (
            f"shadow_{shadow_id:02d}"
        )

        path = (
            result_root
            / name
            / "membership_scores.csv"
        )

        if not path.exists():

            raise RuntimeError(
                f"Missing score file: {path}"
            )

        frames.append(
            pd.read_csv(path)
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    output = (
        result_root
        / "all_shadow_membership_scores.csv"
    )

    combined.to_csv(
        output,
        index=False,
    )

    print(
        "Combined examples:",
        len(combined),
    )

    print(
        "Members:",
        int(
            combined[
                "membership"
            ].sum()
        ),
    )

    print(
        "Nonmembers:",
        int(
            (
                combined["membership"]
                == 0
            ).sum()
        ),
    )

    print("Saved:", output)


# ============================================================
# Main
# ============================================================

def main():

    MULTISEED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("GEMMA SHADOW-LoRA MULTI-SEED PIPELINE")
    print("=" * 70)

    print("\nBase model:", MODEL_ID)
    print("Seeds:", SEEDS)
    print("Shadows per seed:", N_SHADOWS)
    print("Total shadow LoRAs:", len(SEEDS) * N_SHADOWS)
    print("Precision: BF16")
    print("Quantization: NONE")

    print("\nShadow datasets remain FIXED across seeds.")
    print("Only stochastic training state changes.")
    print("\nLoading tokenizer once...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipeline_start = time.time()

    for seed_index, seed in enumerate(SEEDS, start=1):

        result_root = (
            MULTISEED_ROOT
            / f"seed_{seed}"
            / "shadows"
        )
        result_root.mkdir(parents=True, exist_ok=True)

        print("\n\n" + "=" * 70)
        print(
            f"MULTI-SEED SHADOW RUN "
            f"{seed_index}/{len(SEEDS)}"
        )
        print(f"SEED = {seed}")
        print("=" * 70)

        seed_start = time.time()

        for shadow_id in range(N_SHADOWS):

            shadow_name = f"shadow_{shadow_id:02d}"

            # Same shadow datasets for every training seed.
            data_dir = DATA_ROOT / shadow_name

            # Separate outputs prevent cross-seed overwrites.
            result_dir = result_root / shadow_name
            result_dir.mkdir(parents=True, exist_ok=True)

            print("\n\n" + "#" * 70)
            print(
                f"# Seed {seed} | {shadow_name} "
                f"({shadow_id + 1}/{N_SHADOWS})"
            )
            print("#" * 70)

            train_shadow(
                shadow_name,
                data_dir,
                result_dir,
                tokenizer,
                seed,
            )

            score_shadow(
                shadow_name,
                data_dir,
                result_dir,
                tokenizer,
            )

        combine_shadow_scores(result_root)

        seed_elapsed = time.time() - seed_start

        print("\n" + "=" * 70)
        print(f"SEED {seed} SHADOW PIPELINE COMPLETE")
        print("=" * 70)
        print(
            f"Seed runtime: "
            f"{seed_elapsed / 3600:.2f} hours"
        )
        print(
            "Combined attack data:",
            result_root / "all_shadow_membership_scores.csv",
        )

    elapsed = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("MULTI-SEED SHADOW PIPELINE COMPLETE")
    print("=" * 70)
    print("Seeds completed:", SEEDS)
    print("Total shadow LoRAs:", len(SEEDS) * N_SHADOWS)
    print(f"Total runtime: {elapsed / 3600:.2f} hours")

    print(
        "\nEach seed contains its own "
        "60,000-example shadow attack population."
    )


if __name__ == "__main__":
    main()