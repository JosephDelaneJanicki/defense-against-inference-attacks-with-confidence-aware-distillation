import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "google/gemma-3-4b-pt"

DATA_DIR = Path("data/gemma")

# Completely separate from the original experiment.
MULTISEED_ROOT = Path("results/gemma/multiseed")
MULTISEED_ROOT.mkdir(parents=True, exist_ok=True)

# Four independent training seeds.
SEEDS = [
    42,
    123,
    456,
    789,
]

MAX_LENGTH = 256

# LoRA — unchanged from original experiment.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training — unchanged from original experiment.
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM = 8


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.rows = []

        with path.open("r", encoding="utf-8") as f:
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
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


# ============================================================
# Utilities
# ============================================================

def set_seed(seed):
    """
    Seed the major random-number generators used by the
    experiment.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_gpu():
    """
    Explicitly clean up between seed runs so that each model
    begins with a clean GPU state.
    """

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_memory():
    return {
        "allocated_gb":
            torch.cuda.memory_allocated() / 1024**3,

        "reserved_gb":
            torch.cuda.memory_reserved() / 1024**3,

        "peak_allocated_gb":
            torch.cuda.max_memory_allocated() / 1024**3,
    }


# ============================================================
# Train one seed
# ============================================================

def train_seed(
    seed,
    tokenizer,
    train_dataset,
    eval_dataset,
):
    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    seed_root = (
        MULTISEED_ROOT
        / f"seed_{seed}"
    )

    output_dir = (
        seed_root
        / "realistic_lora"
    )

    adapter_dir = (
        output_dir
        / "adapter"
    )

    summary_path = (
        output_dir
        / "training_summary.json"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Resume / skip completed runs
    # --------------------------------------------------------

    if (
        adapter_dir.exists()
        and summary_path.exists()
    ):
        print("\n" + "=" * 70)
        print(
            f"SEED {seed} ALREADY COMPLETE — SKIPPING"
        )
        print("=" * 70)

        print("Adapter:", adapter_dir)
        print("Summary:", summary_path)

        return

    # --------------------------------------------------------
    # Clean state
    # --------------------------------------------------------

    cleanup_gpu()
    set_seed(seed)

    print("\n\n")
    print("#" * 70)
    print(
        f"# GEMMA REALISTIC LoRA — SEED {seed}"
    )
    print("#" * 70)

    print("\nModel:", MODEL_ID)
    print("Seed:", seed)
    print("Precision: BF16")
    print("Quantization: NONE")
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Output:", output_dir)

    # --------------------------------------------------------
    # Base model
    # --------------------------------------------------------

    print("\nLoading fresh Gemma in BF16...")

    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    lora_config = LoraConfig(
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
        lora_config,
    )

    print("\nLoRA configuration:")
    print("Rank:", LORA_R)
    print("Alpha:", LORA_ALPHA)
    print("Dropout:", LORA_DROPOUT)
    print("Targets: q/k/v/o projections")

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    # --------------------------------------------------------
    # Collator
    # --------------------------------------------------------

    data_collator = (
        DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
    )

    # --------------------------------------------------------
    # Trainer configuration
    #
    # Training recipe is intentionally identical to the
    # original realistic-LoRA experiment.
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(
            output_dir / "checkpoints"
        ),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=
            TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=
            EVAL_BATCH_SIZE,

        gradient_accumulation_steps=
            GRAD_ACCUM,

        learning_rate=
            LEARNING_RATE,

        bf16=True,
        fp16=False,

        logging_strategy="steps",
        logging_steps=50,

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2,

        load_best_model_at_end=True,

        metric_for_best_model=
            "eval_loss",

        greater_is_better=False,

        gradient_checkpointing=True,

        optim="adamw_torch",

        seed=seed,
        data_seed=seed,

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # --------------------------------------------------------
    # Initial validation
    # --------------------------------------------------------

    print(
        "\nEvaluating BEFORE fine-tuning..."
    )

    initial_eval = trainer.evaluate()

    print(
        "Initial validation loss:",
        initial_eval.get("eval_loss"),
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        f"\nBeginning realistic LoRA training "
        f"for seed {seed}...\n"
    )

    start_time = time.time()

    train_result = trainer.train()

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print(
        "\nEvaluating AFTER fine-tuning..."
    )

    final_eval = trainer.evaluate()

    print(
        "Final validation loss:",
        final_eval.get("eval_loss"),
    )

    # --------------------------------------------------------
    # Save adapter
    # --------------------------------------------------------

    trainer.model.save_pretrained(
        adapter_dir
    )

    tokenizer.save_pretrained(
        adapter_dir
    )

    print("\nSaved LoRA adapter:")
    print(adapter_dir)

    # --------------------------------------------------------
    # Experiment summary
    # --------------------------------------------------------

    memory = gpu_memory()

    summary = {
        "experiment":
            "gemma_3_4b_realistic_lora_multiseed",

        "seed":
            seed,

        "base_model":
            MODEL_ID,

        "precision":
            "bfloat16",

        "quantization":
            None,

        "membership_definition":
            "Fine-tuning membership in members.jsonl",

        "train_examples":
            len(train_dataset),

        "validation_examples":
            len(eval_dataset),

        "max_length":
            MAX_LENGTH,

        "lora": {
            "r":
                LORA_R,

            "alpha":
                LORA_ALPHA,

            "dropout":
                LORA_DROPOUT,

            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        },

        "training": {
            "epochs":
                NUM_EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "batch_size":
                TRAIN_BATCH_SIZE,

            "gradient_accumulation":
                GRAD_ACCUM,

            "effective_batch_size":
                TRAIN_BATCH_SIZE
                * GRAD_ACCUM,

            "optimizer":
                "adamw_torch",

            "seed":
                seed,

            "data_seed":
                seed,
        },

        "initial_eval_loss":
            initial_eval.get(
                "eval_loss"
            ),

        "final_eval_loss":
            final_eval.get(
                "eval_loss"
            ),

        "train_loss":
            train_result.metrics.get(
                "train_loss"
            ),

        "training_seconds":
            elapsed,

        "gpu":
            torch.cuda.get_device_name(0),

        "gpu_memory":
            memory,
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\nSaved training summary:")
    print(summary_path)

    print("\nGPU memory:")

    print(
        f"Peak allocated: "
        f"{memory['peak_allocated_gb']:.2f} GB"
    )

    print(
        f"\nSEED {seed} TRAINING COMPLETE"
    )

    # --------------------------------------------------------
    # Destroy this model before loading the next one.
    # --------------------------------------------------------

    del trainer
    del model
    del data_collator

    cleanup_gpu()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(
        "GEMMA 3 4B PT — REALISTIC LoRA MULTI-SEED"
    )
    print("=" * 70)

    print("\nSeeds:", SEEDS)
    print("Number of runs:", len(SEEDS))

    print(
        "\nTraining data remain FIXED across seeds."
    )

    print(
        "Only stochastic training state changes."
    )

    print(
        "\nOutput root:",
        MULTISEED_ROOT,
    )

    # --------------------------------------------------------
    # Tokenizer only needs to be loaded once.
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_ID
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # --------------------------------------------------------
    # Dataset only needs to be tokenized once.
    # --------------------------------------------------------

    print(
        "Loading controlled datasets..."
    )

    train_dataset = TextDataset(
        DATA_DIR / "members.jsonl",
        tokenizer,
    )

    eval_dataset = TextDataset(
        DATA_DIR / "validation.jsonl",
        tokenizer,
    )

    print(
        "Training members:",
        len(train_dataset),
    )

    print(
        "Validation examples:",
        len(eval_dataset),
    )

    # --------------------------------------------------------
    # Run all seeds sequentially.
    # --------------------------------------------------------

    overall_start = time.time()

    for run_number, seed in enumerate(
        SEEDS,
        start=1,
    ):
        print("\n\n")
        print("=" * 70)

        print(
            f"MULTI-SEED RUN "
            f"{run_number}/{len(SEEDS)}"
        )

        print(
            f"SEED = {seed}"
        )

        print("=" * 70)

        train_seed(
            seed=seed,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    overall_elapsed = (
        time.time()
        - overall_start
    )

    # --------------------------------------------------------
    # Completion summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MULTI-SEED TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        f"Total runtime: "
        f"{overall_elapsed / 3600:.2f} hours"
    )

    print(
        "\nResults saved under:"
    )

    for seed in SEEDS:
        print(
            " ",
            MULTISEED_ROOT
            / f"seed_{seed}"
            / "realistic_lora"
        )

    print(
        "\nNext step:"
        "\nScore membership for each trained "
        "target adapter using the same fixed "
        "5,000-member / 5,000-nonmember population."
    )


if __name__ == "__main__":
    main()