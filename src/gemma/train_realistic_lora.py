import json
import time
from pathlib import Path

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


MODEL_ID = "google/gemma-3-4b-pt"

DATA_DIR = Path("data/gemma")
OUTPUT_DIR = Path("results/gemma/realistic_lora")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_LENGTH = 256

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM = 8


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


def gpu_memory():
    return {
        "allocated_gb":
            torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb":
            torch.cuda.memory_reserved() / 1024**3,
        "peak_allocated_gb":
            torch.cuda.max_memory_allocated() / 1024**3,
    }


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("=" * 60)
    print("GEMMA 3 4B PT — REALISTIC LoRA")
    print("=" * 60)

    print("\nModel:", MODEL_ID)
    print("Precision: BF16")
    print("Quantization: NONE")
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print("Loading controlled datasets...")

    train_dataset = TextDataset(
        DATA_DIR / "members.jsonl",
        tokenizer,
    )

    eval_dataset = TextDataset(
        DATA_DIR / "validation.jsonl",
        tokenizer,
    )

    print("Training members:", len(train_dataset))
    print("Validation examples:", len(eval_dataset))

    # --------------------------------------------------
    # Base model
    # --------------------------------------------------

    print("\nLoading Gemma in BF16...")

    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )

    # Saves activation memory during training.
    model.gradient_checkpointing_enable()

    # Required/recommended when using gradient checkpointing.
    model.config.use_cache = False

    # --------------------------------------------------
    # LoRA
    # --------------------------------------------------

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",

        # Explicit rather than relying on PEFT defaults.
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

    # --------------------------------------------------
    # Collator
    # --------------------------------------------------

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --------------------------------------------------
    # Trainer configuration
    # --------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        learning_rate=LEARNING_RATE,

        bf16=True,
        fp16=False,

        logging_strategy="steps",
        logging_steps=50,

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        gradient_checkpointing=True,

        optim="adamw_torch",

        seed=SEED,
        data_seed=SEED,

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # --------------------------------------------------
    # Initial validation loss
    # --------------------------------------------------

    print("\nEvaluating BEFORE fine-tuning...")

    initial_eval = trainer.evaluate()

    print(
        "Initial validation loss:",
        initial_eval.get("eval_loss"),
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("\nBeginning realistic LoRA training...\n")

    start_time = time.time()

    train_result = trainer.train()

    elapsed = time.time() - start_time

    # --------------------------------------------------
    # Final evaluation
    # --------------------------------------------------

    print("\nEvaluating AFTER fine-tuning...")

    final_eval = trainer.evaluate()

    print(
        "Final validation loss:",
        final_eval.get("eval_loss"),
    )

    # --------------------------------------------------
    # Save adapter
    # --------------------------------------------------

    adapter_dir = OUTPUT_DIR / "adapter"

    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("\nSaved LoRA adapter:")
    print(adapter_dir)

    # --------------------------------------------------
    # Experiment summary
    # --------------------------------------------------

    memory = gpu_memory()

    summary = {
        "experiment": "gemma_3_4b_realistic_lora",
        "base_model": MODEL_ID,
        "precision": "bfloat16",
        "quantization": None,

        "membership_definition":
            "Fine-tuning membership in members.jsonl",

        "train_examples": len(train_dataset),
        "validation_examples": len(eval_dataset),

        "max_length": MAX_LENGTH,

        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        },

        "training": {
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUM,
            "effective_batch_size":
                TRAIN_BATCH_SIZE * GRAD_ACCUM,
            "optimizer": "adamw_torch",
            "seed": SEED,
        },

        "initial_eval_loss":
            initial_eval.get("eval_loss"),

        "final_eval_loss":
            final_eval.get("eval_loss"),

        "train_loss":
            train_result.metrics.get("train_loss"),

        "training_seconds": elapsed,

        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory": memory,
    }

    summary_path = OUTPUT_DIR / "training_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved training summary:")
    print(summary_path)

    print("\nGPU memory:")
    print(
        f"Peak allocated: "
        f"{memory['peak_allocated_gb']:.2f} GB"
    )

    print("\nREALISTIC LoRA TRAINING COMPLETE")


if __name__ == "__main__":
    main()