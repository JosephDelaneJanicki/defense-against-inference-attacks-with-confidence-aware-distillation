import argparse
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
)
from peft import LoraConfig, get_peft_model


MODEL_ID = "google/gemma-3-4b-pt"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4
NUM_EPOCHS = 2

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM = 8


class BlackBoxDataset(Dataset):

    def __init__(self, cache_path):

        cache = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        self.examples = cache["examples"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        row = self.examples[idx]

        input_ids = row["input_ids"].long()
        pseudo_targets = row["pseudo_targets"].long()

        # Trainer expects labels aligned with input_ids.
        #
        # CausalLM internally shifts:
        # logits[t] -> labels[t+1]
        #
        # Therefore:
        # labels[1:] = target predicted from logits[:-1]
        labels = torch.full_like(
            input_ids,
            -100,
        )

        labels[1:] = pseudo_targets

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }


class BlackBoxCollator:

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):

        max_len = max(
            len(x["input_ids"])
            for x in examples
        )

        batch_ids = []
        batch_masks = []
        batch_labels = []

        for x in examples:

            length = len(x["input_ids"])
            pad = max_len - length

            batch_ids.append(
                torch.cat([
                    x["input_ids"],
                    torch.full(
                        (pad,),
                        self.pad_token_id,
                        dtype=torch.long,
                    ),
                ])
            )

            batch_masks.append(
                torch.cat([
                    x["attention_mask"],
                    torch.zeros(
                        pad,
                        dtype=torch.long,
                    ),
                ])
            )

            batch_labels.append(
                torch.cat([
                    x["labels"],
                    torch.full(
                        (pad,),
                        -100,
                        dtype=torch.long,
                    ),
                ])
            )

        return {
            "input_ids": torch.stack(batch_ids),
            "attention_mask": torch.stack(batch_masks),
            "labels": torch.stack(batch_labels),
        }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("=" * 72)
    print("BLACK-BOX ORDINARY LoRA SHADOW")
    print("=" * 72)

    print("Training cache:", args.cache)
    print("Validation cache:", args.validation_cache)
    print("Seed:", args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = BlackBoxDataset(args.cache)
    eval_dataset = BlackBoxDataset(args.validation_cache)

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(eval_dataset))

    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

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

    model.print_trainable_parameters()

    collator = BlackBoxCollator(
        tokenizer.pad_token_id
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),

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

        seed=args.seed,
        data_seed=args.seed,

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print("\nEvaluating before training...")
    initial_eval = trainer.evaluate()

    print(
        "Initial validation loss:",
        initial_eval.get("eval_loss"),
    )

    print("\nTraining black-box shadow...\n")

    start = time.time()

    result = trainer.train()

    elapsed = time.time() - start

    final_eval = trainer.evaluate()

    adapter_dir = output_dir / "adapter"

    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    summary = {
        "experiment": "black_box_shadow",
        "base_model": MODEL_ID,
        "label_source": "released_defended_student_argmax",
        "soft_teacher_outputs": False,
        "defense_reproduced": False,

        "seed": args.seed,

        "train_examples": len(train_dataset),
        "validation_examples": len(eval_dataset),

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
            "optimizer": "adamw_torch",
        },

        "initial_eval_loss":
            initial_eval.get("eval_loss"),

        "final_eval_loss":
            final_eval.get("eval_loss"),

        "train_loss":
            result.metrics.get("train_loss"),

        "training_seconds": elapsed,

        "peak_vram_gb":
            torch.cuda.max_memory_allocated()
            / 1024**3,
    }

    with (
        output_dir / "training_summary.json"
    ).open("w") as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\nSaved adapter:")
    print(adapter_dir)

    print("\nBLACK-BOX SHADOW COMPLETE")


if __name__ == "__main__":
    main()