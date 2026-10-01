import json
import math
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_ID = "google/gemma-3-4b-pt"
CACHE_PATH = Path("data/gemma/defense/seed_42/teacher_topk/teacher_topk_cache.pt")
VALIDATION_PATH = Path("data/gemma/validation.jsonl")
OUTPUT_DIR = Path("results/gemma/defense/seed_42/raw_soft")
ADAPTER_DIR = OUTPUT_DIR / "adapter"

SEED = 42
MAX_LENGTH = 256

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 1
GRAD_ACCUM = 8
WEIGHT_DECAY = 0.0
TEMPERATURE = 1.0


class TeacherCacheDataset(Dataset):
    def __init__(self, path):
        print("Loading teacher cache:", path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.metadata = payload["metadata"]
        self.rows = payload["sequences"]
        print("Cached sequences:", len(self.rows))
        print("Teacher Top-K:", self.metadata["top_k"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class ValidationDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                enc = tokenizer(
                    row["text"],
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=False,
                    add_special_tokens=True,
                )
                self.rows.append({
                    "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_teacher(batch):
    if len(batch) != 1:
        raise RuntimeError("This cache collator expects TRAIN_BATCH_SIZE=1.")
    row = batch[0]
    return {
        "input_ids": row["input_ids"].long().unsqueeze(0),
        "topk_ids": row["topk_ids"].long().unsqueeze(0),
        "topk_logits": row["topk_logits"].float().unsqueeze(0),
        "topk_mass": row["topk_mass"].float().unsqueeze(0),
    }


def evaluate_validation_nll(model, dataset):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for row in dataset:
            input_ids = row["input_ids"].unsqueeze(0).cuda()
            attention_mask = row["attention_mask"].unsqueeze(0).cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:].bool()

            losses = F.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                reduction="none",
            )
            valid = losses[shift_mask]
            total_loss += float(valid.sum().item())
            total_tokens += int(valid.numel())

    model.train()
    return total_loss / total_tokens


def raw_soft_topk_other_loss(
    student_logits,
    teacher_topk_ids,
    teacher_topk_logits,
    teacher_topk_mass,
):
    """
    Raw-soft KD at T=1.

    Teacher Top-K raw logits are converted to relative probabilities, then
    multiplied by the exact cached Top-K probability mass. The omitted
    vocabulary is represented as one residual 'other' bucket.

    Student probabilities are normalized over the full vocabulary. Its
    teacher-Top-K probabilities are gathered explicitly and all remaining
    probability mass is collapsed into the same 'other' bucket.
    """
    student_shift_logits = student_logits[:, :-1, :].float()

    if student_shift_logits.shape[:2] != teacher_topk_ids.shape[:2]:
        raise RuntimeError(
            f"Alignment mismatch: student={student_shift_logits.shape[:2]}, "
            f"teacher={teacher_topk_ids.shape[:2]}"
        )

    teacher_within_topk = F.softmax(teacher_topk_logits, dim=-1)
    teacher_mass = teacher_topk_mass.clamp(0.0, 1.0).unsqueeze(-1)
    teacher_topk_prob = teacher_within_topk * teacher_mass
    teacher_other_prob = (1.0 - teacher_topk_mass).clamp(0.0, 1.0)

    student_log_probs = F.log_softmax(student_shift_logits, dim=-1)
    student_topk_log_probs = torch.gather(
        student_log_probs, dim=-1, index=teacher_topk_ids
    )

    student_topk_prob_sum = student_topk_log_probs.exp().sum(dim=-1)
    student_other_prob = (1.0 - student_topk_prob_sum).clamp(1e-8, 1.0)
    student_other_log_prob = torch.log(student_other_prob)

    topk_ce = -(teacher_topk_prob * student_topk_log_probs).sum(dim=-1)
    other_ce = -(teacher_other_prob * student_other_log_prob)

    return (topk_ce + other_ce).mean()


def main():
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GEMMA 3 4B — RAW-SOFT DISTILLED LoRA STUDENT")
    print("=" * 72)
    print("Base model:", MODEL_ID)
    print("Teacher cache:", CACHE_PATH)
    print("Temperature:", TEMPERATURE)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing teacher cache: {CACHE_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TeacherCacheDataset(CACHE_PATH)
    validation_dataset = ValidationDataset(VALIDATION_PATH, tokenizer)

    print("Validation sequences:", len(validation_dataset))
    print("\nLoading fresh Gemma base in BF16...")

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config).cuda()
    model.print_trainable_parameters()

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_teacher,
        generator=torch.Generator().manual_seed(SEED),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
    total_updates = updates_per_epoch * NUM_EPOCHS

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1e-3,
        total_iters=max(total_updates, 1),
    )

    print("\nEvaluating BEFORE distillation...")
    initial_eval = evaluate_validation_nll(model, validation_dataset)
    print(f"Initial validation NLL: {initial_eval:.6f}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()
    loss_history = []
    global_update = 0

    print("\nBeginning raw-soft distillation...\n")

    for epoch in range(NUM_EPOCHS):
        epoch_losses = []

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            topk_ids = batch["topk_ids"].cuda(non_blocking=True)
            topk_logits = batch["topk_logits"].cuda(non_blocking=True)
            topk_mass = batch["topk_mass"].cuda(non_blocking=True)

            attention_mask = torch.ones_like(input_ids)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = raw_soft_topk_other_loss(
                outputs.logits,
                topk_ids,
                topk_logits,
                topk_mass,
            )

            raw_loss = float(loss.detach().item())
            (loss / GRAD_ACCUM).backward()

            loss_history.append(raw_loss)
            epoch_losses.append(raw_loss)

            should_update = (
                (step + 1) % GRAD_ACCUM == 0
                or step + 1 == len(train_loader)
            )

            if should_update:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

                if global_update % 25 == 0:
                    recent = float(np.mean(loss_history[-25:]))
                    print(
                        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                        f"update {global_update}/{total_updates} | "
                        f"KD loss {recent:.6f} | "
                        f"lr {scheduler.get_last_lr()[0]:.8f}"
                    )

            del outputs, loss

        print(
            f"\nEpoch {epoch + 1} mean KD loss: "
            f"{np.mean(epoch_losses):.6f}"
        )

        epoch_eval = evaluate_validation_nll(model, validation_dataset)
        print(f"Validation NLL: {epoch_eval:.6f}\n")

    elapsed = time.time() - start_time

    print("Evaluating AFTER distillation...")
    final_eval = evaluate_validation_nll(model, validation_dataset)
    print(f"Final validation NLL: {final_eval:.6f}")

    print("\nSaving student adapter...")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    summary = {
        "experiment": "gemma_seed42_raw_soft_distilled_lora",
        "defense_variant": "raw_soft",
        "base_model": MODEL_ID,
        "teacher_cache": str(CACHE_PATH),
        "teacher_top_k": int(train_dataset.metadata["top_k"]),
        "teacher_representation": (
            "Exact cached Top-K mass plus one residual other-vocabulary bucket"
        ),
        "temperature": TEMPERATURE,
        "hard_label_loss": False,
        "confidence_clipping": None,
        "confidence_filtering": None,
        "student_lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "training": {
            "seed": SEED,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUM,
            "effective_batch_size": TRAIN_BATCH_SIZE * GRAD_ACCUM,
            "optimizer": "AdamW",
            "train_sequences": len(train_dataset),
            "validation_sequences": len(validation_dataset),
        },
        "initial_validation_nll": initial_eval,
        "final_validation_nll": final_eval,
        "mean_training_kd_loss": float(np.mean(loss_history)),
        "training_seconds": elapsed,
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_gb": peak_gb,
    }

    summary_path = OUTPUT_DIR / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("RAW-SOFT STUDENT TRAINING COMPLETE")
    print("=" * 72)
    print("Adapter:", ADAPTER_DIR)
    print("Summary:", summary_path)
    print(f"Training time: {elapsed / 60:.2f} minutes")
    print(f"Peak allocated VRAM: {peak_gb:.2f} GB")
    print(f"Validation NLL: {initial_eval:.6f} -> {final_eval:.6f}")


if __name__ == "__main__":
    main()
