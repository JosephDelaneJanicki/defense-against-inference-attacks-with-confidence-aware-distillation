import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "google/gemma-3-4b-pt"

DATA_DIR = Path("data/gemma")
TEACHER_DIR = Path(
    "results/gemma/multiseed/seed_42/realistic_lora/adapter"
)
OUTPUT_DIR = Path(
    "data/gemma/defense/seed_42/teacher_topk"
)

# Start with the same 5,000 controlled training-member sequences
# used to fine-tune the realistic target teacher.
CORPUS_PATH = DATA_DIR / "members.jsonl"

MAX_LENGTH = 256
BATCH_SIZE = 1
TOP_K = 64

# Adult-derived reference thresholds. These are measured here,
# not tuned against downstream MIA results.
REFERENCE_THRESHOLDS = [0.85, 0.95]


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            rows.append({
                "index": i,
                "sha256": row.get("sha256", f"row_{i}"),
                "text": row["text"],
            })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = OUTPUT_DIR / "teacher_topk_cache.pt"
    summary_path = OUTPUT_DIR / "teacher_topk_summary.json"

    print("=" * 70)
    print("GEMMA SEED-42 TEACHER TOP-K DISTILLATION CACHE")
    print("=" * 70)

    print("\nBase model:", MODEL_ID)
    print("Teacher adapter:", TEACHER_DIR)
    print("Corpus:", CORPUS_PATH)
    print("Max length:", MAX_LENGTH)
    print("Top-K:", TOP_K)
    print("Batch size:", BATCH_SIZE)

    if not TEACHER_DIR.exists():
        raise FileNotFoundError(
            f"Teacher adapter does not exist: {TEACHER_DIR}"
        )

    rows = load_jsonl(CORPUS_PATH)
    print("Sequences:", len(rows))

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base Gemma in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )

    print("Loading seed-42 realistic LoRA teacher...")
    model = PeftModel.from_pretrained(
        model,
        TEACHER_DIR,
    )
    model = model.cuda()
    model.eval()

    print(
        "VRAM after load:",
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
    )

    cached_sequences = []

    all_confidences = []
    all_topk_mass = []

    total_valid_tokens = 0
    above_counts = {
        threshold: 0
        for threshold in REFERENCE_THRESHOLDS
    }

    print("\nCaching teacher distributions...")

    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start:start + BATCH_SIZE]
        texts = [row["text"] for row in batch_rows]

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

            # Match the established scorer exactly:
            # position t predicts token t+1.
            shift_logits = outputs.logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:].bool()

            # Top-K logits/IDs are enough to reconstruct the
            # renormalized Top-K soft target later.
            topk_logits, topk_ids = torch.topk(
                shift_logits,
                k=TOP_K,
                dim=-1,
            )

            # For exact confidence and captured probability mass,
            # compute logsumexp over the full vocabulary without
            # materializing a full softmax tensor.
            full_log_z = torch.logsumexp(
                shift_logits,
                dim=-1,
            )

            top1_confidence = torch.exp(
                topk_logits[..., 0] - full_log_z
            )

            topk_log_z = torch.logsumexp(
                topk_logits,
                dim=-1,
            )

            topk_mass = torch.exp(
                topk_log_z - full_log_z
            )

        for i, source in enumerate(batch_rows):
            mask = shift_mask[i]

            ids_i = topk_ids[i][mask].to(
                dtype=torch.int32
            ).cpu()

            # FP16 is sufficient for the cached teacher logits
            # and halves cache size versus FP32.
            logits_i = topk_logits[i][mask].to(
                dtype=torch.float16
            ).cpu()

            labels_i = shift_labels[i][mask].to(
                dtype=torch.int32
            ).cpu()

            confidence_i = top1_confidence[i][mask].to(
                dtype=torch.float32
            ).cpu()

            mass_i = topk_mass[i][mask].to(
                dtype=torch.float32
            ).cpu()

            cached_sequences.append({
                "index": source["index"],
                "sha256": source["sha256"],
                "input_ids": input_ids[i][attention_mask[i].bool()]
                    .to(dtype=torch.int32).cpu(),
                "target_ids": labels_i,
                "topk_ids": ids_i,
                "topk_logits": logits_i,
                "top1_confidence": confidence_i,
                "topk_mass": mass_i,
            })

            conf_np = confidence_i.numpy()
            mass_np = mass_i.numpy()

            all_confidences.append(conf_np)
            all_topk_mass.append(mass_np)

            total_valid_tokens += len(conf_np)

            for threshold in REFERENCE_THRESHOLDS:
                above_counts[threshold] += int(
                    np.sum(conf_np > threshold)
                )

        del outputs
        del shift_logits
        del topk_logits
        del topk_ids
        del full_log_z
        del topk_log_z
        del top1_confidence
        del topk_mass
        del input_ids
        del attention_mask
        del shift_labels
        del shift_mask

        if start % 100 == 0:
            print(
                f"Cached "
                f"{min(start + BATCH_SIZE, len(rows))}"
                f"/{len(rows)} sequences"
            )

    confidences = np.concatenate(all_confidences)
    masses = np.concatenate(all_topk_mass)

    percentiles = [50, 75, 90, 95, 99]

    confidence_percentiles = {
        f"p{p}": float(np.percentile(confidences, p))
        for p in percentiles
    }

    mass_percentiles = {
        f"p{p}": float(np.percentile(masses, p))
        for p in percentiles
    }

    fractions_above = {
        str(threshold): (
            above_counts[threshold] / total_valid_tokens
        )
        for threshold in REFERENCE_THRESHOLDS
    }

    summary = {
        "experiment": "gemma_seed42_teacher_topk_cache",
        "base_model": MODEL_ID,
        "teacher_adapter": str(TEACHER_DIR),
        "corpus": str(CORPUS_PATH),
        "num_sequences": len(rows),
        "max_length": MAX_LENGTH,
        "top_k": TOP_K,
        "total_prediction_tokens": total_valid_tokens,
        "teacher_confidence": {
            "mean": float(confidences.mean()),
            "std": float(confidences.std()),
            "min": float(confidences.min()),
            "max": float(confidences.max()),
            "percentiles": confidence_percentiles,
            "fraction_above_threshold": fractions_above,
        },
        "topk_probability_mass": {
            "mean": float(masses.mean()),
            "std": float(masses.std()),
            "min": float(masses.min()),
            "max": float(masses.max()),
            "percentiles": mass_percentiles,
        },
        "cache_format": {
            "topk_ids_dtype": "int32",
            "topk_logits_dtype": "float16",
            "target_ids_dtype": "int32",
            "confidence_dtype": "float32",
            "mass_dtype": "float32",
            "note": (
                "topk_logits are raw teacher logits. "
                "Temperature transformations should be applied "
                "to logits during student training."
            ),
        },
    }

    payload = {
        "metadata": summary,
        "sequences": cached_sequences,
    }

    print("\nSaving cache...")
    torch.save(payload, cache_path)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    size_gb = cache_path.stat().st_size / 1024**3

    print("\n" + "=" * 70)
    print("TEACHER TOP-K CACHE COMPLETE")
    print("=" * 70)

    print("\nCache:", cache_path)
    print("Summary:", summary_path)
    print(f"Cache size: {size_gb:.2f} GB")
    print("Prediction tokens:", total_valid_tokens)

    print("\nTeacher confidence:")
    print(f"  Mean: {confidences.mean():.6f}")
    for p in percentiles:
        print(
            f"  P{p:02d}: "
            f"{confidence_percentiles[f'p{p}']:.6f}"
        )

    for threshold in REFERENCE_THRESHOLDS:
        print(
            f"  Fraction > {threshold:.2f}: "
            f"{fractions_above[str(threshold)]:.6f}"
        )

    print("\nTop-64 captured probability mass:")
    print(f"  Mean: {masses.mean():.6f}")
    for p in percentiles:
        print(
            f"  P{p:02d}: "
            f"{mass_percentiles[f'p{p}']:.6f}"
        )

    print(
        "\nDo NOT tune defense thresholds from MIA results yet. "
        "Inspect this confidence/mass summary first."
    )


if __name__ == "__main__":
    main()
