import json
import hashlib
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_ID = "google/gemma-3-4b-pt"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"

SEED = 42
MAX_TOKENS = 256
MIN_TOKENS = 64

N_MEMBERS = 5000
N_NONMEMBERS = 5000
N_VALIDATION = 1000
N_UTILITY = 1000

OUTPUT_DIR = Path("data/gemma")


def text_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading WikiText-103...")
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
    )

    print("Raw rows:", len(dataset))

    # --------------------------------------------------
    # Build sufficiently long text examples.
    #
    # WikiText contains headings, blank rows, and short
    # fragments, so concatenate text until we have useful
    # causal-LM sequences.
    # --------------------------------------------------

    examples = []
    buffer = ""

    print("Constructing sequences...")

    for row in dataset:
        text = row["text"].strip()

        if not text:
            continue

        if buffer:
            buffer += "\n" + text
        else:
            buffer = text

        token_ids = tokenizer(
            buffer,
            add_special_tokens=False,
        )["input_ids"]

        while len(token_ids) >= MAX_TOKENS:
            chunk_ids = token_ids[:MAX_TOKENS]

            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
            ).strip()

            if len(chunk_ids) >= MIN_TOKENS and chunk_text:
                examples.append({
                    "text": chunk_text,
                    "num_tokens": len(chunk_ids),
                    "sha256": text_hash(chunk_text),
                })

            token_ids = token_ids[MAX_TOKENS:]

            if token_ids:
                buffer = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                )
            else:
                buffer = ""

    print("Constructed examples:", len(examples))

    # Remove accidental duplicates.
    unique = {}
    for example in examples:
        unique[example["sha256"]] = example

    examples = list(unique.values())

    print("Unique examples:", len(examples))

    required = (
        N_MEMBERS
        + N_NONMEMBERS
        + N_VALIDATION
        + N_UTILITY
    )

    if len(examples) < required:
        raise RuntimeError(
            f"Need {required} examples but only "
            f"constructed {len(examples)}."
        )

    # --------------------------------------------------
    # Deterministic random assignment
    # --------------------------------------------------

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(examples))

    examples = [examples[i] for i in indices[:required]]

    cursor = 0

    members = examples[cursor:cursor + N_MEMBERS]
    cursor += N_MEMBERS

    nonmembers = examples[cursor:cursor + N_NONMEMBERS]
    cursor += N_NONMEMBERS

    validation = examples[cursor:cursor + N_VALIDATION]
    cursor += N_VALIDATION

    utility = examples[cursor:cursor + N_UTILITY]

    splits = {
        "members": members,
        "nonmembers": nonmembers,
        "validation": validation,
        "utility_test": utility,
    }

    # --------------------------------------------------
    # Save JSONL files
    # --------------------------------------------------

    for split_name, rows in splits.items():
        path = OUTPUT_DIR / f"{split_name}.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        print(f"{split_name:15s}: {len(rows):5d} -> {path}")

    # --------------------------------------------------
    # Manifest
    # --------------------------------------------------

    manifest = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "source_split": "train",
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "min_tokens": MIN_TOKENS,
        "counts": {
            name: len(rows)
            for name, rows in splits.items()
        },
        "membership_definition": (
            "Members are examples selected for LoRA fine-tuning. "
            "Nonmembers are held-out examples never used for LoRA "
            "fine-tuning. Membership does not refer to Gemma's "
            "original pretraining corpus."
        ),
    }

    manifest_path = OUTPUT_DIR / "split_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nSaved manifest:", manifest_path)

    # --------------------------------------------------
    # Final integrity checks
    # --------------------------------------------------

    hash_sets = {
        name: {row["sha256"] for row in rows}
        for name, rows in splits.items()
    }

    names = list(hash_sets)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = hash_sets[names[i]] & hash_sets[names[j]]

            if overlap:
                raise RuntimeError(
                    f"DATA LEAK: {names[i]} and {names[j]} "
                    f"share {len(overlap)} examples."
                )

    print("\nIntegrity checks passed.")
    print("No examples overlap between splits.")


if __name__ == "__main__":
    main()