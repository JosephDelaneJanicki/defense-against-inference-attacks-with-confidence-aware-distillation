import json
import hashlib
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_ID = "google/gemma-3-4b-pt"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"

DATA_DIR = Path("data/gemma")
SHADOW_DIR = DATA_DIR / "shadows"

SEED = 4242

MAX_TOKENS = 256
MIN_TOKENS = 64

N_SHADOWS = 6

N_MEMBERS = 5000
N_NONMEMBERS = 5000
N_VALIDATION = 1000


def text_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_reserved_hashes():
    """
    Load every example reserved for the target experiment.

    None of these examples may appear anywhere in the
    shadow-model experiment.
    """

    reserved = set()

    files = [
        "members.jsonl",
        "nonmembers.jsonl",
        "validation.jsonl",
        "utility_test.jsonl",
    ]

    for filename in files:
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing target dataset: {path}"
            )

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                reserved.add(row["sha256"])

    return reserved


def build_examples(tokenizer):
    """
    Reconstruct WikiText chunks using exactly the same
    chunking procedure used for the target experiment.
    """

    print("Loading WikiText-103...")

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
    )

    print("Raw rows:", len(dataset))
    print("Reconstructing 256-token sequences...")

    examples = []
    buffer = ""

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

            if (
                len(chunk_ids) >= MIN_TOKENS
                and chunk_text
            ):
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

    # Deduplicate using SHA-256.
    unique = {}

    for example in examples:
        unique[example["sha256"]] = example

    examples = list(unique.values())

    print("Unique examples:", len(examples))

    return examples


def save_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(row) + "\n"
            )


def main():
    SHADOW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("PREPARING GEMMA SHADOW DATASETS")
    print("=" * 60)

    print("\nConfiguration:")
    print("Shadows:", N_SHADOWS)
    print("Members per shadow:", N_MEMBERS)
    print("Nonmembers per shadow:", N_NONMEMBERS)
    print("Validation per shadow:", N_VALIDATION)
    print("Seed:", SEED)

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    # --------------------------------------------------
    # Reconstruct source population
    # --------------------------------------------------

    examples = build_examples(tokenizer)

    # --------------------------------------------------
    # Exclude target experiment
    # --------------------------------------------------

    print("\nLoading target-reserved hashes...")

    reserved = load_reserved_hashes()

    print(
        "Target/reserved examples:",
        len(reserved),
    )

    available = [
        example
        for example in examples
        if example["sha256"] not in reserved
    ]

    print(
        "Available after target exclusion:",
        len(available),
    )

    # --------------------------------------------------
    # Determine required population
    # --------------------------------------------------

    per_shadow = (
        N_MEMBERS
        + N_NONMEMBERS
        + N_VALIDATION
    )

    required = (
        N_SHADOWS
        * per_shadow
    )

    print(
        "\nExamples required per shadow:",
        per_shadow,
    )

    print(
        "Total examples required:",
        required,
    )

    if len(available) < required:
        raise RuntimeError(
            f"Need {required} shadow examples "
            f"but only {len(available)} are available."
        )

    # --------------------------------------------------
    # Deterministic shadow allocation
    # --------------------------------------------------

    rng = np.random.default_rng(SEED)

    order = rng.permutation(
        len(available)
    )

    selected = [
        available[i]
        for i in order[:required]
    ]

    cursor = 0

    all_shadow_hashes = set()

    manifest = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "source_split": "train",

        "seed": SEED,

        "num_shadows": N_SHADOWS,

        "members_per_shadow":
            N_MEMBERS,

        "nonmembers_per_shadow":
            N_NONMEMBERS,

        "validation_per_shadow":
            N_VALIDATION,

        "max_tokens":
            MAX_TOKENS,

        "min_tokens":
            MIN_TOKENS,

        "target_overlap_allowed":
            False,

        "cross_shadow_overlap_allowed":
            False,

        "membership_definition": (
            "Shadow members are examples used to "
            "fine-tune the corresponding shadow LoRA. "
            "Shadow nonmembers are held-out examples "
            "never used for that shadow's fine-tuning "
            "or model selection."
        ),

        "shadows": {},
    }

    # --------------------------------------------------
    # Construct each shadow population
    # --------------------------------------------------

    for shadow_id in range(N_SHADOWS):
        name = f"shadow_{shadow_id:02d}"

        directory = (
            SHADOW_DIR / name
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ------------------------------
        # Members
        # ------------------------------

        members = selected[
            cursor:
            cursor + N_MEMBERS
        ]

        cursor += N_MEMBERS

        # ------------------------------
        # Nonmembers
        # ------------------------------

        nonmembers = selected[
            cursor:
            cursor + N_NONMEMBERS
        ]

        cursor += N_NONMEMBERS

        # ------------------------------
        # Validation
        # ------------------------------

        validation = selected[
            cursor:
            cursor + N_VALIDATION
        ]

        cursor += N_VALIDATION

        member_hashes = {
            row["sha256"]
            for row in members
        }

        nonmember_hashes = {
            row["sha256"]
            for row in nonmembers
        }

        validation_hashes = {
            row["sha256"]
            for row in validation
        }

        # --------------------------------------------------
        # Within-shadow integrity checks
        # --------------------------------------------------

        if member_hashes & nonmember_hashes:
            raise RuntimeError(
                f"Member/nonmember overlap "
                f"detected in {name}"
            )

        if member_hashes & validation_hashes:
            raise RuntimeError(
                f"Member/validation overlap "
                f"detected in {name}"
            )

        if (
            nonmember_hashes
            & validation_hashes
        ):
            raise RuntimeError(
                f"Nonmember/validation overlap "
                f"detected in {name}"
            )

        current_hashes = (
            member_hashes
            | nonmember_hashes
            | validation_hashes
        )

        expected_count = (
            N_MEMBERS
            + N_NONMEMBERS
            + N_VALIDATION
        )

        if len(current_hashes) != expected_count:
            raise RuntimeError(
                f"Unexpected duplicate examples "
                f"inside {name}"
            )

        # --------------------------------------------------
        # Target isolation
        # --------------------------------------------------

        target_overlap = (
            current_hashes
            & reserved
        )

        if target_overlap:
            raise RuntimeError(
                f"TARGET LEAKAGE: {name} contains "
                f"{len(target_overlap)} target examples."
            )

        # --------------------------------------------------
        # Cross-shadow isolation
        # --------------------------------------------------

        cross_overlap = (
            current_hashes
            & all_shadow_hashes
        )

        if cross_overlap:
            raise RuntimeError(
                f"CROSS-SHADOW LEAKAGE: {name} "
                f"shares {len(cross_overlap)} examples "
                f"with an earlier shadow."
            )

        all_shadow_hashes.update(
            current_hashes
        )

        # --------------------------------------------------
        # Save datasets
        # --------------------------------------------------

        save_jsonl(
            directory / "members.jsonl",
            members,
        )

        save_jsonl(
            directory / "nonmembers.jsonl",
            nonmembers,
        )

        save_jsonl(
            directory / "validation.jsonl",
            validation,
        )

        manifest["shadows"][name] = {
            "members": len(members),
            "nonmembers": len(nonmembers),
            "validation": len(validation),
            "total": len(current_hashes),
        }

        print(
            f"{name}: "
            f"{len(members)} members + "
            f"{len(nonmembers)} nonmembers + "
            f"{len(validation)} validation "
            f"= {len(current_hashes)}"
        )

    # --------------------------------------------------
    # Final global integrity checks
    # --------------------------------------------------

    print("\nRunning final integrity checks...")

    if len(all_shadow_hashes) != required:
        raise RuntimeError(
            "Global shadow example count does not "
            "match expected count."
        )

    final_target_overlap = (
        all_shadow_hashes
        & reserved
    )

    if final_target_overlap:
        raise RuntimeError(
            "Target overlap detected during final check."
        )

    # --------------------------------------------------
    # Save manifest
    # --------------------------------------------------

    manifest["total_shadow_examples"] = (
        len(all_shadow_hashes)
    )

    manifest["total_target_reserved"] = (
        len(reserved)
    )

    manifest_path = (
        SHADOW_DIR
        / "shadow_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SHADOW DATASET SUMMARY")
    print("=" * 60)

    print(
        "Shadow models:",
        N_SHADOWS,
    )

    print(
        "Examples per shadow:",
        per_shadow,
    )

    print(
        "Total unique shadow examples:",
        len(all_shadow_hashes),
    )

    print(
        "Target overlap:",
        len(
            all_shadow_hashes
            & reserved
        ),
    )

    print(
        "Cross-shadow overlap: 0"
    )

    print(
        "\nSaved manifest:",
        manifest_path,
    )

    print(
        "\nSHADOW DATASET PREPARATION COMPLETE"
    )


if __name__ == "__main__":
    main()