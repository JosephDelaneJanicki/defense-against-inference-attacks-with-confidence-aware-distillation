import json
import hashlib
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_ID = "google/gemma-3-4b-pt"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"

MAX_TOKENS = 256
MIN_TOKENS = 64

DATA_DIR = Path("data/gemma")
OUTPUT_DIR = DATA_DIR / "neighbor_control"

MEMBERS_PATH = DATA_DIR / "members.jsonl"
NONMEMBERS_PATH = DATA_DIR / "nonmembers.jsonl"
VALIDATION_PATH = DATA_DIR / "validation.jsonl"
UTILITY_PATH = DATA_DIR / "utility_test.jsonl"


def text_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def reconstruct_examples(tokenizer):
    """
    Reproduce the original prepare_mia_dataset.py construction exactly,
    while recording each chunk's position in the constructed sequence stream.

    IMPORTANT:
    The original preparation script concatenated nonblank WikiText rows
    continuously and did not preserve article IDs/boundaries. Therefore this
    control is an ADJACENT-CONTEXT control, not a guaranteed same-article
    control.
    """
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
    )

    print("Raw rows:", len(dataset))
    print("Reconstructing original 256-token sequence stream...")

    examples = []
    buffer = ""
    constructed_index = 0

    for row_index, row in enumerate(dataset):
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
                    "constructed_index": constructed_index,
                    "source_row_at_emission": row_index,
                })
                constructed_index += 1

            token_ids = token_ids[MAX_TOKENS:]

            if token_ids:
                buffer = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                )
            else:
                buffer = ""

    print("Constructed examples:", len(examples))

    # Match original duplicate-removal semantics:
    # dict assignment keeps the LAST value for a duplicate hash, while
    # insertion order remains that of the first insertion.
    unique = {}
    for example in examples:
        unique[example["sha256"]] = example

    unique_examples = list(unique.values())

    print("Unique examples:", len(unique_examples))
    return unique_examples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GEMMA WIKITEXT ADJACENT-CONTEXT MEMBERSHIP CONTROL")
    print("=" * 72)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading original controlled splits...")
    members = load_jsonl(MEMBERS_PATH)
    nonmembers = load_jsonl(NONMEMBERS_PATH)
    validation = load_jsonl(VALIDATION_PATH)
    utility = load_jsonl(UTILITY_PATH)

    print("Members:", len(members))
    print("Original nonmembers:", len(nonmembers))

    stream = reconstruct_examples(tokenizer)

    by_hash = {
        row["sha256"]: row
        for row in stream
    }

    # Use constructed_index for actual adjacency, because the original
    # duplicate-removal step can make list position subtly different.
    by_constructed_index = {
        row["constructed_index"]: row
        for row in stream
    }

    member_hashes = {
        row["sha256"]
        for row in members
    }

    # Reserve every original target/control split from becoming a newly
    # selected neighbor. This keeps the neighbor population distinct from
    # members, original nonmembers, validation, and utility examples.
    reserved_hashes = {
        row["sha256"]
        for split in [members, nonmembers, validation, utility]
        for row in split
    }

    missing_members = [
        h for h in member_hashes
        if h not in by_hash
    ]

    if missing_members:
        raise RuntimeError(
            f"Could not reconstruct {len(missing_members)} member hashes. "
            "Sequence construction no longer matches the original script."
        )

    print("\nSelecting nearest unused adjacent-context controls...")

    used_neighbor_hashes = set()
    matched_pairs = []

    # Deterministic order: preserve original members.jsonl order.
    for member_number, member in enumerate(members):
        member_stream = by_hash[member["sha256"]]
        center = member_stream["constructed_index"]

        chosen = None
        chosen_offset = None

        # Prefer immediate neighbors, then expand only when necessary.
        # Alternate left/right deterministically.
        max_radius = 100

        for radius in range(1, max_radius + 1):
            for offset in (-radius, radius):
                candidate = by_constructed_index.get(
                    center + offset
                )

                if candidate is None:
                    continue

                h = candidate["sha256"]

                if h in reserved_hashes:
                    continue

                if h in used_neighbor_hashes:
                    continue

                chosen = candidate
                chosen_offset = offset
                break

            if chosen is not None:
                break

        if chosen is None:
            raise RuntimeError(
                "Could not find a unique unreserved adjacent control "
                f"within ±{max_radius} chunks for member {member_number}."
            )

        used_neighbor_hashes.add(chosen["sha256"])

        matched_pairs.append({
            "pair_id": member_number,
            "member": {
                **member,
                "constructed_index": center,
            },
            "neighbor": {
                "text": chosen["text"],
                "num_tokens": chosen["num_tokens"],
                "sha256": chosen["sha256"],
                "constructed_index": chosen["constructed_index"],
                "offset_chunks": chosen_offset,
                "absolute_offset_chunks": abs(chosen_offset),
            },
        })

    exact_members = []
    neighbors = []

    for pair in matched_pairs:
        m = dict(pair["member"])
        m["pair_id"] = pair["pair_id"]
        m["control_group"] = "exact_member"
        exact_members.append(m)

        n = dict(pair["neighbor"])
        n["pair_id"] = pair["pair_id"]
        n["control_group"] = "adjacent_nonmember"
        neighbors.append(n)

    # The existing 5k held-out nonmembers are our far/random comparison.
    random_nonmembers = []
    for i, row in enumerate(nonmembers):
        r = dict(row)
        r["pair_id"] = i
        r["control_group"] = "random_nonmember"
        random_nonmembers.append(r)

    # --------------------------------------------------
    # Integrity checks
    # --------------------------------------------------

    exact_hashes = {r["sha256"] for r in exact_members}
    neighbor_hashes = {r["sha256"] for r in neighbors}
    random_hashes = {r["sha256"] for r in random_nonmembers}

    if exact_hashes & neighbor_hashes:
        raise RuntimeError("Exact members overlap adjacent controls.")

    if exact_hashes & random_hashes:
        raise RuntimeError("Exact members overlap random nonmembers.")

    if neighbor_hashes & random_hashes:
        raise RuntimeError("Adjacent controls overlap random nonmembers.")

    if len(neighbor_hashes) != len(neighbors):
        raise RuntimeError("Adjacent controls are not unique.")

    offsets = [
        row["absolute_offset_chunks"]
        for row in neighbors
    ]

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    outputs = {
        "exact_members": exact_members,
        "adjacent_nonmembers": neighbors,
        "random_nonmembers": random_nonmembers,
    }

    for name, rows in outputs.items():
        path = OUTPUT_DIR / f"{name}.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        print(f"{name:22s}: {len(rows):5d} -> {path}")

    pairs_path = OUTPUT_DIR / "matched_pairs.jsonl"

    with pairs_path.open("w", encoding="utf-8") as f:
        for pair in matched_pairs:
            f.write(json.dumps(pair) + "\n")

    summary = {
        "control_type": "adjacent_context",
        "important_limitation": (
            "The original dataset construction concatenated nonblank "
            "WikiText rows continuously and did not preserve article IDs "
            "or article boundaries. Therefore adjacency here means nearest "
            "constructed 256-token chunk in the original corpus stream; "
            "it is not guaranteed to mean the same Wikipedia article."
        ),
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "max_tokens": MAX_TOKENS,
        "counts": {
            name: len(rows)
            for name, rows in outputs.items()
        },
        "neighbor_matching": {
            "reserved_original_splits": [
                "members",
                "nonmembers",
                "validation",
                "utility_test",
            ],
            "unique_neighbors": True,
            "prefer_immediate_chunk": True,
            "mean_absolute_offset_chunks": (
                sum(offsets) / len(offsets)
            ),
            "max_absolute_offset_chunks": max(offsets),
            "immediate_neighbor_fraction": (
                sum(x == 1 for x in offsets) / len(offsets)
            ),
        },
        "planned_comparisons": [
            "exact_member_vs_adjacent_nonmember",
            "exact_member_vs_random_nonmember",
            "adjacent_nonmember_vs_random_nonmember",
        ],
    }

    summary_path = OUTPUT_DIR / "control_manifest.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("NEIGHBOR CONTROL PREPARATION COMPLETE")
    print("=" * 72)

    print("\nMatched pairs:", pairs_path)
    print("Manifest:", summary_path)

    print("\nNeighbor quality:")
    print(
        "  Immediate-neighbor fraction:",
        f"{summary['neighbor_matching']['immediate_neighbor_fraction']:.4f}",
    )
    print(
        "  Mean |offset| in 256-token chunks:",
        f"{summary['neighbor_matching']['mean_absolute_offset_chunks']:.4f}",
    )
    print(
        "  Max |offset| in 256-token chunks:",
        summary["neighbor_matching"]["max_absolute_offset_chunks"],
    )

    print(
        "\nIMPORTANT: this is an adjacent-context control, not a "
        "guaranteed same-article control, because the original preparation "
        "script did not retain WikiText article boundaries."
    )


if __name__ == "__main__":
    main()
