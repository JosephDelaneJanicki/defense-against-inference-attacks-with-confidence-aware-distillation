import argparse
import subprocess
from pathlib import Path

import pandas as pd


SHADOW_COUNT = 6


def run(command):

    print("\n$", " ".join(str(x) for x in command))

    subprocess.run(
        [str(x) for x in command],
        check=True,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--defense",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    defense = args.defense
    seed = args.seed

    target_adapter = Path(
        f"results/gemma/defense/"
        f"seed_{seed}/{defense}/adapter"
    )

    root = Path(
        f"results/gemma/defense/"
        f"seed_{seed}/{defense}/black_box_mia"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not target_adapter.exists():
        raise FileNotFoundError(
            f"Defended target adapter missing: "
            f"{target_adapter}"
        )

    print("=" * 80)
    print("BLACK-BOX MIA PIPELINE")
    print("=" * 80)

    print("Defense:", defense)
    print("Seed:", seed)
    print("Target:", target_adapter)

    all_scores = []

    for i in range(SHADOW_COUNT):

        shadow_name = f"shadow_{i:02d}"

        data_root = Path(
            f"data/gemma/shadows/{shadow_name}"
        )

        shadow_root = (
            root
            / "shadows"
            / shadow_name
        )

        cache_root = (
            shadow_root
            / "black_box_labels"
        )

        train_cache = (
            cache_root
            / "members_hard_labels.pt"
        )

        validation_cache = (
            cache_root
            / "validation_hard_labels.pt"
        )

        adapter = (
            shadow_root
            / "adapter"
        )

        summary = (
            shadow_root
            / "training_summary.json"
        )

        scores = (
            shadow_root
            / "membership_scores.csv"
        )

        print("\n" + "=" * 80)
        print(shadow_name.upper())
        print("=" * 80)

        # ----------------------------------------------------
        # Query target on shadow members
        # ----------------------------------------------------

        if train_cache.exists():

            print(
                "[SKIP] member hard-label cache exists:",
                train_cache,
            )

        else:

            run([
                "python",
                "src/gemma/cache_black_box_labels.py",

                "--student-adapter",
                target_adapter,

                "--corpus",
                data_root / "members.jsonl",

                "--output",
                train_cache,
            ])

        # ----------------------------------------------------
        # Query target on shadow validation
        # ----------------------------------------------------

        if validation_cache.exists():

            print(
                "[SKIP] validation cache exists:",
                validation_cache,
            )

        else:

            run([
                "python",
                "src/gemma/cache_black_box_labels.py",

                "--student-adapter",
                target_adapter,

                "--corpus",
                data_root / "validation.jsonl",

                "--output",
                validation_cache,
            ])

        # ----------------------------------------------------
        # Train ordinary hard-label shadow
        # ----------------------------------------------------

        if adapter.exists() and summary.exists():

            print(
                "[SKIP] black-box shadow exists:",
                adapter,
            )

        else:

            run([
                "python",
                "src/gemma/train_black_box_shadow.py",

                "--cache",
                train_cache,

                "--validation-cache",
                validation_cache,

                "--output-dir",
                shadow_root,

                "--seed",
                str(seed + i),
            ])

        # ----------------------------------------------------
        # Score ordinary shadow
        # ----------------------------------------------------

        if scores.exists():

            print(
                "[SKIP] membership scores exist:",
                scores,
            )

        else:

            run([
                "python",
                "src/gemma/score_defense_student.py",

                "--adapter",
                adapter,

                "--members",
                data_root / "members.jsonl",

                "--nonmembers",
                data_root / "nonmembers.jsonl",

                "--output-dir",
                shadow_root,
            ])

        df = pd.read_csv(scores)

        df.insert(
            0,
            "shadow",
            shadow_name,
        )

        df.insert(
            1,
            "target_seed",
            seed,
        )

        df.insert(
            2,
            "defense",
            defense,
        )

        all_scores.append(df)

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = pd.concat(
        all_scores,
        ignore_index=True,
    )

    combined_path = (
        root
        / (
            f"black_box_mia_{defense}_"
            f"seed_{seed}_"
            f"all_shadow_membership_scores.csv"
        )
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("BLACK-BOX SHADOW PIPELINE COMPLETE")
    print("=" * 80)

    print("Total:", len(combined))

    if "membership" in combined.columns:

        print(
            "Members:",
            int(
                (
                    combined["membership"]
                    == 1
                ).sum()
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

    print("Saved:")
    print(combined_path)

    # --------------------------------------------------------
    # Final Black-Box MIA
    # --------------------------------------------------------

    target_scores = Path(
        f"results/gemma/defense/"
        f"seed_{seed}/{defense}/"
        f"membership_scores.csv"
    )

    if not target_scores.exists():
        raise FileNotFoundError(
            f"Target membership scores missing: "
            f"{target_scores}"
        )

    summary_path = (
        root /
        (
            f"black_box_mia_{defense}_"
            f"seed_{seed}_summary.csv"
        )
    )

    if summary_path.exists():

        print(
            "\n[SKIP] Final Black-Box MIA "
            "already complete:"
        )

        print(summary_path)

    else:

        run([
            "python",
            "src/gemma/black_box_mia_attack.py",

            "--shadow-scores",
            combined_path,

            "--target-scores",
            target_scores,

            "--output-dir",
            root,

            "--defense",
            defense,

            "--seed",
            str(seed),
        ])


if __name__ == "__main__":
    main()