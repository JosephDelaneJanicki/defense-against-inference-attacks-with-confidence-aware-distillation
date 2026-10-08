import argparse
import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable

RESULTS_ROOT = Path("results/gemma")
DEFENSE_ROOT = RESULTS_ROOT / "defense"

BLACK_BOX_PIPELINE = Path(
    "src/gemma/run_black_box_mia_pipeline.py"
)

ANALYSIS_BUILDER = Path(
    "src/gemma/build_analysis_tree.py"
)


def run(command):

    command = [str(x) for x in command]

    print("\n" + "=" * 80)
    print("$", " ".join(command))
    print("=" * 80, flush=True)

    subprocess.run(
        command,
        check=True,
    )


def discover_targets():

    jobs = []

    if not DEFENSE_ROOT.exists():
        return jobs

    for seed_dir in sorted(
        DEFENSE_ROOT.glob("seed_*")
    ):

        try:
            seed = int(
                seed_dir.name.split("_", 1)[1]
            )
        except ValueError:
            continue

        for defense_dir in sorted(
            seed_dir.iterdir()
        ):

            if not defense_dir.is_dir():
                continue

            adapter = defense_dir / "adapter"

            if not adapter.exists():
                continue

            jobs.append({
                "seed": seed,
                "defense": defense_dir.name,
                "root": defense_dir,
                "adapter": adapter,
            })

    return jobs


def black_box_complete(job):

    root = (
        job["root"] /
        "black_box_mia"
    )

    if not root.exists():
        return False

    summaries = list(
        root.rglob(
            "*black_box*summary*.csv"
        )
    )

    return len(summaries) > 0


def run_black_box(job):

    if black_box_complete(job):
        print(
            "[SKIP] Black-Box complete:",
            f"seed={job['seed']}",
            f"defense={job['defense']}",
        )
        return

    run([
        PYTHON,
        BLACK_BOX_PIPELINE,
        "--defense",
        job["defense"],
        "--seed",
        job["seed"],
    ])


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip-black-box",
        action="store_true",
    )

    parser.add_argument(
        "--analysis-only",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GEMMA DEFENSE MASTER ORCHESTRATOR")
    print("=" * 80)

    jobs = discover_targets()

    print(f"\nReady defended targets: {len(jobs)}")

    for job in jobs:
        print(
            f"  seed={job['seed']:<4} "
            f"defense={job['defense']:<24} "
            f"{job['adapter']}"
        )

    if args.analysis_only:

        run([
            PYTHON,
            ANALYSIS_BUILDER,
        ])

        return

    # --------------------------------------------------------
    # GPU work
    #
    # Sequential intentionally.
    # One RTX 4090 -> one Gemma training process at a time.
    # --------------------------------------------------------

    if not args.skip_black_box:

        for index, job in enumerate(
            jobs,
            start=1,
        ):

            print(
                "\n"
                f"TARGET {index}/{len(jobs)}: "
                f"seed={job['seed']} "
                f"defense={job['defense']}"
            )

            run_black_box(job)

    # --------------------------------------------------------
    # Rebuild analysis from whatever actually exists.
    # --------------------------------------------------------

    run([
        PYTHON,
        ANALYSIS_BUILDER,
    ])

    print("\n" + "=" * 80)
    print("MASTER ORCHESTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()