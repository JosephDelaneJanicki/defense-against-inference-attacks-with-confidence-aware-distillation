import argparse
import subprocess
import sys
from pathlib import Path

SHADOWS = [f"shadow_{i:02d}" for i in range(6)]

def find_one(candidates, label):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {label}. Tried:\n" +
        "\n".join(f"  {p}" for p in candidates)
    )

def run(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-root", type=Path,
        default=Path("results/gemma/multiseed/seed_42/shadows"))
    ap.add_argument("--data-root", type=Path,
        default=Path("data/gemma/shadows"))
    ap.add_argument("--output-root", type=Path,
        default=Path("results/gemma/defense/seed_42/raw_soft/shadows"))
    ap.add_argument("--cache-root", type=Path,
        default=Path("data/gemma/defense/seed_42/raw_soft/shadows"))
    ap.add_argument("--start-shadow", type=int, default=0)
    args = ap.parse_args()

    cache_script = Path("src/gemma/cache_teacher_topk_parameterized.py")
    train_script = Path("src/gemma/train_raw_soft_student_parameterized.py")
    score_script = Path("src/gemma/score_defense_student.py")

    for p in (cache_script, train_script, score_script):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing required helper: {p}\n"
                "Install the parameterized cache/training helpers supplied "
                "with this runner before starting."
            )

    for i in range(args.start_shadow, 6):
        shadow = SHADOWS[i]
        print("\n" + "=" * 76)
        print(shadow.upper())
        print("=" * 76)

        d = args.data_root / shadow
        t = args.teacher_root / shadow
        out = args.output_root / shadow
        cache_dir = args.cache_root / shadow / "teacher_topk"

        members = find_one([
            d / "members.jsonl",
            d / "shadow_members.jsonl",
            t / "members.jsonl",
        ], f"{shadow} members")

        nonmembers = find_one([
            d / "nonmembers.jsonl",
            d / "shadow_nonmembers.jsonl",
            t / "nonmembers.jsonl",
        ], f"{shadow} nonmembers")

        validation = find_one([
            d / "validation.jsonl",
            t / "validation.jsonl",
            Path("data/gemma/validation.jsonl"),
        ], f"{shadow} validation")

        teacher = find_one([
            t / "adapter",
            t / "realistic_lora" / "adapter",
            Path("results/gemma/shadows") / shadow / "adapter",
        ], f"{shadow} teacher adapter")

        cache = cache_dir / "teacher_topk_cache.pt"
        student_adapter = out / "adapter"
        scores = out / "membership_scores.csv"

        if not cache.exists():
            run([
                sys.executable, cache_script,
                "--teacher", teacher,
                "--corpus", members,
                "--output-dir", cache_dir,
            ])
        else:
            print("[SKIP] cache exists:", cache)

        if not (student_adapter.exists() and
                (out / "training_summary.json").exists()):
            run([
                sys.executable, train_script,
                "--cache", cache,
                "--validation", validation,
                "--output-dir", out,
                "--seed", str(42 + i),
            ])
        else:
            print("[SKIP] student exists:", student_adapter)

        if not scores.exists():
            run([
                sys.executable, score_script,
                "--adapter", student_adapter,
                "--output-dir", out,
                "--members", members,
                "--nonmembers", nonmembers,
            ])
        else:
            print("[SKIP] scores exist:", scores)

    # Combine only after all six are complete.
    import pandas as pd
    frames = []
    for shadow in SHADOWS:
        p = args.output_root / shadow / "membership_scores.csv"
        if not p.exists():
            raise RuntimeError(f"Incomplete pipeline; missing {p}")
        df = pd.read_csv(p)
        if len(df) != 10000:
            raise RuntimeError(f"{p}: expected 10000 rows, got {len(df)}")
        df.insert(0, "shadow", shadow)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 60000:
        raise RuntimeError("Combined row count is not 60000.")
    if int(combined["membership"].sum()) != 30000:
        raise RuntimeError("Combined member count is not 30000.")

    dest = args.output_root / "all_shadow_membership_scores.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(dest, index=False)

    print("\n" + "=" * 76)
    print("RAW-SOFT ADAPTIVE SHADOW PIPELINE COMPLETE")
    print("=" * 76)
    print("Total:", len(combined))
    print("Members:", int(combined["membership"].sum()))
    print("Nonmembers:", int((combined["membership"] == 0).sum()))
    print("Saved:", dest)

if __name__ == "__main__":
    main()
