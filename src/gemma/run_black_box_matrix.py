#!/usr/bin/env python3
"""Resumable 4x4 Gemma black-box MIA batch; run from any working directory.

Place at src/gemma/run_black_box_matrix.py in MIA_Project.
Uses the existing, previously validated run_black_box_mia_pipeline.py interface:
    --defense NAME --seed INT
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SEEDS = (42, 123, 456, 789)
DEFENSES = ('raw_soft', 'temp_soft', 'clip_soft', 'confidence_filtered')
ROOT = Path(__file__).resolve().parents[2]
GEMMA = ROOT / 'results' / 'gemma'
PIPELINE = ROOT / 'src' / 'gemma' / 'run_black_box_mia_pipeline.py'
ANALYSIS = ROOT / 'src' / 'gemma' / 'build_analysis_tree.py'


def complete_adapter(path):
    return (path / 'adapter_config.json').is_file() and any(
        (path / name).is_file() for name in ('adapter_model.safetensors', 'adapter_model.bin'))


def valid_summary(path):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or 'roc_auc' not in reader.fieldnames:
                return False
            rows = list(reader)
        # Both learned attackers must be present; avoid treating partial results as complete.
        names = {str(r.get('attacker', '')).lower() for r in rows}
        return bool(rows) and any('logistic' in n for n in names) and any(
            'forest' in n for n in names)
    except (OSError, UnicodeError, csv.Error):
        return False


def write_status(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def run_logged(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as fh:
        fh.write('\n' + '=' * 70 + '\n')
        fh.write('Started: ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        fh.write('Command: ' + ' '.join(map(str, cmd)) + '\n')
        fh.flush()
        return subprocess.run([str(x) for x in cmd], cwd=ROOT, stdout=fh,
                              stderr=subprocess.STDOUT, check=False).returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    p.add_argument('--defenses', nargs='+', choices=DEFENSES, default=list(DEFENSES))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true', help='Rerun completed experiments')
    p.add_argument('--fail-fast', action='store_true')
    p.add_argument('--skip-analysis', action='store_true')
    args = p.parse_args()

    if not PIPELINE.is_file():
        p.error(f'Black-box pipeline not found: {PIPELINE}')
    if not args.skip_analysis and not ANALYSIS.is_file():
        p.error(f'Analysis builder not found: {ANALYSIS}')

    failures = []
    skipped = 0
    completed = 0
    planned = 0
    for seed in args.seeds:
        for defense in args.defenses:
            target = GEMMA / 'defense' / f'seed_{seed}' / defense
            adapter = target / 'adapter'
            out = target / 'black_box_mia'
            summary = out / f'black_box_mia_{defense}_seed_{seed}_summary.csv'
            status_path = out / 'batch_status.json'
            if not complete_adapter(adapter):
                print(f'[ERROR] Incomplete target adapter: {adapter}', flush=True)
                failures.append(f'{seed}/{defense}: missing adapter')
                if args.fail_fast: break
                continue
            if valid_summary(summary) and not args.force:
                skipped += 1
                print(f'[SKIP COMPLETE] {seed}/{defense}: {summary}', flush=True)
                continue
            planned += 1
            cmd = [sys.executable, PIPELINE, '--defense', defense, '--seed', str(seed)]
            print(f'[RUN] {seed}/{defense}: {" ".join(map(str, cmd))}', flush=True)
            if args.dry_run:
                continue
            start = time.monotonic()
            write_status(status_path, {'seed': seed, 'defense': defense,
                                       'status': 'running', 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')})
            rc = run_logged(cmd, out / 'pipeline.log')
            elapsed = round(time.monotonic() - start, 1)
            ok = rc == 0 and valid_summary(summary)
            write_status(status_path, {'seed': seed, 'defense': defense,
                                       'status': 'complete' if ok else 'failed',
                                       'returncode': rc, 'elapsed_seconds': elapsed,
                                       'summary': str(summary.relative_to(ROOT)),
                                       'summary_valid': valid_summary(summary)})
            if ok:
                completed += 1
                print(f'[COMPLETE] {seed}/{defense} ({elapsed:.0f}s)', flush=True)
            else:
                failures.append(f'{seed}/{defense}: exit={rc}, valid_summary={valid_summary(summary)}')
                print(f'[FAILED] {failures[-1]}; see {out / "pipeline.log"}', flush=True)
                if args.fail_fast: break
        if args.fail_fast and failures:
            break

    if not args.dry_run and not args.skip_analysis:
        print('[ANALYSIS] Rebuilding analysis tree...', flush=True)
        rc = run_logged([sys.executable, ANALYSIS], GEMMA / 'black_box_matrix_analysis.log')
        if rc:
            failures.append(f'analysis builder: exit={rc}')
            print('[WARNING] Analysis rebuild failed; inspect black_box_matrix_analysis.log', flush=True)

    print(f'\n[BATCH FINISHED] completed={completed} skipped={skipped} planned={planned} failed={len(failures)}', flush=True)
    if failures:
        for failure in failures:
            print(' - ' + failure, flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
