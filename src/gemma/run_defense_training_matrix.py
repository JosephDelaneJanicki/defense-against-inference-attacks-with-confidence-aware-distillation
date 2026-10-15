#!/usr/bin/env python3
"""Resumable sequential Gemma student training matrix. Run from repository root."""
import argparse, json, subprocess, sys, time
from pathlib import Path

SEEDS = (42, 123, 456, 789)
DEFENSES = ('raw_soft', 'temp_soft', 'clip_soft', 'confidence_filtered')
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/gemma'
DATA = ROOT / 'data/gemma'
SCRIPTS = ROOT / 'src/gemma'

def adapter_complete(path):
    return (path / 'adapter_config.json').is_file() and any(
        (path / n).is_file() for n in ('adapter_model.safetensors', 'adapter_model.bin'))

def run(cmd, log, dry):
    print('\n$', ' '.join(map(str,cmd)), flush=True)
    if dry: return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as f:
        f.write('\n\n=== COMMAND ' + time.strftime('%Y-%m-%d %H:%M:%S') + ' ===\n')
        f.write(' '.join(map(str,cmd)) + '\n'); f.flush()
        result = subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')

def cache_matches(summary, teacher, corpus):
    try:
        obj = json.loads(summary.read_text())
        return (Path(obj['teacher_adapter']).resolve() == teacher.resolve()
                and Path(obj['corpus']).resolve() == corpus.resolve()
                and int(obj['num_sequences']) == 5000
                and int(obj['top_k']) == 64)
    except (OSError, ValueError, KeyError, TypeError): return False

def locate_cache(seed, teacher, corpus):
    canonical = RESULTS / 'defense' / f'seed_{seed}' / 'teacher_cache'
    candidates = [canonical]
    # Existing seed-42 cache may live elsewhere; only accept provenance-matched caches.
    for summary in RESULTS.glob('**/teacher_topk_summary.json'):
        if 'shadow' in summary.parts or 'black_box_mia' in summary.parts: continue
        candidates.append(summary.parent)
    for directory in candidates:
        cache = directory / 'teacher_topk_cache.pt'
        summary = directory / 'teacher_topk_summary.json'
        if cache.is_file() and summary.is_file() and cache_matches(summary, teacher, corpus):
            return cache, None
    return canonical / 'teacher_topk_cache.pt', canonical

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--defenses', nargs='+', choices=DEFENSES, default=list(DEFENSES))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--skip-scoring', action='store_true')
    p.add_argument('--no-analysis', action='store_true')
    a = p.parse_args()
    for seed in a.seeds:
        teacher = RESULTS / 'multiseed' / f'seed_{seed}' / 'realistic_lora' / 'adapter'
        corpus = DATA / 'members.jsonl'
        validation = DATA / 'validation.jsonl'
        nonmembers = DATA / 'nonmembers.jsonl'
        for f in (corpus, validation, nonmembers):
            if not f.is_file(): raise FileNotFoundError(f)
        if not adapter_complete(teacher): raise FileNotFoundError(f'Incomplete teacher adapter: {teacher}')
        cache, build_dir = locate_cache(seed, teacher, corpus)
        # Do not generate cache when every requested student is complete.
        missing = [d for d in a.defenses if not adapter_complete(
            RESULTS / 'defense' / f'seed_{seed}' / d / 'adapter')]
        if missing and build_dir is not None:
            run([sys.executable, SCRIPTS / 'cache_teacher_topk_parameterized.py',
                 '--teacher', teacher, '--corpus', corpus, '--output-dir', build_dir],
                build_dir / 'cache.log', a.dry_run)
        for defense in a.defenses:
            out = RESULTS / 'defense' / f'seed_{seed}' / defense
            adapter = out / 'adapter'
            if adapter_complete(adapter):
                print(f'[SKIP TRAIN] seed={seed} defense={defense}', flush=True)
            else:
                if adapter.exists() and any(adapter.iterdir()):
                    raise RuntimeError(f'Incomplete adapter folder; inspect manually: {adapter}')
                # Preserve the established seed-42 raw_soft experiment; use its
                # original parameterized trainer for raw_soft on new seeds.
                if defense == 'raw_soft':
                    script = SCRIPTS / 'train_raw_soft_student_parameterized.py'
                    cmd = [sys.executable, script, '--cache', cache,
                           '--validation', validation, '--output-dir', out, '--seed', seed]
                else:
                    script = SCRIPTS / 'train_defense_student.py'
                    cmd = [sys.executable, script, '--cache', cache,
                           '--validation', validation, '--output-dir', out,
                           '--seed', seed, '--defense', defense]
                run(cmd, out / 'training.log', a.dry_run)
            if a.skip_scoring: continue
            score_dir = out
            scores = score_dir / 'membership_scores.csv'
            auc = score_dir / 'membership_auc.csv'
            if scores.is_file() and auc.is_file():
                print(f'[SKIP SCORE] seed={seed} defense={defense}', flush=True)
            else:
                run([sys.executable, SCRIPTS / 'score_defense_student.py',
                     '--adapter', adapter, '--output-dir', score_dir,
                     '--members', corpus, '--nonmembers', nonmembers],
                    out / 'scoring.log', a.dry_run)
    if not a.no_analysis:
        run([sys.executable, SCRIPTS / 'build_analysis_tree.py'],
            RESULTS / 'defense_training_matrix_analysis.log', a.dry_run)
    print('\n[MATRIX FINISHED]', flush=True)

if __name__ == '__main__': main()
