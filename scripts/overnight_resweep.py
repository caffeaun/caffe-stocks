#!/usr/bin/env python3
"""Overnight supervised consolidation — one-shot batch.

Why: the rolling-W7 leak invalidated every pre-freeze gate verdict (those iters
are now flagged ``contaminated=1`` and the record reads an honest 0 gate passes
/ best clean 3/7). But the 1,777 contaminated iters explored a large HP space
whose *real* (frozen-gate) performance is unknown. This job recovers honest
verdicts and hunts a first real pass before next week's unsupervised pivot.

It stands the ML loops down for the night by holding ``.ml-loop.lock`` with the
tag ``overnight_resweep``: ``train_mode`` exits(0) on contention, and
``claude_mode`` only SIGKILLs a holder tagged ``train_mode`` (scripts/claude_mode.py:61)
— any other tag makes it exit(0) cleanly. No sudo, no cron edits. GPU0 is free
while the loops are down.

Three tiers, then re-promote the panel from the honest (contaminated=0) pool:
  Tier A — re-evaluate each non-foundation family's best historical config(s)
  Tier B — deep fresh HP sweep on the cheap high-EV contenders (time-boxed)
  Tier C — budgeted GPU-foundation slice (toto2/ttm/moirai2/mantis/itransformer)

All new rows are written via feedback.record_iteration(mode='train') and default
contaminated=0, so they flow through every consumer we already filtered.

Launch detached before the night:
  cd ~/projects/caffe-stocks && nohup venv/bin/python scripts/overnight_resweep.py \
    > logs/overnight_resweep.log 2>&1 &
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

import numpy as np

from models.sequence_loader import (
    load_sequences, aggregate_sequence, aggregated_feature_names)
from models.search_spaces import sample as sample_config, list_trainers
from scripts import feedback as fb
from scripts.return_gate import (
    SPLIT_DEFS, evaluate_window, MAX_DD,
    MIN_TRADES_PER_WINDOW, MIN_WR, PASS_FRACTION_REQUIRED,
    avg_annualized_return, is_candidate,
)

LOCK_PATH = BASE / 'models' / '.ml-loop.lock'
ARTIFACT_DIR = BASE / 'models' / 'lgbm' / 'candidates' / 'overnight'

# Heavy GPU foundation trainers (~900–1800s/config) — kept out of Tier A and
# given their own budgeted Tier C so they can't eat the whole night.
FOUNDATION = [
    'torch_toto2', 'torch_ttm', 'torch_moirai2', 'torch_mantis',
    'torch_itransformer',
]

# Cheap, high-EV families to deep-sweep in Tier B (the likeliest source of a
# first honest gate pass). All 4–40s/config.
CONTENDERS = [
    'histgb_monotonic', 'anomaly_gated_histgb', 'kernel_logreg',
    'kernel_anomaly_blend', 'label_spreading', 'xgb_huber_regressor',
    'torch_fastkan',
]


# ---------------------------------------------------------------------------
# Lock — own tag so claude_mode stands down instead of killing us.
# ---------------------------------------------------------------------------
def acquire_lock(path: Path, tag: str = 'overnight_resweep'):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f'{os.getpid()} {tag} {datetime.now().isoformat()}\n'.encode())
    os.fsync(fd)
    return fd


def release_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Per-config evaluation — mirrors train_mode.run_one_config BUT threads X_seq
# so consumes_sequences trainers (Tier C foundation models, fastkan, etc.)
# evaluate correctly instead of raising "X_seq was not threaded".
# ---------------------------------------------------------------------------
def eval_one_config(trainer_name, hp, data, verbose=False):
    X, X_tab, y, dates, symbols, pnl, hold_days, agg_features = data
    started_at = datetime.now().isoformat()
    t0 = time.time()
    results = []
    for tr_s, tr_e, te_s, te_e in SPLIT_DEFS:
        r = evaluate_window(
            X_tab, y, dates, symbols, pnl, hold_days, agg_features,
            tr_s, tr_e, te_s, te_e,
            trainer_name, dict(hp), verbose=verbose, X_seq=X)
        results.append(r)
    elapsed = time.time() - t0

    n_passed = sum(1 for r in results if r.get('passed'))
    prior_best_ann = fb.best_candidate_ann_return()
    gate_passed = is_candidate(results, prior_best_ann)
    gate_result = {
        'gate_passed': gate_passed,
        'pass_fraction_required': PASS_FRACTION_REQUIRED,
        'windows_passed': n_passed,
        'windows_total': len(results),
        'avg_annualized_return': avg_annualized_return(results),
        'prior_best_ann_return': prior_best_ann,
        'max_dd': MAX_DD,
        'min_trades_per_window': MIN_TRADES_PER_WINDOW,
        'min_wr': MIN_WR,
        'results': results,
        'source': 'overnight_resweep',
    }
    return gate_result, started_at, datetime.now().isoformat(), elapsed


def record_and_log(trainer, hp, data, tier, verbose=False):
    """Evaluate one config on the honest gate, persist an honest iteration row.
    Returns the gate_result (or None on failure)."""
    try:
        gate_result, started, finished, elapsed = eval_one_config(
            trainer, hp, data, verbose=verbose)
    except Exception as e:
        print(f'  ✗ {trainer} failed: {e!r}', flush=True)
        return None

    wp = gate_result['windows_passed']
    wt = gate_result['windows_total']
    bests = [r['best'] for r in gate_result['results'] if 'best' in r]
    avg_ann = (sum(b['annualized_return'] for b in bests) / len(bests)) if bests else 0.0
    avg_wr = (sum(b['win_rate'] for b in bests) / len(bests)) if bests else 0.0
    n_tr = sum(b['n_trades'] for b in bests)
    flag = '✓✓ PASS' if gate_result['gate_passed'] else f'{wp}/{wt}'

    iter_dir = ARTIFACT_DIR / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    iter_dir.mkdir(parents=True, exist_ok=True)
    with open(iter_dir / 'gate_result.json', 'w') as f:
        json.dump(gate_result, f, indent=2, default=str)
    with open(iter_dir / 'hyperparams.json', 'w') as f:
        json.dump(hp, f, indent=2, default=str)

    iter_id = fb.record_iteration(
        mode='train', trainer=trainer, hyperparams=hp,
        gate_result=gate_result, model_dir=str(iter_dir),
        started_at=started, finished_at=finished,
        elapsed_seconds=int(elapsed),
        hypothesis=f'overnight_resweep:{tier}',
    )
    print(f'  {flag:8s} {trainer:26s} ann={avg_ann:+.1%} wr={avg_wr:.1%} '
          f'n={n_tr} ({elapsed:.0f}s) → #{iter_id}', flush=True)
    return gate_result


# ---------------------------------------------------------------------------
# Best historical configs per trainer (heuristic: rank by stored windows_passed
# then avg_ann — contaminated rows allowed since we re-score honestly anyway).
# ---------------------------------------------------------------------------
def best_historical_configs(trainer, k=2):
    with fb.get_conn() as conn:
        rows = conn.execute("""
            SELECT hyperparams, windows_passed, avg_annualized_return
            FROM iterations
            WHERE trainer = ? AND total_trades > 0 AND hyperparams IS NOT NULL
            ORDER BY windows_passed DESC, avg_annualized_return DESC
            LIMIT ?
        """, (trainer, k)).fetchall()
    out, seen = [], set()
    for r in rows:
        hp_str = r['hyperparams']
        if hp_str in seen:
            continue
        seen.add(hp_str)
        try:
            out.append(json.loads(hp_str))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------
def tier_a(data, deadline, limit=None):
    """Re-evaluate each non-foundation family's best historical config(s)."""
    registry = [t for t in list_trainers() if t not in FOUNDATION]
    if limit:
        registry = registry[:limit]
    print(f'\n=== Tier A — honest baseline per family ({len(registry)} trainers) ===',
          flush=True)
    n_pass = 0
    for trainer in registry:
        if time.time() >= deadline:
            print('  (Tier A deadline reached — stopping)', flush=True)
            break
        configs = best_historical_configs(trainer, k=2)
        if not configs:
            continue
        for hp in configs:
            if time.time() >= deadline:
                break
            gr = record_and_log(trainer, hp, data, tier='tierA')
            if gr and gr['gate_passed']:
                n_pass += 1
    print(f'  Tier A done — {n_pass} honest pass(es)', flush=True)
    return n_pass


def tier_b(data, deadline, seed=None):
    """Deep fresh HP sweep on the cheap high-EV contenders, time-boxed."""
    print(f'\n=== Tier B — deep sweep on contenders (until {time.strftime("%H:%M", time.localtime(deadline))}) ===',
          flush=True)
    rng = np.random.RandomState(seed)
    n_cfg = n_pass = 0
    i = 0
    while time.time() < deadline:
        trainer = CONTENDERS[i % len(CONTENDERS)]
        i += 1
        try:
            hp = sample_config(trainer, rng)
        except Exception as e:
            print(f'  ✗ sample {trainer}: {e!r}', flush=True)
            continue
        gr = record_and_log(trainer, hp, data, tier='tierB')
        n_cfg += 1
        if gr and gr['gate_passed']:
            n_pass += 1
            print(f'  ★★ HONEST GATE PASS — {trainer} (config #{n_cfg})', flush=True)
    print(f'  Tier B done — {n_cfg} configs, {n_pass} pass(es)', flush=True)
    return n_pass


def tier_c(data, deadline):
    """Budgeted GPU-foundation slice — best 1–2 historical configs each."""
    print(f'\n=== Tier C — GPU foundation slice (cap {time.strftime("%H:%M", time.localtime(deadline))}) ===',
          flush=True)
    n_pass = 0
    for trainer in FOUNDATION:
        if time.time() >= deadline:
            print('  (Tier C cap reached — stopping)', flush=True)
            break
        for hp in best_historical_configs(trainer, k=2):
            if time.time() >= deadline:
                break
            gr = record_and_log(trainer, hp, data, tier='tierC')
            if gr and gr['gate_passed']:
                n_pass += 1
    print(f'  Tier C done — {n_pass} pass(es)', flush=True)
    return n_pass


def re_promote():
    print('\n=== Re-promoting panel from the honest pool ===', flush=True)
    r = subprocess.run(
        [sys.executable, str(BASE / 'scripts' / 'promote_panel.py'), '--k', '10'],
        cwd=str(BASE))
    print(f'  promote_panel exit={r.returncode}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--deadline-hours', type=float, default=9.0,
                   help='Hard overall wall-clock cap')
    p.add_argument('--tier-a-hours', type=float, default=2.5)
    p.add_argument('--tier-b-hours', type=float, default=4.5)
    p.add_argument('--tier-c-hours', type=float, default=3.0)
    p.add_argument('--tiers', default='A,B,C',
                   help='Comma list of tiers to run (e.g. "A" for a quick test)')
    p.add_argument('--limit-a', type=int, default=None,
                   help='Limit Tier A to first N trainers (testing)')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--no-promote', action='store_true')
    args = p.parse_args()

    fd = acquire_lock(LOCK_PATH)
    if fd is None:
        print('Another ML loop process holds the lock — exiting.', flush=True)
        sys.exit(0)

    t_start = time.time()
    overall_deadline = t_start + args.deadline_hours * 3600
    tiers = {t.strip().upper() for t in args.tiers.split(',') if t.strip()}
    print(f'=== overnight_resweep start {datetime.now().isoformat()} '
          f'(deadline +{args.deadline_hours}h, tiers={sorted(tiers)}) ===', flush=True)
    try:
        fb.init_db()
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        X, y, dates, symbols, pnl, hold_days, features = load_sequences(
            seq_len=20, verbose=True)
        X_tab = aggregate_sequence(X)
        agg_features = aggregated_feature_names(features)
        data = (X, X_tab, y, dates, symbols, pnl, hold_days, agg_features)
        print(f'Data: tab={X_tab.shape} seq={X.shape} pos_rate={y.mean():.1%} '
              f'({time.time()-t0:.1f}s)', flush=True)

        total_pass = 0
        if 'A' in tiers:
            dl = min(overall_deadline, time.time() + args.tier_a_hours * 3600)
            total_pass += tier_a(data, dl, limit=args.limit_a)
        if 'B' in tiers:
            dl = min(overall_deadline, time.time() + args.tier_b_hours * 3600)
            total_pass += tier_b(data, dl, seed=args.seed)
        if 'C' in tiers:
            dl = min(overall_deadline, time.time() + args.tier_c_hours * 3600)
            total_pass += tier_c(data, dl)

        if not args.no_promote:
            re_promote()

        with fb.get_conn() as conn:
            honest_passes = conn.execute(
                'SELECT COUNT(*) FROM iterations '
                'WHERE gate_passed=1 AND contaminated=0').fetchone()[0]
            best_wp = conn.execute(
                'SELECT MAX(windows_passed) FROM iterations '
                'WHERE contaminated=0').fetchone()[0]
        print(f'\n=== overnight_resweep done ({(time.time()-t_start)/3600:.1f}h) ===',
              flush=True)
        print(f'  honest gate passes (all-time clean): {honest_passes}', flush=True)
        print(f'  best clean windows_passed: {best_wp}', flush=True)
        print(f'  passes found this run: {total_pass}', flush=True)
    finally:
        release_lock(fd)


if __name__ == '__main__':
    main()
