#!/usr/bin/env python3
"""Train mode — sample N HP configs, gate each, log to feedback DB.

Cannot change code. Cannot add trainers. Picks from models/search_spaces.py
and runs return_gate.py logic per config. Acquires a shared lock so it
never runs concurrently with another train_mode or claude_mode run.
"""
import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

from models.sequence_loader import load_sequences, aggregate_sequence, aggregated_feature_names
from models.search_spaces import sample as sample_config, list_trainers
from scripts import active_case, feedback as fb
from scripts.return_gate import (
    SPLIT_DEFS, evaluate_window, MAX_DD,
    MIN_TRADES_PER_WINDOW, MIN_WR, PASS_FRACTION_REQUIRED,
    avg_annualized_return, is_candidate,
)

# Sentinel used to detect "user did not pass --trainer". argparse can't
# tell us whether a default fired vs the user typed the same string, so we
# default to this sentinel and resolve it after parsing — if it's still the
# sentinel we're free to consult models/active_case.json.
_TRAINER_SENTINEL = '__from_active_case__'
_FALLBACK_TRAINER = 'histgb_monotonic'  # was 'xgboost' (Phase 3C archived)


LOCK_PATH = BASE / 'models' / '.ml-loop.lock'
CANDIDATES_DIR = BASE / 'models' / 'lgbm' / 'candidates' / 'train-mode'


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    os.write(fd, f'{os.getpid()} train_mode {datetime.now().isoformat()}\n'.encode())
    os.fsync(fd)
    return fd


def release_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def run_one_config(trainer_name, hp, X_tab, y, dates, symbols, pnl, hold_days,
                    agg_features, verbose=False):
    """Run one config across all calendar splits and return aggregate result."""
    started_at = datetime.now().isoformat()
    t0 = time.time()
    results = []
    for tr_s, tr_e, te_s, te_e in SPLIT_DEFS:
        r = evaluate_window(
            X_tab, y, dates, symbols, pnl, hold_days, agg_features,
            tr_s, tr_e, te_s, te_e,
            trainer_name, dict(hp),  # copy — evaluate_window reads only
            verbose=verbose)
        results.append(r)
    elapsed = time.time() - t0

    n_passed = sum(1 for r in results if r.get('passed'))
    prior_best_ann = fb.best_candidate_ann_return()
    gate_passed = is_candidate(results, prior_best_ann)

    return {
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
    }, started_at, datetime.now().isoformat(), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=int, default=10,
                        help='Number of HP configs to sample and evaluate')
    # Sentinel default lets us tell "user passed --trainer" apart from "default
    # fired" without grovelling through sys.argv. choices is widened to include
    # the sentinel; we resolve it before any choice-validation that matters.
    parser.add_argument('--trainer', default=_TRAINER_SENTINEL,
                        choices=list_trainers() + [_TRAINER_SENTINEL],
                        help='Which trainer family to sweep within '
                             '(default: read from models/active_case.json, '
                             f'fall back to {_FALLBACK_TRAINER!r})')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for HP sampling (default: time-based)')
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Resolve trainer: explicit CLI > active_case.json > xgboost fallback.
    if args.trainer == _TRAINER_SENTINEL:
        case = active_case.read()
        if case and case.get('trainer') in list_trainers():
            print(f'train_mode: using trainer {case["trainer"]!r} '
                  f'from models/active_case.json (claude_iter_id='
                  f'{case.get("claude_iter_id")!r})')
            args.trainer = case['trainer']
        else:
            if case:
                print(f'train_mode: active_case trainer '
                      f'{case.get("trainer")!r} not in registry — '
                      f'falling back to {_FALLBACK_TRAINER!r}')
            args.trainer = _FALLBACK_TRAINER

    # Acquire shared lock
    fd = acquire_lock(LOCK_PATH)
    if fd is None:
        print('Another ML loop process holds the lock — exiting.')
        sys.exit(0)

    try:
        print(f'=== train_mode: {args.configs} configs of {args.trainer} ===')
        fb.init_db()

        # Load data once for all configs
        t0 = time.time()
        X, y, dates, symbols, pnl, hold_days, features = load_sequences(
            seq_len=args.seq_len, verbose=True)
        X_tab = aggregate_sequence(X)
        agg_features = aggregated_feature_names(features)
        print(f'Data: {X_tab.shape}  pos_rate={y.mean():.1%}  ({time.time()-t0:.1f}s)\n')

        rng = np.random.RandomState(args.seed)
        configs = [sample_config(args.trainer, rng) for _ in range(args.configs)]

        # Run each
        for i, hp in enumerate(configs, 1):
            short = ', '.join(f'{k}={v}' for k, v in hp.items() if k in
                             ('num_leaves', 'max_depth', 'learning_rate',
                              'n_estimators', 'pos_class_weight', 'min_child_weight'))
            print(f'[{i}/{len(configs)}] {short[:80]}')

            try:
                gate_result, started, finished, elapsed = run_one_config(
                    args.trainer, hp, X_tab, y, dates, symbols, pnl, hold_days,
                    agg_features, verbose=args.verbose)
            except Exception as e:
                print(f'  ✗ failed: {e!r}')
                continue

            wp = gate_result['windows_passed']
            wt = gate_result['windows_total']
            pass_frac = wp / max(1, wt)
            agg = {
                'avg_ann': sum(r['best']['annualized_return'] for r in gate_result['results']
                                 if 'best' in r) / max(1, len([r for r in gate_result['results'] if 'best' in r])),
                'avg_wr':  sum(r['best']['win_rate'] for r in gate_result['results']
                                 if 'best' in r) / max(1, len([r for r in gate_result['results'] if 'best' in r])),
                'n_trades': sum(r['best']['n_trades'] for r in gate_result['results'] if 'best' in r),
            }
            flag = '✓' if gate_result['gate_passed'] else '✗'
            print(f'  {flag} windows {wp}/{wt}  ann_avg={agg["avg_ann"]:+.1%}  '
                  f'wr_avg={agg["avg_wr"]:.1%}  trades={agg["n_trades"]}  '
                  f'({elapsed:.1f}s)')

            # Save artifacts in a per-iteration dir
            iter_dir = CANDIDATES_DIR / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            iter_dir.mkdir(parents=True, exist_ok=True)
            with open(iter_dir / 'gate_result.json', 'w') as f:
                json.dump(gate_result, f, indent=2, default=str)
            with open(iter_dir / 'hyperparams.json', 'w') as f:
                json.dump(hp, f, indent=2)

            # Log to db
            iter_id = fb.record_iteration(
                mode='train',
                trainer=args.trainer,
                hyperparams=hp,
                gate_result=gate_result,
                model_dir=str(iter_dir),
                started_at=started,
                finished_at=finished,
                elapsed_seconds=int(elapsed),
            )
            print(f'  → iteration #{iter_id}')

        # Summary
        print('\n=== Run summary ===')
        s = fb.stats()
        print(f'Total iterations in db: {s["total"]} (passed: {s["passed"]})')
        print(f'By trainer: {s["by_trainer"]}')
        print(f'Consecutive failures: {s["consecutive_failures"]}')

    finally:
        release_lock(fd)


if __name__ == '__main__':
    main()
