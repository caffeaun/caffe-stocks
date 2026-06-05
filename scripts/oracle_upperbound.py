#!/usr/bin/env python3
"""Oracle upper-bound — the selection-headroom diagnostic (Part B bridge).

Question it answers: given a window's realized trade P&L, what is the *best
possible* result any per-day selection policy could achieve under the real
constraints (2 slots, hold periods, ≥20 trades), scored through the actual
gate criteria? If even this hindsight oracle can't clear 6/7, then no
selection policy — RL included — can on the current labels/data, and the
bottleneck is upstream (features/labels), not selection. If it clears 6/7
comfortably, the selection layer has headroom worth optimizing.

Model-free by default (the oracle uses realized P&L directly). Pass
``--iter <id>`` to also train that iteration's config per window and print the
model's actual windows-passed for comparison.

Read-only: writes no iterations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

import numpy as np

from models.sequence_loader import (
    load_sequences, aggregate_sequence, aggregated_feature_names)
from scripts import feedback as fb
from scripts.return_gate import (
    SPLIT_DEFS, _within, simulate_window, evaluate_window,
    SCORE_THRESHOLDS, MIN_TRADES_PER_WINDOW, MAX_DD, _min_wr_for_window,
)


def _rank_norm(a):
    """Map values to [0,1] by rank — highest value → 1.0. Ties broken
    arbitrarily but stably."""
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort().argsort().astype(np.float64)
    return order / max(len(order) - 1, 1)


def oracle_window(test_dates, test_symbols, test_pnl, test_hold,
                  test_start, test_end):
    """Best-achievable selection for one window: score = rank-normalized
    realized P&L (so the gate's threshold sweep keeps only the highest-P&L
    trades), swept + judged exactly as evaluate_window does."""
    scores = _rank_norm(test_pnl)
    sims = [simulate_window(scores, test_dates, test_symbols, test_pnl,
                            test_hold, thr) for thr in SCORE_THRESHOLDS]
    best = max(sims, key=lambda s: (s['n_trades'] >= MIN_TRADES_PER_WINDOW,
                                    s['annualized_return']))
    min_wr, src = _min_wr_for_window(test_start, test_end)
    fails = []
    if best['max_drawdown'] > MAX_DD:
        fails.append(f"dd {best['max_drawdown']:.1%}>{MAX_DD:.0%}")
    if best['n_trades'] < MIN_TRADES_PER_WINDOW:
        fails.append(f"n {best['n_trades']}<{MIN_TRADES_PER_WINDOW}")
    if best['win_rate'] < min_wr:
        fails.append(f"wr {best['win_rate']:.1%}<{min_wr:.0%}")
    return best, (len(fails) == 0), fails, min_wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iter', type=int, default=None,
                    help='Also train this iteration id config per window and '
                         'compare its windows-passed to the oracle.')
    args = ap.parse_args()

    X, y, dates, symbols, pnl, hold, feats = load_sequences(seq_len=20, verbose=False)
    X_tab = aggregate_sequence(X)
    agg = aggregated_feature_names(feats)
    print(f'data {X_tab.shape}\n')

    model_cfg = None
    if args.iter is not None:
        with fb.get_conn() as conn:
            row = conn.execute('SELECT trainer, hyperparams FROM iterations '
                               'WHERE id=?', (args.iter,)).fetchone()
        if row:
            model_cfg = (row['trainer'], json.loads(row['hyperparams'] or '{}'))
            print(f'comparison model: iter #{args.iter} {model_cfg[0]}\n')

    oracle_pass = model_pass = 0
    print('  win  test_window              oracle: wr     ann      dd    n   pass')
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(SPLIT_DEFS, 1):
        tm = _within(dates, te_s, te_e)
        best, passed, fails, min_wr = oracle_window(
            dates[tm], symbols[tm], pnl[tm], hold[tm], te_s, te_e)
        oracle_pass += int(passed)
        mark = '✓' if passed else '✗ ' + ','.join(fails)
        print(f'  W{i}   {te_s}..{te_e}   '
              f"{best['win_rate']:.1%}  {best['annualized_return']:+.1%}  "
              f"{best['max_drawdown']:.1%}  {best['n_trades']:>3}   {mark}")

        if model_cfg:
            r = evaluate_window(X_tab, y, dates, symbols, pnl, hold, agg,
                                tr_s, tr_e, te_s, te_e, model_cfg[0],
                                dict(model_cfg[1]), verbose=False, X_seq=X)
            model_pass += int(r.get('passed', False))

    print(f'\nORACLE (perfect-foresight ceiling) windows passed: {oracle_pass}/7')
    if oracle_pass >= 6:
        print('  → Each window contains ≥20 winning trades to find, so trade '
              'availability is NOT the binding constraint. The bottleneck is '
              'identifying them from features — exactly what the RL reward sweep '
              'tests. The realizable signal is the model-vs-ceiling gap below.')
    else:
        print('  → Even perfect foresight cannot clear 6/7 under the constraints '
              '(too few winners / DD blowups). No selection policy can — the '
              'bottleneck is upstream (features/labels), and RL over these labels '
              'will not help. Fix labels/data first.')
    if model_cfg:
        print(f'MODEL  (iter #{args.iter}) windows passed: {model_pass}/7  '
              f'— realizable gap to ceiling = {oracle_pass - model_pass} windows.')


if __name__ == '__main__':
    main()
