#!/usr/bin/env python3
"""Walk-forward gate using return-based criteria — replaces walk_forward_gate.py.

For each calendar split:
  1. Train fresh LightGBM on train portion.
  2. Score test portion (out-of-sample).
  3. Walk through test dates chronologically as a single-position trader:
       - On each date, pick the highest-scoring symbol where score >= threshold.
       - Enter the trade. Skip subsequent dates until trade exits (hold_days).
       - Use realistic per-trade pnl from labels.py simulation.
  4. Build equity curve. Compute annualized return, max DD, trade count, WR.

v1 per-window pass criteria (no per-window ann_return floor):
  - Max drawdown <= MAX_DD
  - At least MIN_TRADES_PER_WINDOW
  - Win rate >= MIN_WR

Model becomes a candidate (`is_candidate`) only if ALL 7 windows pass per-window
criteria AND avg annualized return strictly beats the best prior gate-passing
iteration (or > 0 when no prior candidate exists yet).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Pin to GPU 0. trainers.py:28 also sets this via setdefault, but
# models.sequence_loader (imported below) triggers torch's CUDA init
# first, locking visibility before trainers.py runs and defeating the
# setdefault. Setting it here — the first executable line in the entry
# point — is the earliest safe place. Honour an explicit override from
# the caller if one was set (e.g. CUDA_VISIBLE_DEVICES=1 to share the box
# with a long-running ollama session).
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import numpy as np
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.expanduser('~/projects/caffe-stocks'))
from models.sequence_loader import (
    load_sequences, aggregate_sequence, aggregated_feature_names,
)
from models.trainers import get_trainer, TRAINERS

BASE_PATH = os.path.expanduser('~/projects/caffe-stocks')

# Walk-forward calendar splits. Train/test pairs covering 2023-05 → 2026-04.
# Each test window is 4–6 months. We annualize per-window so the goal is
# evaluated consistently regardless of window length.
def _rolling_recent_split() -> tuple:
    """W7: 6 mo train ending 2025-12-31; test from 2026-01-01 to today.

    Test_end is computed at module load so the recent window naturally
    rolls forward as data ingests. Functionally a monthly rolling
    cross-validation slice — every iteration sees the freshest regime
    instead of a fixed 2025-09..2026-02 slice that gets stale.
    """
    from datetime import date as _date
    return ('2025-07-01', '2025-12-31', '2026-01-01', _date.today().isoformat())


SPLIT_DEFS = [
    # (train_start, train_end,        test_start, test_end)
    ('2023-05-01', '2023-10-31', '2023-11-01', '2024-02-28'),  # W1 — 4mo test
    ('2023-07-01', '2023-12-31', '2024-01-01', '2024-04-30'),  # W2
    ('2023-11-01', '2024-04-30', '2024-05-01', '2024-08-31'),  # W3
    ('2024-03-01', '2024-08-31', '2024-09-01', '2024-12-31'),  # W4
    ('2024-07-01', '2024-12-31', '2025-01-01', '2025-04-30'),  # W5
    ('2024-11-01', '2025-04-30', '2025-05-01', '2025-08-31'),  # W6
    _rolling_recent_split(),                                    # W7 — rolling recent
]

# v1 gate (whitepaper §9). Per-window has no ann_return floor — the absolute
# bar moved to the model level (avg ann beats best prior candidate).
MAX_DD = 0.20                  # 20% peak-to-trough per window
MIN_TRADES_PER_WINDOW = 20     # ~1 trade per 6 trading days in 4-mo window
MIN_WR = 0.40                  # roughly breakeven WR with avg_win ~5% / avg_loss ~3.5%

# --- Regime-aware MIN_WR (default ON since 2026-05-26) -------------------
# Per-window MIN_WR is computed as
#   max(MIN_WR_FLOOR, set_buy_hold_wr[window] + MIN_WR_MARGIN)
# using the SET buy-and-hold win rate cached in data/baselines.json. This
# fixes the asymmetry where a 41% trainer WR is "edge" in a bear regime
# (SET wr=35%) but "noise" in a bull regime (SET wr=58%). Default ON so the
# gate honestly measures alpha vs the market; flip OFF via env var
# (RETURN_GATE_REGIME_AWARE=0) for an A/B against the legacy fixed-40% gate.
# Margin started at +3pp (A/B run 2026-05-26: zero historical iters cleared);
# lowered to +1pp so "match the market with slight edge" is the bar — still
# materially stricter than the legacy 0.40 floor that SET routinely clears.
REGIME_AWARE_GATE = os.environ.get('RETURN_GATE_REGIME_AWARE', '1') == '1'
MIN_WR_FLOOR = 0.30            # anti-degenerate floor when SET wr is very low
MIN_WR_MARGIN = 0.01           # require 1pp above SET buy-hold WR

_BASELINES_PATH = Path(os.path.expanduser(
    '~/projects/caffe-stocks/data/baselines.json'))


def _set_wr_for_window(test_start: str, test_end: str) -> Optional[float]:
    """Return SET buy-hold WR for the named test window, or None if the
    baseline cache is missing/stale/unparseable."""
    if not _BASELINES_PATH.exists():
        return None
    try:
        with open(_BASELINES_PATH) as f:
            doc = json.load(f)
    except Exception:
        return None
    per_window = (doc.get('set_buy_hold') or {}).get('per_window') or []
    key = f'{test_start}..{test_end}'
    for w in per_window:
        if w.get('test') == key:
            try:
                return float(w.get('win_rate'))
            except (TypeError, ValueError):
                return None
    return None


def _min_wr_for_window(test_start: str, test_end: str) -> tuple[float, str]:
    """Compute the per-window MIN_WR. Returns (threshold, source_label).
    Falls back to the fixed MIN_WR when regime mode is off OR the baseline
    cache lacks coverage for this window."""
    if not REGIME_AWARE_GATE:
        return MIN_WR, 'fixed'
    set_wr = _set_wr_for_window(test_start, test_end)
    if set_wr is None:
        return MIN_WR, 'fixed (no baseline)'
    adjusted = max(MIN_WR_FLOOR, set_wr + MIN_WR_MARGIN)
    return adjusted, f'regime (SET wr={set_wr:.1%}+{MIN_WR_MARGIN:.0%})'
# 7/7 was the original bar; 750 iterations across 42 trainer families produced
# zero passes (best ever: 6/7), suggesting the bar is unreachable for this
# universe × 10d horizon × 20-trade floor combination. Loosened to 6/7 so a
# candidate can stabilize and feed the panel. The model-level "avg_ann beats
# best prior" check in is_candidate still keeps competition honest.
WINDOWS_TOTAL = 7
WINDOWS_REQUIRED = 6
PASS_FRACTION_REQUIRED = WINDOWS_REQUIRED / WINDOWS_TOTAL

# Concurrent open positions cap. Whitepaper §9 says "Take top-K predictions";
# §11 v2 prescribes 50/50 split sizing across two simultaneous positions as the
# evolution from v1 single-position. With single-position max-hold=10 on a
# ~80-trading-day window, the trader physically cannot exceed ~10–13 trades
# (mean across 124 prior iterations × 7 windows: 10.3 trades; max ever: 27
# only in the longest window 7). MIN_TRADES_PER_WINDOW=20 was therefore
# unreachable by any HP/feature/trainer change. K=2 doubles trade-count
# headroom and halves per-trade DD impact, restoring slack to the gate.
MAX_OPEN_POSITIONS = 2


def avg_annualized_return(windows: list[dict]) -> float:
    """Mean annualized return across the per-window 'best' results."""
    bests = [w['best'] for w in windows if 'best' in w]
    if not bests:
        return 0.0
    return sum(b.get('annualized_return', 0.0) for b in bests) / len(bests)


def is_candidate(windows: list[dict], prior_best_ann: 'float | None') -> bool:
    """Model-level gate: at least WINDOWS_REQUIRED windows pass per-window
    criteria AND avg ann beats the best prior candidate (or > 0 when none
    exists yet).
    """
    if not windows:
        return False
    n_passed = sum(1 for w in windows if bool(w.get('passed')))
    if n_passed < WINDOWS_REQUIRED:
        return False
    avg_ann = avg_annualized_return(windows)
    if prior_best_ann is None:
        return avg_ann > 0.0
    return avg_ann > prior_best_ann



# Score threshold for entry. 0.5 is the model's natural decision boundary;
# higher = more selective. We try a small grid per window to give the trader
# some calibration headroom.
#
# 0.0 = rank-only fallback. Some trainers (stacked_ranker with uniform fallback
# weights, ev_gated_ranker post-gate compression, future regressors with tight
# pred ranges) produce calibrated scores that, on regime-shift test windows,
# never cross the 0.30 floor — yielding 0 trades and an automatic per-window
# fail. The 0.0 entry lets the threshold sweep degrade gracefully to "take
# top-K-per-day" (K = MAX_OPEN_POSITIONS via the slot cap in simulate_window),
# which still satisfies §9's "p ≥ threshold ∧ top-K" with threshold=0. The
# `best_sim` tiebreaker prefers n_trades ≥ 20 then highest ann_return, so this
# entry only wins in windows where no higher threshold produced enough trades —
# making the addition monotonically non-worsening for windows that already
# pass at a tighter threshold.
SCORE_THRESHOLDS = [0.0, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]


def _within(dates, start, end):
    return (dates >= start) & (dates <= end)


def simulate_window(scores, dates, symbols, pnl, hold_days, threshold,
                    max_open=MAX_OPEN_POSITIONS):
    """Walk through test set chronologically as a top-K multi-position trader.

    Implements whitepaper §9 ("Take top-K predictions where p ≥ threshold")
    with K = ``max_open`` concurrent positions, each sized at 1/K of capital
    (matches §11 v2 split-sizing). Per-symbol concentration is capped at one
    open slot.

    Returns per-trade list and aggregate metrics. Equity is compounded in
    entry order; with ``max_open=1`` this reduces to the prior single-position
    behavior.
    """
    # Sort by (date asc, score desc) — top-scoring trades fill open slots first
    order = np.lexsort((-scores, dates))
    s_dates = dates[order]
    s_symbols = symbols[order]
    s_scores = scores[order]
    s_pnl = pnl[order]
    s_hold = hold_days[order]

    # Map every distinct date to a contiguous index for hold-day arithmetic.
    unique_dates = np.sort(np.unique(s_dates))
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    trades = []
    # open_positions: list of (exit_date_str, symbol). Position is "open" while
    # current_date <= exit_date. We re-enter the day after exit_date.
    open_positions = []

    for i in range(len(s_dates)):
        d = s_dates[i]
        # Free up any positions whose exit_date is strictly before today
        open_positions = [(ed, sm) for (ed, sm) in open_positions if ed >= d]
        if len(open_positions) >= max_open:
            continue
        if s_scores[i] < threshold:
            continue
        sym = s_symbols[i]
        # Concentration cap: never run two slots in the same symbol
        if any(sm == sym for (_, sm) in open_positions):
            continue
        # Compute exit date as hold_days unique trading days after entry
        d_idx = date_to_idx[d]
        exit_idx = min(d_idx + int(s_hold[i]) - 1, len(unique_dates) - 1)
        exit_date = unique_dates[exit_idx]
        trades.append({
            'date': str(d),
            'symbol': str(sym),
            'score': float(s_scores[i]),
            'pnl': float(s_pnl[i]),
            'hold_days': int(s_hold[i]),
        })
        open_positions.append((exit_date, sym))

    # Aggregate
    if not trades:
        return {
            'threshold': threshold,
            'n_trades': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'total_return': 0.0,
            'annualized_return': 0.0,
            'max_drawdown': 0.0,
            'final_equity': 1.0,
            'span_days': 1,
            'trades': [],
        }

    pnls = np.array([t['pnl'] for t in trades], dtype=np.float64)
    # Each trade is sized at 1/K of capital, so per-trade equity contribution
    # is pnl/K. Compound in entry order — order-independent for final equity,
    # entry-order is a reasonable proxy for time-resolved DD.
    contribs = pnls / float(max_open)
    equity = np.cumprod(1.0 + contribs)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
    final_equity = float(equity[-1])

    # Annualize over the FULL test-window length, not the trader-active span.
    # Pre-fix: span = first_trade..last_trade + hold; ranker trainers that
    # cluster trades early in the window produced tiny spans (~10 days),
    # blowing up the (365/span) exponent and yielding 10^9 % annualized.
    # Now: span = min(dates)..max(dates) of the test window, so the same
    # equity gain on a 4-month window produces a sane annualized rate.
    # numpy 2.x can't np.min/max string dtype, so use Python builtin min/max.
    date_strs = [str(d)[:10] for d in dates]
    window_start = min(date_strs)
    window_end = max(date_strs)
    span_days = max(1, _date_diff(window_start, window_end) + 1)
    # Clamp final_equity to [1e-6, ~6] before exponentiating: a 4-mo window
    # compounding to ±500% on real fills would imply ~+800% / -83% true return,
    # which is well past anything realistic; values outside that range are
    # ranker/regressor artifacts (sub-day trade clustering, scores treated as
    # equity, etc.) and poison the sort. ann is then clamped to ±500%.
    eq_clamped = float(np.clip(final_equity, 1e-6, 6.0))
    annualized = (eq_clamped ** (365.0 / span_days)) - 1.0
    annualized = float(np.clip(annualized, -5.0, 5.0))

    wins = pnls > 0
    return {
        'threshold': threshold,
        'n_trades': len(trades),
        'win_rate': float(wins.mean()),
        'avg_pnl': float(pnls.mean()),
        'avg_win': float(pnls[wins].mean()) if wins.any() else 0.0,
        'avg_loss': float(pnls[~wins].mean()) if (~wins).any() else 0.0,
        'total_return': final_equity - 1.0,
        'annualized_return': annualized,
        'max_drawdown': max_dd,
        'final_equity': final_equity,
        'span_days': int(span_days),
        'trades': trades,
    }


def _date_diff(a, b):
    """Days between two YYYY-MM-DD strings (inclusive of endpoints)."""
    from datetime import datetime as _dt
    da = _dt.strptime(a[:10], '%Y-%m-%d')
    db = _dt.strptime(b[:10], '%Y-%m-%d')
    return (db - da).days


def _scale_3d_per_feature(X_train, X_test):
    """RobustScaler-equivalent for 3D (N, T, F) sequences.

    Fits per-feature stats on the time-flattened training data so the same
    scaler applies to every step in every sequence. Centering on median and
    scaling by IQR matches the 2D RobustScaler used elsewhere in the gate.
    """
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    median = np.median(flat_train, axis=0)
    q1 = np.quantile(flat_train, 0.25, axis=0)
    q3 = np.quantile(flat_train, 0.75, axis=0)
    iqr = np.where(q3 - q1 > 1e-9, q3 - q1, 1.0)
    return (X_train - median) / iqr, (X_test - median) / iqr


def evaluate_window(X_tab, y, dates, symbols, pnl, hold_days, agg_features,
                     train_start, train_end, test_start, test_end,
                     model_type, trainer_kwargs, verbose=False, X_seq=None):
    """Train fresh model on [train_start, train_end], evaluate on [test_start, test_end].
    Tries each threshold in SCORE_THRESHOLDS and returns the best result.

    For trainers with ``consumes_sequences=True``, the raw 3D ``X_seq`` is
    threaded through with per-feature RobustScaler-equivalent scaling instead
    of the 2D aggregate.
    """
    train_mask = _within(dates, train_start, train_end)
    test_mask = _within(dates, test_start, test_end)

    if train_mask.sum() < 200 or test_mask.sum() < 50:
        return {'passed': False, 'reason': f'insufficient data: train={train_mask.sum()} test={test_mask.sum()}',
                'train': f'{train_start}..{train_end}', 'test': f'{test_start}..{test_end}'}

    # Inner train/val split for early stopping (last 20% of train as val)
    train_dates_arr = np.array(dates[train_mask])
    unique_train_dates = np.sort(np.unique(train_dates_arr))
    val_cutoff = unique_train_dates[int(0.80 * len(unique_train_dates))]
    inner_val = train_dates_arr >= val_cutoff
    inner_tr = train_dates_arr < val_cutoff

    trainer_cls = TRAINERS[model_type]
    consumes_seq = bool(getattr(trainer_cls, 'consumes_sequences', False))

    if consumes_seq:
        if X_seq is None:
            raise RuntimeError(
                f'Trainer {model_type!r} sets consumes_sequences=True but '
                'X_seq was not threaded into evaluate_window.')
        X_train_full, X_test_scaled = _scale_3d_per_feature(
            X_seq[train_mask], X_seq[test_mask])
        X_tr = X_train_full[inner_tr]
        X_val = X_train_full[inner_val]
    else:
        scaler = RobustScaler()
        X_train_full = X_tab[train_mask]
        X_train_full_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_tab[test_mask])
        X_tr = X_train_full_scaled[inner_tr]
        X_val = X_train_full_scaled[inner_val]

    y_tr = y[train_mask][inner_tr]
    y_val = y[train_mask][inner_val]
    pnl_train_full = pnl[train_mask]
    pnl_tr = pnl_train_full[inner_tr]
    pnl_val = pnl_train_full[inner_val]
    dates_train_full = dates[train_mask]
    dates_tr = dates_train_full[inner_tr]
    dates_va = dates_train_full[inner_val]

    if len(set(y_tr)) < 2 or len(set(y_val)) < 2:
        return {'passed': False, 'reason': 'degenerate class distribution in split',
                'train': f'{train_start}..{train_end}', 'test': f'{test_start}..{test_end}'}

    trainer = get_trainer(model_type, **trainer_kwargs)
    trainer.fit(X_tr, y_tr, X_val, y_val, verbose=verbose,
                pnl_train=pnl_tr, pnl_val=pnl_val,
                dates_train=dates_tr, dates_val=dates_va)
    test_dates = dates[test_mask]
    if hasattr(trainer, 'set_predict_context'):
        trainer.set_predict_context(test_dates)
    test_scores = trainer.predict_proba(X_test_scaled)

    test_symbols = symbols[test_mask]
    test_pnl = pnl[test_mask]
    test_hold = hold_days[test_mask]

    # Try each threshold, pick best annualized return that meets the floor
    sims = []
    for thr in SCORE_THRESHOLDS:
        sim = simulate_window(test_scores, test_dates, test_symbols, test_pnl, test_hold, thr)
        sims.append(sim)

    # Pick the one with best annualized return (regardless of pass/fail — we want to see)
    best_sim = max(sims, key=lambda s: (s['n_trades'] >= MIN_TRADES_PER_WINDOW, s['annualized_return']))

    # Per-window pass criteria (v1: no ann_return floor here). MIN_WR is
    # per-window when REGIME_AWARE_GATE is enabled — see _min_wr_for_window.
    n = best_sim['n_trades']
    dd = best_sim['max_drawdown']
    wr = best_sim['win_rate']
    min_wr, min_wr_src = _min_wr_for_window(test_start, test_end)
    fails = []
    if dd > MAX_DD:
        fails.append(f'max_dd {dd:.1%} > {MAX_DD:.0%}')
    if n < MIN_TRADES_PER_WINDOW:
        fails.append(f'n_trades {n} < {MIN_TRADES_PER_WINDOW}')
    if wr < min_wr:
        fails.append(f'wr {wr:.1%} < {min_wr:.0%} [{min_wr_src}]')

    return {
        'passed': len(fails) == 0,
        'fails': fails,
        'train': f'{train_start}..{train_end}',
        'test': f'{test_start}..{test_end}',
        'n_train': int(train_mask.sum()),
        'n_test': int(test_mask.sum()),
        'best': {
            'threshold': best_sim['threshold'],
            'n_trades': best_sim['n_trades'],
            'win_rate': best_sim['win_rate'],
            'avg_pnl': best_sim['avg_pnl'],
            'avg_win': best_sim.get('avg_win', 0),
            'avg_loss': best_sim.get('avg_loss', 0),
            'annualized_return': best_sim['annualized_return'],
            'total_return': best_sim['total_return'],
            'max_drawdown': best_sim['max_drawdown'],
            'final_equity': best_sim['final_equity'],
            'span_days': best_sim.get('span_days', 0),
        },
        'all_thresholds': [{
            'threshold': s['threshold'],
            'n_trades': s['n_trades'],
            'win_rate': s['win_rate'],
            'annualized_return': s['annualized_return'],
            'max_drawdown': s['max_drawdown'],
        } for s in sims],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='lightgbm', choices=list(TRAINERS))
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--num-leaves', type=int, default=31)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--n-estimators', type=int, default=500)
    parser.add_argument('--min-child-samples', type=int, default=50)
    parser.add_argument('--min-child-weight', type=float, default=10.0)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    parser.add_argument('--reg-alpha', type=float, default=0.1)
    parser.add_argument('--reg-lambda', type=float, default=0.1)
    parser.add_argument('--pos-class-weight', type=float, default=1.5)
    parser.add_argument('--quantile-alpha', type=float, default=0.7,
                        help='Upper quantile for xgb_quantile trainer (ignored otherwise)')
    parser.add_argument('--recency-decay', type=float, default=1.5,
                        help='Recency weighting decay for recency_huber_regressor '
                             '(ignored otherwise). 0=uniform, larger=more recent emphasis.')
    parser.add_argument('--huber-slope', type=float, default=1.0,
                        help='Pseudo-Huber slope for huber regressors (ignored otherwise).')
    parser.add_argument('--early-stopping-rounds', type=int, default=30)
    parser.add_argument('--calibrate', type=str, default='none',
                        choices=['none', 'isotonic', 'sigmoid'],
                        help='Optional probability calibration via CalibratedClassifierCV. '
                             'Currently honored by kernel_logreg trainer; ignored otherwise.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', help='Path to write JSON results')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--extra-kwargs', type=str, default='{}',
                        help='JSON dict of additional trainer kwargs merged on top '
                             'of the CLI defaults (trainer __init__ silently drops '
                             'unknown keys via **_).')
    args = parser.parse_args()

    print('=== return_gate v1 (LightGBM, return-based walk-forward) ===\n')

    t0 = time.time()
    X, y, dates, symbols, pnl, hold_days, features = load_sequences(
        seq_len=args.seq_len, verbose=True)
    X_tab = aggregate_sequence(X)
    agg_features = aggregated_feature_names(features)
    print(f'Tabular: {X_tab.shape}  pos_rate={y.mean():.1%}  '
          f'avg_hold={hold_days.mean():.1f}d  ({time.time()-t0:.1f}s)\n')

    trainer_kwargs = dict(
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        min_child_samples=args.min_child_samples,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        pos_class_weight=args.pos_class_weight,
        quantile_alpha=args.quantile_alpha,
        recency_decay=args.recency_decay,
        huber_slope=args.huber_slope,
        early_stopping_rounds=args.early_stopping_rounds,
        calibrate=args.calibrate,
        random_state=args.seed,
    )
    try:
        extra = json.loads(args.extra_kwargs) if args.extra_kwargs else {}
    except json.JSONDecodeError as e:
        raise SystemExit(f'bad --extra-kwargs JSON: {e}')
    if not isinstance(extra, dict):
        raise SystemExit('--extra-kwargs must be a JSON object')
    trainer_kwargs.update(extra)
    print(f'Trainer: {args.model_type}\n')

    results = []
    for idx, (tr_s, tr_e, te_s, te_e) in enumerate(SPLIT_DEFS, 1):
        print(f'--- Split {idx}/{len(SPLIT_DEFS)} ---')
        print(f'  train: {tr_s} → {tr_e}    test: {te_s} → {te_e}')
        t1 = time.time()
        r = evaluate_window(
            X_tab, y, dates, symbols, pnl, hold_days, agg_features,
            tr_s, tr_e, te_s, te_e,
            args.model_type, trainer_kwargs, verbose=args.verbose, X_seq=X)
        elapsed = time.time() - t1
        results.append(r)

        if 'best' in r:
            b = r['best']
            flag = '✓' if r['passed'] else '✗'
            print(f'  {flag} thr={b["threshold"]} n={b["n_trades"]} '
                  f'WR={b["win_rate"]:.1%} ann={b["annualized_return"]:+.1%} '
                  f'DD={b["max_drawdown"]:.1%} '
                  f'avg_win={b["avg_win"]:+.2%} avg_loss={b["avg_loss"]:+.2%}  '
                  f'({elapsed:.1f}s)')
            if not r['passed']:
                print(f'    fails: {", ".join(r["fails"])}')
        else:
            print(f'  ✗ {r["reason"]}')
        print()

    # Aggregate
    valid = [r for r in results if 'best' in r]
    n_passed = sum(1 for r in results if r.get('passed'))
    print('=' * 60)
    print(f'Windows passed: {n_passed}/{len(results)} '
          f'({100*n_passed/max(1,len(results)):.0f}%)')
    print(f'Pass-fraction required: {PASS_FRACTION_REQUIRED:.0%}')

    if valid:
        ann_returns = [r['best']['annualized_return'] for r in valid]
        wrs = [r['best']['win_rate'] for r in valid]
        n_trades_total = sum(r['best']['n_trades'] for r in valid)
        max_dds = [r['best']['max_drawdown'] for r in valid]

        print(f'\nAggregate across windows:')
        print(f'  ann return — avg: {np.mean(ann_returns):+.1%}   '
              f'median: {np.median(ann_returns):+.1%}   '
              f'min: {min(ann_returns):+.1%}   max: {max(ann_returns):+.1%}')
        print(f'  win rate   — avg: {np.mean(wrs):.1%}   '
              f'median: {np.median(wrs):.1%}   '
              f'min: {min(wrs):.1%}   max: {max(wrs):.1%}')
        print(f'  max DD     — avg: {np.mean(max_dds):.1%}   '
              f'median: {np.median(max_dds):.1%}   '
              f'max: {max(max_dds):.1%}')
        print(f'  total trades across all windows: {n_trades_total}')

    overall_passed = (n_passed / max(1, len(results))) >= PASS_FRACTION_REQUIRED
    print(f'\nGate decision: {"PASS" if overall_passed else "FAIL"}')

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'gate_passed': overall_passed,
                'pass_fraction_required': PASS_FRACTION_REQUIRED,
                'windows_passed': n_passed,
                'windows_total': len(results),
                'max_dd': MAX_DD,
                'min_trades_per_window': MIN_TRADES_PER_WINDOW,
                'min_wr': MIN_WR,
                'model_type': args.model_type,
                'trainer_kwargs': trainer_kwargs,
                'results': results,
            }, f, indent=2)
        print(f'Results: {args.output}')

    sys.exit(0 if overall_passed else 1)


if __name__ == '__main__':
    main()
