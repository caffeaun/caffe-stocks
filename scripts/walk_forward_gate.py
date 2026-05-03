#!/usr/bin/env python3
"""Walk-forward validation gate — Stage 3 of the model gate pipeline.

Expanding-window walk-forward validation. Trains a fresh model per split
on "past" data, tests on "future" data. Checks consistency across splits.

Can be run standalone or imported by model_gate.py.

Usage:
  python walk_forward_gate.py --model path/to/model.h5 --scaler path/to/scaler.pkl
"""
import os
import sys
import json
import argparse
import pickle
import time

import numpy as np
import h5py
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from lstm_model import LSTMModel
from labels import label_trade_with_pnl, COMMISSION_PCT

# Import from scripts/lstm_trainer.py (not models/lstm_trainer.py)
import importlib.util
_trainer_spec = importlib.util.spec_from_file_location(
    'lstm_trainer_scripts',
    os.path.join(os.path.dirname(__file__), 'lstm_trainer.py'))
_trainer_mod = importlib.util.module_from_spec(_trainer_spec)
_trainer_spec.loader.exec_module(_trainer_mod)
load_sequences = _trainer_mod.load_sequences
train_model = _trainer_mod.train_model

# Walk-forward thresholds
# Paper-trading gate: get a directionally useful model into paper trading ASAP.
# Tighten thresholds after we have 2-4 weeks of real paper trading data.
# WR must beat market base rate (alpha > 0), no fixed WR floor needed.
N_SPLITS = 7                  # more splits = better regime coverage
MIN_AVG_WIN_PER_SPLIT = 0.03  # 3.0% avg win
MAX_DD_PER_SPLIT = 0.15       # 15% max drawdown
MIN_EV_PER_SPLIT = -0.01      # allow marginally negative EV (paper trading will reveal truth)
MAX_WR_STD = 0.15             # relaxed for 7 splits (more regime variance expected)
MIN_TRADES_PER_SPLIT = 10     # lowered for 7 splits (shorter test windows)
MIN_VALID_SPLITS = 7          # ALL splits must have enough trades — no free passes
# Monitor-only metrics (logged but don't block promotion)
MONITOR_ALPHA_OVER_BASE = 0.05
MONITOR_DATE_CONCENTRATION = 0.20
MONITOR_SELECTIVITY_RATIO = 0.30


def generate_splits(dates, n_splits=N_SPLITS):
    """Fixed calendar-based splits for walk-forward validation.

    Each split has a fixed 6-month train period and a fixed test period.
    Periods are hardcoded so every model is compared on the exact same dates.
    Train size is constant (6 months) — no expanding window bias.
    Test sizes vary slightly by calendar, but periods are fixed across runs.

    Data range: May 2023 – Feb 2026.
    """
    # Fixed split definitions: (train_start, train_end, test_start, test_end)
    SPLIT_DEFS = [
        ('2023-05-01', '2023-10-31', '2023-11-01', '2024-02-28'),  # train H1→Q3, test Q4+Jan-Feb
        ('2023-07-01', '2023-12-31', '2024-01-01', '2024-04-30'),  # slide 2mo
        ('2023-11-01', '2024-04-30', '2024-05-01', '2024-08-31'),
        ('2024-03-01', '2024-08-31', '2024-09-01', '2024-12-31'),
        ('2024-07-01', '2024-12-31', '2025-01-01', '2025-04-30'),
        ('2024-11-01', '2025-04-30', '2025-05-01', '2025-08-31'),
        ('2025-03-01', '2025-08-31', '2025-09-01', '2026-02-28'),
    ]

    splits = []
    for train_start, train_end, test_start, test_end in SPLIT_DEFS:
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        splits.append((train_mask, test_mask))

    return splits


def compute_split_metrics(pnls):
    """Compute per-split metrics from an array of per-trade P&L percentages."""
    if len(pnls) == 0:
        return None

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / len(pnls)
    ev_per_trade = float(np.mean(pnls)) * 100  # as percentage
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0

    gross_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.001
    profit_factor = gross_wins / gross_losses

    # Max drawdown on equity curve (multiplicative, starting at 1.0)
    equity = np.cumprod(1 + pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak  # as fraction of peak
    max_drawdown_pct = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    return {
        'win_rate': round(win_rate, 4),
        'ev_per_trade': round(ev_per_trade, 2),
        'profit_factor': round(profit_factor, 2),
        'trade_count': len(pnls),
        'max_drawdown_pct': round(max_drawdown_pct * 100, 2),  # as percentage
        'avg_win_pct': round(avg_win * 100, 2),
    }


def train_and_evaluate_split(X, y, dates, early_sl, features, n_features,
                              train_mask, test_mask, model_config, pnl_actual=None,
                              verbose=False):
    """Train fresh model on train_mask, score test_mask, compute per-trade P&L.

    Returns dict with metrics or None on training failure.
    """
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    es_train = early_sl[train_mask] if early_sl is not None else None
    pnl_train_split = pnl_actual[train_mask] if pnl_actual is not None else None

    train_dates = dates[train_mask]
    test_dates = dates[test_mask]

    # Use 15% of training data as validation (from end of training period)
    unique_train_dates = np.sort(np.unique(train_dates))
    val_cutoff = unique_train_dates[int(0.85 * len(unique_train_dates))]
    val_mask_inner = train_dates >= val_cutoff
    train_mask_inner = train_dates < val_cutoff

    X_tr = X_train[train_mask_inner]
    y_tr = y_train[train_mask_inner]
    X_val = X_train[val_mask_inner]
    y_val = y_train[val_mask_inner]
    es_tr = es_train[train_mask_inner] if es_train is not None else None
    es_val = es_train[val_mask_inner] if es_train is not None else None
    pnl_tr = pnl_train_split[train_mask_inner] if pnl_train_split is not None else None
    pnl_val_inner = pnl_train_split[val_mask_inner] if pnl_train_split is not None else None
    dates_tr_inner = train_dates[train_mask_inner]

    if verbose:
        print(f"  Train: {len(X_tr)} samples ({min(train_dates)} to {val_cutoff})")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples ({min(test_dates)} to {max(test_dates)})")

    # Train with patience=15 (faster — we need consistency signal, not optimal).
    # Pass pnl_train so the trainer's pairwise PnL-rank auxiliary loss is active
    # during walk-forward retraining (keeps candidate↔WF objective aligned).
    model, scaler, train_metrics = train_model(
        X_tr, y_tr, X_val, y_val, n_features, features,
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        lr=model_config.get('lr', 1e-3),
        batch_size=model_config.get('batch_size', 64),
        patience=15,
        verbose=verbose,
        early_sl_train=es_tr,
        early_sl_val=es_val,
        survival_weight=model_config.get('survival_weight', 3.0),
        fp_weight=model_config.get('fp_weight', 2.5),
        use_attention=model_config.get('use_attention', False),
        sequence_normalize=model_config.get('sequence_normalize', False),
        pnl_train=pnl_tr,
        pnl_val=pnl_val_inner,
        rank_lambda=0.3,
        rank_pnl_gap=0.02,
        dates_train=dates_tr_inner,
        daily_rank_enabled=True,
        daily_rank_lambda=0.5,
        daily_rank_min_per_day=6,
    )

    if model is None:
        return None

    # Score test set
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()

    # Only evaluate trades the model would actually take (pred >= 0.6, matching live threshold)
    LSTM_THRESHOLD = 0.6
    pred_positive = preds >= LSTM_THRESHOLD

    if pred_positive.sum() == 0:
        return {
            'win_rate': 0, 'ev_per_trade': 0, 'profit_factor': 0,
            'trade_count': 0, 'max_drawdown_pct': 0, 'avg_win_pct': 0,
            'train_period': f"{min(train_dates)} to {max(train_dates)}",
            'test_period': f"{min(test_dates)} to {max(test_dates)}",
        }

    positive_preds = pred_positive.flatten()
    selected_labels = y_test[positive_preds]
    selected_dates = test_dates[positive_preds]

    # Use real P&L (already includes commission) if available, else estimate
    pnl_test = pnl_actual[test_mask] if pnl_actual is not None else None
    if pnl_test is not None:
        pnls = pnl_test[positive_preds]
    else:
        # Fallback: rough estimate (commission already in labels, add to P&L estimate)
        pnls = np.where(
            selected_labels == 1,
            0.05 - COMMISSION_PCT,
            -0.03 - COMMISSION_PCT,
        )

    metrics = compute_split_metrics(pnls)
    metrics['train_period'] = f"{min(train_dates)} to {max(train_dates)}"
    metrics['test_period'] = f"{min(test_dates)} to {max(test_dates)}"

    # Base rate: what % of ALL test samples are positive (market win rate)
    base_rate = float(y_test.mean())
    model_wr = metrics['win_rate']
    metrics['base_rate'] = round(base_rate, 4)
    metrics['alpha_over_base'] = round(model_wr - base_rate, 4)

    # Selectivity: what fraction of test samples the model fires on
    # Low = selective (good), High = spray-and-pray (bad, commission drag)
    metrics['selectivity_ratio'] = round(len(selected_labels) / len(y_test), 4)

    # Date concentration: max fraction of trades on any single date
    # If model fires on 60/70 stocks in one day, it's not discriminating
    if len(selected_dates) > 0:
        unique_dates, date_counts = np.unique(selected_dates, return_counts=True)
        max_date_count = int(date_counts.max())
        metrics['max_trades_per_date'] = max_date_count
        metrics['date_concentration'] = round(max_date_count / len(selected_dates), 4)
        metrics['active_dates'] = len(unique_dates)
    else:
        metrics['max_trades_per_date'] = 0
        metrics['date_concentration'] = 0
        metrics['active_dates'] = 0

    return metrics


def run_walk_forward_validation(candidate_h5, candidate_scaler, n_splits=N_SPLITS,
                                 verbose=False):
    """Main entry point. Run walk-forward validation on a candidate model's config.

    Trains fresh models per split using the candidate's hyperparameters.
    Returns: {passed, reason, splits[], aggregate{}}
    """
    start_time = time.time()

    # Read candidate hyperparameters from HDF5
    try:
        with h5py.File(candidate_h5, 'r') as f:
            model_config = {
                'hidden_size': int(f.attrs.get('hidden_size', 64)),
                'num_layers': int(f.attrs.get('num_layers', 2)),
                'dropout': float(f.attrs.get('dropout', 0.3)),
                'use_attention': bool(int(f.attrs.get('use_attention', 0))),
                'survival_weight': float(f.attrs.get('survival_weight', 3.0)),
                'fp_weight': float(f.attrs.get('fp_weight', 2.5)),
                'seq_len': int(f.attrs.get('seq_len', 20)),
                'sequence_normalize': bool(int(f.attrs.get('sequence_normalize', 0))),
            }
    except Exception as e:
        return {'passed': False, 'reason': f'cannot read candidate H5: {e}',
                'splits': [], 'aggregate': {}}

    # Load all sequences (cached)
    try:
        X, y, dates, early_sl, features, pnl_actual = load_sequences(seq_len=model_config['seq_len'])
    except Exception as e:
        return {'passed': False, 'reason': f'cannot load sequences: {e}',
                'splits': [], 'aggregate': {}}

    n_features = len(features)

    # Sort by date
    sort_idx = np.argsort(dates)
    X, y, dates = X[sort_idx], y[sort_idx], dates[sort_idx]
    if early_sl is not None:
        early_sl = early_sl[sort_idx]

    # Generate expanding-window splits
    splits = generate_splits(dates, n_splits=n_splits)

    if len(splits) == 0:
        return {'passed': False, 'reason': 'no valid splits generated',
                'splits': [], 'aggregate': {}}

    if verbose:
        print(f"Walk-forward: {len(splits)} splits, {len(X)} total samples")

    split_results = []
    reasons = []

    for i, (train_mask, test_mask) in enumerate(splits):
        if verbose:
            print(f"\n--- Split {i+1}/{len(splits)} ---")

        result = train_and_evaluate_split(
            X, y, dates, early_sl, features, n_features,
            train_mask, test_mask, model_config,
            pnl_actual=pnl_actual, verbose=verbose)

        if result is None:
            reasons.append(f"split {i+1}: training failed")
            continue

        split_results.append(result)

        # Skip split if too few trades (not a failure)
        if result['trade_count'] < MIN_TRADES_PER_SPLIT:
            if verbose:
                print(f"  Split {i+1}: {result['trade_count']} trades < {MIN_TRADES_PER_SPLIT} — skipped")
            continue

        # Per-split hard gates (block promotion)
        # WR must beat market base rate — no fixed floor, alpha > 0 is the test
        alpha = result.get('alpha_over_base', 0)
        base_rate = result.get('base_rate', 0)
        if alpha < 0:
            reasons.append(f"split {i+1}: WR {result['win_rate']:.1%} below base rate {base_rate:.1%} (alpha {alpha:.1%})")

        if result['avg_win_pct'] / 100 < MIN_AVG_WIN_PER_SPLIT:
            reasons.append(f"split {i+1}: avg win {result['avg_win_pct']:.1f}% < {MIN_AVG_WIN_PER_SPLIT*100:.1f}%")

        if result['max_drawdown_pct'] / 100 > MAX_DD_PER_SPLIT:
            reasons.append(f"split {i+1}: max DD {result['max_drawdown_pct']:.1f}% > {MAX_DD_PER_SPLIT*100:.0f}%")

        if result['ev_per_trade'] < MIN_EV_PER_SPLIT:
            reasons.append(f"split {i+1}: EV {result['ev_per_trade']:.2f}% < {MIN_EV_PER_SPLIT*100:.1f}%")

        # Monitor-only metrics (logged in results but don't block promotion)
        date_conc = result.get('date_concentration', 0)
        sel_ratio = result.get('selectivity_ratio', 0)
        monitors = []
        if alpha < MONITOR_ALPHA_OVER_BASE:
            monitors.append(f"⚠α={alpha:.1%}")
        if date_conc > MONITOR_DATE_CONCENTRATION:
            monitors.append(f"⚠conc={date_conc:.0%}")
        if sel_ratio > MONITOR_SELECTIVITY_RATIO:
            monitors.append(f"⚠sel={sel_ratio:.0%}")
        if monitors:
            result['monitors'] = ' '.join(monitors)

    # Aggregate metrics across valid splits (with enough trades)
    valid_splits = [s for s in split_results if s['trade_count'] >= MIN_TRADES_PER_SPLIT]

    if len(valid_splits) < MIN_VALID_SPLITS:
        return {'passed': False,
                'reason': f'only {len(valid_splits)} valid splits (need {MIN_VALID_SPLITS}+)',
                'splits': split_results, 'aggregate': {}}

    wrs = [s['win_rate'] for s in valid_splits]
    alphas = [s.get('alpha_over_base', 0) for s in valid_splits]
    aggregate = {
        'avg_win_rate': round(float(np.mean(wrs)), 4),
        'wr_std': round(float(np.std(wrs)), 4),
        'min_wr': round(float(np.min(wrs)), 4),
        'max_wr': round(float(np.max(wrs)), 4),
        'avg_alpha': round(float(np.mean(alphas)), 4),
        'min_alpha': round(float(np.min(alphas)), 4),
        'avg_ev': round(float(np.mean([s['ev_per_trade'] for s in valid_splits])), 2),
        'total_trades': sum(s['trade_count'] for s in valid_splits),
        'valid_splits': len(valid_splits),
        'elapsed_seconds': round(time.time() - start_time, 1),
    }

    # Consistency check: WR std across splits
    if aggregate['wr_std'] > MAX_WR_STD:
        reasons.append(f"WR std {aggregate['wr_std']:.1%} > {MAX_WR_STD:.0%} (inconsistent)")

    passed = len(reasons) == 0
    reason = 'all checks passed' if passed else '; '.join(reasons)

    return {
        'passed': passed,
        'reason': reason,
        'splits': split_results,
        'aggregate': aggregate,
    }


def main():
    parser = argparse.ArgumentParser(description='Walk-forward validation gate')
    parser.add_argument('--model', required=True, help='Path to model H5 file')
    parser.add_argument('--scaler', required=True, help='Path to scaler pkl file')
    parser.add_argument('--splits', type=int, default=N_SPLITS, help='Number of splits')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    result = run_walk_forward_validation(
        args.model, args.scaler, n_splits=args.splits, verbose=args.verbose)

    print(json.dumps(result, indent=2))
    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
