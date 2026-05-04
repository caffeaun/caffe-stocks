#!/usr/bin/env python3
"""Generic v1 trainer — model-agnostic via models.trainers.

Trains a binary classifier on aggregated sequence features. Output is a
candidate model directory containing the trainer-specific artifact files
plus scaler.pkl and metadata.json. The walk-forward gate (return_gate.py)
retrains fresh per split using the same trainer + hyperparameters.

Usage:
  ./venv/bin/python scripts/trainer.py
  ./venv/bin/python scripts/trainer.py --model-type xgboost
  ./venv/bin/python scripts/trainer.py --num-leaves 63 --max-depth 8
"""
import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.expanduser('~/projects/caffe-stocks'))
from models.sequence_loader import (
    load_sequences, aggregate_sequence, aggregated_feature_names,
)
from models.trainers import get_trainer, TRAINERS

BASE_PATH = os.path.expanduser('~/projects/caffe-stocks')


def time_split(dates, train_pct=0.70, val_pct=0.85):
    """Return (train_mask, val_mask, test_mask) using chronological splits on
    unique dates."""
    unique_dates = np.sort(np.unique(dates))
    train_cutoff = unique_dates[int(train_pct * len(unique_dates))]
    val_cutoff = unique_dates[int(val_pct * len(unique_dates))]
    train_mask = dates < train_cutoff
    val_mask = (dates >= train_cutoff) & (dates < val_cutoff)
    test_mask = dates >= val_cutoff
    return train_mask, val_mask, test_mask


def evaluate(trainer, X, y, thresholds=(0.5, 0.55, 0.6, 0.65, 0.7), label=''):
    """Compute AUC + per-threshold precision/recall."""
    if len(y) == 0 or len(set(y)) < 2:
        return None
    preds = trainer.predict_proba(X)
    res = {
        'label': label,
        'n': int(len(y)),
        'pos_rate': float(np.mean(y)),
        'auc': float(roc_auc_score(y, preds)),
    }
    for thr in thresholds:
        mask = preds >= thr
        n = int(mask.sum())
        if n == 0:
            res[f'precision_at_{thr}'] = None
            res[f'recall_at_{thr}'] = None
            res[f'n_at_{thr}'] = 0
        else:
            true_positives = int(((preds >= thr) & (y == 1)).sum())
            res[f'precision_at_{thr}'] = float(np.mean(y[mask]))
            res[f'recall_at_{thr}'] = float(true_positives / max(int((y == 1).sum()), 1))
            res[f'n_at_{thr}'] = n
    return res


def print_eval(metrics, header):
    print(f'\n=== {header} ===')
    if metrics is None:
        print('  (insufficient samples or single-class)')
        return
    print(f'  N={metrics["n"]}  pos_rate={metrics["pos_rate"]:.1%}  AUC={metrics["auc"]:.4f}')
    for thr in (0.5, 0.55, 0.6, 0.65, 0.7):
        n = metrics[f'n_at_{thr}']
        if n == 0:
            print(f'  @{thr}: 0 samples selected')
        else:
            p = metrics[f'precision_at_{thr}']
            r = metrics[f'recall_at_{thr}']
            print(f'  @{thr}: precision {p:.1%}  recall {r:.1%}  n={n}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='lightgbm', choices=list(TRAINERS))
    parser.add_argument('--output-dir', default=os.path.join(BASE_PATH, 'models', 'lgbm', 'candidates'))
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--lookahead', type=int, default=10)
    # Hyperparameters — passed through to the chosen trainer
    parser.add_argument('--num-leaves', type=int, default=31)         # lightgbm
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--n-estimators', type=int, default=500)
    parser.add_argument('--min-child-samples', type=int, default=50)  # lightgbm
    parser.add_argument('--min-child-weight', type=float, default=10.0)  # xgboost
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    parser.add_argument('--reg-alpha', type=float, default=0.1)
    parser.add_argument('--reg-lambda', type=float, default=0.1)
    parser.add_argument('--pos-class-weight', type=float, default=1.5)
    parser.add_argument('--early-stopping-rounds', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print(f'=== v1 Trainer ({args.model_type}) ===')
    print(f'output: {args.output_dir}')

    # Load
    X, y, dates, symbols, pnl, hold_days, features = load_sequences(
        seq_len=args.seq_len, lookahead=args.lookahead, verbose=True)
    print(f'\nSamples: {len(X)} | Pos rate: {y.mean():.1%} | Avg hold: {hold_days.mean():.1f}d')

    # Aggregate to tabular
    X_tab = aggregate_sequence(X)
    agg_features = aggregated_feature_names(features)
    print(f'Tabular: {X_tab.shape}  ({len(agg_features)} aggregated features)')

    # Time-based split
    train_mask, val_mask, test_mask = time_split(dates)
    print(f'\nSplit:  train={train_mask.sum()}  val={val_mask.sum()}  test={test_mask.sum()}')
    print(f'  train: {min(dates[train_mask])} → {max(dates[train_mask])}')
    print(f'  val:   {min(dates[val_mask])} → {max(dates[val_mask])}')
    print(f'  test:  {min(dates[test_mask])} → {max(dates[test_mask])}')

    # Scale
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_tab[train_mask])
    X_val = scaler.transform(X_tab[val_mask])
    X_test = scaler.transform(X_tab[test_mask])
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    # Train
    print(f'\nTraining {args.model_type}...')
    trainer = get_trainer(args.model_type, **vars(args))
    trainer.fit(X_train, y_train, X_val, y_val, verbose=args.verbose)
    if trainer.best_iteration is not None:
        print(f'  best iteration: {trainer.best_iteration}')

    # Eval
    val_metrics = evaluate(trainer, X_val, y_val, label='val')
    test_metrics = evaluate(trainer, X_test, y_test, label='test')
    print_eval(val_metrics, 'Validation')
    print_eval(test_metrics, 'Test (held out)')

    # Top features
    importance = trainer.feature_importance()
    if importance is not None:
        order = np.argsort(importance)[::-1][:10]
        print(f'\nTop 10 features by importance:')
        for i in order:
            print(f'  {agg_features[i]:40s}  {importance[i]:.0f}')

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    extra = {
        'seq_len': args.seq_len,
        'lookahead': args.lookahead,
        'features': list(features),
        'agg_features': list(agg_features),
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'n_train': int(train_mask.sum()),
        'n_val': int(val_mask.sum()),
        'n_test': int(test_mask.sum()),
        'train_date_range': [str(min(dates[train_mask])), str(max(dates[train_mask]))],
        'val_date_range': [str(min(dates[val_mask])), str(max(dates[val_mask]))],
        'test_date_range': [str(min(dates[test_mask])), str(max(dates[test_mask]))],
        'saved_at': datetime.now().isoformat(),
    }
    paths = trainer.save(args.output_dir, extra=extra)
    paths['scaler'] = scaler_path

    print(f'\nSaved:')
    for k, v in paths.items():
        print(f'  {k:10s} {v}')


if __name__ == '__main__':
    main()
