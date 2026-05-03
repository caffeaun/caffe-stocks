#!/usr/bin/env python3
"""Clean-Win Label Candidate Trainer.

Structural change: labels.py now enforces a max-adverse-excursion constraint
on positive labels (clean wins only — no underwater recoveries). This script
wraps the existing #197 baseline trainer but forces the sequence cache to
rebuild so the new label definition actually takes effect.

Hypothesis: training on "clean early-strength" wins teaches the model to
identify patterns that project across regimes better than "eventual winners",
because early strength is observable in features (volume, intraday range,
breadth) while eventual recovery requires regime-specific bull tailwind.
"""
import os
import sys
import glob
import json
import pickle
import argparse
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
import torch

BASE_PATH = '/home/kanoonth-ai/projects/caffe-stocks'
sys.path.insert(0, os.path.join(BASE_PATH, 'models'))
sys.path.insert(0, os.path.join(BASE_PATH, 'scripts'))

# Re-import with the modified labels.py in the loop.
import importlib
import labels as _labels  # noqa: F401
import lstm_trainer_candidate as _candidate

DATA_DIR = os.path.join(BASE_PATH, 'data')


def invalidate_sequence_cache():
    """Delete cached sequence files so the new label definition rebuilds them.
    CLEAN_WIN_MAX_DD is not part of the cache key in lstm_trainer.py, so
    without this the old cached labels would silently persist.
    """
    caches = glob.glob(os.path.join(DATA_DIR, 'sequences_cache_*.npz'))
    for c in caches:
        os.remove(c)
    print(f'Invalidated {len(caches)} sequence caches — new clean-win labels '
          f'will be computed on first load')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',
                        default=os.path.join(BASE_PATH, 'models', 'lstm', 'candidates'))
    parser.add_argument('--hidden-size', type=int, default=48)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seq-len', type=int, default=20)
    args = parser.parse_args()

    print(f'Device: {_candidate.DEVICE}')
    print(f'Labels: MIN_PROFIT_PCT={_labels.MIN_PROFIT_PCT}, '
          f'CLEAN_WIN_MAX_DD={_labels.CLEAN_WIN_MAX_DD}')

    invalidate_sequence_cache()

    # Load with new clean-win labels. load_sequences reads MIN_PROFIT_PCT from
    # labels.py into its cache key — but not CLEAN_WIN_MAX_DD. We already
    # nuked the cache above, so it will rebuild with the new label definition.
    X, y, dates, early_sl, features, pnl = _candidate.load_sequences(seq_len=args.seq_len)
    n_features = len(features)
    print(f'Features ({n_features}): {features}')
    print(f'Total samples: {len(X)} | Positive: {y.sum():.0f} ({y.mean():.1%})')
    print(f'Early stop-loss samples: {int(early_sl.sum())} ({early_sl.mean():.1%})')
    if pnl is not None:
        # Sanity check: how often does a PnL > MIN_PROFIT_PCT trade become a
        # NEGATIVE label due to the clean-win constraint? That's our filter rate.
        profit_mask = pnl > _labels.MIN_PROFIT_PCT
        legacy_pos_rate = float(profit_mask.mean())
        new_pos_rate = float(y.mean())
        filtered_pct = 1.0 - (new_pos_rate / max(legacy_pos_rate, 1e-9))
        print(f'Legacy positive rate: {legacy_pos_rate:.1%} (pnl > {_labels.MIN_PROFIT_PCT})')
        print(f'Clean-win positive rate: {new_pos_rate:.1%}')
        print(f'Filtered out {filtered_pct:.1%} of profitable trades as "rocky wins"')

    # Time split (70/15/15)
    splits = _candidate.time_split(X, y, dates, early_sl=early_sl)
    print(f'Train: {len(splits["X_train"])} | '
          f'Val: {len(splits["X_val"])} | '
          f'Test: {len(splits["X_test"])}')

    # Train with #197 baseline config. The only thing different from #197 is
    # the label definition itself (via labels.py). Everything else — mixup=0.3,
    # pure BCE, hidden=48, dropout=0.4, lr=5e-4, SWA — is unchanged so that
    # any WR shift is attributable to the label change, not to confounds.
    model, scaler, train_metrics = _candidate.train_model(
        splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'],
        n_features, features,
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, lr=args.lr, batch_size=args.batch_size,
        early_sl_train=splits.get('early_sl_train'),
        early_sl_val=splits.get('early_sl_val'),
        survival_weight=1.0, fp_weight=1.0,
        use_attention=False,
        sequence_normalize=False,
        mixup_alpha=0.3,
        swa_window=5)

    if model is None:
        raise RuntimeError(f'Training failed: {train_metrics}')

    print(f'Positive weight applied: {train_metrics["pos_weight"]:.2f}x')

    # Evaluate on test split
    metrics = _candidate.evaluate_model(
        model, scaler, splits['X_test'], splits['y_test'], n_features)
    print(f'\n=== Test Results (0.5 threshold) ===')
    print(f'  Accuracy:  {metrics["accuracy"]:.2%}')
    print(f'  Precision: {metrics["precision"]:.2%}')
    print(f'  Recall:    {metrics["recall"]:.2%}')
    print(f'  TP={metrics["tp"]} FP={metrics["fp"]} '
          f'FN={metrics["fn"]} TN={metrics["tn"]}')
    print(f'  Positive rate (test): {metrics["positive_rate"]:.1%}')

    # Also report precision at live 0.6 threshold (what the gate actually uses)
    X_test_scaled = scaler.transform(
        splits['X_test'].reshape(-1, n_features)).reshape(splits['X_test'].shape)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    mask06 = preds >= 0.6
    if mask06.sum() > 0:
        prec06 = float(splits['y_test'][mask06].mean())
        sel06 = float(mask06.mean())
        print(f'  @0.6 threshold: precision={prec06:.1%}, '
              f'selectivity={sel06:.1%}, n={int(mask06.sum())}')
    else:
        print(f'  @0.6 threshold: 0 samples selected (score distribution too low)')

    # Save — use required candidate_* names for the gate to pick up
    os.makedirs(args.output_dir, exist_ok=True)
    cand_model_path = os.path.join(args.output_dir, 'candidate_model.h5')
    cand_scaler_path = os.path.join(args.output_dir, 'candidate_scaler.pkl')
    trading_model_path = os.path.join(args.output_dir, 'trading_model.h5')
    trading_scaler_path = os.path.join(args.output_dir, 'scaler.pkl')

    _candidate.save_model(model, scaler, features, metrics, train_metrics,
                          args.seq_len, cand_model_path, cand_scaler_path)
    # Also write trading_model.h5/scaler.pkl so any existing tooling that
    # expects those names still resolves.
    _candidate.save_model(model, scaler, features, metrics, train_metrics,
                          args.seq_len, trading_model_path, trading_scaler_path)
    print(f'\nSaved: {cand_model_path}')
    print(f'Saved: {cand_scaler_path}')


if __name__ == '__main__':
    main()
