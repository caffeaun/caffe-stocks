#!/usr/bin/env python3
"""Focal-loss candidate trainer.

Structural change: replace BCE with focal BCE (gamma=2) inside
scripts/lstm_trainer.py::train_model so walk-forward retraining also uses it.
This wrapper just invokes the modified train_model with the Entry #197 config
and saves to candidate_model.h5 / candidate_scaler.pkl.
"""
import os
import sys
import json
import argparse
import importlib.util
from datetime import datetime

# Load the shared trainer module (already patched with focal loss)
SCRIPT = '/home/kanoonth-ai/projects/caffe-stocks/scripts/lstm_trainer.py'
spec = importlib.util.spec_from_file_location('lstm_trainer_patched', SCRIPT)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

load_sequences = trainer.load_sequences
time_split = trainer.time_split
train_model = trainer.train_model
evaluate_model = trainer.evaluate_model
save_model = trainer.save_model
DEVICE = trainer.DEVICE
BASE_PATH = trainer.BASE_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',
                        default='/home/kanoonth-ai/projects/caffe-stocks/models/lstm/candidates')
    parser.add_argument('--seq-len', type=int, default=20)
    args = parser.parse_args()

    print(f'Device: {DEVICE}')
    print('Strategy: Focal BCE (gamma=2) on top of Entry #197 baseline '
          '(hidden=48, layers=1, dropout=0.4, lr=5e-4, batch=256, Mixup=0.3, SWA=5)')

    X, y, dates, early_sl, features, _pnl = load_sequences(seq_len=args.seq_len)
    n_features = len(features)
    print(f'Total samples: {len(X)} | Positive: {y.sum():.0f} ({y.mean():.1%})')

    splits = time_split(X, y, dates, early_sl=early_sl)
    print(f'  Train: {len(splits["X_train"])}  Val: {len(splits["X_val"])}  '
          f'Test: {len(splits["X_test"])}')

    # Entry #197 hyperparameters — these go into the candidate H5 and the
    # walk-forward gate will reuse them to retrain fresh models per split.
    model, scaler, train_metrics = train_model(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val'],
        n_features, features,
        hidden_size=48,
        num_layers=1,
        dropout=0.4,
        lr=5e-4,
        batch_size=256,
        patience=20,
        early_sl_train=splits.get('early_sl_train'),
        early_sl_val=splits.get('early_sl_val'),
        survival_weight=1.0,
        fp_weight=1.0,
        use_attention=False,
        sequence_normalize=False,
        mixup_alpha=0.3,
        swa_window=5,
        verbose=True,
    )

    if model is None:
        raise RuntimeError(f'Training failed: {train_metrics}')

    print(f'Epochs: {train_metrics["epochs"]}  '
          f'best_val_loss: {train_metrics["best_val_loss"]:.4f}  '
          f'focal_gamma_pos: {train_metrics.get("focal_gamma_pos", "n/a")}  '
          f'focal_gamma_neg: {train_metrics.get("focal_gamma_neg", "n/a")}  '
          f'SWA snapshots: {train_metrics.get("swa_snapshots", 0)}')

    metrics = evaluate_model(model, scaler,
                             splits['X_test'], splits['y_test'], n_features)
    print(f'\n=== Test (holdout 15%) ===')
    print(f'  Accuracy:  {metrics["accuracy"]:.2%}')
    print(f'  Precision: {metrics["precision"]:.2%}')
    print(f'  Recall:    {metrics["recall"]:.2%}')
    print(f'  TP={metrics["tp"]} FP={metrics["fp"]} '
          f'FN={metrics["fn"]} TN={metrics["tn"]}')
    print(f'  Positive rate: {metrics["positive_rate"]:.1%}')

    # Score distribution — helps diagnose calibration before WF gate runs
    import numpy as np
    import torch as _t
    with _t.no_grad():
        X_test_scaled = scaler.transform(
            splits['X_test'].reshape(-1, n_features)
        ).reshape(splits['X_test'].shape)
        preds = model(_t.tensor(X_test_scaled, dtype=_t.float32)).numpy()
    q = np.percentile(preds, [50, 75, 90, 95, 99])
    frac_06 = float((preds >= 0.6).mean())
    print(f'  Score distribution: p50={q[0]:.3f} p75={q[1]:.3f} '
          f'p90={q[2]:.3f} p95={q[3]:.3f} p99={q[4]:.3f}')
    print(f'  Fraction >= 0.6 (live threshold): {frac_06:.2%}')

    model_path = os.path.join(args.output_dir, 'candidate_model.h5')
    scaler_path = os.path.join(args.output_dir, 'candidate_scaler.pkl')
    save_model(model, scaler, features, metrics, train_metrics,
               args.seq_len, model_path, scaler_path)
    print(f'\nSaved: {model_path}')
    print(f'Saved: {scaler_path}')


if __name__ == '__main__':
    main()
