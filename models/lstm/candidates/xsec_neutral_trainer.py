#!/usr/bin/env python3
"""Cross-sectional neutralization wrapper around lstm_trainer.py.

Structural change (untried per attempt history): the real modification
lives in scripts/lstm_trainer.py — its load_sequences() now subtracts
the per-day cross-sectional mean from every feature BEFORE building
sequences. Every input the model sees is therefore mean-zero within
each trading date, so the model cannot infer the market regime from
raw feature magnitudes. Market-wide features (sector_breadth,
market_breadth_adv, market_new_highs, market_breadth_above_sma20,
foreign_net_monthly_pctrank, sector_avg_ret_1d, market_sector_aligned)
collapse to zero — they carried the regime signal and are the cause of
train->test distribution shift. What remains is the stock-specific
deviation, which is regime-invariant by construction and should
transfer across all 7 walk-forward splits.

This script simply calls the shared load_sequences / train_model /
evaluate_model / save_model functions (so training, walk-forward gate
re-training, and eval all use the same neutralized input space), then
writes the result to candidate_model.h5 / candidate_scaler.pkl.
Training config matches Entry #197: Mixup alpha=0.3, pure BCE,
hidden=48, layers=1, dropout=0.4, batch=256, lr=5e-4, SWA last-5.
"""
import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime

BASE_PATH = '/home/kanoonth-ai/projects/caffe-stocks'
TRAINER_PATH = os.path.join(BASE_PATH, 'scripts', 'lstm_trainer.py')

# Import the shared trainer (the one the walk-forward gate re-imports).
# Using importlib rather than sys.path hacking so the name doesn't
# collide with models/lstm_trainer.py (another module of the same
# short name).
_spec = importlib.util.spec_from_file_location('lstm_trainer_scripts', TRAINER_PATH)
_trainer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trainer)

load_sequences = _trainer.load_sequences
time_split = _trainer.time_split
train_model = _trainer.train_model
evaluate_model = _trainer.evaluate_model
save_model = _trainer.save_model
DEVICE = _trainer.DEVICE


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

    print(f'Device: {DEVICE}')
    print('Loading sequences (cross-sectional demeaning applied inside load_sequences)...')
    X, y, dates, early_sl, features, pnl = load_sequences(seq_len=args.seq_len)
    n_features = len(features)
    print(f'Features ({n_features}): {features}')
    print(f'Total samples: {len(X)} | Positive: {y.sum():.0f} ({y.mean():.1%})')

    splits = time_split(X, y, dates, early_sl=early_sl, pnl=pnl)
    print(f'  Train: {len(splits["X_train"])} {min(splits["train_dates"])} -> {max(splits["train_dates"])}')
    print(f'  Val:   {len(splits["X_val"])} {min(splits["val_dates"])} -> {max(splits["val_dates"])}')
    print(f'  Test:  {len(splits["X_test"])} {min(splits["test_dates"])} -> {max(splits["test_dates"])}')

    # Entry #197 recipe: Mixup alpha=0.3, pure BCE with capped pos_weight,
    # SWA last-5. No DANN (disabled explicitly — we want xsec neutralization
    # to be the only moving piece vs the known baseline), no daily-rank
    # sampler, no survival/fp reweighting.
    model, scaler, train_metrics = train_model(
        splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'],
        n_features, features,
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, lr=args.lr, batch_size=args.batch_size,
        early_sl_train=splits.get('early_sl_train'),
        early_sl_val=splits.get('early_sl_val'),
        survival_weight=1.0, fp_weight=1.0,
        use_attention=False, sequence_normalize=False,
        mixup_alpha=0.3, swa_window=5,
        pnl_train=splits.get('pnl_train'),
        rank_lambda=0.0, rank_pnl_gap=0.02,
        dann_enabled=False,
        dates_train=splits.get('train_dates'),
        daily_rank_enabled=False)

    if model is None:
        raise RuntimeError(f'Training failed: {train_metrics}')

    print(f'Positive weight: {train_metrics["pos_weight"]:.2f}x')
    metrics = evaluate_model(model, scaler, splits['X_test'], splits['y_test'], n_features)
    print('\n=== Test Results ===')
    print(f'  Accuracy:  {metrics["accuracy"]:.2%}')
    print(f'  Precision: {metrics["precision"]:.2%}')
    print(f'  Recall:    {metrics["recall"]:.2%}')
    print(f'  TP={metrics["tp"]} FP={metrics["fp"]} FN={metrics["fn"]} TN={metrics["tn"]}')
    print(f'  Positive rate: {metrics["positive_rate"]:.1%}')

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'candidate_model.h5')
    scaler_path = os.path.join(args.output_dir, 'candidate_scaler.pkl')
    save_model(model, scaler, features, metrics, train_metrics, args.seq_len,
               model_path, scaler_path)
    print(f'\nModel saved to: {model_path}')
    print(f'Scaler saved to: {scaler_path}')


if __name__ == '__main__':
    main()
