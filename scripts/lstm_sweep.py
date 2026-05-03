#!/usr/bin/env python3
"""Hyperparameter sweep using the same fixed calendar splits as the gate.

Evaluates each config across all 7 walk-forward splits (train on 6 months,
validate on next period). Picks config with best average val loss across splits,
then trains a final model on the most recent 6-month window.

Usage:
  python lstm_sweep.py --output-dir /path/to/output
  python lstm_sweep.py --output-dir /path/to/output --configs 30
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.dirname(__file__))
from lstm_trainer import (
    load_sequences, train_model, evaluate_model,
    save_model, DEVICE, BASE_PATH,
)
from walk_forward_gate import generate_splits, N_SPLITS

# --- Search space (sized for 6-month train windows, ~5K effective samples) ---
SEARCH_SPACE = {
    'hidden_size': [32, 48, 64, 128],
    'num_layers': [1, 2],
    'dropout': [0.1, 0.2, 0.3, 0.4],
    'lr': [3e-4, 5e-4, 1e-3, 2e-3],
    'batch_size': [32, 64, 128],
}


def _is_valid_config(cfg):
    if cfg['num_layers'] == 1:
        cfg['dropout'] = 0.0
    return True


def random_configs(n=20, seed=None):
    """Sample n random configs from the search space."""
    rng = np.random.RandomState(seed)
    configs = []
    for _ in range(n * 3):
        cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        cfg = {k: int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v
               for k, v in cfg.items()}
        if _is_valid_config(cfg) and cfg not in configs:
            configs.append(cfg)
        if len(configs) >= n:
            break
    return configs


def evaluate_config(cfg, X, y, dates, early_sl, features, splits,
                    use_attention=False, verbose=False):
    """Train on each split's train window, validate on test window.
    Returns average val loss across all splits."""
    n_features = len(features)
    val_losses = []

    for fold_idx, (train_mask, test_mask) in enumerate(splits):
        X_full, y_full = X[train_mask], y[train_mask]
        es_full = early_sl[train_mask] if early_sl is not None else None
        train_dates = dates[train_mask]

        # Split train window 85/15 for train/val (same as gate)
        # Test window is NEVER seen during config selection
        unique_td = np.sort(np.unique(train_dates))
        val_cutoff = unique_td[int(0.85 * len(unique_td))]
        inner_train = train_dates < val_cutoff
        inner_val = train_dates >= val_cutoff

        X_train, y_train = X_full[inner_train], y_full[inner_train]
        X_val, y_val = X_full[inner_val], y_full[inner_val]
        es_train = es_full[inner_train] if es_full is not None else None
        es_val = es_full[inner_val] if es_full is not None else None

        if len(X_train) < 100 or len(X_val) < 50:
            continue

        model, scaler, train_metrics = train_model(
            X_train, y_train,
            X_val, y_val,
            n_features, features,
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout'],
            lr=cfg['lr'],
            batch_size=cfg['batch_size'],
            patience=15,
            verbose=False,
            early_sl_train=es_train,
            early_sl_val=es_val,
            use_attention=use_attention,
        )

        if model is None:
            return float('inf'), []

        val_losses.append(train_metrics['best_val_loss'])

        if verbose:
            print(f'    Split {fold_idx+1}: val_loss={train_metrics["best_val_loss"]:.4f} '
                  f'({train_metrics["epochs"]} epochs)')

    if not val_losses:
        return float('inf'), []

    return float(np.mean(val_losses)), val_losses


def main():
    parser = argparse.ArgumentParser(description='LSTM hyperparameter sweep')
    parser.add_argument('--output-dir', required=True, help='Where to save best model')
    parser.add_argument('--configs', type=int, default=20, help='Number of random configs to try')
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--seed', type=int, default=None, help='Random seed for config sampling')
    parser.add_argument('--use-attention', action='store_true', help='Enable temporal attention')
    args = parser.parse_args()

    print(f'Device: {DEVICE}')

    # Load data once
    t0 = time.time()
    X, y, dates, early_sl, features, _pnl = load_sequences(seq_len=args.seq_len)
    n_features = len(features)
    dates_str = dates.astype(str) if dates.dtype.kind != 'U' else dates
    print(f'Data: {len(X)} samples, {n_features} features ({time.time()-t0:.1f}s)')
    print(f'Positive rate: {y.mean():.1%}')

    # Use the same fixed calendar splits as the gate
    splits = generate_splits(dates_str)
    print(f'\nUsing {len(splits)} fixed calendar splits (same as gate):')
    for i, (train_mask, test_mask) in enumerate(splits):
        tr_d = np.sort(np.unique(dates_str[train_mask]))
        te_d = np.sort(np.unique(dates_str[test_mask]))
        print(f'  Split {i+1}: train {tr_d[0]}–{tr_d[-1]} ({train_mask.sum()}), '
              f'test {te_d[0]}–{te_d[-1]} ({test_mask.sum()})')

    # Generate configs
    configs = random_configs(args.configs, seed=args.seed)
    print(f'\nSearching {len(configs)} configs...\n')

    # Evaluate each config across all splits
    results = []
    for i, cfg in enumerate(configs):
        t1 = time.time()
        avg_loss, fold_losses = evaluate_config(
            cfg, X, y, dates_str, early_sl, features, splits,
            use_attention=args.use_attention)
        elapsed = time.time() - t1
        results.append({'config': cfg, 'avg_val_loss': avg_loss, 'fold_losses': fold_losses})

        status = f'[{i+1}/{len(configs)}] avg_val_loss={avg_loss:.4f} ({elapsed:.1f}s)'
        cfg_str = f'h={cfg["hidden_size"]} L={cfg["num_layers"]} d={cfg["dropout"]} lr={cfg["lr"]} bs={cfg["batch_size"]}'
        print(f'  {status} | {cfg_str}')

    # Sort by average val loss
    results.sort(key=lambda r: r['avg_val_loss'])
    best = results[0]
    print(f'\n=== Best config ===')
    print(f'  {best["config"]}')
    print(f'  avg_val_loss: {best["avg_val_loss"]:.4f}')
    print(f'  fold_losses: {[f"{l:.4f}" for l in best["fold_losses"]]}')

    # Train final model on the LAST split's train window (most recent 6 months)
    # This is the model that will be used for live trading
    last_train_mask, last_test_mask = splits[-1]
    X_train, y_train = X[last_train_mask], y[last_train_mask]
    X_test, y_test = X[last_test_mask], y[last_test_mask]
    es_train = early_sl[last_train_mask] if early_sl is not None else None

    # Split train into train/val (85/15) for early stopping
    train_dates = dates_str[last_train_mask]
    unique_train_dates = np.sort(np.unique(train_dates))
    val_cutoff = unique_train_dates[int(0.85 * len(unique_train_dates))]
    inner_train = train_dates < val_cutoff
    inner_val = train_dates >= val_cutoff

    print(f'\nTraining final model on last split (most recent 6 months)...')
    print(f'  Train: {inner_train.sum()} | Val: {inner_val.sum()} | Test: {last_test_mask.sum()}')

    cfg = best['config']
    model, scaler, train_metrics = train_model(
        X_train[inner_train], y_train[inner_train],
        X_train[inner_val], y_train[inner_val],
        n_features, features,
        hidden_size=cfg['hidden_size'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        lr=cfg['lr'],
        batch_size=cfg['batch_size'],
        verbose=True,
        early_sl_train=es_train[inner_train] if es_train is not None else None,
        early_sl_val=es_train[inner_val] if es_train is not None else None,
        use_attention=args.use_attention,
    )

    if model is None:
        print('ERROR: Final training failed')
        sys.exit(1)

    metrics = evaluate_model(model, scaler, X_test, y_test, n_features)
    print(f'\n=== Final Test Results (last split test period) ===')
    print(f'  Accuracy:  {metrics["accuracy"]:.2%}')
    print(f'  Precision: {metrics["precision"]:.2%}')
    print(f'  Recall:    {metrics["recall"]:.2%}')
    print(f'  TP={metrics["tp"]} FP={metrics["fp"]} FN={metrics["fn"]} TN={metrics["tn"]}')

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'trading_model.h5')
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    save_model(model, scaler, features, metrics, train_metrics, args.seq_len,
               model_path, scaler_path)
    print(f'\nModel saved to: {model_path}')

    # Save sweep results for feedback
    sweep_results_path = os.path.join(args.output_dir, 'sweep_results.json')
    with open(sweep_results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'device': str(DEVICE),
            'n_configs': len(configs),
            'n_splits': len(splits),
            'best_config': best['config'],
            'best_avg_val_loss': best['avg_val_loss'],
            'test_metrics': metrics,
            'train_metrics': {k: v for k, v in train_metrics.items() if k != 'pos_weight'},
            'top_5': [{'config': r['config'], 'avg_val_loss': r['avg_val_loss']}
                      for r in results[:5]],
            'total_time_s': round(time.time() - t0, 1),
        }, f, indent=2)
    print(f'Sweep results: {sweep_results_path}')


if __name__ == '__main__':
    main()
