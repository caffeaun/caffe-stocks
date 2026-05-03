#!/usr/bin/env python3
"""LSTM trainer for trading signals.

Can be used standalone (CLI) or imported by lstm_sweep.py.
Key functions: load_sequences(), time_split(), train_model(), save_model().
"""
import os
import sys
import json
import hashlib
import argparse
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
import sqlite3
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from lstm_model import LSTMModel
from feature_eng import prepare_data, BASE_FEATURES
from labels import label_trade, label_trade_with_pnl, check_early_stop, STOP_PCT, TARGET_PCT, TRAILING_TRIGGER, TRAILING_FLOOR, MAX_HOLD, MIN_PROFIT_PCT

BASE_PATH = '/home/kanoonth-ai/projects/caffe-stocks'
DB_PATH = os.path.join(BASE_PATH, 'data', 'candles.db')

# Auto-detect device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_sequences(seq_len=20, lookahead=10):
    """Load data, build sequences with labels, return (X, y, dates, early_sl, features).
    Uses disk cache — rebuilds only when candles.db or params change.
    early_sl: boolean array, True if trade hits stop loss within first 3 days.
    """
    conn = sqlite3.connect(DB_PATH)
    data = pd.read_sql_query('SELECT * FROM candles ORDER BY timestamp', conn)
    conn.close()
    data = data.sort_values('timestamp')

    if len(data) == 0:
        raise RuntimeError('candles table is empty — run compute_indicators.py first')

    missing = [f for f in BASE_FEATURES if f not in data.columns]
    if missing:
        raise RuntimeError(f'Missing features: {missing}')

    data, features = prepare_data(data)
    n_features = len(features)

    # Cache key (v3: includes MIN_PROFIT_PCT for strict-win label)
    cache_key_parts = [
        str(os.path.getmtime(DB_PATH)),
        ','.join(features),
        f'{seq_len},{lookahead}',
        f'{STOP_PCT},{TARGET_PCT},{TRAILING_TRIGGER},{TRAILING_FLOOR},{MAX_HOLD}',
        f'min_profit={MIN_PROFIT_PCT}',
        'v3_strict_win',
    ]
    cache_hash = hashlib.md5('|'.join(cache_key_parts).encode()).hexdigest()[:12]
    cache_path = os.path.join(BASE_PATH, 'data', f'sequences_cache_{cache_hash}.npz')

    if os.path.exists(cache_path):
        print(f'Loading cached sequences: {cache_path}')
        cached = np.load(cache_path, allow_pickle=True)
        pnl = cached['pnl'] if 'pnl' in cached else None
        return cached['X'], cached['y'], cached['dates'], cached['early_sl'], features, pnl

    # Build sequences
    X_all, y_all, pnl_all, dates_all, early_sl_all = [], [], [], [], []

    for symbol in data['symbol'].unique():
        sdf = data[data['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
        if len(sdf) < seq_len + lookahead:
            continue

        for i in range(len(sdf) - seq_len - lookahead):
            seq = sdf[features].iloc[i:i+seq_len].values
            if np.isnan(seq).any():
                continue
            X_all.append(seq)
            dates_all.append(sdf.iloc[i+seq_len-1]['timestamp'])

            entry_price = sdf.iloc[i+seq_len-1]['close']
            window = sdf.iloc[i+seq_len:i+seq_len+lookahead]
            label, pnl = label_trade_with_pnl(
                window['high'].values, window['low'].values,
                window['close'].values, entry_price)
            # Strict-win label: 1 only if net P&L exceeds MIN_PROFIT_PCT (decisive win)
            y_all.append(1 if pnl > MIN_PROFIT_PCT else 0)
            pnl_all.append(pnl)

            # Track early stop loss (first 3 days)
            early_sl = check_early_stop(window['low'].values, entry_price)
            early_sl_all.append(early_sl)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    pnl = np.array(pnl_all, dtype=np.float32)
    dates = np.array(dates_all)
    early_sl = np.array(early_sl_all, dtype=np.float32)

    np.savez(cache_path, X=X, y=y, pnl=pnl, dates=dates, early_sl=early_sl)
    # Clean old caches (keep last 2)
    import glob as globmod
    for old in sorted(globmod.glob(os.path.join(BASE_PATH, 'data', 'sequences_cache_*.npz')))[:-2]:
        os.remove(old)
    print(f'Sequences cached: {cache_path}')

    return X, y, dates, early_sl, features, pnl


def time_split(X, y, dates, early_sl=None, train_pct=0.70, val_pct=0.85):
    """Sort by date, split on unique date boundaries. Returns dict of arrays."""
    sort_idx = np.argsort(dates)
    X, y, dates = X[sort_idx], y[sort_idx], dates[sort_idx]
    if early_sl is not None:
        early_sl = early_sl[sort_idx]

    unique_dates = np.unique(dates)
    train_cutoff = unique_dates[int(train_pct * len(unique_dates))]
    val_cutoff = unique_dates[int(val_pct * len(unique_dates))]

    train_mask = dates < train_cutoff
    val_mask = (dates >= train_cutoff) & (dates < val_cutoff)
    test_mask = dates >= val_cutoff

    splits = {
        'X_train': X[train_mask], 'X_val': X[val_mask], 'X_test': X[test_mask],
        'y_train': y[train_mask], 'y_val': y[val_mask], 'y_test': y[test_mask],
        'train_dates': dates[train_mask], 'val_dates': dates[val_mask], 'test_dates': dates[test_mask],
    }

    if early_sl is not None:
        splits['early_sl_train'] = early_sl[train_mask]
        splits['early_sl_val'] = early_sl[val_mask]
        splits['early_sl_test'] = early_sl[test_mask]

    assert max(splits['train_dates']) < min(splits['val_dates']), "Data leakage: train/val overlap"
    assert max(splits['val_dates']) < min(splits['test_dates']), "Data leakage: val/test overlap"
    return splits


def train_model(X_train, y_train, X_val, y_val, n_features, features,
                hidden_size=48, num_layers=1, dropout=0.4, lr=5e-4,
                batch_size=256, max_epochs=200, patience=20, verbose=True,
                early_sl_train=None, early_sl_val=None,
                survival_weight=1.0, fp_weight=1.0,
                use_attention=False,
                sequence_normalize=False,
                mixup_alpha=0.3,
                swa_window=5):
    """Train LSTM model on GPU/CPU. Returns (model, scaler, metrics_dict).

    Args:
        early_sl_train/val: boolean arrays marking early stop-loss samples (for survival weighting)
        survival_weight: multiplier for early-SL negative samples (default 3.0)
        fp_weight: multiplier for false-positive errors (default 2.5, asymmetric loss)
        use_attention: enable temporal attention over LSTM timesteps
        sequence_normalize: per-window z-score normalization (eliminates scaler drift)

    The model is returned on CPU for saving/inference compatibility.
    """
    device = DEVICE

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)

    # Post-scaling validation
    if verbose:
        for i, feat in enumerate(features):
            col = X_train_scaled.reshape(-1, n_features)[:, i]
            p5, p95 = np.percentile(col, [5, 95])
            if (p95 - p5) < 0.05:
                print(f"  WARNING: '{feat}' compressed after scaling (5-95% range = {p95-p5:.4f})")

    # Fixed seed: ensures consistent LSTM initialization across walk-forward splits.
    # With differential LR (backbone at 1/50 of head), the LSTM barely moves from
    # initialization — so the seed determines the feature projection. Fixed seed
    # eliminates random variance between splits, isolating data-driven differences.
    torch.manual_seed(42)
    np.random.seed(42)

    model = LSTMModel(input_size=n_features, hidden_size=hidden_size,
                       num_layers=num_layers, dropout=dropout,
                       use_attention=use_attention,
                       sequence_normalize=sequence_normalize).to(device)

    # Moderate pos_weight in loss (replaces aggressive WeightedRandomSampler).
    # Prior sampler-balanced approach (attempt 261) destroyed score calibration:
    # with strict labels (~10-13% positive rate) the sampler oversamples positives
    # ~8-10x, shifting the sigmoid output distribution upward. Applying the live
    # 0.6 threshold on a 50/50-trained model produced 0 trades or extremely few
    # trades with near-random WR (scores compressed around 0.5). Using pos_weight
    # in the loss (not in sampling) keeps mini-batch distribution matching the
    # test distribution — so sigmoid outputs stay calibrated — while still giving
    # positives enough gradient weight to avoid majority-class collapse in small
    # splits. sqrt() dampens the upweight so even split 1 with 8% positives gets
    # only ~3.4x weight (vs 11x for raw 1/p), avoiding over-correction.
    pos_rate = y_train.mean()
    if pos_rate == 0 or pos_rate == 1:
        return None, None, {'error': 'degenerate labels'}

    # Full class balance (no sqrt dampening, cap at 10x to avoid extreme values).
    # With sqrt + 3.5 cap the effective balance was only ~2.9x on an 11% positive
    # rate, leaving mean logit at -1.0 which after the lstm_model output_temp=0.5
    # sharpening becomes score 0.12 — well below the 0.6 live threshold. Raw
    # (1-p)/p ~= 8.3x aligns positive/negative gradient mass 50/50 so the model
    # actually commits positive logits for genuine high-signal patterns rather
    # than getting pulled down by the 89% negative class.
    pos_weight = float(min((1.0 - pos_rate) / max(pos_rate, 1e-6), 10.0))

    if verbose:
        print(f'  Class-weighted BCE: pos_rate={pos_rate:.3f}, pos_weight={pos_weight:.2f}')

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=(device.type == 'cuda'))

    # Entry #197 baseline config: weight_decay=0 (pure BCE, no L2).
    # Prior 1e-3 L2 combined with small batches starved the LSTM of gradient
    # signal and drove epoch-0-best patterns. #197 showed the tiny model
    # (hidden=48, layers=1, dropout=0.4) with no L2 was sufficient regularization.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.BCELoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    # SWA-style weight averaging: keep a ring buffer of the last N state_dicts
    # from epochs where val_loss improved. At end, average them to smooth out
    # late-epoch noise. #134 got 48% 3-split WR with SWA; pairing with the
    # stable #197 baseline should translate to 7-split without score collapse.
    swa_states = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Mixup augmentation (#197's key ingredient): convex combinations of
            # two batches with beta-distributed weight. Smooths the decision
            # boundary and prevents memorization of specific (feature, label)
            # pairs — the epoch-0-best pattern that has plagued every attempt.
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                # Ensure lam is on the informative side (>= 0.5) so the
                # "primary" sample dominates — avoids extreme label mixing.
                lam = max(lam, 1.0 - lam)
                perm = torch.randperm(X_batch.size(0), device=device)
                X_mix = lam * X_batch + (1 - lam) * X_batch[perm]
                y_mix = lam * y_batch + (1 - lam) * y_batch[perm]
            else:
                X_mix, y_mix = X_batch, y_batch

            optimizer.zero_grad()
            pred = model(X_mix)
            # Per-sample class weighting: the pos_weight computed above was
            # logged but never applied to the loss — leaving training as pure
            # unweighted BCE on a ~11% positive rate, which collapses scores
            # below the 0.6 threshold (all predictions near-zero, 0 valid
            # splits in attempts #272-#276). Weight each sample by its soft
            # label: positives get pos_weight, negatives get 1.0, mixup-blended
            # targets get the linear interpolation. This preserves mini-batch
            # distribution (no resampling) but gives positive gradients enough
            # pull to escape the majority-class local minimum.
            sample_weight = y_mix * pos_weight + (1.0 - y_mix)
            loss = (criterion(pred, y_mix) * sample_weight).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).mean().item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # SWA: append a snapshot every time val_loss improves; keep last N.
            swa_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
            if len(swa_states) > swa_window:
                swa_states.pop(0)
        else:
            patience_counter += 1

        if verbose and epoch % 10 == 0:
            print(f'  Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f} val_loss={val_loss:.4f}')

        if patience_counter >= patience:
            if verbose:
                print(f'  Early stopping at epoch {epoch}')
            break

    # Restore best model — prefer SWA average if we collected >= 2 snapshots,
    # else fall back to single best_state. Only average float tensors; buffers
    # like _output_temp and _seq_normalize are copied from best_state as-is.
    if len(swa_states) >= 2:
        avg_state = {}
        for key in swa_states[0]:
            t0 = swa_states[0][key]
            if t0.dtype.is_floating_point:
                stacked = torch.stack([s[key].float() for s in swa_states], dim=0)
                avg_state[key] = stacked.mean(dim=0).to(t0.dtype)
            else:
                avg_state[key] = best_state[key].clone()
        model.cpu()
        model.load_state_dict(avg_state)
        if verbose:
            print(f'  SWA: averaged {len(swa_states)} late-epoch checkpoints')
    elif best_state:
        model.cpu()
        model.load_state_dict(best_state)
    else:
        model.cpu()

    return model, scaler, {
        'best_val_loss': best_val_loss,
        'epochs': epoch + 1,
        'pos_weight': pos_weight,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'lr': lr,
        'use_attention': use_attention,
        'survival_weight': survival_weight,
        'fp_weight': fp_weight,
        'sequence_normalize': sequence_normalize,
        'mixup_alpha': mixup_alpha,
        'swa_snapshots': len(swa_states),
    }


def evaluate_model(model, scaler, X_test, y_test, n_features):
    """Evaluate model on test set. Returns metrics dict."""
    X_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

    pred_labels = (preds >= 0.5).astype(int)
    tp = ((pred_labels == 1) & (y_test == 1)).sum()
    fp = ((pred_labels == 1) & (y_test == 0)).sum()
    fn = ((pred_labels == 0) & (y_test == 1)).sum()
    tn = ((pred_labels == 0) & (y_test == 0)).sum()

    return {
        'accuracy': float((pred_labels == y_test).mean()),
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'positive_rate': float(y_test.mean()),
    }


def save_model(model, scaler, features, metrics, train_metrics, seq_len,
               model_path, scaler_path):
    """Atomic save of model HDF5 + scaler pickle."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    tmp_model = model_path + '.tmp'
    with h5py.File(tmp_model, 'w') as f:
        f.attrs['model_type'] = 'pytorch_lstm'
        f.attrs['input_size'] = len(features)
        f.attrs['hidden_size'] = train_metrics.get('hidden_size', 128)
        f.attrs['num_layers'] = train_metrics.get('num_layers', 2)
        f.attrs['dropout'] = train_metrics.get('dropout', 0.3)
        f.attrs['seq_len'] = seq_len
        f.attrs['features'] = ','.join(features)
        f.attrs['test_accuracy'] = metrics['accuracy']
        f.attrs['test_precision'] = metrics['precision']
        f.attrs['test_recall'] = metrics['recall']
        f.attrs['use_attention'] = int(train_metrics.get('use_attention', False))
        f.attrs['survival_weight'] = train_metrics.get('survival_weight', 1.0)
        f.attrs['fp_weight'] = train_metrics.get('fp_weight', 1.0)
        f.attrs['sequence_normalize'] = int(train_metrics.get('sequence_normalize', False))
        f.attrs['trained_at'] = datetime.now().isoformat()
        weights_grp = f.create_group('weights')
        for key, tensor in model.state_dict().items():
            weights_grp.create_dataset(key, data=tensor.cpu().numpy())
    os.replace(tmp_model, model_path)

    tmp_scaler = scaler_path + '.tmp'
    with open(tmp_scaler, 'wb') as f:
        pickle.dump(scaler, f)
    os.replace(tmp_scaler, scaler_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=os.path.join(BASE_PATH, 'models', 'lstm'))
    parser.add_argument('--hidden-size', type=int, default=48)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--use-attention', action='store_true', help='Enable temporal attention')
    parser.add_argument('--survival-weight', type=float, default=1.0, help='Weight for early-SL samples')
    parser.add_argument('--fp-weight', type=float, default=1.0, help='Weight for false-positive errors')
    args = parser.parse_args()

    print(f'Device: {DEVICE}')

    # Load data
    X, y, dates, early_sl, features, _pnl = load_sequences(seq_len=args.seq_len)
    n_features = len(features)
    print(f'Available features auto-detected: {n_features}')
    print(f'Features ({n_features}): {features}')
    print(f'Total samples: {len(X)} | Positive: {y.sum():.0f} ({y.mean():.1%})')
    print(f'Early stop-loss samples: {int(early_sl.sum())} ({early_sl.mean():.1%})')

    # Split
    splits = time_split(X, y, dates, early_sl=early_sl)
    print(f'  Train: {len(splits["X_train"])} samples, {min(splits["train_dates"])} to {max(splits["train_dates"])}')
    print(f'  Val:   {len(splits["X_val"])} samples, {min(splits["val_dates"])} to {max(splits["val_dates"])}')
    print(f'  Test:  {len(splits["X_test"])} samples, {min(splits["test_dates"])} to {max(splits["test_dates"])}')

    # Train
    # Entry #197 baseline restored: Mixup alpha=0.3 + pure BCE + hidden=48/layers=1
    # + dropout=0.4 + batch=256 + lr=5e-4 + weight_decay=0. sequence_normalize=False
    # (not part of #197 — adding it on top caused score collapse in recent attempts).
    # Additional change: SWA-style averaging of last 5 best-val snapshots to
    # smooth late-epoch noise (#134 partial success: 48% 3-split WR).
    model, scaler, train_metrics = train_model(
        splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'],
        n_features, features,
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, lr=args.lr, batch_size=args.batch_size,
        early_sl_train=splits.get('early_sl_train'),
        early_sl_val=splits.get('early_sl_val'),
        survival_weight=args.survival_weight,
        fp_weight=args.fp_weight,
        use_attention=args.use_attention,
        sequence_normalize=False,
        mixup_alpha=0.3,
        swa_window=5)

    if model is None:
        raise RuntimeError(f'Training failed: {train_metrics}')

    print(f'Positive weight: {train_metrics["pos_weight"]:.1f}x')

    # Evaluate
    metrics = evaluate_model(model, scaler, splits['X_test'], splits['y_test'], n_features)
    print(f'\n=== Test Results ===')
    print(f'  Accuracy:  {metrics["accuracy"]:.2%}')
    print(f'  Precision: {metrics["precision"]:.2%}')
    print(f'  Recall:    {metrics["recall"]:.2%}')
    print(f'  TP={metrics["tp"]} FP={metrics["fp"]} FN={metrics["fn"]} TN={metrics["tn"]}')
    print(f'  Positive rate: {metrics["positive_rate"]:.1%}')

    # Save
    model_path = os.path.join(args.output_dir, 'trading_model.h5')
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    save_model(model, scaler, features, metrics, train_metrics, args.seq_len,
               model_path, scaler_path)

    print(f'\nModel saved to: {model_path}')
    print(f'Scaler saved to: {scaler_path}')
    print(f'  Train: {len(splits["X_train"])} | Val: {len(splits["X_val"])} | Test: {len(splits["X_test"])}')

    # Set system status
    status_file = os.path.join(BASE_PATH, 'data', 'system-status.json')
    with open(status_file, 'w') as f:
        json.dump({'status': 'active'}, f)
    print('System status set to: active')

    retrain_file = os.path.join(BASE_PATH, 'models', 'last_retrain.md')
    with open(retrain_file, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d'))
    print(f'Last retrain updated: {datetime.now().strftime("%Y-%m-%d")}')


if __name__ == '__main__':
    main()
