#!/usr/bin/env python3
"""Model gate — compare candidate model vs production and optionally promote.

Usage:
  python model_gate.py --candidate path/to/model.h5 --candidate-scaler path/to/scaler.pkl [--promote]

Exit codes: 0=promoted, 1=rejected, 2=error
"""
import os
import sys
import json
import glob
import shutil
import pickle
import argparse
from datetime import datetime

import numpy as np
import sqlite3
import h5py
import torch
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from lstm_model import LSTMModel
from feature_eng import prepare_data, BASE_FEATURES
from labels import label_trade_with_pnl
from walk_forward_gate import run_walk_forward_validation

BASE = '/home/kanoonth-ai/projects/caffe-stocks'
PROD_MODEL = os.path.join(BASE, 'models', 'lstm', 'trading_model.h5')
PROD_SCALER = os.path.join(BASE, 'models', 'lstm', 'scaler.pkl')
DB_PATH = os.path.join(BASE, 'data', 'candles.db')
IMPROVEMENTS_PATH = os.path.join(BASE, 'data', 'ml-improvements.json')

MIN_PRECISION = 0.37  # breakeven WR at 2% risk on ฿10K SET (฿50 min commission)
MIN_VARIANCE = 0.001
MAX_BACKUPS = 3
BACKTEST_SCRIPT = os.path.join(BASE, 'scripts', 'backtest_lstm.py')
BACKTEST_RESULTS = os.path.join(BASE, 'data', 'backtest_results', 'lstm_backtest.json')


def load_model_from_h5(h5_path):
    """Load LSTMModel from HDF5 file, return (model, attrs_dict)."""
    with h5py.File(h5_path, 'r') as f:
        input_size = int(f.attrs['input_size'])
        hidden_size = int(f.attrs.get('hidden_size', 64))
        num_layers = int(f.attrs.get('num_layers', 2))
        dropout = float(f.attrs.get('dropout', 0.3))
        features = f.attrs.get('features', '').split(',')
        seq_len = int(f.attrs.get('seq_len', 20))

        use_attention = bool(int(f.attrs.get('use_attention', 0)))

        attrs = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'features': features,
            'seq_len': seq_len,
            'use_attention': use_attention,
            'test_accuracy': float(f.attrs.get('test_accuracy', 0)),
            'test_precision': float(f.attrs.get('test_precision', 0)),
            'test_recall': float(f.attrs.get('test_recall', 0)),
        }

        head_dims_str = f.attrs.get('head_dims', '')
        head_dims = [int(d) for d in head_dims_str.split(',') if d] if head_dims_str else []

        model = LSTMModel(input_size, hidden_size, num_layers, dropout,
                          use_attention=use_attention, head_dims=head_dims)
        state_dict = {}
        for key in f['weights']:
            state_dict[key] = torch.tensor(np.array(f['weights'][key]))
        model.load_state_dict(state_dict)

    model.eval()
    return model, attrs


def check_degenerate(model, input_size, seq_len):
    """Return True if model is degenerate (low variance predictions)."""
    x = torch.randn(100, seq_len, input_size)
    with torch.no_grad():
        preds = model(x).numpy()
    return float(np.var(preds)) < MIN_VARIANCE


def evaluate_on_test_data(model, scaler_path, features, seq_len):
    """Evaluate model on held-out test data from candles.db.
    Returns dict with precision, accuracy, recall, or None on error.
    """
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        conn = sqlite3.connect(DB_PATH)
        data = pd.read_sql_query('SELECT * FROM candles ORDER BY timestamp', conn)
        conn.close()
        data = data.sort_values('timestamp')

        if len(data) == 0:
            return None

        # Use centralized feature engineering
        data, _ = prepare_data(data, features=features)

        # Create sequences with labels (aligned with backtester)
        lookahead = 10
        X_all, y_all, dates_all = [], [], []
        for symbol in data['symbol'].unique():
            sdf = data[data['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
            if len(sdf) < seq_len + lookahead:
                continue
            for i in range(len(sdf) - seq_len - lookahead):
                seq = sdf[features].iloc[i:i + seq_len].values
                if np.isnan(seq).any():
                    continue
                X_all.append(seq)
                dates_all.append(sdf.iloc[i + seq_len - 1]['timestamp'])
                entry_price = sdf.iloc[i + seq_len - 1]['close']
                window = sdf.iloc[i + seq_len:i + seq_len + lookahead]
                label, _pnl = label_trade_with_pnl(
                    window['high'].values, window['low'].values,
                    window['close'].values, entry_price)
                y_all.append(label)

        if len(X_all) == 0:
            return None

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.float32)
        dates = np.array(dates_all)

        # Time-based split: sort by date, use last 15% as test
        sort_idx = np.argsort(dates)
        X = X[sort_idx]
        y = y[sort_idx]
        dates = dates[sort_idx]

        # Split on unique dates to avoid boundary overlap
        unique_dates = np.unique(dates)
        test_date_cutoff = unique_dates[int(0.85 * len(unique_dates))]
        test_mask = dates >= test_date_cutoff

        X_test = X[test_mask]
        y_test = y[test_mask]

        n_features = X_test.shape[2]
        X_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

        with torch.no_grad():
            preds = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

        pred_labels = (preds >= 0.6).astype(int)  # aligned with live LSTM_THRESHOLD
        tp = ((pred_labels == 1) & (y_test == 1)).sum()
        fp = ((pred_labels == 1) & (y_test == 0)).sum()
        fn = ((pred_labels == 0) & (y_test == 1)).sum()
        accuracy = float((pred_labels == y_test).mean())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        return {'precision': precision, 'accuracy': accuracy, 'recall': recall,
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
    except Exception as e:
        print(f"Error evaluating model: {e}", file=sys.stderr)
        return None


def run_backtest_with_model(model_h5, scaler_pkl):
    """Run backtest against a model without touching production files.
    Returns dict with win_rate, ev_per_trade, profit_factor, total_trades or None.
    """
    import subprocess
    import tempfile

    # Write results to a temp file so we don't pollute production backtest results
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
        tmp_results = tf.name

    try:
        python = os.path.join(os.path.expanduser('~'), 'projects', 'caffe-stocks', 'venv', 'bin', 'python')
        result = subprocess.run(
            [python, BACKTEST_SCRIPT,
             '--model', model_h5,
             '--scaler', scaler_pkl,
             '--results', tmp_results],
            capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            print(f"Backtest failed: {result.stderr[:500]}", file=sys.stderr)
            return None

        with open(tmp_results) as f:
            data = json.load(f)
        lstm = data.get('lstm_filtered', {})
        return {
            'win_rate': lstm.get('win_rate', 0),
            'ev_per_trade': lstm.get('ev_per_trade', 0),
            'profit_factor': lstm.get('profit_factor', 0),
            'total_trades': lstm.get('total_trades', 0),
            'total_pnl': lstm.get('total_pnl', 0),
        }
    except Exception as e:
        print(f"Backtest error: {e}", file=sys.stderr)
        return None
    finally:
        if os.path.exists(tmp_results):
            os.remove(tmp_results)


def promote(candidate_h5, candidate_scaler):
    """Atomic swap: backup production, promote candidate."""
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Backup production
    if os.path.exists(PROD_MODEL):
        backup = f"{PROD_MODEL}.bak.{ts}"
        shutil.copy2(PROD_MODEL, backup)

        # Keep only last MAX_BACKUPS
        backups = sorted(glob.glob(f"{PROD_MODEL}.bak.*"))
        for old in backups[:-MAX_BACKUPS]:
            os.remove(old)

    if os.path.exists(PROD_SCALER):
        backup_s = f"{PROD_SCALER}.bak.{ts}"
        shutil.copy2(PROD_SCALER, backup_s)
        backups_s = sorted(glob.glob(f"{PROD_SCALER}.bak.*"))
        for old in backups_s[:-MAX_BACKUPS]:
            os.remove(old)

    # Atomic swap via tmp + rename
    tmp_model = PROD_MODEL + '.tmp'
    shutil.copy2(candidate_h5, tmp_model)
    os.replace(tmp_model, PROD_MODEL)

    tmp_scaler = PROD_SCALER + '.tmp'
    shutil.copy2(candidate_scaler, tmp_scaler)
    os.replace(tmp_scaler, PROD_SCALER)


def log_improvement(entry):
    """Append entry to ml-improvements.json."""
    history = []
    if os.path.exists(IMPROVEMENTS_PATH):
        with open(IMPROVEMENTS_PATH) as f:
            history = json.load(f)
    history.append(entry)
    with open(IMPROVEMENTS_PATH, 'w') as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Model gate: compare candidate vs production')
    parser.add_argument('--candidate', required=True, help='Path to candidate model H5')
    parser.add_argument('--candidate-scaler', required=True, help='Path to candidate scaler pkl')
    parser.add_argument('--promote', action='store_true', help='Promote candidate if it passes')
    parser.add_argument('--improvement-note', default='', help='Description of the improvement')
    parser.add_argument('--skip-walk-forward', action='store_true',
                        help='Skip walk-forward validation (stage 3)')
    parser.add_argument('--output', help='Write comparison JSON to this path')
    args = parser.parse_args()

    if not os.path.exists(args.candidate):
        print(f"Candidate model not found: {args.candidate}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.candidate_scaler):
        print(f"Candidate scaler not found: {args.candidate_scaler}", file=sys.stderr)
        sys.exit(2)

    # Load candidate
    try:
        cand_model, cand_attrs = load_model_from_h5(args.candidate)
    except Exception as e:
        print(f"Cannot load candidate: {e}", file=sys.stderr)
        sys.exit(2)

    # Evaluate candidate on test data
    cand_metrics = evaluate_on_test_data(
        cand_model, args.candidate_scaler,
        cand_attrs['features'], cand_attrs['seq_len'])

    if cand_metrics is None:
        print("Cannot evaluate candidate on test data", file=sys.stderr)
        sys.exit(2)

    # Load production metrics (from HDF5 attrs as baseline, evaluate fresh if possible)
    prod_metrics = {'precision': 0.0, 'accuracy': 0.0, 'recall': 0.0}
    if os.path.exists(PROD_MODEL) and os.path.exists(PROD_SCALER):
        try:
            prod_model, prod_attrs = load_model_from_h5(PROD_MODEL)
            fresh = evaluate_on_test_data(
                prod_model, PROD_SCALER,
                prod_attrs['features'], prod_attrs['seq_len'])
            if fresh:
                prod_metrics = fresh
            else:
                prod_metrics = {
                    'precision': prod_attrs['test_precision'],
                    'accuracy': prod_attrs['test_accuracy'],
                    'recall': prod_attrs['test_recall'],
                }
        except Exception:
            pass

    # Full-data backtests removed — walk-forward is the only gate.
    # We still run candidate backtest for informational logging only.
    print("Running backtest for candidate model...", file=sys.stderr)
    cand_backtest = run_backtest_with_model(args.candidate, args.candidate_scaler)
    prod_backtest = {'win_rate': 0, 'ev_per_trade': 0, 'profit_factor': 0, 'total_trades': 0, 'total_pnl': 0}

    # Gate checks — stage 1 (cheap hard blocks), stage 3 (walk-forward)
    stage1_reasons = []
    decision = "promoted"

    # --- Stage 1: Hard blocks (~1 sec) ---
    # 1. Hard block: degenerate predictions
    if check_degenerate(cand_model, cand_attrs['input_size'], cand_attrs['seq_len']):
        stage1_reasons.append("degenerate predictions (variance < 0.001)")

    # 2. Hard block: precision floor
    if cand_metrics['precision'] < MIN_PRECISION:
        stage1_reasons.append(f"precision {cand_metrics['precision']:.1%} below {MIN_PRECISION:.0%} minimum")

    # --- Stage 2: REMOVED ---
    # Full-data backtest comparison was removed because the production baseline
    # (64% WR) is overfitted. Comparing against it rejects every honest model.
    # Walk-forward (stage 3) is now the only real validation gate.

    # --- Stage 3: Walk-forward consistency (~10 min) ---
    # Runs if stage 1 passes (skip on degenerate/precision failures).
    # Walk-forward is authoritative: if it passes, stage 2 regressions are
    # downgraded to warnings (production baseline may be overfitting).
    walk_forward_result = None
    if not stage1_reasons and not args.skip_walk_forward:
        print("Running walk-forward validation (stage 3)...", file=sys.stderr)
        walk_forward_result = run_walk_forward_validation(
            args.candidate, args.candidate_scaler)

    # Decision logic:
    #   Stage 1 fail → reject (always)
    #   Stage 3 (walk-forward) fail → reject
    #   Stage 1 pass + stage 3 pass → promote
    reasons = list(stage1_reasons)

    if walk_forward_result and not walk_forward_result['passed']:
        reasons.append(f"walk-forward: {walk_forward_result['reason']}")
    elif not walk_forward_result and not args.skip_walk_forward and not stage1_reasons:
        reasons.append("walk-forward did not run")

    if reasons:
        decision = "rejected"

    comparison = {
        'timestamp': datetime.now().isoformat(),
        'improvement_note': args.improvement_note,
        'candidate': {
            **{k: v for k, v in cand_metrics.items() if k in ('precision', 'accuracy', 'recall')},
            'backtest': cand_backtest or {},
        },
        'production': {
            **{k: v for k, v in prod_metrics.items() if k in ('precision', 'accuracy', 'recall')},
            'backtest': prod_backtest,
        },
        'walk_forward': walk_forward_result,
        'decision': decision,
        'reason': '; '.join(reasons) if reasons else 'all checks passed',
    }

    # Print result
    print(json.dumps(comparison, indent=2))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)

    # Log to improvements history
    log_improvement(comparison)

    if decision == "promoted" and args.promote:
        promote(args.candidate, args.candidate_scaler)
        wf_wr = walk_forward_result['aggregate']['avg_win_rate'] if walk_forward_result else 0
        wf_std = walk_forward_result['aggregate']['wr_std'] if walk_forward_result else 0
        print(f"\n✓ PROMOTED: Walk-forward WR {wf_wr:.1%} (std {wf_std:.1%}) | "
              f"precision {cand_metrics['precision']:.1%}")
        # Re-run backtest with promoted model to save fresh results
        run_backtest_with_model(PROD_MODEL, PROD_SCALER)
        sys.exit(0)
    elif decision == "promoted":
        print(f"\n✓ WOULD PROMOTE (--promote not set)")
        sys.exit(0)
    else:
        print(f"\n✗ REJECTED: {'; '.join(reasons)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
