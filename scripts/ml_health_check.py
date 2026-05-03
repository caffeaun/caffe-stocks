#!/usr/bin/env python3
"""ML pipeline smoke test — checks model, scaler, features, and predictions.
Run daily after feature refresh. Outputs JSON with per-check pass/fail.
Exit 0 = all pass, exit 1 = at least one failure.
"""
import os
import sys
import json
import pickle
from datetime import datetime, timedelta

import numpy as np
import sqlite3
import h5py
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from lstm_model import LSTMModel
from feature_eng import BASE_FEATURES

BASE = '/home/kanoonth-ai/projects/caffe-stocks'
MODEL_PATH = os.path.join(BASE, 'models', 'lstm', 'trading_model.h5')
SCALER_PATH = os.path.join(BASE, 'models', 'lstm', 'scaler.pkl')
DB_PATH = os.path.join(BASE, 'data', 'candles.db')
STATUS_PATH = os.path.join(BASE, 'data', 'system-status.json')
STALENESS_DAYS = 14


def check_model_exists():
    """Model HDF5 exists and is loadable."""
    if not os.path.exists(MODEL_PATH):
        return False, "Model file not found"
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            if 'weights' not in f:
                return False, "No 'weights' group in HDF5"
            if 'input_size' not in f.attrs:
                return False, "No 'input_size' attribute"
        return True, "OK"
    except Exception as e:
        return False, f"Cannot open HDF5: {e}"


def check_scaler_exists():
    """Scaler pickle exists and has n_features_in_."""
    if not os.path.exists(SCALER_PATH):
        return False, "Scaler file not found"
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        if not hasattr(scaler, 'n_features_in_'):
            return False, "Scaler missing n_features_in_"
        return True, f"OK ({scaler.n_features_in_} features)"
    except Exception as e:
        return False, f"Cannot load scaler: {e}"


def check_feature_match():
    """Scaler n_features_in_ matches HDF5 input_size."""
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            input_size = int(f.attrs['input_size'])
        with open(SCALER_PATH, 'rb') as fp:
            scaler = pickle.load(fp)
        if scaler.n_features_in_ != input_size:
            return False, f"Mismatch: scaler={scaler.n_features_in_}, model={input_size}"
        return True, f"OK ({input_size} features)"
    except Exception as e:
        return False, f"Error: {e}"


def check_feature_names():
    """Feature names in HDF5 match columns in candles table."""
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            features_str = f.attrs.get('features', '')
        if not features_str:
            return False, "No 'features' attribute in HDF5"
        model_features = features_str.split(',')

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute('SELECT * FROM candles LIMIT 1')
        db_columns = [desc[0] for desc in cursor.description]
        conn.close()

        # Base features should be in candles (derived ones are computed at runtime)
        missing = [f for f in BASE_FEATURES if f in model_features and f not in db_columns]
        if missing:
            return False, f"Missing from candles: {missing}"
        return True, f"OK ({len(model_features)} features)"
    except Exception as e:
        return False, f"Error: {e}"


def check_smoke_prediction():
    """Run 1 prediction on latest data, verify output is valid probability."""
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            input_size = int(f.attrs['input_size'])
            hidden_size = int(f.attrs.get('hidden_size', 64))
            num_layers = int(f.attrs.get('num_layers', 2))
            dropout = float(f.attrs.get('dropout', 0.3))
            seq_len = int(f.attrs.get('seq_len', 20))

            use_attention = bool(int(f.attrs.get('use_attention', 0)))
            head_dims_str = f.attrs.get('head_dims', '')
            head_dims = [int(d) for d in head_dims_str.split(',') if d] if head_dims_str else []
            model = LSTMModel(input_size, hidden_size, num_layers, dropout,
                              use_attention=use_attention, head_dims=head_dims)
            state_dict = {}
            for key in f['weights']:
                state_dict[key] = torch.tensor(np.array(f['weights'][key]))
            model.load_state_dict(state_dict)
        model.eval()

        # Random input for smoke test
        x = torch.randn(1, seq_len, input_size)
        with torch.no_grad():
            pred = model(x).item()

        if not (0.0 <= pred <= 1.0):
            return False, f"Prediction {pred} outside [0, 1]"
        return True, f"OK (pred={pred:.4f})"
    except Exception as e:
        return False, f"Error: {e}"


def check_degenerate():
    """100 random sequences — prediction variance > 0.001."""
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            input_size = int(f.attrs['input_size'])
            hidden_size = int(f.attrs.get('hidden_size', 64))
            num_layers = int(f.attrs.get('num_layers', 2))
            dropout = float(f.attrs.get('dropout', 0.3))
            seq_len = int(f.attrs.get('seq_len', 20))

            use_attention = bool(int(f.attrs.get('use_attention', 0)))
            head_dims_str = f.attrs.get('head_dims', '')
            head_dims = [int(d) for d in head_dims_str.split(',') if d] if head_dims_str else []
            model = LSTMModel(input_size, hidden_size, num_layers, dropout,
                              use_attention=use_attention, head_dims=head_dims)
            state_dict = {}
            for key in f['weights']:
                state_dict[key] = torch.tensor(np.array(f['weights'][key]))
            model.load_state_dict(state_dict)
        model.eval()

        x = torch.randn(100, seq_len, input_size)
        with torch.no_grad():
            preds = model(x).numpy()

        var = float(np.var(preds))
        if var < 0.001:
            return False, f"Degenerate: variance={var:.6f} (all predictions ~{np.mean(preds):.4f})"
        return True, f"OK (variance={var:.4f})"
    except Exception as e:
        return False, f"Error: {e}"


def check_staleness():
    """trained_at attribute within STALENESS_DAYS days."""
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            trained_at = f.attrs.get('trained_at', '')
        if not trained_at:
            return False, "No 'trained_at' attribute"
        trained_dt = datetime.fromisoformat(trained_at)
        age = datetime.now() - trained_dt
        if age > timedelta(days=STALENESS_DAYS):
            return False, f"Stale: trained {age.days} days ago ({trained_at})"
        return True, f"OK (trained {age.days}d ago)"
    except Exception as e:
        return False, f"Error: {e}"


def check_status_coherence():
    """If status=active, model should load. If status=retrain, flag it."""
    try:
        if not os.path.exists(STATUS_PATH):
            return True, "OK (no status file)"
        with open(STATUS_PATH) as f:
            status = json.load(f).get('status', 'unknown')
        if status == 'retrain':
            return False, "Status is 'retrain' — model may be stale"
        if status == 'active' and not os.path.exists(MODEL_PATH):
            return False, "Status is 'active' but model file missing"
        return True, f"OK (status={status})"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    skip_staleness = '--skip-staleness' in sys.argv

    checks = [
        ("model_exists", check_model_exists),
        ("scaler_exists", check_scaler_exists),
        ("feature_match", check_feature_match),
        ("feature_names", check_feature_names),
        ("smoke_prediction", check_smoke_prediction),
        ("degenerate", check_degenerate),
        ("staleness", check_staleness),
        ("status_coherence", check_status_coherence),
    ]

    results = {}
    all_pass = True
    for name, fn in checks:
        passed, detail = fn()
        if not passed and skip_staleness and name == 'staleness':
            detail += ' (skipped for improvement run)'
            passed = True
        results[name] = {"pass": passed, "detail": detail}
        if not passed:
            all_pass = False

    # Output JSON
    output = {"timestamp": datetime.now().isoformat(), "all_pass": all_pass, "checks": results}
    print(json.dumps(output, indent=2))

    # Print human-readable summary on failure
    if not all_pass:
        print("\n--- FAILURES ---", file=sys.stderr)
        for name, r in results.items():
            if not r["pass"]:
                print(f"  {name}: {r['detail']}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
