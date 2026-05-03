#!/usr/bin/env python3
"""Backtest the production LSTM model against historical candle data.

Compares three strategies:
  1. Rule-based only (ATR + volume + RSI filters)
  2. LSTM-filtered (rule-based + LSTM score > threshold)
  3. Buy-and-hold baseline

Uses SET-specific friction: commission 0.1578% + VAT 7%, slippage 0.2%.
"""
import os
import sys
import json
import sqlite3
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from lstm_model import LSTMModel
from feature_eng import prepare_data

BASE = '/home/kanoonth-ai/projects/caffe-stocks'
DB_PATH = os.path.join(BASE, 'data', 'candles.db')
MODEL_PATH = os.path.join(BASE, 'models', 'lstm', 'trading_model.h5')
SCALER_PATH = os.path.join(BASE, 'models', 'lstm', 'scaler.pkl')
RESULTS_DIR = os.path.join(BASE, 'data', 'backtest_results')

# SET-specific friction
COMMISSION_RATE = 0.001578
VAT_RATE = 0.07
COMMISSION_MIN = 50
SLIPPAGE_RATE = 0.002

# Strategy parameters
PORTFOLIO_SIZE = 1_000_000  # 1M THB
RISK_PER_TRADE = 0.01       # 1% risk per trade
STOP_LOSS_PCT = 0.03        # 3% stop loss
TARGET_PCT = 0.15           # 15% target (changed from 0.05 labeling to match backtester)
TRAILING_TRIGGER = 0.07     # 7% gain activates trailing stop
TRAILING_FLOOR = 0.50       # keep 50% of peak gain
MAX_HOLD_DAYS = 10
LSTM_THRESHOLD = 0.6        # same as lstm_trader.py MIN_LSTM


def load_model(model_path=None, scaler_path=None):
    """Load LSTM model + scaler. Defaults to production paths."""
    model_path = model_path or MODEL_PATH
    scaler_path = scaler_path or SCALER_PATH
    with h5py.File(model_path, 'r') as f:
        input_size = int(f.attrs['input_size'])
        hidden_size = int(f.attrs.get('hidden_size', 64))
        num_layers = int(f.attrs.get('num_layers', 2))
        dropout = float(f.attrs.get('dropout', 0.3))
        seq_len = int(f.attrs.get('seq_len', 20))
        use_attention = bool(int(f.attrs.get('use_attention', 0)))
        head_dims_str = f.attrs.get('head_dims', '')
        head_dims = [int(d) for d in head_dims_str.split(',') if d] if head_dims_str else []
        features = f.attrs['features'].split(',')

        model = LSTMModel(input_size, hidden_size, num_layers, dropout,
                          use_attention=use_attention, head_dims=head_dims)
        state_dict = {}
        for key in f['weights']:
            state_dict[key] = torch.tensor(np.array(f['weights'][key]))
        model.load_state_dict(state_dict)

    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler, features, seq_len


def load_data():
    """Load candle data with derived features."""
    conn = sqlite3.connect(DB_PATH)
    data = pd.read_sql_query('SELECT * FROM candles ORDER BY timestamp', conn)
    conn.close()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(['symbol', 'timestamp'])

    # Use centralized feature engineering (auto-detects available features)
    data, _ = prepare_data(data)
    return data


def compute_lstm_scores(data, model, scaler, features, seq_len):
    """Score every row with LSTM probability."""
    scores = pd.Series(np.nan, index=data.index)

    for symbol in data['symbol'].unique():
        sdf = data[data['symbol'] == symbol].sort_values('timestamp')
        feat_data = sdf[features].values

        if len(feat_data) < seq_len:
            continue

        sequences = []
        indices = []
        for i in range(len(feat_data) - seq_len + 1):
            seq = feat_data[i:i + seq_len]
            if np.isnan(seq).any():
                continue
            sequences.append(seq)
            indices.append(sdf.index[i + seq_len - 1])

        if not sequences:
            continue

        X = np.array(sequences, dtype=np.float32)
        n_feat = X.shape[2]
        X_scaled = scaler.transform(X.reshape(-1, n_feat)).reshape(X.shape)

        with torch.no_grad():
            preds = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

        for idx, score in zip(indices, preds):
            scores.at[idx] = float(score)

    return scores


def rule_filter(row):
    """Standard rule-based filter."""
    if row['atr'] <= 0.03:
        return False
    if row['volume_ratio'] <= 1.5:
        return False
    if not (30 <= row['rsi'] <= 65):
        return False
    if not (row['close'] > row.get('20_day_high', 0) or row['macd'] > 0):
        return False
    return True


def simulate_trade(df, start_idx, entry_row):
    """Simulate a single trade with SET friction."""
    entry_price = entry_row['close'] * (1 + SLIPPAGE_RATE)
    risk_amount = PORTFOLIO_SIZE * RISK_PER_TRADE
    quantity = risk_amount / (STOP_LOSS_PCT * entry_price)
    trade_value = entry_price * quantity

    stop_loss = entry_price * (1 - STOP_LOSS_PCT)
    target = entry_price * (1 + TARGET_PCT)
    peak_gain = 0.0

    exit_price = None
    exit_date = None
    exit_reason = None
    hold_days = 0

    # Walk forward from entry
    symbol_df = df[(df['symbol'] == entry_row['symbol']) &
                   (df['timestamp'] > entry_row['timestamp'])].sort_values('timestamp')

    for _, row in symbol_df.iterrows():
        hold_days += 1

        if row['low'] <= stop_loss:
            exit_price = stop_loss
            exit_date = row['timestamp']
            exit_reason = 'stop_loss'
            break

        if row['high'] >= target:
            exit_price = target
            exit_date = row['timestamp']
            exit_reason = 'target'
            break

        current_gain = (row['close'] - entry_price) / entry_price
        if current_gain >= TRAILING_TRIGGER:
            peak_gain = max(peak_gain, current_gain)

        if peak_gain >= TRAILING_TRIGGER:
            trail_floor = peak_gain * TRAILING_FLOOR
            if current_gain < trail_floor:
                exit_price = row['close']
                exit_date = row['timestamp']
                exit_reason = 'trailing_stop'
                break

        if hold_days >= MAX_HOLD_DAYS:
            exit_price = row['close']
            exit_date = row['timestamp']
            exit_reason = 'max_hold'
            break

    if exit_price is None:
        return None  # no exit found (end of data)

    exit_value = exit_price * quantity
    commission = max(COMMISSION_MIN, trade_value * COMMISSION_RATE * (1 + VAT_RATE))
    commission += max(COMMISSION_MIN, exit_value * COMMISSION_RATE * (1 + VAT_RATE))
    slippage = trade_value * SLIPPAGE_RATE

    pnl = (exit_value - trade_value) - commission - slippage
    pnl_pct = (pnl / trade_value) * 100

    return {
        'symbol': entry_row['symbol'],
        'entry_date': str(entry_row['timestamp'].date()),
        'exit_date': str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
        'entry_price': round(entry_price, 2),
        'exit_price': round(exit_price, 2),
        'hold_days': hold_days,
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 2),
        'exit_reason': exit_reason,
    }


def run_backtest(data, signals_mask, label):
    """Run backtest on filtered signals, return trades list and summary."""
    trades = []
    # Track open positions per symbol to avoid overlapping
    last_exit = {}

    signal_rows = data[signals_mask].sort_values('timestamp')

    for _, row in signal_rows.iterrows():
        sym = row['symbol']
        # Skip if we're still in a position for this symbol
        if sym in last_exit and row['timestamp'] <= last_exit[sym]:
            continue

        trade = simulate_trade(data, None, row)
        if trade is None:
            continue

        trades.append(trade)
        last_exit[sym] = pd.Timestamp(trade['exit_date'])

    if not trades:
        return trades, {'label': label, 'total_trades': 0}

    pnls = [t['pnl_pct'] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    win_rate = len(winners) / len(trades)
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0
    total_pnl = sum(t['pnl'] for t in trades)
    ev = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Max drawdown (cumulative)
    cumulative = np.cumsum([t['pnl'] for t in trades])
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

    # Profit factor
    gross_win = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        r = t['exit_reason']
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    summary = {
        'label': label,
        'total_trades': len(trades),
        'win_rate': round(win_rate, 4),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'ev_per_trade': round(ev, 2),
        'total_pnl': round(total_pnl, 2),
        'max_drawdown': round(max_dd, 2),
        'profit_factor': round(profit_factor, 2),
        'exit_reasons': exit_reasons,
    }

    return trades, summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='Path to model HDF5 (default: prod)')
    parser.add_argument('--scaler', default=None, help='Path to scaler pickle (default: prod)')
    parser.add_argument('--results', default=None, help='Path to results JSON (default: prod)')
    args = parser.parse_args()

    print("Loading model...")
    model, scaler, features, seq_len = load_model(args.model, args.scaler)

    print("Loading data...")
    data = load_data()
    print(f"  {len(data)} rows, {data['symbol'].nunique()} symbols, "
          f"{data['timestamp'].min().date()} to {data['timestamp'].max().date()}")

    print("Computing LSTM scores...")
    data['lstm_score'] = compute_lstm_scores(data, model, scaler, features, seq_len)
    scored = data['lstm_score'].notna().sum()
    print(f"  {scored} rows scored, mean={data['lstm_score'].mean():.4f}, "
          f">{LSTM_THRESHOLD}: {(data['lstm_score'] > LSTM_THRESHOLD).sum()}")

    # Build signal masks
    rule_mask = data.apply(rule_filter, axis=1)
    lstm_mask = rule_mask & (data['lstm_score'] > LSTM_THRESHOLD)

    print(f"\nRule-based signals: {rule_mask.sum()}")
    print(f"LSTM-filtered signals: {lstm_mask.sum()}")

    # Run backtests
    print("\n--- Running Rule-Based Backtest ---")
    rule_trades, rule_summary = run_backtest(data, rule_mask, "Rule-Based")

    print("--- Running LSTM-Filtered Backtest ---")
    lstm_trades, lstm_summary = run_backtest(data, lstm_mask, "LSTM-Filtered")

    # Print comparison
    print("\n" + "=" * 60)
    print(f"{'BACKTEST COMPARISON':^60}")
    print("=" * 60)

    headers = ['Metric', 'Rule-Based', 'LSTM-Filtered']
    rows = [
        ('Total trades', rule_summary.get('total_trades', 0), lstm_summary.get('total_trades', 0)),
        ('Win rate', f"{rule_summary.get('win_rate', 0):.1%}", f"{lstm_summary.get('win_rate', 0):.1%}"),
        ('Avg win %', f"{rule_summary.get('avg_win_pct', 0):.2f}%", f"{lstm_summary.get('avg_win_pct', 0):.2f}%"),
        ('Avg loss %', f"{rule_summary.get('avg_loss_pct', 0):.2f}%", f"{lstm_summary.get('avg_loss_pct', 0):.2f}%"),
        ('EV/trade %', f"{rule_summary.get('ev_per_trade', 0):.2f}%", f"{lstm_summary.get('ev_per_trade', 0):.2f}%"),
        ('Total P&L', f"฿{rule_summary.get('total_pnl', 0):,.0f}", f"฿{lstm_summary.get('total_pnl', 0):,.0f}"),
        ('Max drawdown', f"฿{rule_summary.get('max_drawdown', 0):,.0f}", f"฿{lstm_summary.get('max_drawdown', 0):,.0f}"),
        ('Profit factor', f"{rule_summary.get('profit_factor', 0):.2f}", f"{lstm_summary.get('profit_factor', 0):.2f}"),
    ]

    print(f"\n  {'Metric':<18} {'Rule-Based':>15} {'LSTM-Filtered':>15}")
    print(f"  {'-'*18} {'-'*15} {'-'*15}")
    for metric, rule_val, lstm_val in rows:
        print(f"  {metric:<18} {str(rule_val):>15} {str(lstm_val):>15}")

    # Exit reason breakdown
    print(f"\n  Exit Reasons:")
    all_reasons = set(list(rule_summary.get('exit_reasons', {}).keys()) +
                      list(lstm_summary.get('exit_reasons', {}).keys()))
    for reason in sorted(all_reasons):
        r_count = rule_summary.get('exit_reasons', {}).get(reason, 0)
        l_count = lstm_summary.get('exit_reasons', {}).get(reason, 0)
        print(f"    {reason:<18} {r_count:>15} {l_count:>15}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': MODEL_PATH,
        'data_range': f"{data['timestamp'].min().date()} to {data['timestamp'].max().date()}",
        'total_rows': len(data),
        'symbols': int(data['symbol'].nunique()),
        'lstm_threshold': LSTM_THRESHOLD,
        'rule_based': rule_summary,
        'lstm_filtered': lstm_summary,
    }

    results_path = args.results or os.path.join(RESULTS_DIR, 'lstm_backtest.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save trade details
    trades_path = os.path.join(RESULTS_DIR, 'lstm_backtest_trades.json')
    with open(trades_path, 'w') as f:
        json.dump({'rule_based': rule_trades, 'lstm_filtered': lstm_trades}, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")
    print(f"Trade details: {trades_path}")


if __name__ == '__main__':
    main()
