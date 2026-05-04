"""Sequence loader for v1 trainer + return_gate.

Reads candles.db, computes engineered features (via feature_eng), labels each
sequence using the realistic-trade simulation in labels.py, returns
(X, y, dates, symbols, pnl, hold_days, features).

Caches to .npz keyed on data mtime + label params. New pipeline — does not
share cache with the legacy LSTM trainer (different hold_days field)."""

import hashlib
import os
import sqlite3

import numpy as np
import pandas as pd

from .feature_eng import prepare_data, BASE_FEATURES
from .labels import (
    STOP_PCT, TARGET_PCT, TRAILING_TRIGGER, TRAILING_FLOOR, MAX_HOLD,
    MIN_PROFIT_PCT, COMMISSION_PCT, CLEAN_WIN_MAX_DD,
)

BASE_PATH = os.path.expanduser('~/projects/caffe-stocks')
DB_PATH = os.path.join(BASE_PATH, 'data', 'candles.db')


def label_trade_with_hold(highs, lows, closes, entry_price,
                          stop_pct=STOP_PCT, target_pct=TARGET_PCT,
                          trailing_trigger=TRAILING_TRIGGER,
                          trailing_floor=TRAILING_FLOOR,
                          max_hold=MAX_HOLD, commission_pct=COMMISSION_PCT):
    """Mirror of labels.label_trade_with_pnl but also returns hold_days.

    Returns (label, pnl, hold_days, exit_reason).
    """
    stop_loss = entry_price * (1 - stop_pct)
    target = entry_price * (1 + target_pct)
    peak_gain = 0.0
    worst_gain = 0.0
    rocky_cap = MIN_PROFIT_PCT / 2.0

    def _clean_constraint(pnl):
        if pnl > MIN_PROFIT_PCT and worst_gain <= -CLEAN_WIN_MAX_DD:
            return rocky_cap
        return pnl

    n = min(len(highs), max_hold)
    for day in range(n):
        day_low_gain = (lows[day] - entry_price) / entry_price
        if day_low_gain < worst_gain:
            worst_gain = day_low_gain

        if lows[day] <= stop_loss:
            pnl = -stop_pct - commission_pct
            return 0, pnl, day + 1, 'stop_loss'

        if highs[day] >= target:
            pnl = target_pct - commission_pct
            pnl = _clean_constraint(pnl)
            label = 1 if pnl > MIN_PROFIT_PCT else 0
            return label, pnl, day + 1, 'target'

        current_gain = (closes[day] - entry_price) / entry_price
        if current_gain >= trailing_trigger:
            peak_gain = max(peak_gain, current_gain)

        if peak_gain >= trailing_trigger:
            trail_floor_level = peak_gain * trailing_floor
            if current_gain < trail_floor_level:
                pnl = current_gain - commission_pct
                pnl = _clean_constraint(pnl)
                label = 1 if pnl > MIN_PROFIT_PCT else 0
                return label, pnl, day + 1, 'trailing'

    if n > 0:
        final_gain = (closes[n - 1] - entry_price) / entry_price
        pnl = final_gain - commission_pct
        pnl = _clean_constraint(pnl)
        label = 1 if pnl > MIN_PROFIT_PCT else 0
        return label, pnl, n, 'max_hold'

    return 0, -commission_pct, 0, 'no_data'


def load_sequences(seq_len=20, lookahead=10, db_path=DB_PATH, cache_dir=None,
                    verbose=True):
    """Load (X, y, dates, symbols, pnl, hold_days, features). Cached.

    X: (N, seq_len, F) float32
    y: (N,) float32 — 1 if pnl > MIN_PROFIT_PCT else 0
    dates: (N,) entry dates as 'YYYY-MM-DD' strings
    symbols: (N,) symbol strings
    pnl: (N,) per-trade P&L fraction (0.05 = +5%)
    hold_days: (N,) integer days held
    features: list of feature names in column order of X
    """
    if cache_dir is None:
        cache_dir = os.path.join(BASE_PATH, 'data')
    os.makedirs(cache_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    data = pd.read_sql_query('SELECT * FROM candles ORDER BY timestamp', conn)
    conn.close()
    data = data.sort_values('timestamp')

    if len(data) == 0:
        raise RuntimeError('candles table is empty')

    missing = [f for f in BASE_FEATURES if f not in data.columns]
    if missing:
        raise RuntimeError(f'Missing base features in candles: {missing}')

    data, features = prepare_data(data)

    cache_key_parts = [
        str(os.path.getmtime(db_path)),
        ','.join(features),
        f'{seq_len},{lookahead}',
        f'{STOP_PCT},{TARGET_PCT},{TRAILING_TRIGGER},{TRAILING_FLOOR},{MAX_HOLD}',
        f'min_profit={MIN_PROFIT_PCT},clean_dd={CLEAN_WIN_MAX_DD},comm={COMMISSION_PCT}',
        'v1_with_hold',
    ]
    cache_hash = hashlib.md5('|'.join(cache_key_parts).encode()).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f'sequences_v1_{cache_hash}.npz')

    if os.path.exists(cache_path):
        if verbose:
            print(f'Loading cached sequences: {cache_path}')
        cached = np.load(cache_path, allow_pickle=True)
        return (cached['X'], cached['y'], cached['dates'], cached['symbols'],
                cached['pnl'], cached['hold_days'], features)

    if verbose:
        print(f'Building sequences (cache miss). seq_len={seq_len} lookahead={lookahead} '
              f'features={len(features)}')

    X_all, y_all, pnl_all, hold_all = [], [], [], []
    dates_all, symbols_all = [], []

    for symbol in data['symbol'].unique():
        sdf = (data[data['symbol'] == symbol]
               .sort_values('timestamp').reset_index(drop=True))
        if len(sdf) < seq_len + lookahead:
            continue

        for i in range(len(sdf) - seq_len - lookahead):
            seq = sdf[features].iloc[i:i + seq_len].values
            if np.isnan(seq).any():
                continue

            entry_price = sdf.iloc[i + seq_len - 1]['close']
            window = sdf.iloc[i + seq_len:i + seq_len + lookahead]
            label, pnl, hold, _reason = label_trade_with_hold(
                window['high'].values, window['low'].values,
                window['close'].values, entry_price)

            X_all.append(seq)
            y_all.append(1 if pnl > MIN_PROFIT_PCT else 0)
            pnl_all.append(pnl)
            hold_all.append(hold)
            dates_all.append(str(sdf.iloc[i + seq_len - 1]['timestamp'])[:10])
            symbols_all.append(symbol)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    pnl = np.array(pnl_all, dtype=np.float32)
    hold_days = np.array(hold_all, dtype=np.int32)
    dates = np.array(dates_all)
    symbols = np.array(symbols_all)

    np.savez(cache_path,
             X=X, y=y, pnl=pnl, hold_days=hold_days,
             dates=dates, symbols=symbols)

    if verbose:
        print(f'  cached: {cache_path}')
        print(f'  N={len(X)}  pos_rate={y.mean():.1%}  avg_hold={hold_days.mean():.1f} days')

    return X, y, dates, symbols, pnl, hold_days, features


def aggregate_sequence(X):
    """Convert (N, seq_len, F) to (N, 4F) tabular: [last, mean, std, last-mean]."""
    last = X[:, -1, :].astype(np.float32)
    mean = X.mean(axis=1).astype(np.float32)
    std = X.std(axis=1).astype(np.float32)
    return np.concatenate([last, mean, std, last - mean], axis=1)


def aggregated_feature_names(features):
    """Return the 4F feature names matching aggregate_sequence output order."""
    names = []
    for kind in ('last', 'mean', 'std', 'dev'):
        names.extend(f'{kind}__{f}' for f in features)
    return names
