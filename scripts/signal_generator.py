#!/usr/bin/env python3
"""Daily signal generator — produces the 17:15 Telegram alert.

v1 architecture (per docs/ml-training.md and docs/ml-loop.md):
  - The strategy is the rules + sizing + exits. The ML model is a *filter*.
  - The 3-model paper-trade panel (`production_panel` in ml-feedback.db) is
    the source of signals. Each panel rank produces ONE signal per day —
    its top-scored symbol after applying the rule-based screen.
  - When the panel is empty, the daily scan emits no signals and reports it.
    Run `scripts/promote_panel.py` to (re)populate.

Saved panel artifacts live in `models/panel/rank{1,2,3}/` — produced by
`scripts/promote_panel.py` retraining each candidate on the full dataset.
The artifacts include the trainer's own files plus a `scaler.pkl` matching
the feature standardization used at training time.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

DB_PATH = BASE / 'data' / 'candles.db'
FB_DB_PATH = BASE / 'data' / 'ml-feedback.db'
STATUS_FILE = BASE / 'data' / 'system-status.json'
PANEL_DIR = BASE / 'models' / 'panel'

# Rule-based screen — applied before panel scoring (whitepaper §3 baseline).
MIN_ATR_PCT = 0.03
MIN_VOL_RATIO = 2.0
RSI_MIN, RSI_MAX = 30, 65
# BLS tick is ฿0.01 — that's 1% of a ฿1 stock and 5% of a ฿0.20 stock.
# Sub-฿1 names suffer tick-noise the strategy can't profitably absorb.
MIN_CLOSE = 1.0


def _system_active() -> bool:
    if not STATUS_FILE.exists():
        return True
    try:
        with open(STATUS_FILE) as f:
            return json.load(f).get('status') == 'active'
    except Exception:
        return True


def load_panel_models() -> list[dict]:
    """Return [{rank, trainer, scorer, scaler, features}] for each panel
    member whose artifact dir exists. Empty list when panel is unpopulated.
    """
    if not FB_DB_PATH.exists():
        return []
    conn = sqlite3.connect(FB_DB_PATH)
    try:
        rows = conn.execute("""
            SELECT pp.rank, pp.iteration_id, it.trainer, it.avg_win_rate,
                   it.avg_annualized_return
            FROM production_panel pp
            JOIN iterations it ON it.id = pp.iteration_id
            ORDER BY pp.rank
        """).fetchall()
    finally:
        conn.close()

    from models.trainers import TRAINERS
    panel = []
    for r in rows:
        rank, iter_id, trainer_name, wr, ann = r
        art_dir = PANEL_DIR / f'rank{rank}'
        if not art_dir.exists():
            print(f'panel rank {rank}: artifact dir missing {art_dir}', file=sys.stderr)
            continue
        cls = TRAINERS.get(trainer_name)
        if cls is None:
            print(f'panel rank {rank}: trainer {trainer_name!r} not in registry', file=sys.stderr)
            continue
        try:
            scorer = cls.load(str(art_dir))
            scaler = joblib.load(art_dir / 'scaler.pkl')
            with open(art_dir / 'metadata.json') as f:
                meta = json.load(f)
        except Exception as e:
            print(f'panel rank {rank}: load failed: {e!r}', file=sys.stderr)
            continue
        panel.append({
            'rank': rank,
            'iteration_id': iter_id,
            'trainer': trainer_name,
            'wr': wr,
            'ann': ann,
            'scorer': scorer,
            'scaler': scaler,
            'features': meta.get('features', []),
        })
    return panel


def _load_latest_features(seq_len: int = 20):
    """Build aggregated tabular features for **inference today** — i.e. the
    last `seq_len`-day window per symbol, ending at the latest close in
    `candles`. Bypasses sequence_loader.load_sequences (which only emits
    fully-labeled sequences and therefore lags by `lookahead` days).

    Forward-fills monthly features per symbol so the latest day inherits
    the last published value — matches the implicit model assumption that
    last month's foreign-flow data is the best available "today" estimate.

    Output shape matches `aggregate_sequence(X)` from sequence_loader so
    the saved panel artifacts can score it directly.
    """
    from models.feature_eng import (
        get_available_features,
        compute_derived_features, compute_percentile_ranks,
        compute_cross_sectional_ranks, compute_cross_features,
    )
    from models.sequence_loader import aggregate_sequence

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query('SELECT * FROM candles ORDER BY symbol, timestamp', conn)
    finally:
        conn.close()
    df = df[~df['symbol'].str.startswith('^')]  # drop indices

    # Replicate prepare_data minus the global dropna — that drop is fine
    # for training but kills inference on the latest day (lagged monthly
    # features publish ~1 month behind).
    df = compute_derived_features(df)
    df = compute_percentile_ranks(df)
    df = compute_cross_sectional_ranks(df)
    df = compute_cross_features(df)

    feats = get_available_features(df)

    # Forward-fill monthly features per symbol — they only update at month
    # boundaries; today's NaN is yesterday's value carried forward.
    LAGGED = [c for c in df.columns
              if 'monthly' in c or c.endswith('_pctrank') and 'monthly' in c]
    if LAGGED:
        df[LAGGED] = df.groupby('symbol', sort=False)[LAGGED].ffill()

    latest_date = df['date'].max()

    Xs, kept_syms = [], []
    skipped_short = skipped_stale = skipped_nan = 0
    for sym, sdf in df.groupby('symbol', sort=False):
        sdf = sdf.sort_values('timestamp').tail(seq_len)
        if len(sdf) < seq_len:
            skipped_short += 1
            continue
        if sdf['date'].iloc[-1] != latest_date:
            skipped_stale += 1
            continue
        block = sdf[feats].values.astype(np.float32)
        if not np.isfinite(block).all():
            skipped_nan += 1
            continue
        Xs.append(block)
        kept_syms.append(sym)

    print(f'_load_latest_features: kept {len(Xs)}  '
          f'(short={skipped_short}, stale={skipped_stale}, nan={skipped_nan})',
          file=sys.stderr)

    if not Xs:
        return (np.zeros((0, len(feats) * 4), dtype=np.float32),
                np.array([]), feats, str(latest_date))

    X = np.stack(Xs, axis=0)
    X_tab = aggregate_sequence(X)
    return X_tab, np.array(kept_syms), feats, str(latest_date)


def _apply_rule_screen(symbols_today: np.ndarray, latest_date: str) -> set[str]:
    """Return the set of symbols passing the volume/ATR/RSI screen on
    `latest_date`. We read these from `candles` (post-indicator table)
    because they're only used to filter, not to score."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            'SELECT symbol, close, atr, volume_ratio, rsi FROM candles WHERE date = ?',
            conn, params=[latest_date])
    finally:
        conn.close()
    df = df[df['symbol'].isin(symbols_today)]
    df = df.dropna(subset=['atr', 'close', 'volume_ratio', 'rsi'])
    df = df[
        (df['close'] >= MIN_CLOSE) &
        ((df['atr'] / df['close']) > MIN_ATR_PCT) &
        (df['volume_ratio'] > MIN_VOL_RATIO) &
        (df['rsi'].between(RSI_MIN, RSI_MAX))
    ]
    return set(df['symbol'])


def generate_signals():
    panel = load_panel_models()
    if not panel:
        return [], '?', 0, panel, set()

    X_today, symbols_today, features, latest_date = _load_latest_features()
    if len(symbols_today) == 0:
        return [], latest_date, 0, panel, set()

    rule_pass = _apply_rule_screen(symbols_today, latest_date)

    signals = []
    for member in panel:
        try:
            X_scaled = member['scaler'].transform(X_today)
            scores = member['scorer'].predict_proba(X_scaled)
        except Exception as e:
            print(f"panel rank {member['rank']}: predict failed: {e!r}", file=sys.stderr)
            continue
        # Restrict to rule-passing symbols, then take the top-scored row
        mask = np.array([s in rule_pass for s in symbols_today])
        if not mask.any():
            continue
        valid_scores = scores[mask]
        valid_symbols = symbols_today[mask]
        top_idx = int(np.argmax(valid_scores))
        signals.append({
            'rank': member['rank'],
            'trainer': member['trainer'],
            'iteration_id': member['iteration_id'],
            'wr': member['wr'],
            'symbol': str(valid_symbols[top_idx]),
            'score': float(valid_scores[top_idx]),
            'date': latest_date,
        })

    return signals, latest_date, int(len(symbols_today)), panel, rule_pass


def _enrich_with_price(signals: list[dict], latest_date: str) -> list[dict]:
    if not signals:
        return signals
    syms = list({s['symbol'] for s in signals})
    qmark = ','.join('?' * len(syms))
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            f"SELECT symbol, close, atr, volume_ratio, rsi FROM candles "
            f"WHERE date = ? AND symbol IN ({qmark})",
            conn, params=[latest_date] + syms)
    finally:
        conn.close()
    info = {r['symbol']: r for _, r in df.iterrows()}
    for s in signals:
        row = info.get(s['symbol'], {})
        s['price'] = float(row.get('close', 0) or 0)
        s['rsi'] = round(float(row.get('rsi', 0) or 0), 1)
        s['atr_pct'] = float((row.get('atr', 0) or 0) / max(row.get('close', 1) or 1, 1e-9))
        s['vol_ratio'] = float(row.get('volume_ratio', 0) or 0)
    return signals


def format_telegram(signals: list[dict], latest_date: str, n_symbols: int,
                     panel: list[dict], rule_pass: set[str]) -> str:
    if not panel:
        return (f'📊 *Signal Scan — {latest_date}*\n'
                f'Panel empty — run `scripts/promote_panel.py` to populate.')

    panel_summary = ', '.join(
        f"#{m['rank']} {m['trainer']} (WR {m['wr']:.0%})" for m in panel
    )
    header = (
        f'📊 *Signal Scan — {latest_date}*\n'
        f'Scanned {n_symbols} symbols, {len(rule_pass)} passed rule screen\n'
        f'Panel: {panel_summary}'
    )
    if not signals:
        return f'{header}\n\nNo signals: no rule-screen pass had a positive panel score.'

    lines = [header, '']
    for s in signals:
        lines.append(
            f'🟢 *{s["symbol"].replace(".BK", "")}* — ฿{s.get("price", 0):.2f}  '
            f'(rank #{s["rank"]} {s["trainer"]})'
        )
        lines.append(
            f'   score {s["score"]:.2f}  RSI {s.get("rsi", "?")}  '
            f'ATR {s.get("atr_pct", 0):.1%}  Vol×{s.get("vol_ratio", 0):.1f}'
        )
        lines.append('')
    return '\n'.join(lines)


def main():
    if not _system_active():
        print('Skipping signal generation: system status is not active')
        return
    signals, latest_date, n_symbols, panel, rule_pass = generate_signals()
    signals = _enrich_with_price(signals, latest_date)
    print(format_telegram(signals, latest_date, n_symbols, panel, rule_pass))


if __name__ == '__main__':
    main()
