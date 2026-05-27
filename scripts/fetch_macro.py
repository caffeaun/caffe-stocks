#!/usr/bin/env python3
"""Daily macro-feature fetcher.

Pulls 6 yfinance symbols (close-only daily) into `macro_daily` in
data/ml-feedback.db. Trainers see these as 18 derived features
({sym}_z60, {sym}_ret_5d, {sym}_above_sma20) appended to CURATED_FEATURES
by models/feature_eng.py.

Symbols (rationale):
  ^VIX   — global risk-off
  THB=X  — USD/THB; Thai equity / currency coupling
  GC=F   — gold (flight-to-safety)
  CL=F   — oil (energy sector + trade balance)
  ^TNX   — US 10Y yield (global discount rate)
  ^STI   — Singapore Straits Times (ASEAN regional regime)

Leak check: each macro symbol closes at NYC EOD ~04:00 BKK, before next
Thai market open at 10:00. feature_eng.py joins via `.shift(1)` so date
D's macro = previous trading day's close (D-1 in NYC).

Run daily via cron (06:15 BKK) AND on first install as a backfill.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import yfinance as yf

DB_PATH = os.path.expanduser('~/projects/caffe-stocks/data/ml-feedback.db')
BACKFILL_START = date(2023, 1, 1)  # spans earliest train window (2023-05-01)

MACRO_SYMBOLS = [
    ('^VIX',  'vix'),
    ('THB=X', 'usd_thb'),
    ('GC=F',  'gold'),
    ('CL=F',  'oil'),
    ('^TNX',  'us_10y'),
    ('^STI',  'sti'),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS macro_daily (
  date TEXT PRIMARY KEY,
  vix REAL, usd_thb REAL, gold REAL, oil REAL, us_10y REAL, sti REAL
);
"""


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def _latest_date(conn, col: str) -> date | None:
    row = conn.execute(
        f"SELECT MAX(date) FROM macro_daily WHERE {col} IS NOT NULL"
    ).fetchone()
    if not row or not row[0]:
        return None
    y, m, d = row[0].split('-')
    return date(int(y), int(m), int(d))


def _fetch_close_series(symbol: str, start: date, end: date) -> dict[str, float]:
    """Return {YYYY-MM-DD: close} for a single symbol across [start, end]."""
    try:
        df = yf.download(
            symbol,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval='1d',
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as e:
        print(f'  WARN {symbol} download failed: {e}', file=sys.stderr)
        return {}
    if df is None or df.empty:
        print(f'  WARN {symbol} empty response', file=sys.stderr)
        return {}
    if hasattr(df.columns, 'get_level_values'):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            pass
    closes = {}
    for ts, row in df.iterrows():
        try:
            closes[ts.strftime('%Y-%m-%d')] = float(row['Close'])
        except Exception:
            continue
    return closes


def _upsert(conn, dates_closes_by_col: dict[str, dict[str, float]]):
    """dates_closes_by_col: {col_name: {date: close}}. Upsert per date."""
    all_dates = set()
    for d in dates_closes_by_col.values():
        all_dates.update(d.keys())
    if not all_dates:
        return 0
    for dt in sorted(all_dates):
        cols = ['date']
        vals: list = [dt]
        for col, series in dates_closes_by_col.items():
            cols.append(col)
            vals.append(series.get(dt))
        placeholders = ','.join('?' for _ in cols)
        col_list = ','.join(cols)
        update_clause = ','.join(
            f'{c}=COALESCE(excluded.{c}, macro_daily.{c})'
            for c in cols if c != 'date'
        )
        conn.execute(
            f'INSERT INTO macro_daily ({col_list}) VALUES ({placeholders}) '
            f'ON CONFLICT(date) DO UPDATE SET {update_clause}',
            vals,
        )
    conn.commit()
    return len(all_dates)


def main():
    conn = _conn()
    today = date.today()

    # For each symbol, determine its incremental window. If the col was
    # never populated, start from BACKFILL_START. Otherwise from the day
    # after the latest known value (idempotent re-runs).
    series_by_col: dict[str, dict[str, float]] = {}
    for ticker, col in MACRO_SYMBOLS:
        latest = _latest_date(conn, col)
        start = BACKFILL_START if latest is None else (latest + timedelta(days=1))
        if start > today:
            print(f'{ticker} -> {col}: up-to-date (last={latest})')
            series_by_col[col] = {}
            continue
        print(f'{ticker} -> {col}: fetching {start.isoformat()}..{today.isoformat()}')
        s = _fetch_close_series(ticker, start, today)
        print(f'  got {len(s)} bars')
        series_by_col[col] = s
        time.sleep(0.3)  # be polite to yfinance

    n = _upsert(conn, series_by_col)
    print(f'\nUpserted {n} distinct dates into macro_daily.')

    # Summary
    cur = conn.execute(
        'SELECT date, vix, usd_thb, gold, oil, us_10y, sti '
        'FROM macro_daily ORDER BY date DESC LIMIT 3'
    )
    print('\nMost recent rows:')
    for row in cur:
        print(' ', row)

    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
