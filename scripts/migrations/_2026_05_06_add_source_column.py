"""Idempotent: adds `source` column to `candles_raw` and backfills existing
rows with 'yfinance' (factually correct — current DB was populated entirely
by scripts/fetch_ohlcv.py).

Safe to run multiple times. Returns True iff a change was made.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(os.path.expanduser('~/projects/caffe-stocks/data/candles.db'))


def run(db_path: Path = DEFAULT_DB) -> bool:
    if not db_path.exists():
        # No DB to migrate (fresh install) — orchestrator's first run will
        # create it via fetch_ohlcv.py's CREATE TABLE IF NOT EXISTS.
        return False
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cols = {r[1] for r in cur.execute('PRAGMA table_info(candles_raw)')}
        if not cols:
            return False  # table doesn't exist yet
        if 'source' in cols:
            return False
        cur.execute("ALTER TABLE candles_raw ADD COLUMN source TEXT DEFAULT 'yfinance'")
        cur.execute("UPDATE candles_raw SET source = 'yfinance' WHERE source IS NULL")
        conn.commit()
        return True
    finally:
        conn.close()


if __name__ == '__main__':
    import sys
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    changed = run(db)
    print(f'migration {"applied" if changed else "no-op"} on {db}')
