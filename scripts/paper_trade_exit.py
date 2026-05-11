#!/usr/bin/env python3
"""Evaluate stop / target / trailing / max-hold exits on all `open` paper
trades. Called from scripts/daily-close-check.sh at 20:00 Mon-Fri (after
SET closes at 16:30 and candles.db has today's bar).
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

from scripts import paper_book

CANDLES_DB = BASE / 'data' / 'candles.db'


def _fetch_ohlc(symbol: str, date_str: str) -> 'dict | None':
    """Return {open, high, low, close} for symbol on date_str, or None."""
    if not CANDLES_DB.exists():
        return None
    conn = sqlite3.connect(CANDLES_DB)
    try:
        row = conn.execute(
            'SELECT open, high, low, close FROM candles '
            'WHERE symbol = ? AND date = ? LIMIT 1',
            (symbol, date_str)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return {'open': row[0], 'high': row[1], 'low': row[2], 'close': row[3]}


def main():
    today = datetime.now().date().isoformat()
    print(f'paper_trade_exit: {today}')
    exits = paper_book.evaluate_exits(today, _fetch_ohlc)
    if not exits:
        print('  no exits triggered')
        return
    for e in exits:
        print(f"  exit: rank={e['rank']} {e['symbol']:14s} pnl={e['pnl_pct']:+.2%}  "
              f"(฿{e['pnl_thb']:+,.0f})  reason={e['reason']}")


if __name__ == '__main__':
    main()
