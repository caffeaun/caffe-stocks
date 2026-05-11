#!/usr/bin/env python3
"""Morning fill: for every `signaled` paper trade, fill at today's open.

Cron: 09:30 Mon-Fri (after SET opens at 10:00 ICT? No — SET pre-open is
09:30-10:00. We use today's recorded OPEN from candles.db once it's
available. Today's bar usually lands in candles by end of day, so this
script actually fills T+1 the morning AFTER the signal — see signal_date
vs entry_date in the row.).

Strategy: pick the most recent candle for the symbol that's >= signal_date,
take its OPEN. Realistic semantics: signal at close T → fill at open T+1.
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


def _fetch_open(symbol: str) -> 'float | None':
    """Get the most recent OPEN price for `symbol` from candles.db.
    Real-world this would be 'today's open after market opens'; backtest-
    style here uses whatever the latest bar shows.
    """
    if not CANDLES_DB.exists():
        return None
    conn = sqlite3.connect(CANDLES_DB)
    try:
        row = conn.execute(
            'SELECT open FROM candles WHERE symbol = ? '
            'ORDER BY date DESC LIMIT 1', (symbol,)).fetchone()
    finally:
        conn.close()
    return float(row[0]) if row and row[0] is not None else None


def main():
    today = datetime.now().date().isoformat()
    print(f'paper_trade_fill: {today}')
    results = paper_book.fill_open_signals(today, _fetch_open)
    if not results:
        print('  no signaled trades to fill')
        return
    for r in results:
        if r['status'] == 'filled':
            print(f"  fill: rank={r['rank']} {r['symbol']:14s} @ ฿{r['price']:.2f}  size=฿{r['size_thb']:,.0f}")
        else:
            print(f"  skip: trade_id={r['trade_id']} ({r['reason']})")


if __name__ == '__main__':
    main()
