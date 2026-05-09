#!/usr/bin/env python3
"""Multi-source historical OHLCV ingester. Replaces scripts/fetch_ohlcv.py.

Modes:
  --mode daily     ~1-3 latest bars per symbol; runs in the 17:00 cron
  --mode backfill  full 3y per symbol; runs by hand or weekly Sunday cron

For each symbol, tries adapters in priority order (`config/sources.py` —
`PRIORITY_DAILY` or `PRIORITY_BACKFILL`, with per-symbol overrides). First
adapter to return non-empty data wins; lower-priority adapters fill only the
gaps thanks to `INSERT OR IGNORE`.

Per-row provenance recorded in `candles_raw.source`. Schema migration runs
idempotently at startup.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

from config import sources as src_cfg
from config.symbols import ALL_SYMBOLS
from scripts.migrations import _2026_05_06_add_source_column as add_source_col
from scripts.sources.base import Bar, Source

DB_PATH = BASE / 'data' / 'candles.db'
LOG_DIR = BASE / 'logs'
RUN_LOG_DIR = BASE / 'data' / 'multi_source_logs'

log = logging.getLogger('fetch_multi_source')

# ---------- adapter registry --------------------------------------------

def build_adapters(needed: set[str]) -> dict[str, Source]:
    """Lazily construct only the adapters in `needed`. Skips Selenium-based
    adapters when daily mode doesn't need them (fast cold-start)."""
    adapters: dict[str, Source] = {}
    if 'yfinance' in needed:
        from scripts.sources.yfinance_source import YFinanceSource
        adapters['yfinance'] = YFinanceSource()
    if 'settrade' in needed:
        try:
            from scripts.sources.settrade_source import SettradeSource
            adapters['settrade'] = SettradeSource()
        except ImportError:
            log.warning('settrade adapter not yet installed; skipping')
    if 'stooq' in needed:
        try:
            from scripts.sources.stooq_source import StooqSource
            adapters['stooq'] = StooqSource()
        except ImportError:
            log.warning('stooq adapter not yet installed; skipping')
    if 'set_csv' in needed:
        try:
            from scripts.sources.set_csv_source import SetCsvSource
            adapters['set_csv'] = SetCsvSource()
        except ImportError:
            log.warning('set_csv adapter not yet installed; skipping')
    if 'investing' in needed:
        try:
            from scripts.sources.investing_source import InvestingSource
            adapters['investing'] = InvestingSource()
        except ImportError:
            log.warning('investing adapter not yet installed; skipping')
    return adapters


def priority_for(symbol: str, mode: str) -> list[str]:
    if symbol in src_cfg.SYMBOL_OVERRIDES:
        return src_cfg.SYMBOL_OVERRIDES[symbol]
    return src_cfg.PRIORITY_BACKFILL if mode == 'backfill' else src_cfg.PRIORITY_DAILY


# ---------- DB plumbing -------------------------------------------------

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS candles_raw (
    timestamp TEXT,
    symbol TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    source TEXT DEFAULT 'yfinance'
);
"""
CREATE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_symbol_timestamp
    ON candles_raw(symbol, timestamp);
"""


def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_TABLE)
    conn.execute(CREATE_INDEX)
    add_source_col.run(DB_PATH)
    return conn


def existing_range(conn: sqlite3.Connection, symbol: str) -> tuple[Optional[str], Optional[str]]:
    row = conn.execute(
        'SELECT MIN(timestamp), MAX(timestamp) FROM candles_raw WHERE symbol = ?',
        (symbol,),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def persist(conn: sqlite3.Connection, symbol: str, bars: list[Bar], source: str) -> int:
    """INSERT OR IGNORE — first-write-wins. Returns rows actually inserted."""
    if not bars:
        return 0
    rows = [
        (b['timestamp'], symbol, b['open'], b['high'], b['low'], b['close'], b['volume'], source)
        for b in bars
    ]
    cur = conn.cursor()
    cur.executemany(
        'INSERT OR IGNORE INTO candles_raw '
        '(timestamp, symbol, open, high, low, close, volume, source) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        rows,
    )
    inserted = cur.rowcount
    conn.commit()
    return inserted


# ---------- orchestration -----------------------------------------------

def orchestrate(*, mode: str, symbols: Iterable[str],
                since: date, until: date,
                dry_run: bool = False) -> dict:
    today = datetime.now().date()
    symbols = list(symbols)
    log.info('mode=%s  symbols=%d  range=%s..%s  dry_run=%s',
             mode, len(symbols), since, until, dry_run)

    # Build only the adapters we'll actually need
    needed = set()
    for sym in symbols:
        needed.update(priority_for(sym, mode))
    adapters = build_adapters(needed)
    log.info('adapters loaded: %s', sorted(adapters))

    conn = open_db()

    summary = {
        'mode': mode, 'started_at': datetime.now().isoformat(),
        'symbols_total': len(symbols),
        'inserted_total': 0,
        'by_source': {n: 0 for n in adapters},
        'symbols_failed': [],
        'symbols_skipped': [],
    }

    for i, sym in enumerate(symbols, 1):
        # Resumability — if we already have what we need, skip.
        if mode == 'daily':
            _, max_ts = existing_range(conn, sym)
            if max_ts and max_ts[:10] >= today.isoformat():
                summary['symbols_skipped'].append(sym)
                continue

        chain = priority_for(sym, mode)
        wrote = False
        for source_name in chain:
            adapter = adapters.get(source_name)
            if adapter is None:
                continue
            try:
                if mode == 'backfill':
                    bars = adapter.fetch_history(sym, since, until)
                else:
                    one = adapter.fetch_daily(sym)
                    bars = [one] if one else []
            except Exception as e:
                log.warning('  %s/%s raised: %r', source_name, sym, e)
                bars = []

            if not bars:
                continue

            if dry_run:
                log.info('[%d/%d] %s -> %s [%d bars] DRY-RUN', i, len(symbols), sym, source_name, len(bars))
            else:
                inserted = persist(conn, sym, bars, source_name)
                summary['inserted_total'] += inserted
                summary['by_source'][source_name] = summary['by_source'].get(source_name, 0) + inserted
                log.info('[%d/%d] %s -> %s [%d bars, %d new]',
                         i, len(symbols), sym, source_name, len(bars), inserted)
            wrote = True
            break

        if not wrote:
            log.warning('[%d/%d] %s -> ALL SOURCES FAILED', i, len(symbols), sym)
            summary['symbols_failed'].append(sym)

    summary['finished_at'] = datetime.now().isoformat()
    conn.close()

    # Persist run log for audit
    if not dry_run:
        RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = RUN_LOG_DIR / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{mode}.json'
        log_path.write_text(json.dumps(summary, indent=2))
        log.info('run log: %s', log_path)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['daily', 'backfill'], required=True)
    parser.add_argument('--symbols', nargs='+', help='subset; default = ALL_SYMBOLS from config/symbols.py')
    parser.add_argument('--since', help='YYYY-MM-DD; backfill only (default: 3y ago)')
    parser.add_argument('--until', help='YYYY-MM-DD; backfill only (default: today)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )

    today = datetime.now().date()
    if args.mode == 'backfill':
        since = date.fromisoformat(args.since) if args.since else today - timedelta(days=3*365)
        until = date.fromisoformat(args.until) if args.until else today
    else:
        # daily mode — short window, only the last bar matters
        since = today - timedelta(days=5)
        until = today

    symbols = args.symbols or ALL_SYMBOLS
    summary = orchestrate(
        mode=args.mode, symbols=symbols,
        since=since, until=until, dry_run=args.dry_run,
    )
    print(json.dumps(
        {k: v for k, v in summary.items() if k != 'symbols_skipped'},
        indent=2,
    ))


if __name__ == '__main__':
    main()
