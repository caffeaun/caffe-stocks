"""Idempotent: create paper_portfolios + paper_trades tables in paper-live.db.

3 portfolios (one per panel rank) each starting at ฿10,000.
Trades flow: signaled (from signal_generator) → open (after fill) → closed.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

DEFAULT_DB = Path(os.path.expanduser('~/projects/caffe-stocks/data/paper-live.db'))
STARTING_CAPITAL_THB = 10_000.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_portfolios (
    rank INTEGER PRIMARY KEY CHECK(rank IN (1, 2, 3)),
    iteration_id INTEGER,
    trainer TEXT,
    starting_capital REAL NOT NULL,
    current_capital REAL NOT NULL,
    created_at TEXT NOT NULL,
    refreshed_at TEXT
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_rank INTEGER NOT NULL,
    signal_date TEXT NOT NULL,
    entry_date TEXT,
    symbol TEXT NOT NULL,
    entry_score REAL,
    entry_price REAL,
    position_size_thb REAL,
    stop_price REAL,
    target_price REAL,
    high_watermark REAL,
    trailing_floor_price REAL,
    exit_date TEXT,
    exit_price REAL,
    exit_reason TEXT,
    pnl_pct REAL,
    pnl_thb REAL,
    status TEXT NOT NULL CHECK(status IN ('signaled', 'open', 'closed')),
    hold_days_actual INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_status      ON paper_trades(status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_portfolio   ON paper_trades(portfolio_rank, status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_signal_date ON paper_trades(signal_date);
"""


def run(db_path: Path = DEFAULT_DB) -> bool:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        # Seed 3 portfolios if missing
        now = datetime.now().isoformat()
        for rank in (1, 2, 3):
            cur = conn.execute(
                'SELECT 1 FROM paper_portfolios WHERE rank = ?', (rank,))
            if cur.fetchone() is None:
                conn.execute("""
                    INSERT INTO paper_portfolios
                    (rank, starting_capital, current_capital, created_at, refreshed_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (rank, STARTING_CAPITAL_THB, STARTING_CAPITAL_THB, now, now))
        conn.commit()
        return True
    finally:
        conn.close()


if __name__ == '__main__':
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    run(p)
    print(f'paper-trade schema applied to {p}')
