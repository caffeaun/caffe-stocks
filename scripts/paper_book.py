"""Paper-trade book — library used by signal_generator (insert),
the morning fill script, the close-check exit evaluator, and the daily
summary.

3 portfolios (one per panel rank), each ฿10,000 starting. Per-trade size
33% of current_capital (whitepaper §11). Max K=2 concurrent per port.
Stop -3%, target +15%, trailing trigger +7% / floor 50% of peak gain,
max-hold 10 trading days (whitepaper §12 / labels.py).

All operations write to data/paper-live.db. (The pre-existing
scripts/paper_trade.py is a separate backtest simulator — unrelated.)
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
DB_PATH = BASE / 'data' / 'paper-live.db'

# Reuse the canonical trade-rule constants
from models.labels import (
    STOP_PCT, TARGET_PCT, TRAILING_TRIGGER, TRAILING_FLOOR, MAX_HOLD,
    COMMISSION_PCT,
)

POSITION_FRACTION = 0.33   # whitepaper §11 — 33% of port capital per trade
MAX_OPEN_PER_PORT = 2      # whitepaper §11 v2 split sizing


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Idempotent — invoked by every entry point."""
    from scripts.migrations._2026_05_11_paper_trade_schema import run
    run(DB_PATH)


def record_signal(*, portfolio_rank: int, signal_date: str, symbol: str,
                   entry_score: float) -> int:
    """signal_generator → here: one row per panel rank per emit. Returns
    paper_trades.id. NO dedup — 3 ranks emit 3 separate signals even if
    they pick the same symbol."""
    init_db()
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO paper_trades
            (portfolio_rank, signal_date, symbol, entry_score, status)
            VALUES (?, ?, ?, ?, 'signaled')
        """, (portfolio_rank, signal_date, symbol, entry_score))
        return cur.lastrowid


def fill_open_signals(today_str: str,
                       fetch_open_price: Callable[[str], Optional[float]]) -> list[dict]:
    """For each `signaled` row, look up entry-day open price via
    `fetch_open_price(symbol) -> float | None`. Apply the K=2 concurrency
    cap per port (skip rows that would exceed). Returns summary list of
    {trade_id, status, reason}.
    """
    init_db()
    out = []
    with get_conn() as conn:
        signaled = conn.execute("""
            SELECT t.id, t.portfolio_rank, t.signal_date, t.symbol, t.entry_score,
                   p.current_capital
            FROM paper_trades t
            JOIN paper_portfolios p ON p.rank = t.portfolio_rank
            WHERE t.status = 'signaled'
            ORDER BY t.signal_date, t.portfolio_rank, t.id
        """).fetchall()

        open_count = {1: 0, 2: 0, 3: 0}
        for r in conn.execute("SELECT portfolio_rank, COUNT(*) AS n FROM paper_trades "
                              "WHERE status='open' GROUP BY portfolio_rank"):
            open_count[r['portfolio_rank']] = r['n']

        for r in signaled:
            rank = r['portfolio_rank']
            if open_count[rank] >= MAX_OPEN_PER_PORT:
                conn.execute("UPDATE paper_trades SET status='closed', "
                             "exit_reason='skipped_full_port', exit_date=? "
                             "WHERE id=?", (today_str, r['id']))
                out.append({'trade_id': r['id'], 'status': 'skipped',
                            'reason': f'port {rank} at K={MAX_OPEN_PER_PORT}'})
                continue

            price = fetch_open_price(r['symbol'])
            if price is None or price <= 0:
                conn.execute("UPDATE paper_trades SET status='closed', "
                             "exit_reason='no_fill', exit_date=? "
                             "WHERE id=?", (today_str, r['id']))
                out.append({'trade_id': r['id'], 'status': 'skipped',
                            'reason': 'no open price'})
                continue

            size_thb = float(r['current_capital']) * POSITION_FRACTION
            stop_price = price * (1 - STOP_PCT)
            target_price = price * (1 + TARGET_PCT)

            conn.execute("""
                UPDATE paper_trades SET
                    status = 'open',
                    entry_date = ?,
                    entry_price = ?,
                    position_size_thb = ?,
                    stop_price = ?,
                    target_price = ?,
                    high_watermark = ?,
                    trailing_floor_price = NULL
                WHERE id = ?
            """, (today_str, price, size_thb, stop_price, target_price, price, r['id']))
            open_count[rank] += 1
            out.append({'trade_id': r['id'], 'status': 'filled',
                        'symbol': r['symbol'], 'price': price,
                        'rank': rank, 'size_thb': size_thb})
    return out


def evaluate_exits(today_str: str,
                    fetch_ohlc: Callable[[str, str], Optional[dict]]) -> list[dict]:
    """For each `open` trade, check today's high/low/close against stop /
    target / trailing-floor / max-hold. Exit on FIRST trigger this session.

    Pricing semantics (intraday — pessimistic):
      - Stop hit:    low <= stop_price            → exit at stop_price
      - Target hit:  high >= target_price          → exit at target_price
      - Trailing:    once gain>=7%, lock 50% of peak gain;
                     low <= trailing_floor_price  → exit at floor
      - Max hold:    >= MAX_HOLD trading days     → exit at close
    """
    init_db()
    out = []
    with get_conn() as conn:
        opens = conn.execute("""
            SELECT id, portfolio_rank, signal_date, entry_date, symbol,
                   entry_price, position_size_thb, stop_price, target_price,
                   high_watermark, trailing_floor_price
            FROM paper_trades
            WHERE status = 'open'
            ORDER BY entry_date, portfolio_rank, id
        """).fetchall()

        for r in opens:
            bar = fetch_ohlc(r['symbol'], today_str)
            if bar is None:
                continue

            entry = float(r['entry_price'])
            high_today = float(bar['high'])
            low_today = float(bar['low'])
            close_today = float(bar['close'])

            new_watermark = max(float(r['high_watermark'] or entry), high_today)
            new_floor = r['trailing_floor_price']
            gain_at_peak = (new_watermark - entry) / entry
            if gain_at_peak >= TRAILING_TRIGGER:
                new_floor = entry * (1 + gain_at_peak * TRAILING_FLOOR)

            entry_d = datetime.fromisoformat(r['entry_date']).date()
            today_d = datetime.fromisoformat(today_str).date()
            days_held = max(0, (today_d - entry_d).days)
            trading_days_held = int(round(days_held * 5 / 7))

            exit_reason = None
            exit_price = None

            if low_today <= float(r['stop_price']):
                exit_reason = 'stop'
                exit_price = float(r['stop_price'])
            elif high_today >= float(r['target_price']):
                exit_reason = 'target'
                exit_price = float(r['target_price'])
            elif new_floor is not None and low_today <= new_floor:
                exit_reason = 'trailing'
                exit_price = new_floor
            elif trading_days_held >= MAX_HOLD:
                exit_reason = 'max_hold'
                exit_price = close_today
            else:
                conn.execute("""
                    UPDATE paper_trades SET high_watermark = ?, trailing_floor_price = ?
                    WHERE id = ?
                """, (new_watermark, new_floor, r['id']))
                continue

            gross_pct = (exit_price - entry) / entry
            pnl_pct = gross_pct - COMMISSION_PCT
            pnl_thb = float(r['position_size_thb']) * pnl_pct

            conn.execute("""
                UPDATE paper_trades SET
                    status = 'closed',
                    exit_date = ?,
                    exit_price = ?,
                    exit_reason = ?,
                    pnl_pct = ?,
                    pnl_thb = ?,
                    high_watermark = ?,
                    trailing_floor_price = ?,
                    hold_days_actual = ?
                WHERE id = ?
            """, (today_str, exit_price, exit_reason, pnl_pct, pnl_thb,
                  new_watermark, new_floor, trading_days_held, r['id']))

            conn.execute("""
                UPDATE paper_portfolios SET
                    current_capital = current_capital + ?,
                    refreshed_at = ?
                WHERE rank = ?
            """, (pnl_thb, today_str, r['portfolio_rank']))

            out.append({'trade_id': r['id'], 'symbol': r['symbol'],
                        'rank': r['portfolio_rank'], 'pnl_pct': pnl_pct,
                        'pnl_thb': pnl_thb, 'reason': exit_reason})
    return out


def daily_summary(today_str: str) -> dict:
    """Aggregate per-port stats for the Telegram summary."""
    init_db()
    with get_conn() as conn:
        portfolios = [dict(r) for r in conn.execute(
            'SELECT * FROM paper_portfolios ORDER BY rank').fetchall()]
        per_port = []
        for p in portfolios:
            rank = p['rank']
            today_closed = conn.execute("""
                SELECT COUNT(*) AS n,
                       COALESCE(SUM(pnl_thb), 0) AS pnl_thb_today,
                       COALESCE(SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END), 0) AS wins_today
                FROM paper_trades
                WHERE portfolio_rank = ? AND status = 'closed' AND exit_date = ?
            """, (rank, today_str)).fetchone()

            recent_closed = conn.execute("""
                SELECT COUNT(*) AS n,
                       COALESCE(SUM(pnl_thb), 0) AS pnl_thb_30d,
                       COALESCE(AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0.0 END), 0) AS wr_30d
                FROM paper_trades
                WHERE portfolio_rank = ? AND status = 'closed'
                  AND exit_date >= date(?, '-30 days')
            """, (rank, today_str)).fetchone()

            opens = [dict(r) for r in conn.execute("""
                SELECT id, symbol, entry_date, entry_price, target_price,
                       stop_price, high_watermark, position_size_thb
                FROM paper_trades
                WHERE portfolio_rank = ? AND status = 'open'
                ORDER BY entry_date
            """, (rank,))]

            per_port.append({
                'rank': rank,
                'iteration_id': p['iteration_id'],
                'trainer': p['trainer'],
                'starting_capital': p['starting_capital'],
                'current_capital': p['current_capital'],
                'pnl_pct_total': (p['current_capital'] - p['starting_capital']) / p['starting_capital'],
                'today_closed': today_closed['n'],
                'pnl_thb_today': today_closed['pnl_thb_today'],
                'recent_closed': recent_closed['n'],
                'pnl_thb_30d': recent_closed['pnl_thb_30d'],
                'wr_30d': recent_closed['wr_30d'],
                'opens': opens,
            })

    return {'date': today_str, 'portfolios': per_port}


def update_panel_metadata() -> None:
    """Sync paper_portfolios.iteration_id + trainer from production_panel.
    Called after promote_panel.py runs so the daily summary correctly
    labels each port. Does NOT reset current_capital (panel change is
    a strategy update, not a portfolio reset)."""
    import sqlite3 as _sqlite3
    fb_db = BASE / 'data' / 'ml-feedback.db'
    if not fb_db.exists():
        return
    init_db()
    fb_conn = _sqlite3.connect(fb_db)
    fb_conn.row_factory = _sqlite3.Row
    try:
        rows = fb_conn.execute("""
            SELECT pp.rank, pp.iteration_id, it.trainer
            FROM production_panel pp
            JOIN iterations it ON it.id = pp.iteration_id
            ORDER BY pp.rank
        """).fetchall()
    finally:
        fb_conn.close()

    with get_conn() as conn:
        now = datetime.now().isoformat()
        for r in rows:
            conn.execute("""
                UPDATE paper_portfolios SET iteration_id = ?, trainer = ?,
                       refreshed_at = ?
                WHERE rank = ?
            """, (r['iteration_id'], r['trainer'], now, r['rank']))
