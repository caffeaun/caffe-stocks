import json
import sqlite3
import os
from datetime import datetime, timedelta

# Database paths
CANDLES_DB = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'
PAPER_DB = '/home/kanoonth-ai/projects/caffe-stocks/data/paper-silent.db'
PAPER_STATUS_FILE = '/home/kanoonth-ai/projects/caffe-stocks/data/paper-status.json'


def simulate_trade(symbol, entry_price, entry_date):
    """Simulate trade exit based on actual candle data"""
    conn = sqlite3.connect(CANDLES_DB)
    cursor = conn.cursor()
    
    # Fetch candles starting from entry_date
    cursor.execute('''
        SELECT date, open, high, low, close
        FROM candles
        WHERE symbol = ? AND date >= ?
        ORDER BY date
        LIMIT 10
    ''', (symbol, entry_date))
    candles = cursor.fetchall()
    conn.close()

    if not candles:
        return None

    # Initialize tracking
    peak_gain = 0.0
    exit_price = None

    # Simulate 10 days or until exit
    for i, candle in enumerate(candles):
        date, open_price, high, low, close = candle

        # Check stop loss
        if low <= entry_price * 0.97:
            exit_price = low
            break

        # Check target
        if high >= entry_price * 1.15:
            exit_price = high
            break

        # Track peak gain for trailing stop (after 7% gain)
        high_gain = (high - entry_price) / entry_price
        if high_gain >= 0.07:
            peak_gain = max(peak_gain, high_gain)

        # Check trailing stop (50% pullback from peak gain)
        if peak_gain >= 0.07:
            low_gain = (low - entry_price) / entry_price
            if low_gain < peak_gain * 0.5:
                exit_price = low
                break
        
        # Day 10 exit
        if i == 9:
            exit_price = close
            break

    if exit_price is None:
        return None

    # Calculate PNL
    pnl_pct = (exit_price / entry_price - 1) * 100
    return exit_price, pnl_pct

def get_signals_from_candles():
    """Read real signals from candles.db using same filter criteria as score_signals.py"""
    conn = sqlite3.connect(CANDLES_DB)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT symbol, date, close
        FROM candles
        WHERE volume_ratio > 1.5 AND atr > 0.03
        ORDER BY date
    ''')
    signals = [(row[0], row[2], row[1]) for row in cursor.fetchall()]
    conn.close()
    return signals


def main():
    # Connect to paper-silent database
    conn = sqlite3.connect(PAPER_DB)
    cursor = conn.cursor()

    # Create table if needed (matches existing schema)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            symbol TEXT,
            entry_date TEXT,
            entry_price REAL,
            stop_loss REAL,
            target REAL,
            position_size INTEGER,
            status TEXT,
            pnl_pct REAL
        )
    ''')

    # Get signals from real candles data filtered by signal criteria
    signals = get_signals_from_candles()
    print(f'Loaded {len(signals)} real signals from candles.db')

    # Process each signal
    for symbol, entry_price, entry_date in signals:
        result = simulate_trade(symbol, entry_price, entry_date)
        if result:
            exit_price, pnl_pct = result
            stop_loss = round(entry_price * 0.97, 4)
            target = round(entry_price * 1.15, 4)
            cursor.execute('''
                INSERT INTO trades (symbol, entry_date, entry_price, stop_loss, target, position_size, status, pnl_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, entry_date, entry_price, stop_loss, target, 1, 'closed', pnl_pct))

    conn.commit()
    conn.close()

    # Verify results
    conn = sqlite3.connect(PAPER_DB)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM trades')
    total_trades = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(DISTINCT pnl_pct) FROM trades')
    distinct_pnl = cursor.fetchone()[0]
    print(f'Processed {total_trades} trades with {distinct_pnl} distinct pnl_pct values')

    # Update paper circuit breaker flags
    update_paper_circuit_breaker(conn)
    conn.close()


def update_paper_circuit_breaker(conn):
    """Check paper trading drawdowns and write paper-status.json."""
    cursor = conn.cursor()

    # Build daily equity curve from trades
    cursor.execute('SELECT entry_date, pnl_pct FROM trades ORDER BY entry_date')
    rows = cursor.fetchall()
    if not rows:
        return

    # Aggregate daily P&L
    daily_pnl = {}
    for date, pnl in rows:
        daily_pnl.setdefault(date, []).append(pnl)

    # Build cumulative equity (starting at 100%)
    capital = 100.0
    equity_by_date = {}
    for date in sorted(daily_pnl.keys()):
        for pnl in daily_pnl[date]:
            capital *= (1 + pnl / 100)
        equity_by_date[date] = capital

    dates = sorted(equity_by_date.keys())
    if len(dates) < 2:
        _write_paper_status('active', equity_by_date, rows)
        return

    latest = equity_by_date[dates[-1]]

    # Daily loss > 3%
    prev_date = dates[-2] if len(dates) >= 2 else None
    daily_loss = 0
    if prev_date:
        prev_val = equity_by_date[prev_date]
        daily_loss = (prev_val - latest) / prev_val if prev_val > 0 else 0

    # Weekly loss > 5%
    week_ago = (datetime.strptime(dates[-1], '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
    week_val = next((equity_by_date[d] for d in dates if d <= week_ago), None)
    weekly_loss = (week_val - latest) / week_val if week_val and week_val > 0 else 0

    # Monthly loss > 8%
    month_ago = (datetime.strptime(dates[-1], '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
    month_val = next((equity_by_date[d] for d in dates if d <= month_ago), None)
    monthly_loss = (month_val - latest) / month_val if month_val and month_val > 0 else 0

    # Rolling 30-trade win rate
    recent = rows[-30:] if len(rows) >= 30 else rows
    wins = sum(1 for _, pnl in recent if pnl > 0)
    win_rate = wins / len(recent) if recent else 0

    # Determine status
    if monthly_loss > 0.08:
        status = 'stopped'
    elif weekly_loss > 0.05:
        status = 'paused'
    elif daily_loss > 0.03:
        status = 'paused'
    elif len(rows) >= 30 and win_rate < 0.40:
        status = 'retrain'
    else:
        status = 'active'

    _write_paper_status(status, equity_by_date, rows)
    print(f'Paper circuit breaker: {status} (daily={daily_loss:.1%}, weekly={weekly_loss:.1%}, monthly={monthly_loss:.1%}, WR={win_rate:.1%})')


def _write_paper_status(status, equity_by_date, trades):
    dates = sorted(equity_by_date.keys())
    data = {
        'status': status,
        'updated': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'total_trades': len(trades),
        'final_equity_pct': round(equity_by_date[dates[-1]], 2) if dates else 100.0,
    }
    with open(PAPER_STATUS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()