#!/home/kanoonth-ai/shared-venv/bin/python
"""Day-by-day paper trading simulation using real candles data."""
import argparse
import sqlite3

CANDLES_DB = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'

STOP_LOSS_PCT = 0.97   # 3% stop loss
TARGET_PCT = 1.15      # 15% target
TRAILING_TRIGGER = 0.07  # Activate trailing stop after 7% gain
TRAILING_PULLBACK = 0.5  # Exit if gain drops to 50% of peak
MAX_HOLD_DAYS = 10

SIGNAL_FILTERS = 'volume_ratio > 1.5 AND atr > 0.03'


def get_trading_days(conn, limit):
    cursor = conn.cursor()
    cursor.execute(
        f'SELECT DISTINCT date FROM candles WHERE {SIGNAL_FILTERS} ORDER BY date DESC LIMIT ?',
        (limit,)
    )
    days = [row[0] for row in cursor.fetchall()]
    return sorted(days)


def get_signals_for_day(conn, date):
    cursor = conn.cursor()
    cursor.execute(
        f'SELECT symbol, close FROM candles WHERE date = ? AND {SIGNAL_FILTERS} ORDER BY volume_ratio DESC',
        (date,)
    )
    return cursor.fetchall()


def simulate_trade(conn, symbol, entry_price, entry_date):
    """Simulate trade exit using subsequent candle data. Returns (exit_price, pnl_pct, reason)."""
    cursor = conn.cursor()
    cursor.execute(
        '''SELECT date, high, low, close FROM candles
           WHERE symbol = ? AND date > ?
           ORDER BY date LIMIT ?''',
        (symbol, entry_date, MAX_HOLD_DAYS)
    )
    candles = cursor.fetchall()

    if not candles:
        return None

    peak_gain = 0.0

    for i, (date, high, low, close) in enumerate(candles):
        # Stop loss (check low)
        if low <= entry_price * STOP_LOSS_PCT:
            exit_price = entry_price * STOP_LOSS_PCT
            pnl_pct = (exit_price / entry_price - 1) * 100
            return exit_price, pnl_pct, 'stop_loss'

        # Target (check high)
        if high >= entry_price * TARGET_PCT:
            exit_price = entry_price * TARGET_PCT
            pnl_pct = (exit_price / entry_price - 1) * 100
            return exit_price, pnl_pct, 'target'

        # Trailing stop
        high_gain = (high - entry_price) / entry_price
        if high_gain >= TRAILING_TRIGGER:
            peak_gain = max(peak_gain, high_gain)

        if peak_gain >= TRAILING_TRIGGER:
            low_gain = (low - entry_price) / entry_price
            if low_gain < peak_gain * TRAILING_PULLBACK:
                exit_price = low
                pnl_pct = (exit_price / entry_price - 1) * 100
                return exit_price, pnl_pct, 'trailing_stop'

        # Day-10 exit (max hold)
        if i == MAX_HOLD_DAYS - 1:
            pnl_pct = (close / entry_price - 1) * 100
            return close, pnl_pct, 'max_hold'

    # Ran out of candles before max hold
    last_close = candles[-1][3]
    pnl_pct = (last_close / entry_price - 1) * 100
    return last_close, pnl_pct, 'end_of_data'


def main():
    parser = argparse.ArgumentParser(description='Paper trading simulation using real candles data')
    parser.add_argument('--days', type=int, default=20, help='Number of trading days to simulate')
    args = parser.parse_args()

    conn = sqlite3.connect(CANDLES_DB)
    trading_days = get_trading_days(conn, args.days)

    if not trading_days:
        print('No signal days found in candles.db')
        conn.close()
        return

    print(f'Simulating {len(trading_days)} trading days ({trading_days[0]} to {trading_days[-1]})')
    print('-' * 60)

    trades = []
    for day in trading_days:
        signals = get_signals_for_day(conn, day)
        if not signals:
            continue

        # Take top signal by volume_ratio
        symbol, entry_price = signals[0]
        result = simulate_trade(conn, symbol, entry_price, day)
        if result is None:
            print(f'Day {day}: {symbol} @ {entry_price:.2f} — no exit data, skipped')
            continue

        exit_price, pnl_pct, reason = result
        trades.append(pnl_pct)
        sign = '+' if pnl_pct >= 0 else ''
        print(f'Day {day}: {symbol} entry={entry_price:.2f} exit={exit_price:.2f} '
              f'PnL={sign}{pnl_pct:.2f}% [{reason}]')

    conn.close()

    if not trades:
        print('No completed trades.')
        return

    wins = [p for p in trades if p > 0]
    losses = [p for p in trades if p <= 0]
    win_rate = len(wins) / len(trades) * 100
    avg_pnl = sum(trades) / len(trades)
    total_pnl = sum(trades)

    print('-' * 60)
    print(f'Trades: {len(trades)} | Wins: {len(wins)} | Losses: {len(losses)} | '
          f'Win rate: {win_rate:.1f}%')
    print(f'Avg PnL: {avg_pnl:+.2f}% | Total PnL: {total_pnl:+.2f}%')


if __name__ == '__main__':
    main()
