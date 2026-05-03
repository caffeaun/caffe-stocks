import sqlite3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'
RESULTS_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/backtest_results/latest.json'

# SET-specific friction parameters
COMMISSION_MIN = 50
COMMISSION_RATE = 0.001578
VAT_RATE = 0.07
SLIPPAGE_RATE = 0.002


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM candles ORDER BY timestamp', conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def apply_filters(row):
    # ATR > 3% of close
    if row['atr'] <= 0.03:
        return False
    
    # Volume ratio > 2.0
    if row['volume_ratio'] <= 2.0:
        return False
    
    # RSI between 30-65
    if not (30 <= row['rsi'] <= 65):
        return False
    
    # Breakout: price > 20-day high OR MACD > 0
    if not (row['close'] > row['20_day_high'] or row['macd'] > 0):
        return False
    
    return True

def calculate_commission(entry_price, quantity):
    trade_value = entry_price * quantity
    commission = trade_value * COMMISSION_RATE
    total_commission = commission * (1 + VAT_RATE)
    return max(COMMISSION_MIN, total_commission)

def calculate_slippage(entry_price, quantity):
    return (entry_price * quantity) * SLIPPAGE_RATE

def simulate_trade(df, start_index):
    # Entry: close + 0.5% (limit price) + 0.2% slippage
    entry_price = df.iloc[start_index]['close'] * 1.005 * (1 + SLIPPAGE_RATE)
    
    # Position size: 1% risk / 3% SL = 33% of portfolio
    portfolio_value = 10000  # Starting capital
    risk = portfolio_value * 0.01
    sl = 0.03
    quantity = (risk / (sl * entry_price)) * portfolio_value
    
    # Track position with peak_gain tracking
    position = {
        'entry_price': entry_price,
        'quantity': quantity,
        'entry_date': df.iloc[start_index]['timestamp'],
        'stop_loss': entry_price * 0.97,
        'target': entry_price * 1.15,
        'peak_gain': 0.0  # Track peak gain for trailing stop
    }
    
    # Exit conditions
    exit_price = None
    exit_date = None
    exit_reason = None
    for i in range(start_index + 1, len(df)):
        current = df.iloc[i]
        
        # Check SL
        if current['low'] <= position['stop_loss']:
            exit_price = position['stop_loss']
            exit_date = current['timestamp']
            exit_reason = 'stop_loss'
            break

        # Check target
        if current['high'] >= position['target']:
            exit_price = position['target']
            exit_date = current['timestamp']
            exit_reason = 'target'
            break

        # Track peak gain for trailing stop
        current_gain = (current['close'] - position['entry_price']) / position['entry_price']
        if current_gain >= 0.07:
            position['peak_gain'] = max(position['peak_gain'], current_gain)

        # Check trailing stop (50% of peak gain)
        if position['peak_gain'] >= 0.07:
            trail_floor = position['peak_gain'] * 0.5
            if current_gain < trail_floor:
                exit_price = current['close']
                exit_date = current['timestamp']
                exit_reason = 'trailing_stop'
                break

        # Day 10 exit
        if (current['timestamp'] - position['entry_date']).days >= 10:
            exit_price = current['close']
            exit_date = current['timestamp']
            exit_reason = 'day10'
            break

    if exit_price is None:
        # No exit found (shouldn't happen in real data)
        exit_price = df.iloc[-1]['close']
        exit_date = df.iloc[-1]['timestamp']
        exit_reason = 'end_of_data'
    
    # Calculate P&L
    trade_value = entry_price * quantity
    exit_value = exit_price * quantity
    commission = calculate_commission(entry_price, quantity)
    slippage = calculate_slippage(entry_price, quantity)
    
    pnl = (exit_value - trade_value) - commission - slippage
    pnl_pct = (pnl / trade_value) * 100
    
    return {
        'entry_date': position['entry_date'],
        'exit_date': exit_date,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'quantity': quantity,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'exit_reason': exit_reason
    }

def main():
    df = load_data()
    
    # Add 20-day high for breakout filter
    df['20_day_high'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(20).max())
    
    # Track trades
    trades = []
    in_position = False
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Check for signal
        if apply_filters(row):
            if not in_position:
                # Start new position
                trade = simulate_trade(df, i)
                trades.append(trade)
                in_position = True
        else:
            # Exit if in position
            if in_position:
                in_position = False
    
    # Calculate metrics
    if not trades:
        print('No trades found in backtest')
        return

    total_trades = len(trades)
    win_rate = sum(1 for t in trades if t['pnl_pct'] > 0) / total_trades
    winners = [t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]
    losers = [t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0]
    avg_win = sum(winners) / len(winners) if winners else 0
    avg_loss = sum(losers) / len(losers) if losers else 0
    max_drawdown = min(t['pnl_pct'] for t in trades)
    
    # Calculate EV
    win_rate = sum(1 for t in trades if t['pnl_pct'] > 0) / total_trades
    ev = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Save results
    results = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win_pct': avg_win,
        'max_drawdown_pct': max_drawdown,
        'ev_per_trade': ev
    }
    
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print('Backtest completed:')
    print(f'  Total trades: {total_trades}')
    print(f'  Win rate: {win_rate:.2%}')
    print(f'  Avg win: {avg_win:.2f}%')
    print(f'  Max drawdown: {max_drawdown:.2f}%')
    print(f'  EV per trade: {ev:.2f}%')

if __name__ == '__main__':
    main()