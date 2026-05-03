import json
import sqlite3
import random
import argparse
import os
from datetime import datetime, timedelta

# Load config from paper.env
PAPER_ENV_PATH = '/home/kanoonth-ai/projects/caffe-stocks/config/paper.env'
config = {}
with open(PAPER_ENV_PATH) as f:
    for line in f:
        if '=' in line:
            key, value = line.strip().split('=', 1)
            config[key] = value

PAPER_START_CAPITAL = float(config['PAPER_START_CAPITAL'])
PAPER_COMMISSION = float(config['PAPER_COMMISSION'])

# Database setup
DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/paper-live.db'


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            signal_id TEXT,
            symbol TEXT,
            entry_price REAL,
            exit_price REAL,
            quantity REAL,
            pnl REAL,
            portfolio_value REAL,
            entry_date TEXT,
            exit_date TEXT
        )
    ''')
    conn.commit()
    conn.close()

def load_signal():
    with open('/home/kanoonth-ai/projects/caffe-stocks/data/latest_signal.json') as f:
        return json.load(f)

def simulate_fill(limit_price):
    slippage = random.uniform(0, 0.005)  # 0-0.5%
    return limit_price * (1 + slippage)

def calculate_commission(entry_price, quantity):
    trade_value = entry_price * quantity
    return trade_value * PAPER_COMMISSION

def check_exit(current_price, entry_price, peak_gain, entry_date):
    """Check all exit conditions"""
    current_gain = (current_price - entry_price) / entry_price
    
    # Stop-loss
    if current_price <= entry_price * 0.97:
        return 'stop_loss'
    
    # Target
    if current_price >= entry_price * 1.15:
        return 'target'
    
    # Trailing stop (after 7% gain)
    if current_gain >= 0.07:
        peak_gain = max(peak_gain, current_gain)
        if current_gain < peak_gain * 0.5:
            return 'trailing_stop'
    
    # Day-10 exit
    days_since = (datetime.now() - datetime.strptime(entry_date, '%Y-%m-%d')).days
    if days_since >= 10:
        return 'MAX_HOLD'
    
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', action='store_true', help='Simulate fill and print signal')
    parser.add_argument('--status', action='store_true', help='Print current portfolio status')
    parser.add_argument('--test', action='store_true', help='Run exit rule tests')
    args = parser.parse_args()

    init_db()
    
    if args.test:
        run_exit_tests()
        return

    if args.simulate:
        signal = load_signal()
        fill_price = simulate_fill(signal['limit_price'])
        
        print(f"\nSignal: {signal['symbol']} - Buy at {signal['limit_price']} (limit), {fill_price:.2f} (filled)")
        print(f"Portfolio: Cash {signal['cash']}, Positions {signal['positions']}, Distance to ฿40K {signal['distance']}")
        
        # Record trade
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (signal_id, symbol, entry_price, quantity, entry_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            signal['id'],
            signal['symbol'],
            fill_price,
            signal['quantity'],
            datetime.now().strftime('%Y-%m-%d')
        ))
        conn.commit()
        conn.close()
        
        print(f"Trade recorded in paper-live.db (entry: {fill_price})")
        return

    if args.status:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT portfolio_value FROM trades ORDER BY id DESC LIMIT 1')
        result = cursor.fetchone()
        print(f"Current portfolio value: {result[0]:.2f} (from paper-live.db)")
        conn.close()


def run_exit_tests():
    print('\n=== Running Exit Rule Tests ===')
    
    # Test 1: Stop-loss
    entry_price = 100
    current_price = 96.5  # 3.5% below
    exit_reason = check_exit(current_price, entry_price, 0.0, '2026-02-28')
    assert exit_reason == 'stop_loss', f'Expected stop_loss, got {exit_reason}'
    print('  STOP_LOSS: Passed')

    # Test 2: Target
    current_price = 116  # 16% above
    exit_reason = check_exit(current_price, entry_price, 0.0, '2026-02-28')
    assert exit_reason == 'target', f'Expected target, got {exit_reason}'
    print('  TARGET: Passed')

    # Test 3: Trailing stop
    entry_price = 100
    current_price = 95  # 5% below peak (peak=7%)
    exit_reason = check_exit(current_price, entry_price, 0.07, '2026-02-28')
    assert exit_reason == 'trailing_stop', f'Expected trailing_stop, got {exit_reason}'
    print('  TRAILING_STOP: Passed')

    # Test 4: Day-10
    entry_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    exit_reason = check_exit(100, entry_price, 0.0, entry_date)
    assert exit_reason == 'MAX_HOLD', f'Expected MAX_HOLD, got {exit_reason}'
    print('  MAX_HOLD: Passed')
    
    print('All exit rule tests passed!')

if __name__ == '__main__':
    main()