import json
import os
import sys
from datetime import datetime, timedelta

# Path to portfolio history
HISTORY_FILE = '/home/kanoonth-ai/projects/caffe-stocks/data/portfolio_history.json'
STATUS_FILE = '/home/kanoonth-ai/projects/caffe-stocks/data/system-status.json'
PAPER_STATUS_FILE = '/home/kanoonth-ai/projects/caffe-stocks/data/paper-status.json'
MEMORY_FILE = '/home/kanoonth-ai/projects/caffe-stocks/MEMORY.md'


def get_portfolio_history():
    """Load or initialize portfolio history"""
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)
        return []
    with open(HISTORY_FILE, 'r') as f:
        return json.load(f)

def append_portfolio_snapshot(portfolio_value):
    """Append current portfolio value to history"""
    history = get_portfolio_history()
    today = datetime.now().strftime('%Y-%m-%d')
    history.append({'date': today, 'value': portfolio_value})
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def update_system_status(status):
    """Update system status in system-status.json"""
    with open(STATUS_FILE, 'w') as f:
        json.dump({'status': status}, f)

def update_paper_status(status):
    """Update paper status in paper-status.json"""
    with open(PAPER_STATUS_FILE, 'w') as f:
        json.dump({'status': status}, f)

def check_daily_loss(history):
    """Check if daily loss exceeds 3%"""
    if not history:
        return False
    today = datetime.now().strftime('%Y-%m-%d')
    today_entry = next((entry for entry in reversed(history) if entry['date'] == today), None)
    if not today_entry:
        return False
    prev_entry = next((entry for entry in reversed(history[:-1]) if entry['date'] != today), None)
    if not prev_entry:
        return False
    return (prev_entry['value'] - today_entry['value']) / prev_entry['value'] > 0.03

def check_weekly_loss(history):
    """Check if weekly loss exceeds 5%"""
    if not history:
        return False
    today = datetime.now().strftime('%Y-%m-%d')
    today_entry = next((entry for entry in reversed(history) if entry['date'] == today), None)
    if not today_entry:
        return False
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    week_entry = next((entry for entry in history if entry['date'] == week_ago), None)
    if not week_entry:
        return False
    return (week_entry['value'] - today_entry['value']) / week_entry['value'] > 0.05

def check_monthly_loss(history):
    """Check if monthly loss exceeds 8%"""
    if not history:
        return False
    today = datetime.now().strftime('%Y-%m-%d')
    today_entry = next((entry for entry in reversed(history) if entry['date'] == today), None)
    if not today_entry:
        return False
    month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    month_entry = next((entry for entry in history if entry['date'] == month_ago), None)
    if not month_entry:
        return False
    return (month_entry['value'] - today_entry['value']) / month_entry['value'] > 0.08

def check_win_rate():
    """Check if rolling 30-trade win rate < 40%"""
    if not os.path.exists(MEMORY_FILE):
        return False
    
    with open(MEMORY_FILE, 'r') as f:
        lines = f.readlines()

    # Find start of Trade Log
    trade_log_start = None
    for i, line in enumerate(lines):
        if line.strip() == '## Trade Log':
            trade_log_start = i + 1
            break
    if trade_log_start is None:
        return False

    trades = []
    i = trade_log_start
    while i < len(lines) and not lines[i].startswith('##'):
        if lines[i].startswith('### Trade'):
            entry_price = None
            exit_price = None
            j = i + 1
            while j < len(lines) and not lines[j].startswith('### Trade'):
                line = lines[j].strip()
                if line.startswith('Entry:'):
                    entry_price = float(line.split(':')[1].strip().replace('฿', '').replace(',', ''))
                elif line.startswith('Exit:'):
                    exit_price = float(line.split(':')[1].strip().replace('฿', '').replace(',', ''))
                j += 1
            if entry_price is not None and exit_price is not None:
                trades.append((entry_price, exit_price))
            i = j
        else:
            i += 1

    # Take most recent 30 trades
    recent_trades = trades[-30:] if len(trades) >= 30 else trades
    if len(recent_trades) < 30:
        return False

    wins = sum(1 for (entry, exit) in recent_trades if exit > entry)
    win_rate = wins / len(recent_trades)
    return win_rate < 0.40

def main():
    # Load current portfolio value
    with open('/home/kanoonth-ai/projects/caffe-stocks/memory/trade_state.json', 'r') as f:
        trade_state = json.load(f)
    portfolio_value = trade_state.get('portfolio_value', 0)

    # Append snapshot to history
    append_portfolio_snapshot(portfolio_value)

    # Check thresholds
    history = get_portfolio_history()
    if check_monthly_loss(history):
        if os.getenv('PAPER_MODE') == 'true':
            update_paper_status('stopped')
        else:
            update_system_status('stopped')
    elif check_weekly_loss(history):
        if os.getenv('PAPER_MODE') == 'true':
            update_paper_status('paused')
        else:
            update_system_status('paused')
    elif check_daily_loss(history):
        if os.getenv('PAPER_MODE') == 'true':
            update_paper_status('paused')
        else:
            update_system_status('paused')
    elif check_win_rate():
        if os.getenv('PAPER_MODE') == 'true':
            update_paper_status('retrain')
        else:
            update_system_status('retrain')

if __name__ == '__main__':
    if '--trigger' in sys.argv and 'retrain' in sys.argv:
        if os.getenv('PAPER_MODE') == 'true':
            update_paper_status('retrain')
        else:
            update_system_status('retrain')
    else:
        main()