import os
import re
from datetime import datetime, timedelta
import subprocess

MEMORY_FILE = "/home/kanoonth-ai/projects/caffe-stocks/MEMORY.md"
LAST_RETRAIN_FILE = "/home/kanoonth-ai/projects/caffe-stocks/models/last_retrain.md"


def compute_win_rate():
    """Returns (win_rate, trade_count) for last 30 trades, or (0.0, 0) if no data."""
    if not os.path.exists(MEMORY_FILE):
        return 0.0, 0

    with open(MEMORY_FILE, 'r') as f:
        content = f.read()

    # Extract trade entries
    trades = re.findall(r"- Outcome: (WIN|LOSS)\n", content)
    if not trades:
        return 0.0, 0

    # Calculate win rate for last 30 trades
    recent_trades = trades[-30:]
    wins = recent_trades.count('WIN')
    return (wins / len(recent_trades), len(recent_trades)) if recent_trades else (0.0, 0)

def append_weekly_summary():
    win_rate, trade_count = compute_win_rate()
    today = datetime.now().strftime("%Y-%m-%d")

    summary = f"## Weekly Analysis\n- Win rate (last 30 trades): {win_rate:.1%}\n- Last updated: {today}\n\n"

    with open(MEMORY_FILE, 'a') as f:
        f.write(summary)


def check_retrain_needed():
    if not os.path.exists(LAST_RETRAIN_FILE):
        return True

    with open(LAST_RETRAIN_FILE, 'r') as f:
        content = f.read().strip()

    if not content:
        return True

    # Extract YYYY-MM-DD date from content (handles plain date or markdown)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
    if not date_match:
        return True

    try:
        last_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
    except ValueError:
        return True
    if (datetime.now() - last_date).days > 30:
        print("RETRAIN NEEDED")
        return True
    return False


def trigger_retrain():
    subprocess.run(['/home/kanoonth-ai/shared-venv/bin/python',
                    '/home/kanoonth-ai/projects/caffe-stocks/scripts/circuit_breaker.py',
                    '--trigger', 'retrain'],
                   check=True)


def main():
    append_weekly_summary()

    # WR drift circuit breaker: rolling 30-trade win rate < 40% triggers retrain
    win_rate, trade_count = compute_win_rate()
    if trade_count >= 30 and win_rate < 0.40:
        trigger_retrain()
        return

    if check_retrain_needed():
        trigger_retrain()

if __name__ == '__main__':
    main()