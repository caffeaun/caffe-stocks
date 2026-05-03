import sqlite3
import os
import sys

# Paths
PAPER_SILENT_DB = '/home/kanoonth-ai/projects/caffe-stocks/data/paper-silent.db'
PAPER_LIVE_DB = '/home/kanoonth-ai/projects/caffe-stocks/data/paper-live.db'
SYSTEM_STATUS = '/home/kanoonth-ai/projects/caffe-stocks/data/system-status.json'
CIRCUIT_BREAKER = '/home/kanoonth-ai/projects/caffe-stocks/scripts/circuit_breaker.py'
TRADE_MANAGER = '/home/kanoonth-ai/projects/caffe-stocks/scripts/trade_manager.py'
PAPER_ENV = '/home/kanoonth-ai/projects/caffe-stocks/config/paper.env'


def check_trades(db_path, min_count):
    if not os.path.exists(db_path):
        return False
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM trades')
    count = cursor.fetchone()[0]
    conn.close()
    return count >= min_count

def check_win_rate(db_path, min_rate):
    if not os.path.exists(db_path):
        return False
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT pnl_pct FROM trades')
    pcts = [row[0] for row in cursor.fetchall() if row[0] is not None]
    conn.close()
    if not pcts:
        return False
    wins = sum(1 for p in pcts if p > 0)
    win_rate = wins / len(pcts)
    return win_rate >= min_rate

def check_max_drawdown(db_path, max_dd):
    if not os.path.exists(db_path):
        return False
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT MIN(pnl_pct) FROM trades')
    min_pct = cursor.fetchone()[0]
    conn.close()
    if min_pct is None:
        return False
    return min_pct > -max_dd

def calculate_ev(db_path):
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT pnl_pct FROM trades')
    pcts = [row[0] for row in cursor.fetchall() if row[0] is not None]
    conn.close()
    if not pcts:
        return None
    wins = [p for p in pcts if p > 0]
    losses = [p for p in pcts if p <= 0]
    if not wins or not losses:
        return None
    win_rate = len(wins) / len(pcts)
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    ev = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) - 1.1
    return ev

def check_system_files():
    checks = {
        'system-status.json': os.path.exists(SYSTEM_STATUS),
        'circuit_breaker.py': os.path.exists(CIRCUIT_BREAKER),
        'trade_manager.py has trailing stop': False,
        'paper.env': os.path.exists(PAPER_ENV)
    }
    
    # Check for trailing stop in trade_manager.py
    if os.path.exists(TRADE_MANAGER):
        with open(TRADE_MANAGER, 'r') as f:
            content = f.read()
            if 'check_trailing_stop' in content:
                checks['trade_manager.py has trailing stop'] = True

    return checks


def main():
    # Paper Trading Checks
    paper_checks = {
        'Port A: 50+ trades': check_trades(PAPER_SILENT_DB, 50),
        'Port A: WR >= 50%': check_win_rate(PAPER_SILENT_DB, 0.5),
        'Port A: Max drawdown < 15%': check_max_drawdown(PAPER_SILENT_DB, 15),
                'Port A: EV > 0%': (calculate_ev(PAPER_SILENT_DB) or 0) > 0,
        'Port B: 20+ trades': check_trades(PAPER_LIVE_DB, 20),
        'Port B: WR >= 50%': check_win_rate(PAPER_LIVE_DB, 0.5),
        'Port B: Max drawdown < 15%': check_max_drawdown(PAPER_LIVE_DB, 15),
        'Port B: EV > 0%': (calculate_ev(PAPER_LIVE_DB) or 0) > 0
    }

    # System Checks
    system_checks = check_system_files()

    # Print results
    print("\nGO-LIVE CHECKLIST")
    print("================")
    
    # Paper checks
    for i, (name, passed) in enumerate(paper_checks.items(), 1):
        status = 'PASS' if passed else 'FAIL'
        print(f"{i}. {name}: {status}")

    # System checks
    for i, (name, passed) in enumerate(system_checks.items(), len(paper_checks) + 1):
        status = 'PASS' if passed else 'FAIL'
        print(f"{i}. {name}: {status}")

    # Calculate total passed
    total_passed = sum(paper_checks.values()) + sum(system_checks.values())
    total_checks = len(paper_checks) + len(system_checks)
    
    print(f"\n{total_passed}/{total_checks} criteria passed.")
    if total_passed == total_checks:
        print("✅ READY FOR GO-LIVE")
    else:
        print("❌ NOT READY — fix failing criteria first")

if __name__ == '__main__':
    main()