#!/home/kanoonth-ai/projects/caffe-stocks/venv/bin/python
import argparse
import os
import re

MEMORY_FILE = "/home/kanoonth-ai/projects/caffe-stocks/memory/MEMORY.md"


def init_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'w') as f:
            f.write("# Trading System Memory\n\n")
            f.write("## Portfolio Status\n")
            f.write("Current State: IDLE\n")
            f.write("Cash: ฿10,000 | Positions: 0 | SCB: 0 shares\n\n")
            f.write("## Trade History\n")
            f.write("<!-- Template for each trade:\n")
            f.write("### Trade #001 | 2026-03-15 | COM7\n")
            f.write("Entry: ฿45.50 | Stop: ฿44.14 (−3%) | Target: ฿52.33 (+15%)\n")
            f.write("Position: ฿3,333 (33%) | Risk: ฿100 (1%)\n")
            f.write("Exit: ฿49.25 (+8.2%) @ trailing stop | Port gain: +2.7%\n")
            f.write("LSTM: 0.72 | RSI: 42 | Volume: 1.8x\n")
            f.write("Lesson: Trail captured partial gain; stock reversed after +10%.\n")
            f.write("-->\n\n")
            f.write("## Lessons Learned\n")
            f.write("- (Agent logs every loss with root cause)\n")


def get_next_trade_number():
    if not os.path.exists(MEMORY_FILE):
        return 1
    with open(MEMORY_FILE, 'r') as f:
        content = f.read()
    # Strip HTML comments before counting to avoid matching template entries
    content_no_comments = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    matches = re.findall(r'### Trade #(\d+)', content_no_comments)
    if not matches:
        return 1
    return max(int(m) for m in matches) + 1


def is_duplicate_trade(symbol, entry_date):
    if not os.path.exists(MEMORY_FILE):
        return False
    with open(MEMORY_FILE, 'r') as f:
        content = f.read()
    content_no_comments = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    pattern = rf'### Trade #\d+\n- Symbol: {re.escape(symbol)}\n- Entry: [^\n]+ on {re.escape(entry_date)}'
    return bool(re.search(pattern, content_no_comments))


def append_trade(symbol, fill_price, entry_date, sell_price, exit_date, pnl, pnl_pct, outcome):
    if is_duplicate_trade(symbol, entry_date):
        print(f"Duplicate trade detected: {symbol} on {entry_date}. Skipping.")
        return None
    n = get_next_trade_number()
    with open(MEMORY_FILE, 'a') as f:
        f.write(f"\n### Trade #{n:03d}\n")
        f.write(f"- Symbol: {symbol}\n")
        f.write(f"- Entry: ฿{fill_price} on {entry_date}\n")
        f.write(f"- Exit: ฿{sell_price} on {exit_date}\n")
        f.write(f"- P&L: ฿{pnl} ({pnl_pct}%)\n")
        f.write(f"- Outcome: {outcome}\n")
    return n


def log_trade():
    init_memory()
    n = append_trade(
        symbol="COM7",
        fill_price=45.50,
        entry_date="2026-03-15",
        sell_price=49.25,
        exit_date="2026-03-16",
        pnl=3.75,
        pnl_pct=8.24,
        outcome="WIN",
    )
    if n is not None:
        print(f"Test trade logged as Trade #{n:03d} to MEMORY.md.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', action='store_true', help='Initialize MEMORY.md')
    parser.add_argument('--test', action='store_true', help='Log a test trade')
    args = parser.parse_args()
    
    if args.init:
        init_memory()
        print("MEMORY.md initialized with template.")
    elif args.test:
        log_trade()
    else:
        print("Use --init or --test")

if __name__ == '__main__':
    main()