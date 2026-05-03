import re
import os

MEMORY_FILE = '/home/kanoonth-ai/projects/caffe-stocks/MEMORY.md'


def main():
    if not os.path.exists(MEMORY_FILE):
        print('No trade history yet')
        return

    with open(MEMORY_FILE, 'r') as f:
        content = f.read()

    # Regex pattern to match trade entries
    pattern = r'## Trade #(\d+)\n- Symbol: ([A-Z]+)\n- Entry: ([\d.]+) on ([\d-]+)\n- Exit: ([\d.]+) on ([\d-]+)\n- P&L: [\d.]+ ([\d.]+%)\n- Outcome: (WIN|LOSS)'

    matches = re.findall(pattern, content)
    
    if not matches:
        print('No trade history yet')
        return

    # Take last 10 trades
    recent_trades = matches[-10:]
    
    print('📜 TRADE HISTORY (last 10)')
    for i, trade in enumerate(recent_trades):
        # trade = (number, symbol, entry_price, entry_date, exit_price, exit_date, pnl_pct, outcome)
        entry_date = trade[3]
        symbol = trade[1]
        entry_price = trade[2]
        exit_price = trade[4]
        pnl_pct = trade[6]
        outcome = trade[7]
        
        # Format P&L to show positive/negative
        if pnl_pct.startswith('0.0'):
            pnl_fmt = '0.0%'
        else:
            pnl_fmt = pnl_pct
        
        print(f"#{trade[0]} | {entry_date} | {symbol} | ฿{entry_price} → ฿{exit_price} | {pnl_fmt} | {outcome}")

if __name__ == '__main__':
    main()