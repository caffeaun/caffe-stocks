import json
import os
from datetime import datetime

def main():
    state_file = '/home/kanoonth-ai/projects/caffe-stocks/memory/trade_state.json'
    if not os.path.exists(state_file):
        state = {'state': 'IDLE', 'portfolio_value': 10000}
    else:
        with open(state_file, 'r') as f:
            state = json.load(f)
    
    portfolio = state.get('portfolio_value', 10000)
    to_40k = 40000 - portfolio
    
    output = f"📊 TRADING STATUS\nState: {state['state']}\nPortfolio: ฿{portfolio:,.0f}\nTo ฿40K: ฿{to_40k:,.0f}"
    
    if state['state'] != 'IDLE':
        entry_date = state.get('entry_date', 'N/A')
        days_held = 0
        if entry_date is not None and entry_date != 'N/A':
            entry_date_obj = datetime.strptime(entry_date, '%Y-%m-%d')
            days_held = (datetime.now() - entry_date_obj).days

        symbol = state.get('symbol') or 'N/A'
        fill_price = state.get('fill_price')
        stop_loss = state.get('stop_loss')
        target = state.get('target')
        output += f"\n\nState: {state['state']}\nSymbol: {symbol}"
        if fill_price is not None:
            output += f"\nEntry: ฿{fill_price:.2f} on {entry_date}"
            if stop_loss is not None:
                output += f"\nStop-Loss: ฿{stop_loss:.2f} (−3%)"
            if target is not None:
                output += f"\nTarget: ฿{target:.2f} (+15%)"
            output += f"\nDays Held: {days_held}"
        output += f"\nTrailing: {'Active' if state.get('trailing_active', False) else 'Inactive'}"
        output += f"\nSignal: {state.get('signal_id', 'N/A')}"
    
    print(output)

if __name__ == '__main__':
    main()