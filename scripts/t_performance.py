import re
import os

MEMORY_FILE = '/home/kanoonth-ai/projects/caffe-stocks/MEMORY.md'


def main():
    if not os.path.exists(MEMORY_FILE):
        print('No trade history yet')
        return

    with open(MEMORY_FILE, 'r') as f:
        content = f.read()

    # Regex to find all trade entries
    pattern = r'### Trade #\d+\n- Symbol: [A-Z]+\n- Entry: [\d.]+ on [\d-]+\n- Exit: [\d.]+ on [\d-]+\n- P\&L: [\d.]+ ([\-\d.]+%)\n- Outcome: (WIN|LOSS)'

    matches = re.findall(pattern, content)
    
    if not matches:
        print('No trade history yet')
        return

    # Extract percentages and outcomes
    pcts = []
    outcomes = []
    for match in matches:
        pct_str = match[0].rstrip('%')
        pct_val = float(pct_str)
        pcts.append(pct_val)
        outcomes.append(match[1])

    n = len(pcts)
    if n < 5:
        print(f"Not enough trades for statistics (need 5+, have {n})")
        return

    # Separate wins and losses
    wins = [p for p, o in zip(pcts, outcomes) if o == 'WIN']
    losses = [p for p, o in zip(pcts, outcomes) if o == 'LOSS']

    wins_count = len(wins)
    losses_count = len(losses)
    win_rate = wins_count / n

    avg_win = sum(wins) / wins_count if wins_count > 0 else 0
    avg_loss = sum(losses) / losses_count if losses_count > 0 else 0

    # EV calculation (subtracting 1.1% commission friction)
    ev = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) - 1.1

    # Max drawdown = most negative P&L
    max_drawdown = min(pcts) if pcts else 0

    # Format output
    print('📈 PERFORMANCE REPORT')
    print(f'Total Trades: {n}')
    print(f'Win Rate: {win_rate:.1%} ({wins_count}W / {losses_count}L)')
    print(f'Avg Win: +{avg_win:.1f}%')
    print(f'Avg Loss: {avg_loss:.1f}%')
    print(f'EV per Trade: {ev:+.2f}%')
    print(f'Max Drawdown: {max_drawdown:.1f}%')
    print('Commission Impact: ~1.1% per trade')

if __name__ == '__main__':
    main()