#!/bin/bash
# Weekly trading performance report — runs via system cron (Saturday 08:00 ICT)
# No LLM — just reads actual data files and sends to Telegram

set -euo pipefail

PYTHON="$HOME/shared-venv/bin/python"
TRADE_STATE="$HOME/projects/caffe-stocks/memory/trade_state.json"
PORTFOLIO_HISTORY="$HOME/projects/caffe-stocks/data/portfolio_history.json"
SYSTEM_STATUS="$HOME/projects/caffe-stocks/data/system-status.json"
LOG="$HOME/projects/caffe-stocks/logs/weekly-report.log"

# Telegram config
source "$HOME/kanoonth/scripts/telegram.conf"
BOT_TOKEN="${TELEGRAM_BOT_TOKEN//\"/}"
CHAT_ID="${TELEGRAM_CHAT_ID//\"/}"

send_telegram() {
    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d chat_id="$CHAT_ID" \
        -d text="$1" \
        -d parse_mode="Markdown" \
        --max-time 10 > /dev/null 2>&1 || true
}

echo "$(date '+%Y-%m-%d %H:%M') Starting weekly trading report" >> "$LOG"

# Gather all data via Python
REPORT=$($PYTHON << 'PYEOF'
import json, os
from datetime import datetime

trade_state_file = os.path.expanduser("~/projects/caffe-stocks/memory/trade_state.json")
portfolio_file = os.path.expanduser("~/projects/caffe-stocks/data/portfolio_history.json")
system_status_file = os.path.expanduser("~/projects/caffe-stocks/data/system-status.json")
memory_file = os.path.expanduser("~/projects/caffe-stocks/MEMORY.md")

# System status
sys_status = "unknown"
if os.path.isfile(system_status_file):
    with open(system_status_file) as f:
        sys_status = json.load(f).get("status", "unknown")

# Trade state
state = {}
if os.path.isfile(trade_state_file):
    with open(trade_state_file) as f:
        state = json.load(f)

current_state = state.get("state", "UNKNOWN")
portfolio_value = state.get("portfolio_value", 0)
cash = state.get("cash", 0)
fill_price = state.get("fill_price")
stop_loss = state.get("stop_loss")
target = state.get("target")
entry_date = state.get("entry_date")

# Portfolio history
history = []
if os.path.isfile(portfolio_file):
    with open(portfolio_file) as f:
        history = json.load(f)

# Weekly change
if len(history) >= 2:
    latest = history[-1]["value"]
    week_ago_val = history[0]["value"]
    for h in history:
        if h["date"] <= (datetime.now().strftime("%Y-%m-%d")):
            week_ago_val = h["value"]
        if len(history) > 7:
            week_ago_val = history[-7]["value"]
    weekly_change = latest - week_ago_val
    weekly_pct = (weekly_change / week_ago_val * 100) if week_ago_val else 0
else:
    weekly_change = 0
    weekly_pct = 0

# Trade history from MEMORY.md
import re
wins = 0
losses = 0
if os.path.isfile(memory_file):
    with open(memory_file) as f:
        content = f.read()
    outcomes = re.findall(r"- Outcome: (WIN|LOSS)", content)
    wins = outcomes.count("WIN")
    losses = outcomes.count("LOSS")

total_trades = wins + losses
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

# Build report
today = datetime.now().strftime("%Y-%m-%d")
lines = [f"📊 *Weekly Trading Report* ({today})", ""]
lines.append(f"*System status:* {sys_status}")
lines.append(f"*Trade state:* {current_state}")
lines.append(f"*Portfolio value:* ฿{portfolio_value:,.2f}")
lines.append(f"*Cash:* ฿{cash:,.2f}")

if current_state == "POSITION_OPEN" and fill_price:
    lines.append("")
    lines.append("*Open position:*")
    lines.append(f"  Entry: ฿{fill_price} on {entry_date or 'N/A'}")
    if stop_loss:
        lines.append(f"  Stop loss: ฿{stop_loss}")
    if target:
        lines.append(f"  Target: ฿{target:.2f}")

lines.append("")
if total_trades > 0:
    lines.append(f"*All-time trades:* {total_trades} ({wins}W / {losses}L, {win_rate:.0f}%)")
else:
    lines.append("*All-time trades:* None completed yet")

lines.append(f"*Target:* ฿40,000 ({portfolio_value/40000*100:.1f}%)")

print("\n".join(lines))
PYEOF
)

send_telegram "$REPORT"
echo "$(date '+%Y-%m-%d %H:%M') Weekly trading report sent" >> "$LOG"
echo "$REPORT" >> "$LOG"
