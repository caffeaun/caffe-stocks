#!/usr/bin/env bash
set -euo pipefail

# Daily close check — runs Mon-Fri after market close
# Replaces Qwen-based OpenClaw cron job trd-daily-close-check

LOGFILE="$HOME/projects/caffe-stocks/logs/daily-close.log"
mkdir -p "$(dirname "$LOGFILE")"
exec >> "$LOGFILE" 2>&1
echo "=== $(date '+%Y-%m-%d %H:%M:%S') daily-close-check ==="

# Telegram config
source "$HOME/kanoonth/scripts/telegram.conf"
BOT_TOKEN="${TELEGRAM_BOT_TOKEN//\"/}"
CHAT_ID="${TELEGRAM_CHAT_ID//\"/}"

PYTHON="$HOME/projects/caffe-stocks/venv/bin/python"
STATE_FILE="$HOME/projects/caffe-stocks/memory/trade_state.json"
CANDLES_DB="$HOME/projects/caffe-stocks/data/candles.db"

send_telegram() {
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d chat_id="${CHAT_ID}" \
        -d parse_mode=Markdown \
        -d text="${msg}" > /dev/null
}

if [ ! -f "$STATE_FILE" ]; then
    echo "ERROR: trade_state.json not found"
    send_telegram "Daily Close Check: trade_state.json not found"
    exit 1
fi

MSG=$("$PYTHON" - "$STATE_FILE" "$CANDLES_DB" <<'PYEOF'
import json, sys, sqlite3, os
from datetime import datetime

state_file = sys.argv[1]
candles_db = sys.argv[2]

with open(state_file, 'r') as f:
    state = json.load(f)

current_state = state.get('state', 'IDLE')
today = datetime.now().strftime('%Y-%m-%d')

if current_state == 'IDLE':
    print(f"📋 *Daily Close Check* ({today})\n\nNo open position. Portfolio idle.")
    sys.exit(0)

if current_state in ('SIGNAL_SENT', 'ORDER_PENDING'):
    symbol = state.get('symbol') or 'N/A'
    signal_id = state.get('signal_id') or 'N/A'
    print(f"📋 *Daily Close Check* ({today})\n\nStatus: *{current_state}*\nSymbol: {symbol}\nSignal: {signal_id}\n\n⏳ Waiting for order fill.")
    sys.exit(0)

# POSITION_OPEN
symbol = state.get('symbol', 'N/A')
fill_price = state.get('fill_price')
entry_date = state.get('entry_date', 'N/A')
stop_loss = state.get('stop_loss')
target = state.get('target')
trailing = state.get('trailing_active', False)
peak_price = state.get('peak_price')

# Days held
days_held = 0
if entry_date and entry_date != 'N/A':
    try:
        days_held = (datetime.now() - datetime.strptime(entry_date, '%Y-%m-%d')).days
    except ValueError:
        pass

# Latest closing price from candles.db
latest_close = None
close_date = None
if os.path.exists(candles_db) and symbol != 'N/A':
    try:
        conn = sqlite3.connect(candles_db)
        row = conn.execute(
            "SELECT close, date FROM candles WHERE symbol = ? ORDER BY date DESC LIMIT 1",
            (symbol,)
        ).fetchone()
        conn.close()
        if row:
            latest_close = row[0]
            close_date = row[1]
    except Exception as e:
        print(f"DB error: {e}", file=sys.stderr)

# Build message
lines = [f"📋 *Daily Close Check* ({today})", ""]
lines.append(f"Symbol: *{symbol}*")
lines.append(f"State: *{current_state}*")

if fill_price is not None:
    lines.append(f"Entry: ฿{fill_price:.2f} ({entry_date})")
    lines.append(f"Days Held: {days_held}")
    if stop_loss is not None:
        lines.append(f"Stop Loss: ฿{stop_loss:.2f}")
    if target is not None:
        lines.append(f"Target: ฿{target:.2f}")
    lines.append(f"Trailing: {'Active' if trailing else 'Inactive'}")
    if peak_price is not None:
        lines.append(f"Peak: ฿{peak_price:.2f}")

    lines.append("")

    if latest_close is not None:
        pnl = ((latest_close - fill_price) / fill_price) * 100
        emoji = "🟢" if pnl >= 0 else "🔴"
        lines.append(f"Latest Close: ฿{latest_close:.2f} ({close_date})")
        lines.append(f"Unrealized P&L: {emoji} {pnl:+.2f}%")

        # Check stop loss / target hits
        alerts = []
        if stop_loss is not None and latest_close <= stop_loss:
            alerts.append(f"⚠️ *STOP LOSS HIT* — close ฿{latest_close:.2f} <= SL ฿{stop_loss:.2f}")
        if target is not None and latest_close >= target:
            alerts.append(f"🎯 *TARGET HIT* — close ฿{latest_close:.2f} >= target ฿{target:.2f}")

        if alerts:
            lines.append("")
            lines.extend(alerts)
    else:
        lines.append("⚠️ No candle data found for symbol")
else:
    lines.append("(fill price not yet recorded)")

print("\n".join(lines))
PYEOF
)

echo "Message: $MSG"
send_telegram "$MSG"
echo "Sent OK"

# v1 paper-trade panel: evaluate exits on all open paper trades. Silent —
# no per-trade Telegram alerts. The daily summary at 20:30 reports outcomes.
echo "--- paper_trade_exit ---"
export PYTHONPATH="$HOME/projects/caffe-stocks"
"$PYTHON" "$HOME/projects/caffe-stocks/scripts/paper_trade_exit.py" || \
    echo "WARN: paper_trade_exit.py failed"
