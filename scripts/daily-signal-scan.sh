#!/usr/bin/env bash
set -euo pipefail

LOGFILE="$HOME/projects/caffe-stocks/logs/signal-scan.log"
PYTHON="$HOME/shared-venv/bin/python"
SCRIPT="$HOME/projects/caffe-stocks/scripts/signal_generator.py"

# Telegram config
source "$HOME/kanoonth/scripts/telegram.conf"
BOT_TOKEN="${TELEGRAM_BOT_TOKEN//\"/}"
CHAT_ID="${TELEGRAM_CHAT_ID//\"/}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOGFILE"
}

send_telegram() {
    local msg="$1"
    curl -s -X POST \
        "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d chat_id="${CHAT_ID}" \
        -d text="${msg}" \
        -d parse_mode="Markdown" \
        > /dev/null 2>&1 || log "WARNING: Failed to send Telegram message"
}

log "Starting daily signal scan"

# Run signal generator and capture output + exit code
set +e
OUTPUT=$("$PYTHON" "$SCRIPT" 2>&1)
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -ne 0 ]; then
    log "ERROR: signal_generator.py exited with code $EXIT_CODE"
    log "Output: $OUTPUT"
    send_telegram "⚠️ Signal scan error (exit $EXIT_CODE): ${OUTPUT:0:500}"
    exit 1
fi

log "signal_generator.py completed successfully"

if echo "$OUTPUT" | grep -q "Skipping"; then
    # Extract status from output (e.g., "Skipping: system status is INACTIVE")
    STATUS=$(echo "$OUTPUT" | grep "Skipping" | head -1)
    MSG="Signal scan skipped: ${STATUS}"
    log "$MSG"
    send_telegram "$MSG"
else
    log "Sending signal results to Telegram"
    send_telegram "$OUTPUT"
fi

log "Daily signal scan complete"
