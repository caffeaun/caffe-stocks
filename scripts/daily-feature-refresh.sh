#!/usr/bin/env bash
set -euo pipefail

LOGFILE="$HOME/projects/caffe-stocks/logs/feature-refresh.log"
PYTHON="$HOME/projects/caffe-stocks/venv/bin/python"
SCRIPT="$HOME/projects/caffe-stocks/scripts/compute_indicators.py"

# Telegram config
source "$HOME/kanoonth/scripts/telegram.conf"
BOT_TOKEN="${TELEGRAM_BOT_TOKEN//\"/}"
CHAT_ID="${TELEGRAM_CHAT_ID//\"/}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOGFILE"
}

send_telegram() {
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d chat_id="${CHAT_ID}" \
        -d text="${msg}" \
        -d parse_mode="Markdown" \
        > /dev/null 2>&1 || log "WARN: Failed to send Telegram message"
}

log "Starting daily feature refresh"

# Fetch supplemental market data (intraday, sector indices, foreign flows)
log "Fetching supplemental market data"
set +e
MARKET_OUTPUT=$("$PYTHON" "$HOME/projects/caffe-stocks/scripts/fetch_market_data.py" 2>&1)
MARKET_EXIT=$?
set -e
if [ $MARKET_EXIT -ne 0 ]; then
    log "WARNING: fetch_market_data.py failed (exit $MARKET_EXIT): $MARKET_OUTPUT"
fi
[ -n "$MARKET_OUTPUT" ] && log "Market data: $MARKET_OUTPUT"

# Backfill foreign_flows_monthly from daily foreign_flows. The monthly
# XLS import is a manual SET-publication workflow with a ~1-month lag;
# this aggregates daily into monthly so the feature_eng monthly-pctrank
# feature stays fresh and prepare_data doesn't drop recent rows. Cheap
# (<1s), idempotent.
log "Backfilling foreign_flows_monthly from daily aggregate"
set +e
BACKFILL_OUTPUT=$("$PYTHON" "$HOME/projects/caffe-stocks/scripts/backfill_foreign_monthly.py" 2>&1)
BACKFILL_EXIT=$?
set -e
if [ $BACKFILL_EXIT -ne 0 ]; then
    log "WARNING: backfill_foreign_monthly.py failed (exit $BACKFILL_EXIT): $BACKFILL_OUTPUT"
else
    log "Backfill: $BACKFILL_OUTPUT"
fi

# Run compute_indicators.py, capture output and exit code
set +e
OUTPUT=$("$PYTHON" "$SCRIPT" 2>&1)
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ]; then
    log "Feature refresh completed successfully"

    # Check for warnings in output — notify only if present
    if echo "$OUTPUT" | grep -qi "warn"; then
        log "Warnings detected in output:"
        echo "$OUTPUT" >> "$LOGFILE"
        send_telegram "⚠️ *Feature Refresh* — completed with warnings:
\`\`\`
$(echo "$OUTPUT" | grep -i "warn" | head -20)
\`\`\`"
    else
        # Quiet success — log only
        [ -n "$OUTPUT" ] && log "Output: $OUTPUT"
    fi
else
    log "ERROR: Feature refresh failed (exit code $EXIT_CODE)"
    echo "$OUTPUT" >> "$LOGFILE"

    # Truncate output for Telegram (max ~3000 chars)
    TRIMMED=$(echo "$OUTPUT" | tail -30 | head -c 3000)
    send_telegram "🔴 *Feature Refresh FAILED* (exit $EXIT_CODE)
\`\`\`
${TRIMMED}
\`\`\`"
fi

# ML health check (after feature refresh)
log "Running ML health check"
set +e
HEALTH=$("$PYTHON" "$HOME/projects/caffe-stocks/scripts/ml_health_check.py" 2>&1)
HEALTH_EXIT=$?
set -e

if [ $HEALTH_EXIT -ne 0 ]; then
    log "ML health check FAILED"
    echo "$HEALTH" >> "$LOGFILE"
    # Extract failure lines from stderr output
    FAILURES=$(echo "$HEALTH" | grep -A999 "FAILURES" | tail -n +2 | head -10)
    send_telegram "🔴 *ML Health Check FAILED*
${FAILURES}"
else
    log "ML health check passed"
fi

# Modal request approval ping — surfaces/re-pings any pending Modal-trainer
# requests to Telegram with a cost estimate (see scripts/modal_request_ping.py
# + models/modal_requests.json). Self-contained: sends its own Telegram asks
# and never touches a GPU. Non-fatal.
log "Modal request ping"
# Modal's workspace-level spend cap (independent of our modal_budget tracker;
# the cap value isn't exposed by Modal's API). Set here so the ping shows a
# spent/cap/left readout — change this one line if the workspace cap changes.
export MODAL_WORKSPACE_CAP_USD=30
set +e
"$PYTHON" "$HOME/projects/caffe-stocks/scripts/modal_request_ping.py" >> "$LOGFILE" 2>&1 \
    || log "WARN: modal_request_ping.py failed"
set -e

log "Done"
