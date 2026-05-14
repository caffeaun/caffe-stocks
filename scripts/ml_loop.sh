#!/usr/bin/env bash
# ml_loop.sh — top-level dispatcher for the ML iteration loop.
#
# Usage:
#   ml_loop.sh train [--configs N] [--trainer xgboost|lightgbm]
#   ml_loop.sh claude
#   ml_loop.sh research                # daily SOTA research + scaffold install
#   ml_loop.sh panel [--show-only]
#   ml_loop.sh status
#
# Cron entries (in ops/cron/trading.cron, post-migration):
#   0  */3 * * *  bash ~/projects/caffe-stocks/scripts/ml_loop.sh claude
#   30 *   * * *  bash ~/projects/caffe-stocks/scripts/ml_loop.sh train --configs 5
#   0  3   * * *  bash ~/projects/caffe-stocks/scripts/ml_loop.sh research
#   1  0   1 * *  bash ~/projects/caffe-stocks/scripts/ml_loop.sh panel
#
# Hard wall-times:
#   claude:   30 min  (1800s)
#   train:    55 min  (3300s) — must finish before next :30 fires
#   research: 60 min  (3600s) — Claude wall is 55, +5 for ml_scaffold post-script

set -uo pipefail

unset CLAUDECODE 2>/dev/null || true

if [ $# -lt 1 ]; then
    echo "Usage: $0 {train|claude|research|panel|status} [args...]" >&2
    exit 1
fi
MODE="$1"
shift

BASE="$HOME/projects/caffe-stocks"
LOG="$BASE/logs/ml-loop.log"
PY="$BASE/venv/bin/python"

mkdir -p "$BASE/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$MODE] $*" | tee -a "$LOG"; }

cd "$BASE"
export PYTHONPATH="$BASE"

case "$MODE" in
    train)
        log "starting (args: $*)"
        timeout 3300 "$PY" scripts/train_mode.py "$@" 2>&1 | tee -a "$LOG"
        EX=${PIPESTATUS[0]}
        log "exit $EX"
        exit "$EX"
        ;;

    claude)
        log "starting"
        timeout 1800 "$PY" scripts/claude_mode.py "$@" 2>&1 | tee -a "$LOG"
        EX=${PIPESTATUS[0]}
        log "exit $EX"
        exit "$EX"
        ;;

    research)
        log "starting"
        timeout 3600 "$PY" scripts/research_mode.py "$@" 2>&1 | tee -a "$LOG"
        EX=${PIPESTATUS[0]}
        log "exit $EX"
        exit "$EX"
        ;;

    panel)
        log "refreshing panel"
        "$PY" scripts/promotion.py "$@" 2>&1 | tee -a "$LOG"
        ;;

    status)
        "$PY" scripts/feedback.py
        ;;

    *)
        echo "Usage: $0 {train|claude|research|panel|status} [args...]" >&2
        exit 1
        ;;
esac
