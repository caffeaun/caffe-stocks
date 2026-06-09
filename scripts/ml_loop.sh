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
#   claude:   120 min (7200s) — outer kill; inner claude_mode HARD_TIMEOUT=5400s
#   train:    55 min  (3300s) — must finish before next :30 fires
#   research: 60 min  (3600s) — Claude wall is 55, +5 for ml_scaffold post-script

set -uo pipefail

unset CLAUDECODE 2>/dev/null || true

if [ $# -lt 1 ]; then
    echo "Usage: $0 {train|claude|research|panel|status|leaderboard} [args...]" >&2
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

# Source per-trainer token files. Each is sourceable shell (KEY=value lines)
# and only exports keys that aren't already set (so an outer env wins).
for ENVFILE in "$BASE/.env.tabfpn" "$BASE/.env.tabpfn"; do
    [ -f "$ENVFILE" ] || continue
    while IFS='=' read -r k v; do
        [ -z "$k" ] && continue
        case "$k" in \#*) continue ;; esac
        eval ": \${$k:=$v}"
        export "$k"
    done < "$ENVFILE"
done

case "$MODE" in
    train)
        # GPU policy (2026-06-09): GPU 0 is the *training* GPU, shared with
        # other projects — so the sweep runs daytime only (08:00–22:00) to
        # avoid overnight contention, plus a one-off blackout date. Cron still
        # fires hourly at :30; outside the window this is a clean no-op (no GPU
        # touch, no work). Override via ML_TRAIN_{START,END}_HOUR / _BLACKOUT_DATE.
        TRAIN_START_HOUR="${ML_TRAIN_START_HOUR:-8}"
        TRAIN_END_HOUR="${ML_TRAIN_END_HOUR:-22}"
        BLACKOUT_DATE="${ML_TRAIN_BLACKOUT_DATE:-2026-06-09}"
        NOW_DATE="$(date +%F)"
        NOW_HOUR="$((10#$(date +%H)))"   # 10# = force base-10 (avoid octal on 08/09)
        if [ "$NOW_DATE" = "$BLACKOUT_DATE" ]; then
            log "blackout date $BLACKOUT_DATE — skipping train sweep (no GPU touch)"
            exit 0
        fi
        if [ "$NOW_HOUR" -lt "$TRAIN_START_HOUR" ] || [ "$NOW_HOUR" -ge "$TRAIN_END_HOUR" ]; then
            log "outside training window ${TRAIN_START_HOUR}:00-${TRAIN_END_HOUR}:00 (hour=$NOW_HOUR) — skipping (no GPU touch)"
            exit 0
        fi
        export CUDA_VISIBLE_DEVICES=0          # pin the sweep to the training GPU
        log "starting (args: $*) [GPU 0, window ${TRAIN_START_HOUR}:00-${TRAIN_END_HOUR}:00]"
        timeout 3300 "$PY" scripts/train_mode.py "$@" 2>&1 | tee -a "$LOG"
        EX=${PIPESTATUS[0]}
        log "exit $EX"
        exit "$EX"
        ;;

    claude)
        # GPU 0 = training sweep, GPU 1 = inference. claude_mode must not touch
        # the local GPUs: CUDA is hidden so a stray local torch run can't grab
        # one, and ML_FORCE_MODAL signals it to route any real training to
        # modal.com (see the prompt's compute-constraint section).
        export CUDA_VISIBLE_DEVICES=''
        export ML_FORCE_MODAL=1
        log "starting [no local GPU; ML_FORCE_MODAL=1 -> modal.com for training]"
        timeout 7200 "$PY" scripts/claude_mode.py "$@" 2>&1 | tee -a "$LOG"
        EX=${PIPESTATUS[0]}
        log "exit $EX"
        exit "$EX"
        ;;

    research)
        # Same GPU policy as claude_mode: no local GPU. research/ml_scaffold
        # only import-smoke-tests new trainers (no GPU needed); any actual
        # training routes to modal.com.
        export CUDA_VISIBLE_DEVICES=''
        export ML_FORCE_MODAL=1
        log "starting [no local GPU; modal.com for training]"
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

    leaderboard)
        "$PY" scripts/leaderboard.py "$@"
        ;;

    *)
        echo "Usage: $0 {train|claude|research|panel|status|leaderboard} [args...]" >&2
        exit 1
        ;;
esac
