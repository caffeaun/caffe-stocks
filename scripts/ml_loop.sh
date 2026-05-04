#!/usr/bin/env bash
# ml_loop.sh — top-level dispatcher for the ML iteration loop.
#
# Usage:
#   ml_loop.sh train [--configs N] [--trainer xgboost|lightgbm]
#   ml_loop.sh claude
#   ml_loop.sh panel [--show-only]
#   ml_loop.sh status
#   ml_loop.sh resume   # clear the consecutive-failure pause
#
# Cron entries (in ops/cron/trading.cron, post-migration):
#   0  */3 * * *  bash ~/projects/caffe-stocks/scripts/ml_loop.sh claude
#   30 *   * * *  bash ~/projects/caffe-stocks/scripts/ml_loop.sh train --configs 5
#   1  0   1 * *  bash ~/projects/caffe-stocks/scripts/ml_loop.sh panel
#
# Hard wall-times:
#   claude: 30 min  (1800s)
#   train:  55 min  (3300s) — must finish before next :30 fires

set -uo pipefail

unset CLAUDECODE 2>/dev/null || true

if [ $# -lt 1 ]; then
    echo "Usage: $0 {train|claude|panel|status|resume} [args...]" >&2
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

    panel)
        log "refreshing panel"
        "$PY" scripts/promotion.py "$@" 2>&1 | tee -a "$LOG"
        ;;

    status)
        "$PY" scripts/feedback.py
        ;;

    resume)
        # Insert a synthetic 'passed' iteration to clear consecutive_failures.
        # Actually simpler: directly mutate. But cleaner to just count off.
        "$PY" - <<'PYEOF'
import sys, os, sqlite3
from pathlib import Path
DB = Path(os.path.expanduser('~/projects/caffe-stocks/data/ml-feedback.db'))
if not DB.exists():
    print('No DB yet — nothing to resume.')
    sys.exit(0)
conn = sqlite3.connect(DB)
# Soft-acknowledge: mark recent iterations so consecutive_failures returns 0.
# We do this by inserting a marker row that is "passed" but trainer=manual.
import datetime
now = datetime.datetime.now().isoformat()
conn.execute("""
    INSERT INTO iterations
    (started_at, finished_at, mode, trainer, hyperparams, gate_passed,
     windows_passed, windows_total, avg_annualized_return, avg_win_rate,
     avg_max_dd, total_trades, model_dir, elapsed_seconds, full_result)
    VALUES (?, ?, 'train', 'manual_resume', '{}', 1, 0, 0, 0, 0, 0, 0, '', 0, '{}')
""", (now, now))
conn.commit()
conn.close()
print('Pause cleared. Next train_mode/claude_mode run will proceed.')
PYEOF
        ;;

    *)
        echo "Usage: $0 {train|claude|panel|status|resume} [args...]" >&2
        exit 1
        ;;
esac
