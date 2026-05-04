#!/usr/bin/env bash
# ml-improve.sh — Automated ML improvement cycle
#
# Two modes:
#   sweep  (default) — GPU hyperparameter sweep with walk-forward CV, every 3 hours
#   claude           — Claude makes structural changes (architecture, loss, features), daily
#
# Usage:
#   ml-improve.sh              # sweep mode (default)
#   ml-improve.sh --mode sweep
#   ml-improve.sh --mode claude

set -euo pipefail

# Allow nested Claude invocation
unset CLAUDECODE 2>/dev/null || true

MODE="${1:-sweep}"
if [ "$MODE" = "--mode" ]; then
    MODE="${2:-sweep}"
fi

BASE="$HOME/projects/caffe-stocks"
LOCKFILE="$BASE/models/.ml-improve.lock"
LOGFILE="$BASE/logs/ml-improve.log"
PYTHON="$HOME/projects/caffe-stocks/venv/bin/python"
CLAUDE_BIN="$HOME/.local/bin/claude"
CANDIDATE_DIR="$BASE/models/lstm/candidates"
PROD_MODEL="$BASE/models/lstm/trading_model.h5"
PROD_SCALER="$BASE/models/lstm/scaler.pkl"
IMPROVEMENTS="$BASE/data/ml-improvements.json"
FEEDBACK_FILE="$BASE/data/ml-feedback.json"
STATUS_FILE="$BASE/data/system-status.json"
TRAINER_SRC="$BASE/scripts/lstm_trainer.py"
TRAINER_CANDIDATE="$BASE/scripts/lstm_trainer_candidate.py"
BACKTEST_FILE="$BASE/data/backtest_results/lstm_backtest.json"
BACKTEST_SCRIPT="$BASE/scripts/backtest_lstm.py"
PROMPT_BUILDER="$BASE/scripts/ml_prompt_builder.py"
FEEDBACK_EXTRACTOR="$BASE/scripts/ml_feedback_extractor.py"
SWEEP_SCRIPT="$BASE/scripts/lstm_sweep.py"

mkdir -p "$CANDIDATE_DIR" "$(dirname "$LOGFILE")"

# Telegram config
source "$HOME/kanoonth/scripts/telegram.conf"
BOT_TOKEN="${TELEGRAM_BOT_TOKEN//\"/}"
CHAT_ID="${TELEGRAM_CHAT_ID//\"/}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

send_telegram() {
    local msg="$1"
    [ -z "$BOT_TOKEN" ] && return
    [ -z "$CHAT_ID" ] && return
    local result
    result=$(curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=$CHAT_ID" \
        --data-urlencode "text=$msg" \
        --data-urlencode "parse_mode=Markdown" 2>&1)
    if echo "$result" | grep -q '"ok":true'; then
        return 0
    fi
    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=$CHAT_ID" \
        --data-urlencode "text=$msg" \
        > /dev/null 2>&1 || log "WARN: Failed to send Telegram"
}

cleanup() {
    rm -f "$TRAINER_CANDIDATE"
    rm -f /tmp/ml-improve-prompt-*.txt
    [ "$MODE" = "sweep" ] && rm -f "$BASE/models/.ml-improve-sweep.pid"
}
trap cleanup EXIT

# --- Lock & Priority ---
# Claude mode has priority — kills running sweep if needed
MODE_LOCKFILE="$BASE/models/.ml-improve-${MODE}.lock"
SWEEP_LOCKFILE="$BASE/models/.ml-improve-sweep.lock"
SWEEP_PID_FILE="$BASE/models/.ml-improve-sweep.pid"

if [ "$MODE" = "claude" ]; then
    # Check if sweep is running — if so, kill it to take priority
    if ! flock -n 9 9>"$SWEEP_LOCKFILE" 2>/dev/null; then
        SWEEP_PID=""
        [ -f "$SWEEP_PID_FILE" ] && SWEEP_PID=$(cat "$SWEEP_PID_FILE")
        if [ -n "$SWEEP_PID" ] && kill -0 "$SWEEP_PID" 2>/dev/null; then
            log "Claude mode preempting sweep (PID $SWEEP_PID)"
            # Capture sweep progress before killing
            SWEEP_LOG_LATEST=$(ls -t "$BASE/logs/ml-sweep-"*.log 2>/dev/null | head -1)
            SWEEP_PROGRESS=""
            if [ -n "$SWEEP_LOG_LATEST" ] && [ -s "$SWEEP_LOG_LATEST" ]; then
                SWEEP_PROGRESS=$(tail -3 "$SWEEP_LOG_LATEST" 2>/dev/null)
            fi
            # Kill sweep process tree
            kill -- -"$SWEEP_PID" 2>/dev/null || kill "$SWEEP_PID" 2>/dev/null
            sleep 2
            kill -9 "$SWEEP_PID" 2>/dev/null || true
            rm -f "$SWEEP_PID_FILE"
            send_telegram "🔄 *Claude mode preempted sweep*
Sweep was running — killed to give Claude priority.
Last sweep progress:
${SWEEP_PROGRESS:-no output yet}"
            log "Sweep killed, proceeding with Claude mode"
        fi
    fi
fi

exec 9>"$MODE_LOCKFILE"
if ! flock -n 9; then
    log "Another $MODE run is active. Skipping."
    exit 0
fi

# Save PID for sweep so Claude can kill it later
if [ "$MODE" = "sweep" ]; then
    echo $$ > "$SWEEP_PID_FILE"
fi

log "=== ML Improve cycle starting (mode: $MODE) ==="

# --- Pre-flight: health check ---
# Sweep mode: blocks on failure (no point sweeping a broken pipeline).
# Claude mode: ALWAYS runs — health failures are exactly when we need code-level
# intervention. Failure details get passed into Claude's prompt as recovery context.
set +e
HEALTH=$("$PYTHON" "$BASE/scripts/ml_health_check.py" --skip-staleness 2>&1)
HEALTH_EXIT=$?
set -e

HEALTH_FAILURES=""
if [ $HEALTH_EXIT -ne 0 ]; then
    HEALTH_FAILURES=$(echo "$HEALTH" | grep -A999 'FAILURES' | tail -n +2 | head -10)
    if [ "$MODE" = "sweep" ]; then
        log "Pre-flight health check FAILED — sweep skipping"
        send_telegram "⚠️ *ML Sweep skipped* — health check failed:
$HEALTH_FAILURES"
        exit 0
    else
        log "Pre-flight health check FAILED — Claude mode proceeding to recover"
        send_telegram "🔧 *ML Claude recovering* — health check failed, invoking Claude to fix:
$HEALTH_FAILURES"
    fi
fi

# --- Check system status ---
if [ -f "$STATUS_FILE" ]; then
    STATUS=$(python3 -c "import json; print(json.load(open('$STATUS_FILE')).get('status','unknown'))" 2>/dev/null || echo "unknown")
    if [ "$STATUS" != "active" ]; then
        log "System status is '$STATUS', not 'active'. Skipping."
        exit 0
    fi
fi

# ============================================================
# SWEEP MODE — GPU hyperparameter sweep, no Claude needed
# ============================================================
if [ "$MODE" = "sweep" ]; then
    log "Running hyperparameter sweep (GPU)"

    SWEEP_LOG="$BASE/logs/ml-sweep-$(date '+%Y%m%d-%H%M%S').log"

    set +e
    timeout 7200 "$PYTHON" "$SWEEP_SCRIPT" \
        --output-dir "$CANDIDATE_DIR" \
        --configs 20 \
        --use-attention \
        >> "$SWEEP_LOG" 2>&1
    SWEEP_EXIT=$?
    set -e

    if [ $SWEEP_EXIT -eq 124 ]; then
        log "Sweep timed out after 2 hours"
        send_telegram "⚠️ *ML Sweep* — timed out (2h limit)"
        exit 1
    fi

    if [ $SWEEP_EXIT -ne 0 ]; then
        log "Sweep failed (exit $SWEEP_EXIT)"
        send_telegram "⚠️ *ML Sweep* — failed (exit $SWEEP_EXIT)
See: $SWEEP_LOG"
        exit 1
    fi

    CAND_MODEL="$CANDIDATE_DIR/trading_model.h5"
    CAND_SCALER="$CANDIDATE_DIR/scaler.pkl"

    if [ ! -f "$CAND_MODEL" ] || [ ! -f "$CAND_SCALER" ]; then
        log "Sweep produced no candidate"
        exit 1
    fi

    # Extract sweep summary for improvement note
    SWEEP_RESULTS="$CANDIDATE_DIR/sweep_results.json"
    IMPROVEMENT_NOTE="sweep"
    if [ -f "$SWEEP_RESULTS" ]; then
        IMPROVEMENT_NOTE=$("$PYTHON" -c "
import json
with open('$SWEEP_RESULTS') as f:
    r = json.load(f)
cfg = r['best_config']
t = r.get('total_time_s', 0)
print(f'HP sweep ({r[\"n_configs\"]} configs, {r.get(\"n_splits\", r.get(\"n_folds\", \"?\"))} splits, {t:.0f}s): h={cfg[\"hidden_size\"]} L={cfg[\"num_layers\"]} d={cfg[\"dropout\"]} lr={cfg[\"lr\"]} bs={cfg[\"batch_size\"]}')
" 2>/dev/null || echo "HP sweep")
    fi

# ============================================================
# CLAUDE MODE — structural improvements via Claude
# ============================================================
elif [ "$MODE" = "claude" ]; then
    log "Building dynamic prompt from feedback history"

    # Use last near-successful candidate if available (preserves code improvements
    # across runs instead of always starting from scratch)
    BEST_CANDIDATE="$CANDIDATE_DIR/best_candidate_trainer.py"
    if [ -f "$BEST_CANDIDATE" ]; then
        log "Starting from best previous candidate (not production)"
        cp "$BEST_CANDIDATE" "$TRAINER_CANDIDATE"
    else
        cp "$TRAINER_SRC" "$TRAINER_CANDIDATE"
    fi

    PROMPT_FILE=$(mktemp /tmp/ml-improve-prompt-XXXXXX.txt)
    "$PYTHON" "$PROMPT_BUILDER" \
        --backtest-file "$BACKTEST_FILE" \
        --model-file "$PROD_MODEL" \
        --feedback-file "$FEEDBACK_FILE" \
        --trainer-src "$TRAINER_SRC" \
        --candidate-path "$TRAINER_CANDIDATE" \
        --candidate-dir "$CANDIDATE_DIR" \
        --python-path "$PYTHON" \
        --backtest-script "$BACKTEST_SCRIPT" \
        > "$PROMPT_FILE" 2>/dev/null

    if [ ! -s "$PROMPT_FILE" ]; then
        log "ERROR: Prompt builder produced empty output"
        send_telegram "⚠️ *ML Improve* — prompt builder failed"
        exit 1
    fi

    # If health check failed, prepend a recovery banner so Claude pivots from
    # "improve the model" to "fix the broken pipeline first".
    if [ -n "$HEALTH_FAILURES" ]; then
        RECOVERY_BANNER=$(mktemp /tmp/ml-recovery-banner-XXXXXX.txt)
        cat > "$RECOVERY_BANNER" <<EOF
=== RECOVERY MODE — PIPELINE IS BROKEN ===

The pre-flight health check failed. The production model or scaler is corrupt,
missing, or unloadable. Your job in this run is NOT to improve the model — it
is to GET THE PIPELINE BACK TO A WORKING STATE so the normal improvement loop
can resume next run.

HEALTH CHECK FAILURES:
$HEALTH_FAILURES

RECOVERY STRATEGY (in priority order):
1. Diagnose: read ml_health_check.py to understand which check failed and why.
2. Inspect production files: ~/projects/caffe-stocks/models/lstm/trading_model.h5 and
   scaler.pkl. Run 'file' on them, try opening with h5py, check size, check
   if it's actually a torch.save zip instead of HDF5.
3. If model file is corrupt: re-train fresh by running
   ~/projects/caffe-stocks/venv/bin/python ~/projects/caffe-stocks/scripts/lstm_trainer.py
   --output-dir ~/projects/caffe-stocks/models/lstm
   This produces a valid baseline. The hyperparameter improvement loop will
   tune from there.
4. If features mismatch: check feature_eng.py CURATED_FEATURES vs what the
   saved model expects. Either revert features or retrain.
5. After recovery, verify: ~/projects/caffe-stocks/venv/bin/python ~/projects/caffe-stocks/scripts/ml_health_check.py
   should exit 0.
6. Save a forensic copy of the corrupt file as
   trading_model.h5.corrupted-\$(date +%Y%m%d) before overwriting it.

Do NOT attempt the normal improvement workflow below until the pipeline is
healthy. The instructions below assume a working baseline; ignore them and
focus on recovery first.

=== END RECOVERY BANNER ===

EOF
        cat "$RECOVERY_BANNER" "$PROMPT_FILE" > "${PROMPT_FILE}.new"
        mv "${PROMPT_FILE}.new" "$PROMPT_FILE"
        rm -f "$RECOVERY_BANNER"
        log "Recovery banner prepended to Claude prompt"
    fi

    log "Invoking Claude Opus for ML improvement"
    CLAUDE_LOG="$BASE/logs/ml-improve-claude-$(date '+%Y%m%d-%H%M%S').log"

    set +e
    timeout 1800 "$CLAUDE_BIN" -p \
        --model claude-opus-4-7 \
        --dangerously-skip-permissions \
        --allowedTools "Bash Edit Write Read Glob Grep" \
        < "$PROMPT_FILE" \
        >> "$CLAUDE_LOG" 2>&1
    CLAUDE_EXIT=$?
    set -e

    if [ $CLAUDE_EXIT -eq 124 ]; then
        log "Claude timed out after 30 minutes"
        send_telegram "⚠️ *ML Improve* — Claude timed out (30 min limit)"
        exit 1
    fi

    rm -f "$PROMPT_FILE"
    log "Claude exit code: $CLAUDE_EXIT (log: $CLAUDE_LOG)"

    if [ $CLAUDE_EXIT -ne 0 ]; then
        log "Claude invocation failed"
        send_telegram "⚠️ *ML Improve* — Claude failed (exit $CLAUDE_EXIT)
See: $CLAUDE_LOG"
        exit 1
    fi

    CAND_MODEL="$CANDIDATE_DIR/candidate_model.h5"
    CAND_SCALER="$CANDIDATE_DIR/candidate_scaler.pkl"

    if [ ! -f "$CAND_MODEL" ]; then
        if [ -f "$CANDIDATE_DIR/trading_model.h5" ]; then
            CAND_MODEL="$CANDIDATE_DIR/trading_model.h5"
            CAND_SCALER="$CANDIDATE_DIR/scaler.pkl"
        fi
    fi

    if [ ! -f "$CAND_MODEL" ] || [ ! -f "$CAND_SCALER" ]; then
        log "Candidate model not found"
        send_telegram "⚠️ *ML Improve* — no candidate model produced
See: $CLAUDE_LOG"
        exit 1
    fi

    IMPROVEMENT_NOTE=$("$PYTHON" "$FEEDBACK_EXTRACTOR" extract-note --claude-log "$CLAUDE_LOG" 2>/dev/null || echo "ML improvement attempt")
else
    log "Unknown mode: $MODE"
    exit 1
fi

# ============================================================
# COMMON: Gate + Promote + Notify
# ============================================================
# Shared gate lock — prevents sweep and claude from running gate simultaneously
exec 8>"$LOCKFILE"
flock 8
log "Running model gate"
GATE_JSON_FILE=$(mktemp /tmp/ml-gate-XXXXXX.json)
set +e
GATE_OUTPUT=$("$PYTHON" "$BASE/scripts/model_gate.py" \
    --candidate "$CAND_MODEL" \
    --candidate-scaler "$CAND_SCALER" \
    --improvement-note "$IMPROVEMENT_NOTE" \
    --promote 2>&1)
GATE_EXIT=$?
set -e

log "Gate exit code: $GATE_EXIT"
log "Gate output: $GATE_OUTPUT"

echo "$GATE_OUTPUT" | sed '/^Running backtest/d' | sed -n '/^{/,/^}/p' > "$GATE_JSON_FILE"

# Extract structured feedback (for Claude mode)
if [ "$MODE" = "claude" ] && [ -n "${CLAUDE_LOG:-}" ]; then
    log "Extracting structured feedback"
    "$PYTHON" "$FEEDBACK_EXTRACTOR" extract \
        --claude-log "$CLAUDE_LOG" \
        --gate-output "$GATE_JSON_FILE" \
        --gate-exit "$GATE_EXIT" \
        >> "$LOGFILE" 2>&1 || log "WARN: Feedback extraction failed"
fi

# Parse gate output for Telegram
GATE_SUMMARY=$("$PYTHON" - "$GATE_JSON_FILE" <<'PYEOF'
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    c = data['candidate']
    print(f"Precision: {c['precision']:.1%}")
    wf = data.get('walk_forward')
    if wf and wf.get('splits'):
        agg = wf.get('aggregate', {})
        status = '✅' if wf.get('passed') else '❌'
        if agg:
            print(f"Walk-forward: {status} avg WR {agg.get('avg_win_rate',0):.1%} "
                  f"(std {agg.get('wr_std',0):.1%}, "
                  f"alpha {agg.get('avg_alpha',0):.1%}, "
                  f"{agg.get('total_trades',0)} trades, "
                  f"{agg.get('valid_splits',0)}/7 splits)")
        else:
            print(f"Walk-forward: {status} {wf.get('reason','')}")
        # Per-split breakdown — always show all 7
        for i, s in enumerate(wf.get('splits', []), 1):
            tc = s.get('trade_count', 0)
            if tc == 0:
                period = s.get('test_period', '?')
                print(f"  S{i}: 0 trades ({period})")
            else:
                wr = s.get('win_rate', 0)
                ev = s.get('ev_per_trade', 0)
                alpha = s.get('alpha_over_base', 0)
                base = s.get('base_rate', 0)
                sel = s.get('selectivity_ratio', 0)
                conc = s.get('date_concentration', 0)
                # Flag failures
                flags = []
                if wr < 0.25: flags.append('WR')
                if alpha < 0.05: flags.append('α')
                if ev <= 0: flags.append('EV')
                if sel > 0.30: flags.append('sel')
                if conc > 0.20: flags.append('conc')
                if s.get('max_drawdown_pct', 0) / 100 > 0.15: flags.append('DD')
                flag_str = f" ⚠{','.join(flags)}" if flags else " ✅"
                print(f"  S{i}: WR {wr:.0%} α{alpha:+.0%} EV {ev:+.1f}% "
                      f"({tc}t, sel {sel:.0%}, base {base:.0%}){flag_str}")
    elif wf:
        print(f"Walk-forward: ❌ {wf.get('reason','no data')}")
    else:
        print("Walk-forward: skipped (failed stage 1)")
except Exception as e:
    print(f"parse error: {e}")
PYEOF
)

REJECT_REASON=$("$PYTHON" - "$GATE_JSON_FILE" <<'PYEOF'
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    print(data.get('reason', 'unknown'))
except:
    print("unknown")
PYEOF
)

# Save near-successful candidate code for next run to build on (Claude mode only)
# If candidate WR improved over production, keep the code even if gate rejected
if [ "$MODE" = "claude" ] && [ -f "$TRAINER_CANDIDATE" ]; then
    BEST_CANDIDATE="$CANDIDATE_DIR/best_candidate_trainer.py"
    SHOULD_SAVE=$("$PYTHON" - "$GATE_JSON_FILE" 2>/dev/null <<'PYEOF' || echo "no"
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    wf = data.get('walk_forward', {})
    wf_avg = wf.get('aggregate', {}).get('avg_win_rate', 0)
    # Save if walk-forward avg WR > 35% (shows promise)
    print("yes" if wf_avg > 0.35 else "no")
except:
    print("no")
PYEOF
    )
    if [ "$SHOULD_SAVE" = "yes" ] || [ $GATE_EXIT -eq 0 ]; then
        cp "$TRAINER_CANDIDATE" "$BEST_CANDIDATE"
        log "Saved near-successful candidate trainer for next run"
    fi
    if [ $GATE_EXIT -eq 0 ]; then
        rm -f "$BEST_CANDIDATE"
    fi
fi

rm -f "$GATE_JSON_FILE"

# Release shared gate lock
flock -u 8

# Telegram alerts
MODE_TAG=$(echo "$MODE" | tr '[:lower:]' '[:upper:]')
if [ $GATE_EXIT -eq 0 ]; then
    send_telegram "🟢 *ML Model Promoted* ($MODE_TAG)
$IMPROVEMENT_NOTE

$GATE_SUMMARY"
elif [ $GATE_EXIT -eq 1 ]; then
    send_telegram "🟡 *ML Candidate Rejected* ($MODE_TAG)
$IMPROVEMENT_NOTE

$GATE_SUMMARY
Reason: $REJECT_REASON"
else
    send_telegram "🔴 *ML Gate Error* ($MODE_TAG)
$(echo "$GATE_OUTPUT" | tail -3)"
fi

# Data requests (Claude mode only)
if [ "$MODE" = "claude" ]; then
    DATA_REQ_FILE="$BASE/data/ml-data-requests.txt"
    if [ -f "$DATA_REQ_FILE" ]; then
        DATA_REQ=$(cat "$DATA_REQ_FILE")
        if [ -n "$DATA_REQ" ]; then
            send_telegram "💡 *ML Data Request*
The ML improvement system suggests:

$DATA_REQ

_Reply if you can provide this data. It may help reach 52% walk-forward WR._"
            log "Data request sent: $DATA_REQ"
            mv "$DATA_REQ_FILE" "$BASE/data/ml-data-requests-$(date '+%Y%m%d-%H%M%S').txt"
        fi
    fi
fi

# Cleanup
rm -f "$TRAINER_CANDIDATE"
find "$CANDIDATE_DIR" -name "*.h5" -mtime +7 -delete 2>/dev/null || true
find "$CANDIDATE_DIR" -name "*.pkl" -mtime +7 -delete 2>/dev/null || true

log "=== ML Improve cycle complete (mode: $MODE) ==="
