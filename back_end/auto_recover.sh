#!/usr/bin/env bash
# auto_recover.sh — Polls log every 60s, restarts on crash (up to 3 attempts)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
LOG="$PROJECT_ROOT/logs/training_run.log"
RECOVERY_LOG="$PROJECT_ROOT/logs/recovery.log"
PID_FILE="$PROJECT_ROOT/logs/training_pid.txt"
CKPT_DIR="$SCRIPT_DIR/checkpoints"
BACK_END="$SCRIPT_DIR"

MAX_RETRIES=3
STALE_TIMEOUT=2400  # seconds with no log output = stale (large STEP files can take >900s)
POLL_INTERVAL=60

attempt=0

log_recovery() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RECOVERY_LOG"
}

get_last_completed_fold() {
    local max_fold=-1
    for f in "$CKPT_DIR"/best_fold*.pt; do
        [ -f "$f" ] || continue
        fold_num=$(echo "$f" | grep -o '[0-9]' | tail -1)
        if [ "$fold_num" -gt "$max_fold" ]; then
            max_fold=$fold_num
        fi
    done
    echo $((max_fold + 1))
}

start_training() {
    local start_fold="${1:-0}"
    local extra_args=""
    if [ "$start_fold" -gt 0 ]; then
        extra_args="--start-fold $start_fold"
    fi
    log_recovery "Starting training (attempt $((attempt+1))/$MAX_RETRIES, start-fold=$start_fold)"

    cd "$BACK_END"
    echo "=== Recovery restart at $(date -u '+%Y-%m-%d %H:%M:%S UTC') (attempt $((attempt+1)), fold $start_fold) ===" >> "$LOG"
    PYTHONUNBUFFERED=1 $PYTHON -u train.py --config config.yaml $extra_args 2>&1 | while IFS= read -r line; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
    done >> "$LOG" &

    local pid=$!
    echo "$pid" > "$PID_FILE"
    log_recovery "Training PID: $pid"
}

check_for_traceback() {
    # Check last 20 lines for Python traceback
    tail -20 "$LOG" 2>/dev/null | grep -q "Traceback\|Error:" && return 0 || return 1
}

log_recovery "auto_recover.sh started. Monitoring $LOG every ${POLL_INTERVAL}s"

while true; do
    sleep $POLL_INTERVAL

    # Check if training process is alive
    TRAIN_PID=""
    [ -f "$PID_FILE" ] && TRAIN_PID=$(cat "$PID_FILE" 2>/dev/null) || true

    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        # Process alive — check staleness
        NOW=$(date +%s)
        MTIME=$(stat -f %m "$LOG" 2>/dev/null || echo 0)
        STALE=$((NOW - MTIME))

        if [ $STALE -gt $STALE_TIMEOUT ]; then
            log_recovery "Log stale ${STALE}s > ${STALE_TIMEOUT}s — killing and restarting"
            kill "$TRAIN_PID" 2>/dev/null || true
            sleep 3

            if [ $attempt -ge $MAX_RETRIES ]; then
                log_recovery "FAILED: max retries ($MAX_RETRIES) exhausted. Giving up."
                exit 1
            fi
            attempt=$((attempt + 1))
            next_fold=$(get_last_completed_fold)
            log_recovery "Restarting from fold $next_fold after stale timeout"
            start_training "$next_fold"
        fi
    else
        # Process dead — check if it completed normally
        if grep -q "Results saved" "$LOG" 2>/dev/null; then
            log_recovery "Training completed normally. Exiting auto_recover."
            exit 0
        fi

        if check_for_traceback; then
            log_recovery "Traceback detected in log"
        fi

        if [ $attempt -ge $MAX_RETRIES ]; then
            log_recovery "FAILED: max retries ($MAX_RETRIES) exhausted. Giving up."
            exit 1
        fi
        attempt=$((attempt + 1))
        next_fold=$(get_last_completed_fold)
        log_recovery "Restarting from fold $next_fold (attempt $((attempt+1))/$MAX_RETRIES)"
        start_training "$next_fold"
    fi
done
