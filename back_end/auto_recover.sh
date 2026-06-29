#!/usr/bin/env bash
# auto_recover.sh — Monitor training and auto-recover on failure
# Polls log every 60s, restarts on crash (up to 3 attempts)

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACK_END="$PROJECT_ROOT/back_end"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
LOG="$PROJECT_ROOT/logs/training_run.log"
PID_FILE="$PROJECT_ROOT/logs/training_pid.txt"
RECOVERY_LOG="$PROJECT_ROOT/logs/recovery.log"
CKPT_DIR="$BACK_END/checkpoints"

MAX_RETRIES=3
STALE_TIMEOUT=900  # 15 minutes with no output = stale (allows slow STEP parsers)
POLL_INTERVAL=60

attempt=0
start_time=$(date +%s)

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

is_training_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

last_log_mod() {
    stat -f %m "$LOG" 2>/dev/null || echo 0
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
    tail -20 "$LOG" 2>/dev/null | grep -q "Traceback\|Error:\|FAILED"
}

log_recovery "Auto-recovery monitor started"

while true; do
    sleep $POLL_INTERVAL

    elapsed=$(( $(date +%s) - start_time ))

    if is_training_running; then
        # Check for stale output
        last_mod=$(last_log_mod)
        now=$(date +%s)
        stale_seconds=$((now - last_mod))

        if [ "$stale_seconds" -gt "$STALE_TIMEOUT" ]; then
            log_recovery "WARNING: No log output for ${stale_seconds}s (threshold: ${STALE_TIMEOUT}s)"
            pid=$(cat "$PID_FILE")
            kill "$pid" 2>/dev/null
            sleep 2

            attempt=$((attempt + 1))
            if [ "$attempt" -ge "$MAX_RETRIES" ]; then
                log_recovery "FATAL: Max retries ($MAX_RETRIES) exceeded. Giving up."
                echo "=== Training FAILED after $MAX_RETRIES recovery attempts at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ===" >> "$LOG"
                exit 1
            fi

            next_fold=$(get_last_completed_fold)
            log_recovery "Restarting from fold $next_fold after stale timeout"
            start_training "$next_fold"
        fi
    else
        # Process exited — check if it succeeded or failed
        if grep -q "CV Summary" "$LOG" && grep -q "Results saved" "$LOG"; then
            log_recovery "Training completed successfully! Elapsed: ${elapsed}s"
            exit 0
        fi

        if check_for_traceback; then
            log_recovery "Training crashed (traceback detected)"
        else
            log_recovery "Training process exited unexpectedly"
        fi

        attempt=$((attempt + 1))
        if [ "$attempt" -ge "$MAX_RETRIES" ]; then
            log_recovery "FATAL: Max retries ($MAX_RETRIES) exceeded. Giving up."
            echo "=== Training FAILED after $MAX_RETRIES recovery attempts at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ===" >> "$LOG"
            exit 1
        fi

        next_fold=$(get_last_completed_fold)
        log_recovery "Restarting from fold $next_fold (attempt $((attempt+1))/$MAX_RETRIES)"
        start_training "$next_fold"
    fi
done
