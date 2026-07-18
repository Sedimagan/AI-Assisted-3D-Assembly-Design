#!/usr/bin/env bash
# train_monitor.sh — Start training with monitoring, auto-recovery, and live status
# Usage: bash train_monitor.sh
set -u

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACK_END="$PROJ_ROOT/back_end"
LOG_FILE="$PROJ_ROOT/logs/training_run.log"
RECOVERY_LOG="$PROJ_ROOT/logs/recovery.log"
CKPT_DIR="$BACK_END/checkpoints"
CONFIG="$BACK_END/config.yaml"
PYTHON="$PROJ_ROOT/.venv/bin/python"

MAX_RETRIES=3
POLL_INTERVAL=60          # seconds
STALL_TIMEOUT=300         # 5 minutes with no new output = stalled
N_FOLDS=5

attempt=0
start_fold=0
TRAINING_PID=""
OVERALL_START=$(date +%s)

ts() { date "+%Y-%m-%d %H:%M:%S"; }

log_recovery() {
    echo "[$(ts)] $1" | tee -a "$RECOVERY_LOG"
}

# Determine the last completed fold from checkpoints
last_completed_fold() {
    local last=-1
    for i in $(seq 0 $((N_FOLDS - 1))); do
        if [ -f "$CKPT_DIR/best_fold${i}.pt" ]; then
            last=$i
        else
            break
        fi
    done
    echo $last
}

start_training() {
    local sf=$1
    echo ""
    echo "=================================================================="
    echo "[$(ts)] Starting training (attempt $((attempt+1))/$MAX_RETRIES, start_fold=$sf)"
    echo "=================================================================="

    if [ "$sf" -gt 0 ]; then
        cd "$BACK_END" && "$PYTHON" train.py --config "$CONFIG" --start-fold "$sf" 2>&1 | while IFS= read -r line; do echo "[$(date '+%H:%M:%S')] $line"; done >> "$LOG_FILE" &
    else
        cd "$BACK_END" && "$PYTHON" train.py --config "$CONFIG" 2>&1 | while IFS= read -r line; do echo "[$(date '+%H:%M:%S')] $line"; done >> "$LOG_FILE" &
    fi
    TRAINING_PID=$!
    echo "$TRAINING_PID" > "$PROJ_ROOT/logs/training_pid.txt"
    log_recovery "Training PID=$TRAINING_PID started (start_fold=$sf, attempt=$((attempt+1)))"
}

print_fold_summary() {
    local fold_num=$1
    local now=$(date +%s)
    local elapsed=$(( now - OVERALL_START ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    # Parse best AUC for this fold from log
    local best_auc=$(grep "Fold ${fold_num} new best AUC" "$LOG_FILE" | tail -1 | grep -oE 'AUC=[0-9.]+' | tail -1 | cut -d= -f2)
    local test_auc=$(grep "Fold ${fold_num} test" "$LOG_FILE" | tail -1 | grep -oE 'AUC=[0-9.]+' | head -1 | cut -d= -f2)
    local test_ap=$(grep "Fold ${fold_num} test" "$LOG_FILE" | tail -1 | grep -oE 'AP=[0-9.]+' | head -1 | cut -d= -f2)

    # Estimate remaining time
    local folds_done=$fold_num
    if [ "$folds_done" -gt 0 ]; then
        local est_total=$(( elapsed * N_FOLDS / folds_done ))
        local est_remaining=$(( est_total - elapsed ))
        local est_mins=$(( est_remaining / 60 ))
    else
        local est_mins="?"
    fi

    echo ""
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  FOLD $fold_num/$N_FOLDS COMPLETE                                          │"
    echo "│  Best val AUC: ${best_auc:-N/A}   Test AUC: ${test_auc:-N/A}   Test AP: ${test_ap:-N/A}"
    echo "│  Elapsed: ${mins}m ${secs}s   Est. remaining: ${est_mins}m"
    echo "└──────────────────────────────────────────────────────────────┘"
}

# ── Clear old log and start fresh ──
echo "[$(ts)] ═══ Training Monitor Started ═══" > "$LOG_FILE"
echo "[$(ts)] ═══ Recovery Log ═══" > "$RECOVERY_LOG"

# ── Initial launch ──
start_training 0

# ── Monitor loop ──
last_size=0
stall_count=0
last_fold_seen=0

while true; do
    sleep "$POLL_INTERVAL"

    # Check if process is still alive
    if ! kill -0 "$TRAINING_PID" 2>/dev/null; then
        # Process exited — check if it was successful
        wait "$TRAINING_PID" 2>/dev/null
        exit_code=$?

        if grep -q "CV Summary" "$LOG_FILE" 2>/dev/null; then
            echo ""
            echo "=================================================================="
            echo "[$(ts)] Training completed successfully!"
            echo "=================================================================="

            # Print final summary from log
            grep -A 20 "CV Summary" "$LOG_FILE" | head -25

            # Print overall timing
            local_end=$(date +%s)
            total_elapsed=$(( local_end - OVERALL_START ))
            total_mins=$(( total_elapsed / 60 ))
            total_secs=$(( total_elapsed % 60 ))
            echo ""
            echo "Total training time: ${total_mins}m ${total_secs}s"
            echo "Attempts used: $((attempt + 1))/$MAX_RETRIES"
            log_recovery "Training completed successfully in ${total_mins}m ${total_secs}s"
            exit 0
        fi

        # Training failed
        attempt=$((attempt + 1))
        log_recovery "Training process exited with code $exit_code"

        # Check for traceback
        if grep -q "Traceback" "$LOG_FILE" 2>/dev/null; then
            log_recovery "Python traceback detected in log"
            tail -20 "$LOG_FILE" >> "$RECOVERY_LOG"
        fi

        if [ "$attempt" -ge "$MAX_RETRIES" ]; then
            log_recovery "FATAL: All $MAX_RETRIES recovery attempts exhausted. Giving up."
            echo ""
            echo "=================================================================="
            echo "[$(ts)] TRAINING FAILED after $MAX_RETRIES attempts"
            echo "=================================================================="
            echo "TRAINING_STATUS=FAILED" > "$PROJ_ROOT/logs/training_status.txt"
            exit 1
        fi

        # Determine resume fold
        lcf=$(last_completed_fold)
        if [ "$lcf" -ge 0 ]; then
            start_fold=$((lcf + 1))
            log_recovery "Found checkpoint up to fold $lcf. Resuming from fold $start_fold."
        else
            start_fold=0
            log_recovery "No checkpoints found. Restarting from scratch."
        fi

        echo "" >> "$LOG_FILE"
        echo "[$(ts)] ═══ RECOVERY ATTEMPT $((attempt))/$MAX_RETRIES (start_fold=$start_fold) ═══" >> "$LOG_FILE"
        start_training "$start_fold"
        stall_count=0
        last_size=$(wc -c < "$LOG_FILE" 2>/dev/null || echo 0)
        continue
    fi

    # Check for stall (no new output)
    current_size=$(wc -c < "$LOG_FILE" 2>/dev/null || echo 0)
    if [ "$current_size" -eq "$last_size" ]; then
        stall_count=$((stall_count + 1))
        stall_seconds=$((stall_count * POLL_INTERVAL))
        if [ "$stall_seconds" -ge "$STALL_TIMEOUT" ]; then
            log_recovery "STALL detected: no output for ${stall_seconds}s. Killing PID $TRAINING_PID."
            kill "$TRAINING_PID" 2>/dev/null
            sleep 2
            kill -9 "$TRAINING_PID" 2>/dev/null
            # Will be caught on next iteration as process exit
        fi
    else
        stall_count=0
        last_size=$current_size
    fi

    # Track fold completions for summaries
    for f in $(seq 1 $N_FOLDS); do
        if [ "$f" -gt "$last_fold_seen" ]; then
            if grep -q "Fold ${f} test" "$LOG_FILE" 2>/dev/null; then
                print_fold_summary "$f"
                last_fold_seen=$f
            fi
        fi
    done

    # Print current epoch progress
    latest_ep=$(grep "Ep " "$LOG_FILE" | tail -1 | sed 's/^\[[^]]*\] *//')
    if [ -n "$latest_ep" ]; then
        now=$(date +%s)
        elapsed=$(( now - OVERALL_START ))
        mins=$(( elapsed / 60 ))
        secs=$(( elapsed % 60 ))
        echo "[${mins}m${secs}s] $latest_ep"
    fi
done
