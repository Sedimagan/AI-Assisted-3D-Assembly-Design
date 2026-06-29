#!/usr/bin/env bash
# run_training_monitored.sh — Start training with timestamped logging
# Usage: bash run_training_monitored.sh [extra train.py args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
LOG_FILE="../logs/training_run.log"
PID_FILE="../logs/training_pid.txt"
RECOVERY_LOG="../logs/recovery.log"

mkdir -p ../logs

echo "=== Training run started at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ===" | tee -a "$LOG_FILE"

# Run training with timestamps on every line; -u forces unbuffered stdout/stderr
PYTHONUNBUFFERED=1 "$PYTHON" -u train.py --config config.yaml "$@" 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
done >> "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "Training PID: $TRAIN_PID (logged to $PID_FILE)"
echo "Log file: $LOG_FILE"

wait $TRAIN_PID
EXIT_CODE=$?
echo "=== Training finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC') with exit code $EXIT_CODE ===" >> "$LOG_FILE"
exit $EXIT_CODE
