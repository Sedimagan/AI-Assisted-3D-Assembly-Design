#!/usr/bin/env bash
# Auto-resume watchdog for the Ph2-final-design Phase 2 (NodeRanker) run.
# train_ranker.py is single-fold with no mid-run checkpoint-resume support
# (unlike train.py's --start-fold), so on death this just restarts the whole
# script from scratch -- cheap, since a full run is only 30 epochs.
set -u
cd "$(dirname "$0")/../back_end"
LOG="../logs/ph2_final_phase2.log"
RESUME_LOG="../logs/watchdog_resume_events.log"
source ../.venv/bin/activate

echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog-p2] started, watching $LOG" >> "$RESUME_LOG"

while true; do
    sleep 60

    if grep -q "Saved → .*node_ranker.pt" "$LOG" 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog-p2] completion marker found — run complete, exiting watchdog." >> "$RESUME_LOG"
        break
    fi

    if pgrep -f "train_ranker.py" > /dev/null 2>&1; then
        continue
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog-p2] train_ranker.py not running and no completion marker — restarting from scratch" >> "$RESUME_LOG"

    nohup python3 -u train_ranker.py >> "$LOG" 2>&1 &
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog-p2] relaunched, PID $!" >> "$RESUME_LOG"

    sleep 20
done
