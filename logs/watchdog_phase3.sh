#!/usr/bin/env bash
# Auto-resume watchdog for the Ph2-final-design Phase 3 (shape-gen VAE) run.
# train_shape_gen.py is single-fold with no mid-run checkpoint-resume (like
# train_ranker.py) -- on death this restarts the whole script from scratch.
set -u
cd "$(dirname "$0")/../back_end"
LOG="../logs/ph2_final_phase3.log"
RESUME_LOG="../logs/watchdog_resume_events.log"
source ../.venv/bin/activate

echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog-p3] started, watching $LOG" >> "$RESUME_LOG"

while true; do
    sleep 60

    if grep -q "Saved → .*shape_vae.pt" "$LOG" 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog-p3] completion marker found — run complete, exiting watchdog." >> "$RESUME_LOG"
        break
    fi

    if pgrep -f "train_shape_gen.py" > /dev/null 2>&1; then
        continue
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog-p3] train_shape_gen.py not running and no completion marker — restarting from scratch" >> "$RESUME_LOG"

    nohup python3 -u train_shape_gen.py >> "$LOG" 2>&1 &
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog-p3] relaunched, PID $!" >> "$RESUME_LOG"

    sleep 20
done
