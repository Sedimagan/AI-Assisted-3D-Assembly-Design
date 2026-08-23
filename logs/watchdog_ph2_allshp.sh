#!/usr/bin/env bash
# Auto-resume watchdog for the ph2-all-shp-gen retrain: 749-model corpus
# (renamed/balanced/deduped 2026-08-23/24) with the new 34-dim hole
# features (through/blind, counterbore, indentation, occupancy,
# flat-vs-curved surface). Polls every 60s; if train.py has died without a
# final "Mean AUC" summary, determines the correct --start-fold from how
# many folds have a genuine "Fold N test — AUC=" line in the log
# (replayed/checkpoint-loaded folds on resume print a different
# "(from checkpoint...)" line and don't recount), and relaunches WITHOUT
# --force-reload (the corpus cache is already built after the first
# successful parse pass; force-reload would wipe data.pt and re-trigger
# the slow per-file timeout retries across all 749 models).
set -u
cd "$(dirname "$0")/../back_end"
LOG="../logs/train_ph2_allshp_r1.log"
RESUME_LOG="../logs/watchdog_resume_events.log"
source ../.venv/bin/activate

echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog] started, watching $LOG" >> "$RESUME_LOG"

while true; do
    sleep 60

    if grep -q "Mean AUC" "$LOG" 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog] Mean AUC summary found — run complete, exiting watchdog." >> "$RESUME_LOG"
        break
    fi

    if pgrep -f "train.py" > /dev/null 2>&1; then
        continue
    fi

    completed=$(grep -cE "Fold [0-9]+ test — AUC=" "$LOG" 2>/dev/null || echo 0)
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog] train.py not running, $completed fold(s) confirmed complete — resuming with --start-fold $completed" >> "$RESUME_LOG"

    nohup python3 -u train.py --start-fold "$completed" >> "$LOG" 2>&1 &
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog] relaunched, PID $!" >> "$RESUME_LOG"

    sleep 20
done
