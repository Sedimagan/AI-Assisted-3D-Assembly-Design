#!/usr/bin/env bash
# Watches the ph2-all-shp-gen reload log for the dataset-parsing-complete
# marker ("[2/4] Building model" -- train.py only prints this once
# AssemblyDataset.process() has returned and processed/data.pt exists) and,
# the moment it appears, makes a full timestamped copy of the processed
# dataset directory (data.pt, categories.json, sources.json, pre_filter.pt,
# pre_transform.pt, and the per-graph cache) before the 5-fold CV training
# phase starts. Training is the long, crash-prone part of this run -- if it
# dies partway through and something ALSO corrupts the live processed dir
# (or a future run passes --force-reload again before this corpus is
# reused), this backup means the ~749-file parse doesn't have to be redone.
set -u
cd "$(dirname "$0")/../back_end"
LOG="../logs/train_ph2_allshp_r1.log"
MARK_LOG="../logs/backup_ph2allshp_events.log"
PROCESSED_DIR="data/processed/processed"
BACKUP_ROOT="data/processed_backup_ph2allshp_$(date +%Y%m%d_%H%M%S)"

echo "$(date '+%Y-%m-%d %H:%M:%S')  [backup-watcher] started, watching $LOG for parse-complete" >> "$MARK_LOG"

while true; do
    sleep 30

    if grep -q "\[2/4\] Building model" "$LOG" 2>/dev/null; then
        if [ -f "$PROCESSED_DIR/data.pt" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S')  [backup-watcher] parse complete, backing up $PROCESSED_DIR -> $BACKUP_ROOT" >> "$MARK_LOG"
            mkdir -p "$BACKUP_ROOT"
            cp -R "$PROCESSED_DIR/." "$BACKUP_ROOT/"
            n_cache=$(ls "$BACKUP_ROOT/graph_cache" 2>/dev/null | wc -l | tr -d ' ')
            echo "$(date '+%Y-%m-%d %H:%M:%S')  [backup-watcher] done -- $n_cache cached graphs + data.pt backed up to $BACKUP_ROOT" >> "$MARK_LOG"
            break
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S')  [backup-watcher] saw the marker but data.pt missing yet -- retrying" >> "$MARK_LOG"
        fi
    fi

    # Give up only after train.py has been gone for several consecutive
    # checks (a real abort), not on a single missed poll -- a manual
    # restart (kill old train.py, relaunch fresh) has a brief window where
    # the process genuinely isn't running yet, and this watcher used to
    # mistake that gap for the run being aborted and exit permanently,
    # silently leaving no backup coverage until manually noticed and
    # relaunched (happened once, 2026-08-25). Five misses (~2.5min) is
    # comfortably longer than any restart transition seen this session.
    if ! pgrep -f "train.py" > /dev/null 2>&1; then
        if ! grep -q "\[2/4\] Building model" "$LOG" 2>/dev/null; then
            miss_count=$((${miss_count:-0} + 1))
            if [ "$miss_count" -ge 5 ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S')  [backup-watcher] train.py gone for $miss_count consecutive checks and parse never completed -- exiting without backup" >> "$MARK_LOG"
                break
            fi
        fi
    else
        miss_count=0
    fi
done
