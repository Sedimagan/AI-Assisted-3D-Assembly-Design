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
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$0")/../back_end"
LOG="../logs/train_ph2_allshp_r1.log"
RESUME_LOG="../logs/watchdog_resume_events.log"
source ../.venv/bin/activate

# TEMPORARY (2026-08-30): proactive stall-restart, folds 3-5 only.
# Every crash-triggered restart this run has cleared some accumulated
# swap/MPS degradation and dropped per-epoch time by 10-20x (30-120min/epoch
# -> 3-6min/epoch) -- restarting isn't just recovery, it's a real speedup.
# Fold 2 is excluded: it's already being handled by a separate reduced-
# patience workaround and shouldn't have a second restart mechanism racing
# it. Two independent triggers, either one fires a restart:
#   1. STALL_TIMEOUT: total log silence (no new bytes at all) -- catches a
#      genuine hang like the fold-2 test-eval stall (1h+ with zero output).
#   2. EPOCH_STALL_TIMEOUT: the CURRENT epoch alone has been running longer
#      than this, even while the log keeps growing with batch-progress
#      lines -- added 2026-08-30 per user request, since epochs were
#      observed climbing from ~5min to 30-70min across a single fold as
#      swap pressure built up, well before 20min of total silence would
#      ever trigger. Tracked via a small state file recording the last
#      completed epoch number and when we first saw it; if that number
#      hasn't advanced within EPOCH_STALL_TIMEOUT seconds, the epoch
#      after it is considered stuck.
# NOTE: with epoch times already reaching 8-30min even during otherwise
# healthy training later in a fold, an 8min threshold WILL fire repeatedly
# through the back half of a fold, not just on a truly pathological epoch --
# accepted as a reasonable trade since every restart this run has been a
# net win (checkpoint-safe, pace snaps back afterward).
STALL_TIMEOUT=1200
EPOCH_STALL_TIMEOUT=480
STALL_MIN_FOLD=3
EPOCH_STATE_FILE="../logs/.fold3plus_epoch_watch"

do_stall_restart() {
    local reason="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog] $reason -- committing progress and restarting" >> "$RESUME_LOG"
    (
        cd "$REPO_ROOT" && \
        git add back_end/checkpoints/ back_end/data/processed/ logs/train_ph2_allshp_r1.log logs/watchdog_resume_events.log 2>/dev/null && \
        git commit -m "Auto-restart backup: $reason" 2>/dev/null && \
        git push 2>/dev/null
    ) >> "$RESUME_LOG" 2>&1
    pkill -9 -f "train.py" 2>/dev/null
    sleep 2
}

echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog] started, watching $LOG" >> "$RESUME_LOG"

while true; do
    sleep 60

    if grep -q "Mean AUC" "$LOG" 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog] Mean AUC summary found — run complete, exiting watchdog." >> "$RESUME_LOG"
        break
    fi

    if pgrep -f "train.py" > /dev/null 2>&1; then
        cur_fold=$(grep -oE "FOLD [0-9]+ / 5" "$LOG" 2>/dev/null | tail -1 | sed -E 's/FOLD ([0-9]+) \/ 5/\1/')
        cur_fold=${cur_fold:-0}
        case "$cur_fold" in ''|*[!0-9]*) cur_fold=0 ;; esac
        if [ "$cur_fold" -ge "$STALL_MIN_FOLD" ] && [ -f "$LOG" ]; then
            log_mtime=$(stat -f %m "$LOG" 2>/dev/null || echo 0)
            now_ts=$(date +%s)
            silence=$(( now_ts - log_mtime ))
            if [ "$silence" -ge "$STALL_TIMEOUT" ]; then
                do_stall_restart "fold $cur_fold stalled ${silence}s (total log silence), restarting to clear swap/MPS degradation"
                rm -f "$EPOCH_STATE_FILE"
                continue
            fi

            cur_epoch=$(grep -oE "Ep +[0-9]+/200" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1)
            cur_epoch=${cur_epoch:-0}
            case "$cur_epoch" in ''|*[!0-9]*) cur_epoch=0 ;; esac
            stored_epoch=""
            stored_ts="$now_ts"
            if [ -f "$EPOCH_STATE_FILE" ]; then
                read -r stored_epoch stored_ts < "$EPOCH_STATE_FILE" 2>/dev/null
                stored_ts=${stored_ts:-$now_ts}
            fi
            if [ "$cur_epoch" != "$stored_epoch" ]; then
                echo "$cur_epoch $now_ts" > "$EPOCH_STATE_FILE"
            else
                epoch_elapsed=$(( now_ts - stored_ts ))
                if [ "$epoch_elapsed" -ge "$EPOCH_STALL_TIMEOUT" ]; then
                    do_stall_restart "fold $cur_fold epoch $((cur_epoch + 1)) running ${epoch_elapsed}s (>= ${EPOCH_STALL_TIMEOUT}s), restarting to clear swap/MPS degradation"
                    echo "$cur_epoch $now_ts" > "$EPOCH_STATE_FILE"
                fi
            fi
        fi
        continue
    fi
    rm -f "$EPOCH_STATE_FILE"

    # grep -c always prints a count (even "0") and still exits 1 when that
    # count is zero -- the old `|| echo 0` fallback fired on that exit code
    # too, appending a SECOND "0" and producing a literal two-line "0\n0"
    # that crashed train.py's argparse (invalid int). Only fall back to 0
    # when grep produced no output at all (e.g. the log file is missing).
    completed=$(grep -cE "Fold [0-9]+ test — AUC=" "$LOG" 2>/dev/null)
    completed=${completed:-0}
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog] train.py not running, $completed fold(s) confirmed complete — resuming with --start-fold $completed" >> "$RESUME_LOG"

    nohup python3 -u train.py --start-fold "$completed" --true-5way-test >> "$LOG" 2>&1 &
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog] relaunched, PID $!" >> "$RESUME_LOG"

    sleep 20
done
