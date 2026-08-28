#!/usr/bin/env bash
# Auto-continuation pipeline for ph2-all-shp-gen: waits for the live Phase 1
# true-5way retrain (train.py, watched by watchdog_ph2_allshp.sh) to finish
# all 5 folds, then runs the downstream steps that depend on the new frozen
# encoder, in order:
#   1. Backup + rebuild the part bank (corpus expanded 238->749 models,
#      so the existing part_bank/ built from the smaller corpus is stale)
#   2. Phase 2 -- NodeRanker retrain (true-5way, matching Phase 1's protocol)
#   3. Phase 3 -- Shape-gen VAE retrain (uses the rebuilt part bank + the
#      newly-promoted best_serving.pt automatically, per each script's
#      own --serving-ckpt default)
#
# Completion signal mirrors the watchdog's own: the literal "Mean AUC" line
# train.py prints only once, at the very end after all 5 folds + promotion
# logic have run. Polls every 60s.
set -u
cd "$(dirname "$0")/../back_end"
TRAIN_LOG="../logs/train_ph2_allshp_r1.log"
EVENTS_LOG="../logs/phase23_pipeline_events.log"
source ../.venv/bin/activate

echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] waiting for Phase 1 (train.py) to complete all 5 folds..." >> "$EVENTS_LOG"

while true; do
    sleep 60
    if grep -q "Mean AUC" "$TRAIN_LOG" 2>/dev/null; then
        break
    fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] Phase 1 complete (Mean AUC line found). Starting downstream pipeline." >> "$EVENTS_LOG"

# ── Step 1: backup + rebuild part bank ─────────────────────────────────────
BACKUP_DIR="data/part_bank_backup_pre_749corpus_$(date '+%Y%m%d_%H%M%S')"
if [ -d "data/part_bank" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] backing up current part bank to $BACKUP_DIR" >> "$EVENTS_LOG"
    cp -r data/part_bank "$BACKUP_DIR"
fi

echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] rebuilding part bank from full corpus..." >> "$EVENTS_LOG"
python3 -u part_bank.py --out-dir data/part_bank >> ../logs/part_bank_rebuild_r1.log 2>&1
PB_STATUS=$?
if [ $PB_STATUS -ne 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] part bank rebuild FAILED (exit $PB_STATUS) -- see logs/part_bank_rebuild_r1.log. Aborting pipeline." >> "$EVENTS_LOG"
    exit 1
fi
echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] part bank rebuild complete." >> "$EVENTS_LOG"

# ── Step 2: Phase 2 -- NodeRanker retrain ──────────────────────────────────
echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] starting Phase 2 (NodeRanker, --true-5way-test)..." >> "$EVENTS_LOG"
python3 -u train_ranker.py --true-5way-test >> ../logs/train_ranker_r1.log 2>&1
TR_STATUS=$?
if [ $TR_STATUS -ne 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] Phase 2 (NodeRanker) FAILED (exit $TR_STATUS) -- see logs/train_ranker_r1.log. Continuing to Phase 3 anyway." >> "$EVENTS_LOG"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] Phase 2 (NodeRanker) complete." >> "$EVENTS_LOG"
fi

# ── Step 3: Phase 3 -- Shape-gen VAE retrain ───────────────────────────────
echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] starting Phase 3 (Shape-gen VAE)..." >> "$EVENTS_LOG"
python3 -u train_shape_gen.py >> ../logs/train_shapegen_r1.log 2>&1
SG_STATUS=$?
if [ $SG_STATUS -ne 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] Phase 3 (Shape-gen) FAILED (exit $SG_STATUS) -- see logs/train_shapegen_r1.log." >> "$EVENTS_LOG"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] Phase 3 (Shape-gen) complete." >> "$EVENTS_LOG"
fi

echo "$(date '+%Y-%m-%d %H:%M:%S')  [phase23] pipeline finished (part bank rebuild, Phase 2, Phase 3 all attempted)." >> "$EVENTS_LOG"
