#!/usr/bin/env bash
# Learning-curve experiment (task 23) — does Phase 1 AUC scale with corpus
# size? Same 2-fold screening protocol as tasks 11/12, using the current
# best hyperparameters (dropout=0.3, the task-11 winner) so this reflects
# "if we had more/less data with today's best setup," not the old default.
# val/test are NEVER subsampled (--train-frac only touches the train split)
# so every point is evaluated on the same held-out data. --no-promote
# throughout -- this experiment should never touch best_serving.pt.
set -u
cd "$(dirname "$0")/../back_end"
source ../.venv/bin/activate

run() {
    local name=$1; shift
    echo "=================================================="
    echo "  LEARNING CURVE: $name"
    echo "=================================================="
    python3 -u train.py --dropout 0.3 --n-folds 2 --epochs 60 --patience 15 --no-promote "$@" \
        > "../logs/learning_curve_${name}.log" 2>&1
    cp results/cv_summary.json "../logs/learning_curve_${name}_cv_summary.json" 2>/dev/null
    echo "  Done: $name (exit $?)"
}

run frac25  --train-frac 0.25
run frac50  --train-frac 0.50
run frac75  --train-frac 0.75
run frac100

echo "ALL LEARNING CURVE RUNS COMPLETE"
