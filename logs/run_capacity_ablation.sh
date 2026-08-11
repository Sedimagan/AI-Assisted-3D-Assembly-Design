#!/usr/bin/env bash
# Capacity ablation (task 11) — one hyperparameter at a time from the R37
# baseline (hidden_dim=128, dropout=0.2, weight_decay=5e-4), same 2-fold
# screening protocol as the task-12 encoder benchmark so results are
# directly comparable to that run's "rgat" baseline (Mean AUC=0.6358+/-0.0205).
# --no-promote throughout -- lesson learned from task 12's near-miss:
# a 2-fold screening run should never be allowed to promote itself.
set -u
cd "$(dirname "$0")/../back_end"
source ../.venv/bin/activate

run() {
    local name=$1; shift
    echo "=================================================="
    echo "  CAPACITY ABLATION: $name"
    echo "=================================================="
    python3 -u train.py --n-folds 2 --epochs 60 --patience 15 --no-promote "$@" \
        > "../logs/capacity_bench_${name}.log" 2>&1
    cp results/cv_summary.json "../logs/capacity_bench_${name}_cv_summary.json" 2>/dev/null
    echo "  Done: $name (exit $?)"
}

run hidden64   --hidden-dim 64
run hidden256  --hidden-dim 256
run dropout01  --dropout 0.1
run dropout03  --dropout 0.3
run wd1e-4     --weight-decay 0.0001
run wd1e-3     --weight-decay 0.001

echo "ALL CAPACITY ABLATION RUNS COMPLETE"
