#!/usr/bin/env bash
# Sequential encoder benchmark (task 12) — one architecture at a time to
# avoid the resource contention that's crashed prior long runs. Each run's
# own best_serving.pt promotion is expected to correctly decline (R37 is
# the real incumbent) — this script is purely to collect comparable
# cv_summary numbers per architecture, not to promote anything.
set -u
cd "$(dirname "$0")/../back_end"
source ../.venv/bin/activate

for enc in rgat gatv2 sage gin; do
    echo "=================================================="
    echo "  ENCODER BENCHMARK: $enc"
    echo "=================================================="
    python3 -u train.py --encoder-type "$enc" --n-folds 2 --epochs 60 --patience 15 --no-promote \
        > "../logs/encoder_bench_${enc}.log" 2>&1
    cp results/cv_summary.json "../logs/encoder_bench_${enc}_cv_summary.json" 2>/dev/null
    echo "  Done: $enc (exit $?)"
done
echo "ALL ENCODER BENCHMARKS COMPLETE"
