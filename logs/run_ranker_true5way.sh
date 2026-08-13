#!/usr/bin/env bash
# NodeRanker under true-5-way CV, previous (pre-16/30) architecture --
# train_ranker.py doesn't have a built-in multi-fold CV summary (main()
# trains one --fold-idx per invocation), so run all 5 explicitly and
# aggregate afterward from each run's saved ranker_test_metrics.json.
set -u
cd "$(dirname "$0")/../back_end"
source ../.venv/bin/activate

for fold in 0 1 2 3 4; do
    echo "=================================================="
    echo "  NODERANKER TRUE-5WAY: fold $fold"
    echo "=================================================="
    python3 -u train_ranker.py --true-5way-test --fold-idx "$fold" \
        > "../logs/ranker_true5way_fold${fold}.log" 2>&1
    cp results/ranker_test_metrics.json "../logs/ranker_true5way_fold${fold}_metrics.json" 2>/dev/null
    echo "  Done: fold $fold (exit $?)"
done

echo "ALL NODERANKER TRUE-5WAY FOLDS COMPLETE"
