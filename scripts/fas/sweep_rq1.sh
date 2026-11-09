#!/bin/bash
# RQ1: DPPO vs DPPO-uniform vs DPPO-FAS, matched budget, several seeds. Runs sequentially;
# on a cluster submit each line as its own job instead.
#   bash scripts/fas/sweep_rq1.sh hopper-v2 "0 1 2"
TASK=${1:-hopper-v2}; SEEDS=${2:-"0 1 2"}
for SEED in $SEEDS; do
  for METHOD in dppo uniform fas; do
    echo ">>> $TASK $METHOD seed $SEED"
    bash "$(dirname "$0")/finetune.sh" "$TASK" "$METHOD" "$SEED"
  done
done
