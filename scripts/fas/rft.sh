#!/bin/bash
# RQ3: rejection-sampling fine-tuning from the pretrained checkpoint on the successful episodes that
# a DPPO(-FAS) run saved (fas.save_success_trajs=True). No env interaction.
#   bash scripts/fas/rft.sh <task> <label> "<run dir glob>" [seed] [extra hydra overrides...]
#   bash scripts/fas/rft.sh walker2d-v2 fas_a3_s0 "$DPPO_LOG_DIR/gym-finetune/walker2d-medium-v2_rq3-fas*/*_0" 0
# Evaluate the result like any other policy:  bash scripts/fas/evaluate.sh <task> <rft run>/checkpoint/state_10.pt
set -e
TASK=$1; LABEL=$2; DIRS=$3; SEED=${4:-0}; shift $(( $# < 4 ? $# : 4 ))
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
python script/run.py --config-dir=cfg/gym/rft/$TASK --config-name=rft_diffusion_mlp \
  source=$LABEL "traj_dirs=$DIRS" seed=$SEED "$@"
