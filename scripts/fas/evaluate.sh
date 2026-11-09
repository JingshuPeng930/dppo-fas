#!/bin/bash
# Evaluate a policy on the fixed disturbance test grid (identical for all methods).
#   bash scripts/fas/evaluate.sh <task> pretrained            [train|ood] [overrides...]
#   bash scripts/fas/evaluate.sh <task> <path/to/state_X.pt>  [train|ood] [overrides...]
# Results: $DPPO_LOG_DIR/gym-eval-disturbance/<name>/<time>/result.json
set -e
TASK=$1; CKPT=$2; GRID=${3:-train}; shift $(( $# < 3 ? $# : 3 ))
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

CFG=eval_diffusion_mlp_disturbance
[ "$GRID" = "ood" ] && CFG=eval_diffusion_mlp_disturbance_ood

if [ "$CKPT" = "pretrained" ]; then
  ARGS=(ft_denoising_steps=0)
else
  ARGS=(base_policy_path=$(realpath "$CKPT") ft_denoising_steps=10)
fi
python script/run.py --config-dir=cfg/gym/eval/$TASK --config-name=$CFG "${ARGS[@]}" "$@"
