#!/bin/bash
# Launch one fine-tuning run.
#   bash scripts/fas/finetune.sh <task> <method> <seed> [extra hydra overrides...]
#     task   : hopper-v2 | walker2d-v2 | halfcheetah-v2
#     method : dppo     -- upstream DPPO, nominal env only (reproduction)
#              uniform  -- DPPO + uniform disturbance sampling (beta = 0)
#              fas      -- DPPO + Failure-Aware Sampling
#   N_ITR (env var) sets the interaction budget: N_ITR x n_envs(40) x n_steps(500) x act_steps(4)
#   env steps; default 200 iterations = 16M steps (upstream config: 1000).
#
# Examples:
#   bash scripts/fas/finetune.sh hopper-v2 fas 0
#   bash scripts/fas/finetune.sh hopper-v2 fas 0 fas.alpha=2.0 fas.beta=0.8
#   N_ITR=5 bash scripts/fas/finetune.sh hopper-v2 fas 0 env.n_envs=4 train.n_steps=50   # smoke test
set -e
TASK=$1; METHOD=$2; SEED=${3:-0}; shift $(( $# < 3 ? $# : 3 ))
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

case $METHOD in
  dppo)    CFG=ft_ppo_diffusion_mlp ;;
  uniform) CFG=ft_ppo_diffusion_mlp_uniform ;;
  fas)     CFG=ft_ppo_diffusion_mlp_fas ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac
ENV_NAME=${TASK%-v2}-medium-v2

WANDB_ARGS=()
if [ -z "$DPPO_WANDB_ENTITY" ]; then
  WANDB_ARGS=(wandb=null)
else
  WANDB_ARGS=(wandb.project=dppo-fas-$ENV_NAME "wandb.run=${METHOD}_seed${SEED}")
fi

python script/run.py \
  --config-dir=cfg/gym/finetune/$TASK --config-name=$CFG \
  seed=$SEED train.n_train_itr=${N_ITR:-200} \
  "${WANDB_ARGS[@]}" "$@"
