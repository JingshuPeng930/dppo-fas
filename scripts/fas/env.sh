#!/bin/bash
# Non-interactive replacement for script/set_path.sh.  Usage:  source scripts/fas/env.sh
# Override any variable by exporting it before sourcing.
_REPO=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)
export DPPO_DATA_DIR=${DPPO_DATA_DIR:-$_REPO/data}
export DPPO_LOG_DIR=${DPPO_LOG_DIR:-$_REPO/log}
# export DPPO_WANDB_ENTITY=<your wandb team>   # leave unset to run with wandb=null
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export D4RL_SUPPRESS_IMPORT_ERROR=1
echo "DPPO_DATA_DIR=$DPPO_DATA_DIR  DPPO_LOG_DIR=$DPPO_LOG_DIR  WANDB=${DPPO_WANDB_ENTITY:-<off>}"
