#!/bin/bash
# Self-contained setup on a Linux GPU machine, no sudo, nothing written outside ROOT
# (no conda init, no ~/.bashrc edits, MuJoCo not in ~/.mujoco). Safe on a shared account.
#
#   bash scripts/fas/setup_linux.sh [ROOT]        # default ROOT=~/dppo_fas
#   source $ROOT/activate.sh                      # in every new shell
#
# Result:
#   $ROOT/miniforge3/envs/dppo   python 3.8 env with dppo[gym]
#   $ROOT/.mujoco/mujoco210      MuJoCo 2.1 for mujoco_py
#   $ROOT/dppo-fas               this repo (cloned if missing)
#   $ROOT/{data,log}             DPPO_DATA_DIR / DPPO_LOG_DIR
#   $ROOT/activate.sh
#
# mujoco_py needs these system packages (present on the SafeAI machines; otherwise ask an admin):
#   libosmesa6-dev libgl1-mesa-glx libglew-dev build-essential
set -e
ROOT=${1:-$HOME/dppo_fas}
REPO_URL=https://github.com/JingshuPeng930/dppo-fas.git
mkdir -p "$ROOT" && cd "$ROOT"

if [ ! -x "$ROOT/miniforge3/bin/conda" ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge_$$.sh
  bash /tmp/miniforge_$$.sh -b -p "$ROOT/miniforge3" && rm /tmp/miniforge_$$.sh
fi
if [ ! -d "$ROOT/.mujoco/mujoco210" ]; then
  mkdir -p "$ROOT/.mujoco"
  wget -q https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O /tmp/mj210_$$.tgz
  tar -xzf /tmp/mj210_$$.tgz -C "$ROOT/.mujoco" && rm /tmp/mj210_$$.tgz
fi
[ -d "$ROOT/dppo-fas" ] || git clone "$REPO_URL" "$ROOT/dppo-fas"

cat > "$ROOT/activate.sh" <<'EOF'
# Shared DPPO-FAS environment.   source <ROOT>/activate.sh
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $ROOT/miniforge3/etc/profile.d/conda.sh
conda activate $ROOT/miniforge3/envs/dppo
export MUJOCO_PY_MUJOCO_PATH=$ROOT/.mujoco/mujoco210
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia
export DPPO_DATA_DIR=$ROOT/data
export DPPO_LOG_DIR=$ROOT/log
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
cd $ROOT/dppo-fas
echo "[dppo_fas] env ready | repo $(git rev-parse --short HEAD) | DPPO_LOG_DIR=$DPPO_LOG_DIR | wandb entity: ${DPPO_WANDB_ENTITY:-<off, wandb=null>}"
EOF

source "$ROOT/miniforge3/etc/profile.d/conda.sh"
[ -d "$ROOT/miniforge3/envs/dppo" ] || conda create -y -q -p "$ROOT/miniforge3/envs/dppo" python=3.8
source "$ROOT/activate.sh"
pip install -e ".[gym]" pytest

# first import compiles mujoco_py (a few minutes)
python -c "import gym, d4rl.gym_mujoco; e = gym.make('hopper-medium-v2'); e.reset(); print('MuJoCo OK')"
python -m pytest tests -q
echo "Done. In every new shell:  source $ROOT/activate.sh"
