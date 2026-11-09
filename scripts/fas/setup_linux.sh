#!/bin/bash
# One-time setup on a Linux machine with an NVIDIA GPU (DPPO + Gym/MuJoCo).
#   bash scripts/fas/setup_linux.sh [conda_env_name]
# System packages for mujoco_py (needs sudo once; ask your admin on a cluster):
#   sudo apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf
set -e
ENV_NAME=${1:-dppo}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

eval "$(conda shell.bash hook)"
if ! conda env list | grep -q "^${ENV_NAME} "; then
  conda create -n "$ENV_NAME" python=3.8 -y
fi
conda activate "$ENV_NAME"

# MuJoCo 2.1.0 binaries for mujoco_py
if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
  mkdir -p "$HOME/.mujoco"
  wget -q https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz
  tar -xzf /tmp/mujoco210.tar.gz -C "$HOME/.mujoco/"
fi
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

cd "$REPO"
pip install -e ".[gym]"
pip install pytest

# First import compiles mujoco_py (takes a few minutes)
python -c "import mujoco_py, d4rl.gym_mujoco, gym; e = gym.make('hopper-medium-v2'); e.reset(); print('MuJoCo OK')"
python -m pytest tests -q

echo
echo "Done. In every new shell run:  conda activate $ENV_NAME && source scripts/fas/env.sh"
