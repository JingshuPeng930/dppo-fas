# (1) FAS gated by cumulative progress (fas.score=failure_stall; post hoc, after hopper's main-grid negative
#     result): hopper and walker2d, seeds 0-4, alpha 3, beta 0.9, margin 0.1, floor 0.25, warmup 40.
# (2) RQ2 reward sparsity to >= 5 seeds per cell: hopper K=25, walker2d K=25 and K=100, seeds 3-4, FAS + uniform.
# Each job trains 200 iterations, then evaluates state_199 on the train and OOD grids.
#   usage: bash scripts/fas/run_stall.sh [max parallel jobs, default 8]   (inside tmux)
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
P=${1:-8}
job() {  # task kind seed gpu
  local T=$1 M=$2 S=$3 G=$4 E=${1%-v2}-medium-v2 TS="taskset -c 64-127,192-255"
  case $M in
    *:*) bash scripts/fas/run_sparse.sh $T ${M#*:} ${M%%:*} $S $G | sed "s|^|$T sparse$M $S: |"; return ;;  # K:method
  esac
  mkdir -p ~/dppo_fas/runs/$T
  CUDA_VISIBLE_DEVICES=$G N_ITR=200 $TS bash scripts/fas/finetune.sh $T fas $S method=fasstall fas.score=failure_stall \
    fas.alpha=3.0 fas.beta=0.9 ++fas.stall_margin=0.1 ++fas.stall_floor=0.25 ++fas.stall_warmup=40 ++fas.stall_init=10 \
    > ~/dppo_fas/runs/$T/fasstall_seed$S.log 2>&1
  echo "$T $M $S: train exit $?"
  local CK=$(ls -td $DPPO_LOG_DIR/gym-finetune/${E}_ppo_diffusion_mlp_fasstall_*/*_$S | head -1)/checkpoint/state_199.pt
  for SPLIT in train ood; do
    CUDA_VISIBLE_DEVICES=$G $TS bash scripts/fas/evaluate.sh $T $CK $SPLIT > /dev/null 2>&1; echo "$T $M $S: eval $SPLIT exit $?"
  done
}
export -f job
{
  for S in 0 1 2 3 4; do echo "hopper-v2 stall $S"; echo "walker2d-v2 stall $S"; done
  for S in 3 4; do for M in fas uniform; do
    echo "hopper-v2 25:$M $S"; echo "walker2d-v2 25:$M $S"; echo "walker2d-v2 100:$M $S"
  done; done
} | awk '{print $0, (NR-1)%4}' | xargs -P $P -L 1 bash -c 'job "$@"' _
echo ALL DONE
