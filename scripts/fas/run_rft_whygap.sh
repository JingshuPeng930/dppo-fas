# RQ3: why does RFT recover most of RL's gain on walker2d but only about half on hopper / halfcheetah?
# Two tests on the same source runs as Table 2 (walker2d seeds 0-2, hopper 3-5, halfcheetah 5-7):
#   late    -- staleness: RFT only on successes from RL iterations >= 150 (the last quarter), same
#              gradient steps as the full RFT (agent/dataset/rft.py: min_itr + match_full_steps)
#   distill -- imitation fidelity: roll out the *final* RL policy on the train grid (100 episodes per bin,
#              seed 2000+s so not the eval episodes), keep its successes, and train the same diffusion BC
#              with the same gradient steps as the full RFT (draws_per_epoch). This is RFT's ceiling when
#              the data come from the final policy itself.
# Every policy is evaluated on the train and OOD grids.
#   usage: bash scripts/fas/run_rft_whygap.sh   (inside tmux)
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
L=$DPPO_LOG_DIR/gym-finetune
C=$DPPO_LOG_DIR/gym-distill-collect
TS="taskset -c 64-127,192-255"
mkdir -p ~/dppo_fas/runs/rft $C
evals() {  # gpu task rft_label env seed
  local CK=$(ls -td $DPPO_LOG_DIR/gym-rft/${4}_rft_${3}_ta4_*/*_$5 | head -1)/checkpoint/state_10.pt
  for sp in train ood; do
    CUDA_VISIBLE_DEVICES=$1 $TS bash scripts/fas/evaluate.sh $2 $CK $sp >/dev/null 2>&1
    echo "$2 $3 $5: eval $sp exit $?"
  done
}
job() {  # idx kind task env src rl_key seed
  local G=$(($1 % 4)) KIND=$2 T=$3 E=$4 SRC=$5 K=$6 S=$7 LAB=$5_$2
  local D=$(ls -d $L/${E}_ppo_diffusion_mlp_${K}_*/*_$S | tail -1)
  local LOG=~/dppo_fas/runs/rft/${T%-v2}_${LAB}_seed$S.log
  if [ $KIND = late ]; then
    CUDA_VISIBLE_DEVICES=$G $TS bash scripts/fas/rft.sh $T $LAB "$D" $S min_itr=150 > $LOG 2>&1
  else
    local OUT=$C/${E}_${K}_$S
    CUDA_VISIBLE_DEVICES=$G $TS bash scripts/fas/evaluate.sh $T $D/checkpoint/state_199.pt train \
      n_episodes_per_condition=100 ++save_success_trajs=true logdir=$OUT seed=$((2000 + S)) > ${LOG%.log}_collect.log 2>&1
    echo "$T $LAB $S: collect exit $?"
    local N=$(python -c "import glob, numpy as np; print(sum(len(np.load(f)['ep']) for f in glob.glob('$D/success_trajs/itr_*.npz')))")
    CUDA_VISIBLE_DEVICES=$G $TS bash scripts/fas/rft.sh $T $LAB "$OUT" $S draws_per_epoch=$N > $LOG 2>&1
  fi
  echo "$T $LAB $S: rft exit $?"
  evals $G $T $LAB $E $S
}
export -f job evals; export L C TS
{
  i=0
  for kind in distill late; do
    for m in fas uniform; do
      K=rq3uniform_ta4; [ $m = fas ] && K=rq3fas_a3.0_b0.9_ta4
      for s in 0 1 2; do echo "$i $kind walker2d-v2 walker2d-medium-v2 $m $K $s"; i=$((i+1)); done
      for s in 5 6 7; do echo "$i $kind halfcheetah-v2 halfcheetah-medium-v2 $m $K $s"; i=$((i+1)); done
      K=rq3uniform_gv2_ta4; [ $m = fas ] && K=rq3fas_gv2_a3.0_b0.9_ta4
      for s in 3 4 5; do echo "$i $kind hopper-v2 hopper-medium-v2 $m $K $s"; i=$((i+1)); done
    done
  done
} | xargs -P 6 -L 1 bash -c 'job "$@"' _
echo ALL DONE
