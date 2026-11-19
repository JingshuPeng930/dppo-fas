# RQ3 sensitivity test: why does the same action-space staleness cost hopper / halfcheetah more RFT recovery
# than walker2d? Evaluate each final RL policy of Table 2 (walker2d seeds 0-2, hopper 3-5, halfcheetah 5-7;
# FAS and uniform sources) on the train grid with N(0, sigma^2) noise on every executed action
# (eval agent: ++action_noise_std). sigma = 0.3 is about the RL exploration noise (scripts/fas/rft_staleness.py).
# The loss is read relative to each task's RL gain over the pretrained policy.
#   usage: bash scripts/fas/run_rft_sensitivity.sh   (inside tmux)
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
L=$DPPO_LOG_DIR/gym-finetune
O=$DPPO_LOG_DIR/gym-sensitivity
TS="taskset -c 64-127,192-255"
mkdir -p $O
job() {  # idx task env rl_key seed sigma [extra override]
  local G=$(($1 % 4)) T=$2 E=$3 K=$4 S=$5 SIG=$6
  local D=$(ls -d $L/${E}_ppo_diffusion_mlp_${K}_*/*_$S | tail -1)
  local CK=$D/checkpoint/state_199.pt
  [ $K = pretrained ] && CK=pretrained
  CUDA_VISIBLE_DEVICES=$G $TS bash scripts/fas/evaluate.sh $T $CK train ++action_noise_std=$SIG \
    logdir=$O/${E}_${K}_${S}_sig$SIG $7 >/dev/null 2>&1
  echo "$T $K $S sigma $SIG: exit $?"
}
export -f job; export L O TS
{
  i=0
  for sig in 0.0 0.1 0.2 0.3; do
    for m in fas uniform; do
      K=rq3uniform_ta4; [ $m = fas ] && K=rq3fas_a3.0_b0.9_ta4
      for s in 0 1 2; do echo "$i walker2d-v2 walker2d-medium-v2 $K $s $sig"; i=$((i+1)); done
      for s in 5 6 7; do echo "$i halfcheetah-v2 halfcheetah-medium-v2 $K $s $sig"; i=$((i+1)); done
      K=rq3uniform_gv2_ta4; [ $m = fas ] && K=rq3fas_gv2_a3.0_b0.9_ta4
      for s in 3 4 5; do echo "$i hopper-v2 hopper-medium-v2 $K $s $sig"; i=$((i+1)); done
    done
    for t in walker2d halfcheetah hopper; do  # the pretrained policy, 3 eval seeds
      for s in 0 1 2; do echo "$i $t-v2 $t-medium-v2 pretrained $s $sig seed=$((1000 + s))"; i=$((i+1)); done
    done
  done
} | xargs -P 8 -L 1 bash -c 'job "$@"' _
echo ALL DONE
