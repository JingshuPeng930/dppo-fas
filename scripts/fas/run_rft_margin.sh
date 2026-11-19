# RQ3 margin test: is RFT weak on hopper / halfcheetah because most saved "successes" barely pass the
# success criterion? Re-run RFT on the same source runs, keeping only successes with a margin:
#   hopper (success = no fall in the first 300 steps): episodes of >= 600 env steps
#   halfcheetah (success = return >= 4000): episodes with return >= 4500
# Gradient steps match the unfiltered RFT (agent/dataset/rft.py: match_full_steps). Each policy is
# evaluated on the train and OOD grids.
#   usage: bash scripts/fas/run_rft_margin.sh   (inside tmux)
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
L=$DPPO_LOG_DIR/gym-finetune
TS="taskset -c 64-127,192-255"
mkdir -p ~/dppo_fas/runs/rft
job() {  # idx task env src rl_key seed label override
  local G=$(($1 % 4)) T=$2 E=$3 SRC=$4 K=$5 S=$6 LAB=$4_$7 OV=$8
  local D=$(ls -d $L/${E}_ppo_diffusion_mlp_${K}_*/*_$S | tail -1)
  CUDA_VISIBLE_DEVICES=$G $TS bash scripts/fas/rft.sh $T $LAB "$D" $S $OV > ~/dppo_fas/runs/rft/${T%-v2}_${LAB}_seed$S.log 2>&1
  echo "$T $LAB $S: rft exit $?"
  local CK=$(ls -td $DPPO_LOG_DIR/gym-rft/${E}_rft_${LAB}_ta4_*/*_$S | head -1)/checkpoint/state_10.pt
  for sp in train ood; do
    CUDA_VISIBLE_DEVICES=$G $TS bash scripts/fas/evaluate.sh $T $CK $sp >/dev/null 2>&1
    echo "$T $LAB $S: eval $sp exit $?"
  done
}
export -f job; export L TS
{
  i=0
  for m in uniform fas; do
    K=rq3uniform_gv2_ta4; [ $m = fas ] && K=rq3fas_gv2_a3.0_b0.9_ta4
    for s in 3 4 5; do echo "$i hopper-v2 hopper-medium-v2 $m $K $s len600 min_ep_len=600"; i=$((i+1)); done
    K=rq3uniform_ta4; [ $m = fas ] && K=rq3fas_a3.0_b0.9_ta4
    for s in 5 6 7; do echo "$i halfcheetah-v2 halfcheetah-medium-v2 $m $K $s ret4500 min_return=4500"; i=$((i+1)); done
  done
} | xargs -P 6 -L 1 bash -c 'job "$@"' _
echo ALL DONE
