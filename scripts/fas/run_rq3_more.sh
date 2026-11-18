# RQ3 completion (Table 2):
#   * hopper: RFT + 60% D4RL demos on the saved successes of the rq3 source runs (seeds 3-5)
#   * halfcheetah: rq3 source runs (uniform / FAS, seeds 5-7, success trajectories saved), then RFT and
#     RFT + 60% demos on each; every policy evaluated on the train and OOD grids
#   usage: bash scripts/fas/run_rq3_more.sh   (inside tmux; ~6 h on 4 GPUs)
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
L=$DPPO_LOG_DIR/gym-finetune
TS="taskset -c 64-127,192-255"
mkdir -p ~/dppo_fas/runs/rft
ev() { CUDA_VISIBLE_DEVICES=$1 $TS bash scripts/fas/evaluate.sh $2 $3 $4 >/dev/null 2>&1; echo "$? eval $2 $4 $3"; }
rft() {  # gpu task env src(uniform|fas) rl_key seed mix(0|1)
  local G=$1 T=$2 E=$3 SRC=$4 K=$5 S=$6 MIX=$7
  local D=$(ls -d $L/${E}_ppo_diffusion_mlp_${K}_*/*_$S | tail -1)
  local LAB=$SRC OV=""
  [ $MIX = 1 ] && { LAB=${SRC}_mix; OV="demo_weight=0.6 train.n_epochs=4"; }
  CUDA_VISIBLE_DEVICES=$G $TS bash scripts/fas/rft.sh $T $LAB "$D" $S $OV > ~/dppo_fas/runs/rft/${T%-v2}_${LAB}_seed$S.log 2>&1
  echo "$? rft $T $LAB $S"
  local CK=$(ls -td $DPPO_LOG_DIR/gym-rft/${E}_rft_${LAB}_ta4_*/*_$S | head -1)/checkpoint
  CK=$(ls $CK/state_*.pt | sort -V | tail -1)
  ev $G $T $CK train; ev $G $T $CK ood
}
hc() {  # gpu method seed: source run (+ its evals) -> RFT -> RFT + demos
  local G=$1 M=$2 S=$3 K=rq3uniform_ta4
  [ $M = fas ] && K=rq3fas_a3.0_b0.9_ta4
  bash scripts/fas/run_rq3.sh halfcheetah-v2 $M $S $G
  rft $G halfcheetah-v2 halfcheetah-medium-v2 $M $K $S 0
  rft $G halfcheetah-v2 halfcheetah-medium-v2 $M $K $S 1
}
i=0
for m in uniform fas; do for s in 5 6 7; do ( hc $((i%4)) $m $s ) & i=$((i+1)); done; done
for m in uniform fas; do
  K=rq3uniform_gv2_ta4; [ $m = fas ] && K=rq3fas_gv2_a3.0_b0.9_ta4
  for s in 3 4 5; do ( rft $((i%4)) hopper-v2 hopper-medium-v2 $m $K $s 1 ) & i=$((i+1)); done
done
wait; echo ALL DONE
