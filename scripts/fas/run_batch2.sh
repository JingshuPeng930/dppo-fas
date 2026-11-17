# Batch 2 (starts when the hopper RQ3 source runs are done):
#   * RQ3 on hopper: RFT from the EMA init on each source run's successful train episodes, eval train/OOD
#   * halfcheetah uniform / FAS seeds 3-4 (5 seeds total), eval train/OOD
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
L=$DPPO_LOG_DIR/gym-finetune
until [ "$(cat ~/dppo_fas/runs/hopper-v2/rq3*_chain_s*.log 2>/dev/null | grep -c 'eval ood')" -ge 6 ]; do sleep 120; done
mkdir -p ~/dppo_fas/runs/rft ~/dppo_fas/runs/halfcheetah-v2
TS="taskset -c 64-127,192-255"
ev() { CUDA_VISIBLE_DEVICES=$1 $TS bash scripts/fas/evaluate.sh $2 $3 $4 >/dev/null 2>&1; echo "$? eval $2 $4 $3"; }
rft() {  # gpu src seed
  local D=$(ls -d $L/hopper-medium-v2_ppo_diffusion_mlp_rq3${2}_*/*_$3)
  CUDA_VISIBLE_DEVICES=$1 $TS bash scripts/fas/rft.sh hopper-v2 $2 "$D" $3 > ~/dppo_fas/runs/rft/hopper_${2}_seed$3.log 2>&1
  echo "$? rft hopper $2 $3"
  local CK=$(ls -td $DPPO_LOG_DIR/gym-rft/hopper-medium-v2_rft_${2}_*/*_$3 | head -1)/checkpoint
  CK=$(ls $CK/state_*.pt | sort -V | tail -1)
  ev $1 hopper-v2 $CK train; ev $1 hopper-v2 $CK ood
}
hc() {  # gpu method seed
  local OV="method=uniform"; local BASE=uniform
  [ $2 = fas ] && { BASE=fas; OV="fas.alpha=3.0 fas.beta=0.9"; }
  CUDA_VISIBLE_DEVICES=$1 N_ITR=200 $TS bash scripts/fas/finetune.sh halfcheetah-v2 $BASE $3 $OV > ~/dppo_fas/runs/halfcheetah-v2/${2}_seed$3.log 2>&1
  echo "$? train halfcheetah $2 $3"
  local K=uniform_ta4; [ $2 = fas ] && K=fas_a3.0_b0.9_ta4
  local CK=$(ls -d $L/halfcheetah-medium-v2_ppo_diffusion_mlp_${K}_*/*_$3 | tail -1)/checkpoint/state_199.pt
  ev $1 halfcheetah-v2 $CK train; ev $1 halfcheetah-v2 $CK ood
}
i=0
for s in uniform fas; do for seed in 3 4 5; do ( rft $((i%4)) $s $seed ) & i=$((i+1)); done; done
for m in uniform fas; do for seed in 3 4; do ( hc $((i%4)) $m $seed ) & i=$((i+1)); done; done
wait; echo ALL DONE
