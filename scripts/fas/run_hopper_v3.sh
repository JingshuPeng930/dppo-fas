# Hypothesis test: hopper grid v3 = v2 without the unlearnable 300 N pushes (impulse {0, 100, 200} N).
# Prediction (written before the runs): FAS > uniform here, unlike on v2.
#   usage: bash run_hopper_v3.sh <uniform|fas|fasp> <seed> <gpu>
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
M=$1; S=$2; G=$3; E=hopper-medium-v2
W=env.wrappers.mujoco_disturbance.condition_space
GRID="$W.impulse_force=[0.0,100.0,200.0]"
case $M in
  uniform) BASE=uniform; OV="grid=v3" ;;
  fas)     BASE=fas; OV="grid=v3 fas.alpha=3.0 fas.beta=0.9" ;;
  fasp)    BASE=fas; OV="grid=v3 method=fasp1 fas.score=failure_progress fas.alpha=3.0 fas.beta=0.9 ++fas.lp_floor=0.25" ;;
esac
mkdir -p ~/dppo_fas/runs/hopper-v2
CUDA_VISIBLE_DEVICES=$G N_ITR=200 taskset -c 64-127,192-255 bash scripts/fas/finetune.sh hopper-v2 $BASE $S $OV $GRID \
  > ~/dppo_fas/runs/hopper-v2/v3_${M}_seed$S.log 2>&1
echo "train exit $?"
K=${M}_gv3; [ $M = fasp ] && K=fasp1_gv3
CK=$(ls -d $DPPO_LOG_DIR/gym-finetune/${E}_ppo_diffusion_mlp_${K}_*/*_$S | tail -1)/checkpoint/state_199.pt
TS="taskset -c 64-127,192-255"
CUDA_VISIBLE_DEVICES=$G $TS bash scripts/fas/evaluate.sh hopper-v2 $CK train eval_grid=v3 $GRID > /dev/null 2>&1; echo "eval v3 exit $?"
CUDA_VISIBLE_DEVICES=$G $TS bash scripts/fas/evaluate.sh hopper-v2 $CK ood > /dev/null 2>&1; echo "eval ood exit $?"
