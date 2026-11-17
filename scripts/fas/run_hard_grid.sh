# walker2d on a harder 36-bin grid (most bins mid-difficulty), then eval state_199 on that grid and on the
# original train grid.  usage: bash hard_grid.sh <uniform|fas|sgs> <seed> <gpu>
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
M=$1; S=$2; G=$3
W=env.wrappers.mujoco_disturbance.condition_space
GRID="$W.mass_scale=[1.0,1.25,1.5,1.75] $W.friction_scale=[0.35,0.5,1.0] $W.impulse_force=[150.0,300.0,450.0]"
case $M in
  uniform) BASE=uniform; OV="method=hard_uniform" ;;
  fas)     BASE=fas; OV="method=hard_fas fas.alpha=3.0 fas.beta=0.9" ;;
  sgs)     BASE=fas; OV="method=hard_sgs fas.score=frontier fas.beta=1.0" ;;
esac
mkdir -p ~/dppo_fas/runs/walker2d-v2
CUDA_VISIBLE_DEVICES=$G N_ITR=200 bash scripts/fas/finetune.sh walker2d-v2 $BASE $S $OV $GRID > ~/dppo_fas/runs/walker2d-v2/hard_${M}_seed$S.log 2>&1
echo "train exit $?"
CK=$(ls -d $DPPO_LOG_DIR/gym-finetune/walker2d-medium-v2_ppo_diffusion_mlp_hard_${M}_*/*_$S | tail -1)/checkpoint/state_199.pt
CUDA_VISIBLE_DEVICES=$G bash scripts/fas/evaluate.sh walker2d-v2 $CK train eval_grid=hard $GRID > /dev/null 2>&1; echo "eval hard exit $?"
CUDA_VISIBLE_DEVICES=$G bash scripts/fas/evaluate.sh walker2d-v2 $CK train > /dev/null 2>&1; echo "eval train exit $?"
