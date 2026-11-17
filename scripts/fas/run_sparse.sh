# RQ2 reward sparsity: dense reward paid every K env steps (episode return unchanged), then eval state_199
# on the task's train and OOD grids.   usage: bash run_sparse.sh <task> <uniform|fas> <K> <seed> <gpu>
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
T=$1; M=$2; K=$3; S=$4; G=$5; E=${T%-v2}-medium-v2
case $M in
  uniform) BASE=uniform; OV="method=sparse${K}_uniform" ;;
  fas)     BASE=fas; OV="method=sparse${K}_fas fas.alpha=3.0 fas.beta=0.9" ;;
esac
mkdir -p ~/dppo_fas/runs/$T
CUDA_VISIBLE_DEVICES=$G N_ITR=200 taskset -c 64-127,192-255 bash scripts/fas/finetune.sh $T $BASE $S $OV ++env.wrappers.mujoco_disturbance.reward_delay=$K > ~/dppo_fas/runs/$T/sparse${K}_${M}_seed$S.log 2>&1
echo "train exit $?"
CK=$(ls -d $DPPO_LOG_DIR/gym-finetune/${E}_ppo_diffusion_mlp_sparse${K}_${M}_*/*_$S | tail -1)/checkpoint/state_199.pt
for SPLIT in train ood; do
  CUDA_VISIBLE_DEVICES=$G taskset -c 64-127,192-255 bash scripts/fas/evaluate.sh $T $CK $SPLIT > /dev/null 2>&1; echo "eval $SPLIT exit $?"
done
