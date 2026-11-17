# FAS gated by learnability (fas.score=failure_progress, alpha 3, beta 0.9, lp_floor 0.25), optional
# reward delay K (1 = dense); then eval state_199 on train and OOD grids.
#   usage: bash run_fasp.sh <task> <K> <seed> <gpu>
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
T=$1; K=$2; S=$3; G=$4; E=${T%-v2}-medium-v2
mkdir -p ~/dppo_fas/runs/$T
CUDA_VISIBLE_DEVICES=$G N_ITR=200 taskset -c 64-127,192-255 bash scripts/fas/finetune.sh $T fas $S method=fasp$K \
  fas.score=failure_progress fas.alpha=3.0 fas.beta=0.9 ++fas.lp_floor=0.25 ++env.wrappers.mujoco_disturbance.reward_delay=$K \
  > ~/dppo_fas/runs/$T/fasp${K}_seed$S.log 2>&1
echo "train exit $?"
CK=$(ls -d $DPPO_LOG_DIR/gym-finetune/${E}_ppo_diffusion_mlp_fasp${K}_*/*_$S | tail -1)/checkpoint/state_199.pt
for SPLIT in train ood; do
  CUDA_VISIBLE_DEVICES=$G taskset -c 64-127,192-255 bash scripts/fas/evaluate.sh $T $CK $SPLIT > /dev/null 2>&1; echo "eval $SPLIT exit $?"
done
