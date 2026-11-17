# RQ3 source run: uniform / FAS (alpha 3, beta 0.9) with fas.save_success_trajs=True (training is unchanged,
# so these also count as extra seeds of the main comparison); then eval state_199 on train and OOD grids.
#   usage: bash run_rq3.sh <task> <uniform|fas> <seed> <gpu>
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
T=$1; M=$2; S=$3; G=$4; E=${T%-v2}-medium-v2
case $M in
  uniform) BASE=uniform; OV="method=rq3uniform" ;;
  fas)     BASE=fas; OV="method=rq3fas fas.alpha=3.0 fas.beta=0.9" ;;
esac
mkdir -p ~/dppo_fas/runs/$T
CUDA_VISIBLE_DEVICES=$G N_ITR=200 taskset -c 64-127,192-255 bash scripts/fas/finetune.sh $T $BASE $S $OV fas.save_success_trajs=true \
  > ~/dppo_fas/runs/$T/rq3${M}_seed$S.log 2>&1
echo "train exit $?"
CK=$(ls -d $DPPO_LOG_DIR/gym-finetune/${E}_ppo_diffusion_mlp_rq3${M}_*/*_$S | tail -1)/checkpoint/state_199.pt
for SPLIT in train ood; do
  CUDA_VISIBLE_DEVICES=$G taskset -c 64-127,192-255 bash scripts/fas/evaluate.sh $T $CK $SPLIT > /dev/null 2>&1; echo "eval $SPLIT exit $?"
done
