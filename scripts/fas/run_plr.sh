# PLR baseline: positive-value-loss score (Jiang et al. 2021) in the FAS sampler (alpha 3, beta 0.9).
# Then eval state_199 on the task's train and OOD grids.   usage: bash run_plr.sh <task> <seed> <gpu>
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
T=$1; S=$2; G=$3; E=${T%-v2}-medium-v2
mkdir -p ~/dppo_fas/runs/$T
CUDA_VISIBLE_DEVICES=$G N_ITR=200 taskset -c 64-127,192-255 bash scripts/fas/finetune.sh $T fas $S method=plr fas.score=value fas.alpha=3.0 fas.beta=0.9 > ~/dppo_fas/runs/$T/plr_seed$S.log 2>&1
echo "train exit $?"
CK=$(ls -d $DPPO_LOG_DIR/gym-finetune/${E}_ppo_diffusion_mlp_plr_*/*_$S | tail -1)/checkpoint/state_199.pt
for SPLIT in train ood; do
  CUDA_VISIBLE_DEVICES=$G taskset -c 64-127,192-255 bash scripts/fas/evaluate.sh $T $CK $SPLIT > /dev/null 2>&1; echo "eval $SPLIT exit $?"
done
