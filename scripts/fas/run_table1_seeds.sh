# Table 1 to >= 5 seeds per cell: DPPO (nominal), SGS, LP, PLR, FAS+LP seeds 3-4 where missing, and the
# PLR baseline on halfcheetah (seeds 0-4, previously not run). Same overrides as the existing runs of each
# method (read from their .hydra/overrides.yaml). Each job trains, then evaluates state_199 on train + OOD.
#   usage: bash scripts/fas/run_table1_seeds.sh [max parallel jobs, default 7]   (inside tmux)
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
P=${1:-7}
job() {  # task method seed gpu
  local T=$1 M=$2 S=$3 G=$4 E=${1%-v2}-medium-v2 TS="taskset -c 64-127,192-255" BASE OV PAT
  mkdir -p ~/dppo_fas/runs/$T
  case $M in
    plr)  bash scripts/fas/run_plr.sh $T $S $G | sed "s|^|$T $M $S: |"; return ;;
    fasp) bash scripts/fas/run_fasp.sh $T 1 $S $G | sed "s|^|$T $M $S: |"; return ;;
    dppo) BASE=dppo; OV=""; PAT=ta4 ;;
    sgs)  BASE=fas; OV="method=sgs fas.score=frontier fas.beta=1.0"; PAT=sgs ;;
    lp)   BASE=fas; OV="method=lp fas.score=progress fas.alpha=1.0 fas.beta=0.8"; PAT=lp ;;
  esac
  CUDA_VISIBLE_DEVICES=$G N_ITR=200 $TS bash scripts/fas/finetune.sh $T $BASE $S $OV > ~/dppo_fas/runs/$T/${M}_seed$S.log 2>&1
  echo "$T $M $S: train exit $?"
  local CK=$(ls -d $DPPO_LOG_DIR/gym-finetune/${E}_ppo_diffusion_mlp_${PAT}_*/*_$S | tail -1)/checkpoint/state_199.pt
  for SPLIT in train ood; do
    CUDA_VISIBLE_DEVICES=$G $TS bash scripts/fas/evaluate.sh $T $CK $SPLIT > /dev/null 2>&1; echo "$T $M $S: eval $SPLIT exit $?"
  done
}
export -f job
{
  for T in walker2d-v2 hopper-v2 halfcheetah-v2; do for M in dppo sgs lp; do for S in 3 4; do echo "$T $M $S"; done; done; done
  for T in walker2d-v2 hopper-v2; do for S in 3 4; do echo "$T plr $S"; done; done
  for S in 0 1 2 3 4; do echo "halfcheetah-v2 plr $S"; done
  for T in hopper-v2 halfcheetah-v2; do for S in 3 4; do echo "$T fasp $S"; done; done
} | awk '{print $0, (NR-1)%4}' | xargs -P $P -L 1 bash -c 'job "$@"' _
echo ALL DONE
