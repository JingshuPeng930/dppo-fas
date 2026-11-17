# RQ2: per-bin critic error (rq2_critic.py) for the reward-sparsity runs (K = 25, 100; uniform vs FAS;
# walker2d + hopper; seeds 0-2; itr 100 and 199). Waits until the GPUs are free of the evening batch.
source ~/dppo_fas/activate.sh >/dev/null
cd ~/dppo_fas/dppo-fas
until [ "$(cat ~/dppo_fas/runs/hopper-v2/v3_*_chain_s*.log ~/dppo_fas/runs/walker2d-v2/fasp1_chain_s[34].log ~/dppo_fas/runs/halfcheetah-v2/fasp1_chain_s*.log 2>/dev/null | grep -c 'eval ood')" -ge 14 ] \
      && grep -q "ALL DONE" ~/dppo_fas/runs/batch2.log; do sleep 120; done
F=$DPPO_LOG_DIR/gym-finetune
OUT=~/dppo_fas/rq2; mkdir -p $OUT
JOBS=()
for pat in {walker2d,hopper}-medium-v2_ppo_diffusion_mlp_sparse{25,100}_{uniform,fas}_; do
  for d in $(ls -d $F/${pat}*/*_[012] 2>/dev/null); do
    for itr in 100 199; do [ -f $d/checkpoint/state_$itr.pt ] && JOBS+=("$d $itr"); done
  done
done
echo "${#JOBS[@]} jobs"
for w in 0 1 2 3; do
  (
    for ((j = w; j < ${#JOBS[@]}; j += 4)); do
      read -r d itr <<< "${JOBS[$j]}"
      name=$(basename $(dirname $d))__$(basename $d)__$itr
      [ -f $OUT/$name.json ] && continue
      CUDA_VISIBLE_DEVICES=$w taskset -c 64-127,192-255 python scripts/fas/rq2_critic.py $d $itr --episodes 8 --out $OUT/$name.json > $OUT/$name.log 2>&1
      echo "$? $name"
    done
  ) &
done
wait; echo ALL DONE
