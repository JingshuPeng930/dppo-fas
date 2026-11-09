#!/bin/bash
# Run the RQ1 grid (methods x seeds) in parallel: one tmux session per GPU, each running its
# share of the jobs sequentially. Concurrency = number of GPUs (each job uses n_envs=40 CPU procs).
#
#   bash scripts/fas/launch_parallel.sh <task> "<seeds>" "<gpu ids>" [methods] [extra overrides...]
#   bash scripts/fas/launch_parallel.sh hopper-v2 "0 1 2" "0 1 2 3"
#
# Monitor:  tmux ls ;  tail -f $DPPO_LOG_DIR/../runs/<task>/*.log
# Stop:     tmux kill-session -t fas_<task>_gpu<id>
set -e
TASK=$1; SEEDS=${2:-"0 1 2"}; GPUS=(${3:-0 1 2 3}); METHODS=${4:-"dppo uniform fas"}
shift $(( $# < 4 ? $# : 4 ))
EXTRA="$*"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUNDIR=${DPPO_LOG_DIR:?source activate.sh first}/../runs/$TASK
mkdir -p "$RUNDIR"

JOBS=()
for S in $SEEDS; do for M in $METHODS; do JOBS+=("$M $S"); done; done

NG=${#GPUS[@]}
for gi in "${!GPUS[@]}"; do
  G=${GPUS[$gi]}
  CMDS=""
  for ((j = gi; j < ${#JOBS[@]}; j += NG)); do
    read -r M S <<< "${JOBS[$j]}"
    LOG=$RUNDIR/${M}_seed${S}.log
    CMDS+="echo '>>> $TASK $M seed $S on GPU $G' ; CUDA_VISIBLE_DEVICES=$G N_ITR=${N_ITR:-200} bash scripts/fas/finetune.sh $TASK $M $S $EXTRA > $LOG 2>&1 ; echo \"<<< exit \$? $M $S\" ; "
  done
  [ -z "$CMDS" ] && continue
  SESSION=fas_${TASK}_gpu${G}
  tmux new-session -d -s "$SESSION" "source $(cd $DPPO_LOG_DIR/.. && pwd)/activate.sh >/dev/null; cd $REPO && $CMDS echo ALL DONE; sleep 1d"
  echo "started $SESSION"
done
echo "logs in $RUNDIR"
