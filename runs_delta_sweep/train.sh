#!/bin/bash
# Trains ICPE (stochastic Gaussian bandits, fixed confidence, K = 4, neurips_submission settings) once per
# delta = exp(-x), x in 0 2 4 6 8 10. Each run trains in runs_delta_sweep/log<x>/ (train.log, wandb/, models/).
#
# Settings (override on the command line, e.g.  HORIZON=400 GPUS="0 1" bash runs_delta_sweep/train.sh):
#   HORIZON  ICPE's maximum trajectory length N (its transformers have N input positions). Choose it so that
#            ICPE never reaches it; eval_fixed_instance.py reports any trajectory that does (hit_cap).
#            Replay buffer RAM is about 0.04 * N * (K+1) GB per run: 60 GB for N = 300, K = 4.
#   GPUS     one queue per GPU; the six runs are dealt round-robin to the queues, each queue runs its
#            runs one after the other, all queues run at the same time.
#   WINDOW   p_hat window of the lambda update: "auto" = ceil(0.1 / delta) training steps (so the window
#            holds about 10 / delta episodes: 1, 1, 6, 41, 299, 2203); "1" = the authors' update.
set -u
cd "$(dirname "$0")/.."                        # repo root
PY=${PY:-python}
HORIZON=${HORIZON:-300}
GPUS=${GPUS:-"0 1 2"}
STEPS=${STEPS:-30000}
SEED=${SEED:-1}
K=${K:-4}
WINDOW=${WINDOW:-auto}
LOGS="0 2 4 6 8 10"

run() {  # run <gpu> <x>
    local gpu=$1 x=$2
    local delta window
    delta=$($PY -c "import math; print(repr(math.exp(-$x)))")
    if [ "$WINDOW" = auto ]; then window=$($PY -c "import math; print(max(1, math.ceil(0.1 / math.exp(-$x))))"); else window=$WINDOW; fi
    mkdir -p runs_delta_sweep/log$x
    echo "log(1/delta)=$x (delta=$delta, N=$HORIZON, p_hat_window=$window) started on GPU $gpu at $(date)"
    ( cd runs_delta_sweep/log$x && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline PYTHONUNBUFFERED=1 \
        $PY ../../main.py --num_actions $K --seed $SEED --delta $delta --horizon $HORIZON \
        --max_training_steps $STEPS --p_hat_window $window > train.log 2>&1 < /dev/null )
    echo "log(1/delta)=$x finished on GPU $gpu at $(date) (exit code $?)"
}

read -ra gpu_list <<< "$GPUS"
declare -a queues
i=0
for x in $LOGS; do
    q=$(( i % ${#gpu_list[@]} ))
    queues[$q]="${queues[$q]:-} $x"
    i=$((i + 1))
done
for q in "${!gpu_list[@]}"; do
    ( for x in ${queues[$q]}; do run "${gpu_list[$q]}" "$x"; done ) &
done
wait
echo "all runs finished at $(date)"
