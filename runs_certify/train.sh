#!/bin/bash
# Certified training (Proposition B.12, App. B.1.6): train ICPE for delta and stop training when the test
# certifies delta' with confidence parameter eta. Three runs, one per GPU, all in parallel:
#     delta' = 10 %, 5 %, 1 %   (certified)      delta = 9 %, 4 %, 0.5 %   (training target)
# Each run trains in runs_certify/dp<delta'>/ (train.log, wandb/, models/). Its model folder gets
#     certificate_<seed>.csv   one row per test epoch: X_t, running average, boundary
#     certificate_<seed>.json  certified or not, stopping step, file name of the certified model
#
# Settings (override on the command line, e.g.  GPUS="4 5 1" HORIZON=300 bash runs_certify/train.sh):
#   HORIZON     maximum trajectory length N (default 300; replay buffer RAM about 0.04*N*(K+1) GB per run)
#   GPUS        three GPU ids, one per run
#   BUDGET      maximum training steps if the test never certifies (default 60000)
#   ETA         Prop. B.12 eta (default 0.1)
#   ROLLOUTS    fresh rollouts K per test epoch (default 1000)
#   EVERY       training steps per test epoch (default 10)
#   START       training step at which the test starts (default 10000, fixed before training)
set -u
cd "$(dirname "$0")/.."                        # repo root
PY=${PY:-python}
HORIZON=${HORIZON:-300}
GPUS=${GPUS:-"4 5 1"}
BUDGET=${BUDGET:-60000}
ETA=${ETA:-0.1}
ROLLOUTS=${ROLLOUTS:-1000}
EVERY=${EVERY:-10}
START=${START:-10000}
SEED=${SEED:-1}
K=${K:-4}
PAIRS="0.1:0.09 0.05:0.04 0.01:0.005"          # delta_prime:delta

read -ra gpu_list <<< "$GPUS"
i=0
for pair in $PAIRS; do
    dp=${pair%%:*}; d=${pair##*:}
    gpu=${gpu_list[$((i % ${#gpu_list[@]}))]}
    window=$($PY -c "import math; print(max(1, math.ceil(0.1 / $d)))")   # p_hat over ~10/delta episodes
    dir=runs_certify/dp$dp
    mkdir -p "$dir"
    echo "delta'=$dp delta=$d eta=$ETA N=$HORIZON p_hat_window=$window on GPU $gpu, started $(date)"
    ( cd "$dir" && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline PYTHONUNBUFFERED=1 \
        $PY ../../main.py --num_actions $K --seed $SEED --delta $d --horizon $HORIZON \
        --max_training_steps $BUDGET --p_hat_window $window \
        --delta_prime $dp --eta $ETA --cert_rollouts $ROLLOUTS --cert_every $EVERY --cert_start $START \
        > train.log 2>&1 < /dev/null; echo "delta'=$dp finished at $(date) (exit code $?)" ) &
    i=$((i + 1))
done
wait
echo "all certified runs finished at $(date)"
