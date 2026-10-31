#!/bin/bash
# Polite launcher for the certified runs (delta' = 10 %, 5 %, 1 %; delta = 9 %, 4 %, 0.5 %; eta = 0.1).
# It never takes a busy GPU: a run starts only when
#   * fewer than MAX_RUNS of our runs are active,
#   * some GPU (not already used by one of our runs) has >= MIN_FREE_MB free memory and <= MAX_UTIL % load,
#   * the server has >= MIN_RAM_GB of available RAM (each run's replay buffer grows to ~0.04*N*(K+1) GB).
# Runs use low CPU / IO priority (nice, ionice). Checks every WAIT seconds; one log line per decision.
# Start once from the repo root (keeps running after you disconnect):
#   nohup bash runs_certify/launch_polite.sh > runs_certify/launcher.log 2>&1 &  disown
# (no "set -u": empty associative arrays count as unset in some bash versions)
cd "$(dirname "$0")/.."                        # repo root
PY=${PY:-python}
HORIZON=${HORIZON:-300}
MAX_RUNS=${MAX_RUNS:-2}
MIN_FREE_MB=${MIN_FREE_MB:-20000}
MAX_UTIL=${MAX_UTIL:-30}
MIN_RAM_GB=${MIN_RAM_GB:-80}
WAIT=${WAIT:-300}
SETTLE=${SETTLE:-180}
BUDGET=${BUDGET:-60000}
ETA=${ETA:-0.1}
ROLLOUTS=${ROLLOUTS:-1000}
EVERY=${EVERY:-10}
START=${START:-10000}
SEED=${SEED:-1}
K=${K:-4}
PAIRS=${PAIRS:-"0.1:0.09 0.05:0.04 0.01:0.005"}   # delta_prime:delta, started in this order

declare -A GPU_OF                              # our running jobs: pid -> GPU

reap() { for pid in "${!GPU_OF[@]}"; do kill -0 "$pid" 2>/dev/null || unset "GPU_OF[$pid]"; done; }

pick_gpu() {                                   # first free GPU not used by one of our runs
    local taken=" ${GPU_OF[*]:-} "
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
    while IFS=', ' read -r idx mem util; do
        if [ "$mem" -ge "$MIN_FREE_MB" ] && [ "$util" -le "$MAX_UTIL" ] && [[ "$taken" != *" $idx "* ]]; then
            echo "$idx"; break
        fi
    done
}

echo "polite launcher started $(date): runs {$PAIRS}, N=$HORIZON, max $MAX_RUNS at once, GPU needs >=${MIN_FREE_MB} MiB free and <=${MAX_UTIL}% load, RAM >=${MIN_RAM_GB} GB"
for pair in $PAIRS; do
    dp=${pair%%:*}; d=${pair##*:}
    while true; do
        reap
        ram=$(free -g | awk '/^Mem:/{print $7}')
        gpu=$(pick_gpu)
        if [ "${#GPU_OF[@]}" -lt "$MAX_RUNS" ] && [ -n "$gpu" ] && [ "$ram" -ge "$MIN_RAM_GB" ]; then break; fi
        echo "$(date '+%F %T') waiting for delta'=$dp: our runs ${#GPU_OF[@]}/$MAX_RUNS, free GPU '${gpu:-none}', RAM available ${ram} GB"
        sleep "$WAIT"
    done
    window=$($PY -c "import math; print(max(1, math.ceil(0.1 / $d)))")
    dir=runs_certify/dp$dp
    mkdir -p "$dir"
    echo "$(date '+%F %T') START delta'=$dp delta=$d on GPU $gpu (p_hat_window=$window)"
    ( cd "$dir" && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline PYTHONUNBUFFERED=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        nice -n 10 ionice -c 3 $PY ../../main.py --num_actions $K --seed $SEED --delta $d --horizon $HORIZON \
        --max_training_steps $BUDGET --p_hat_window $window \
        --delta_prime $dp --eta $ETA --cert_rollouts $ROLLOUTS --cert_every $EVERY --cert_start $START \
        > train.log 2>&1 < /dev/null
      echo "$(date '+%F %T') END delta'=$dp (exit code $?)" ) &
    GPU_OF[$!]=$gpu
    sleep "$SETTLE"                            # let it allocate memory before the next check
done
wait
echo "all certified runs finished $(date)"
