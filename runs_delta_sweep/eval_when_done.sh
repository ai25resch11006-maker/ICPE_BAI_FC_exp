#!/bin/bash
# Waits for the ICPE trainings of runs_delta_sweep/train.sh. As soon as the run for one delta = exp(-x) has
# finished (final checkpoint saved and "Training complete." in its train.log), it evaluates TaS, TTPS,
# Uniform and ICPE for that delta on the fixed instance (500 trajectories, seeds 1..500), then rebuilds
# the results table and the plots. It checks every SLEEP seconds and exits when all deltas are done.
#
# Start once, from the repo root (it keeps running after you close the terminal):
#   nohup bash runs_delta_sweep/eval_when_done.sh > runs_delta_sweep/eval_all.log 2>&1 &  disown
# Settings: GPU (a GPU not used by the trainings), HORIZON (as in train.sh), NUM_TRAJ, LOGS, SLEEP.
set -u
cd "$(dirname "$0")/.."                        # repo root
PY=${PY:-python}
GPU=${GPU:-2}
HORIZON=${HORIZON:-300}
K=${K:-4}
STEPS=${STEPS:-30000}
SEED=${SEED:-1}
NUM_TRAJ=${NUM_TRAJ:-500}
OUT=${OUT:-runs_delta_sweep/fixed_instance}
LOGS=${LOGS:-"0 2 4 6 8 10"}
SLEEP=${SLEEP:-600}

echo "watching log(1/delta) in {$LOGS} (N=$HORIZON, K=$K, step $STEPS, $NUM_TRAJ trajectories, GPU $GPU), started $(date)"
pending="$LOGS"
while [ -n "$pending" ]; do
    still=""
    for x in $pending; do
        model=runs_delta_sweep/log$x/models/sequential/StochasticGaussianBanditFixedConfidenceSequential_${HORIZON}_${K}/icpe_${STEPS}_${SEED}.pkl
        if [ -f "$model" ] && grep -q "Training complete." "runs_delta_sweep/log$x/train.log" 2>/dev/null; then
            echo "log(1/delta)=$x: training finished, evaluating at $(date)"
            CUDA_VISIBLE_DEVICES=$GPU $PY eval_fixed_instance.py --algos TaS TTPS Uniform ICPE --logs "$x" \
                --num_traj "$NUM_TRAJ" --icpe_N "$HORIZON" --steps "$STEPS" --train_seed "$SEED" --K "$K" --out "$OUT"
            status=$?
            $PY make_results_table.py --dir "$OUT" > /dev/null
            $PY plot_fixed_instance.py --dir "$OUT" > /dev/null
            echo "log(1/delta)=$x: evaluation finished at $(date) (exit code $status); table: $OUT/results_table.md"
        else
            still="$still $x"
        fi
    done
    pending=$(echo $still)
    if [ -n "$pending" ]; then sleep "$SLEEP"; fi
done
echo "all deltas evaluated at $(date)"
