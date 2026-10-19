#!/bin/bash
# Trains ICPE (stochastic Gaussian bandits, fixed confidence) for K = 14, 12, 10, 8, 6, 4 with delta = 0.1, seed 1.
# Two runs per GPU, one after the other (replay buffer RAM is about 4*(K+1) GB per run; peak about 156 GB):
#   GPU 0: K=14 then K=8    GPU 1: K=12 then K=6    GPU 2: K=10 then K=4
# Each run trains in its own folder runs_stochastic_K/K<K>/ (train.log, wandb/, models/).
cd /DATA1/rf1365/BAI_Erasures/algo/runs_stochastic_K
PY=/DATA1/rf1365/anaconda3/envs/bai/bin/python

run() {  # run <gpu> <K>
    mkdir -p K$2
    cd K$2
    echo "K=$2 started on GPU $1 at $(date)"
    CUDA_VISIBLE_DEVICES=$1 WANDB_MODE=offline PYTHONUNBUFFERED=1 \
        $PY ../../main.py --num_actions $2 --seed 1 --delta 0.1 > train.log 2>&1 < /dev/null
    echo "K=$2 finished on GPU $1 at $(date) (exit code $?)"
    cd ..
}

(run 0 14; run 0 8) &
(run 1 12; run 1 6) &
(run 2 10; run 2 4) &
wait
echo "all runs finished at $(date)"
