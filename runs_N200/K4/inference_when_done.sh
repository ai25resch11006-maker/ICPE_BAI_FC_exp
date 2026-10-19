#!/bin/bash
# Waits until the N = 200, K = 4 training run has finished (final model saved and the training process
# has exited), then runs inference_fixed_env.py: 5000 trajectories (seeds 1..5000) on the fixed bandit
# drawn with env_seed 0. Results: models/sequential/StochasticGaussianBanditFixedConfidenceSequential_200_4/inference_env0/
cd /DATA1/rf1365/BAI_Erasures/algo
PY=/DATA1/rf1365/anaconda3/envs/bai/bin/python
RUN=runs_N200/K4
M=$RUN/models/sequential/StochasticGaussianBanditFixedConfidenceSequential_200_4/icpe_30000_1.pkl

until [ -f $M ] && ! pgrep -u $USER -f "main.py --num_actions 4 --seed 1 --delta 0.1" > /dev/null \
      && grep -q "Training complete." $RUN/train.log; do
    sleep 120
done
echo "training finished, model found at $(date)"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 $PY inference_fixed_env.py --model $M --env_seed 0 --num_traj 5000 --seed_start 1
echo "inference finished at $(date) (exit code $?)"
