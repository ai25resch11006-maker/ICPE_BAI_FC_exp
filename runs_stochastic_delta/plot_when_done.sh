#!/bin/bash
# Waits for the five final models (step 30000), evaluates them and plots E[tau] vs log(1/delta),
# then stops the private CUDA MPS daemon used by the training runs.
cd /DATA1/rf1365/BAI_Erasures/algo
export CUDA_MPS_PIPE_DIRECTORY=/DATA1/rf1365/BAI_Erasures/.mps/pipe
M=models/sequential/StochasticGaussianBanditFixedConfidenceSequential_100_4/icpe_30000_1.pkl
until [ -f runs_stochastic_delta/log2/$M ] && [ -f runs_stochastic_delta/log4/$M ] && [ -f runs_stochastic_delta/log6/$M ] \
   && [ -f runs_stochastic_delta/log8/$M ] && [ -f runs_stochastic_delta/log10/$M ]; do sleep 120; done
echo "all models found at $(date)"
CUDA_VISIBLE_DEVICES=2 /DATA1/rf1365/anaconda3/envs/bai/bin/python plot_tau_vs_log_delta.py --logs 2 4 6 8 10
echo "plot done at $(date)"
echo quit | nvidia-cuda-mps-control
echo "MPS daemon stopped at $(date)"
