"""
Expected stopping time vs log(1/delta) for ICPE on stochastic Gaussian bandits (fixed confidence).

NEW CODE (not from the ICPE repository). For each value x = log(1/delta), it loads the model trained with
`main.py --delta exp(-x)` (run from runs_stochastic_delta/log<x>/), plays B new bandits x N trajectories
greedily with icpe_inference.sample_trajectories (Algorithm 1, lines 13-15), and measures, as the
authors' notebook (experiments/stochastic_mab/results_bandit_stochastic_fixed_confidence.ipynb,
neurips_v2 branch) does:
  stopping time  = traj.stopping_time
  correctness    = traj.rew_opt_actions at index stopping_time - 1
Defaults B = 300, N = 15 are the notebook's. 95% confidence intervals are normal-approximation intervals
over the B per-bandit averages (the N trajectories of one bandit are not independent).

Saves <out>.csv (per delta: average stopping time, CI, accuracy, CI, target 1 - delta), <out>.npz (every
trajectory's stopping time and correctness) and <out>.png.

Usage (from this folder, inside the `bai` conda environment):
  python plot_tau_vs_log_delta.py --logs 2 4 6 8 10
"""
import argparse
import math
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from icpe_inference import load, sample_trajectories

PATH = ('runs_stochastic_delta/log{x}/models/sequential/'
        'StochasticGaussianBanditFixedConfidenceSequential_100_{K}/icpe_{steps}_{seed}.pkl')


def evaluate(path: str, B: int, N: int, device: str):
    agent = load(path, device)
    traj = sample_trajectories(agent, B, N).to_numpy()
    tau = traj.stopping_time                                                    # (B, N)
    correct = np.take_along_axis(traj.rew_opt_actions,
                                 (tau - 1)[..., None].astype(np.int64), axis=-1)[..., 0]  # (B, N)
    return tau, correct


def mean_ci(x: np.ndarray):
    per_bandit = x.mean(1)
    return x.mean(), 1.96 * per_bandit.std(ddof=1) / np.sqrt(len(per_bandit))


def main():
    p = argparse.ArgumentParser(description='Expected stopping time vs log(1/delta) (stochastic bandits, fixed confidence)')
    p.add_argument('--logs', nargs='+', default=['2', '4', '6', '8', '10'],
                   help='values of log(1/delta), written as in the run folder names (log<x>)')
    p.add_argument('--K', type=int, default=4)
    p.add_argument('--seed', type=int, default=1, help='training seed of the models')
    p.add_argument('--steps', type=int, default=30000, help='training step of the checkpoint to load')
    p.add_argument('--B', type=int, default=300, help='number of new bandits')
    p.add_argument('--N', type=int, default=15, help='trajectories per bandit')
    p.add_argument('--eval_seed', type=int, default=0)
    p.add_argument('--path', default=PATH, help='model path template with {x}, {K}, {steps}, {seed}')
    p.add_argument('--out', default='runs_stochastic_delta/tau_vs_log_delta')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows, raw = [], {}
    for x in args.logs:
        path = args.path.format(x=x, K=args.K, steps=args.steps, seed=args.seed)
        if not os.path.exists(path):
            print(f'log(1/delta)={x}: missing {path}, skipped')
            continue
        delta = math.exp(-float(x))
        np.random.seed(args.eval_seed)
        torch.manual_seed(args.eval_seed)
        tau, correct = evaluate(path, args.B, args.N, device)
        raw[f'tau_log{x}'], raw[f'correct_log{x}'] = tau, correct
        tau_m, tau_ci = mean_ci(tau)
        acc_m, acc_ci = mean_ci(correct)
        rows.append({'log_inv_delta': float(x), 'delta': delta, 'avg_tau': tau_m, 'tau_ci95': tau_ci,
                     'accuracy': acc_m, 'accuracy_ci95': acc_ci, 'target_accuracy': 1 - delta,
                     'max_tau': tau.max(), 'episodes': tau.size})
        print(f'log(1/delta)={x:>3} (delta={delta:.3g}): avg tau {tau_m:6.2f} +- {tau_ci:.2f}, '
              f'accuracy {acc_m:.4f} +- {acc_ci:.4f} (target {1 - delta:.5f}), {tau.size} episodes', flush=True)

    if not rows:
        print('No models found.')
        return
    df = pd.DataFrame(rows).sort_values('log_inv_delta')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(f'{args.out}.csv', index=False)
    np.savez_compressed(f'{args.out}.npz', **raw)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].errorbar(df.log_inv_delta, df.avg_tau, yerr=df.tau_ci95, marker='o', capsize=3, label='ICPE')
    ax[0].set_xlabel(r'$\log(1/\delta)$')
    ax[0].set_ylabel(r'expected stopping time $\mathbb{E}[\tau]$')
    ax[0].legend()
    ax[1].errorbar(df.log_inv_delta, df.accuracy, yerr=df.accuracy_ci95, marker='o', capsize=3, label='ICPE')
    ax[1].plot(df.log_inv_delta, df.target_accuracy, 'k--', lw=1, marker='x', label=r'target $1-\delta$')
    ax[1].set_xlabel(r'$\log(1/\delta)$')
    ax[1].set_ylabel(r'$P(\hat H_\tau = H^\star)$')
    ax[1].legend(loc='lower right')
    fig.suptitle(rf'Stochastic Gaussian bandits, fixed confidence, K = {args.K}, N = 100 '
                 rf'(seed {args.seed}, {args.steps} steps)')
    fig.tight_layout()
    fig.savefig(f'{args.out}.png', dpi=150)
    print(f'Saved {args.out}.png, {args.out}.csv and {args.out}.npz')


if __name__ == '__main__':
    main()
