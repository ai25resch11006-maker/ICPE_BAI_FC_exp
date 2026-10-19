"""
Expected stopping time vs delta for ICPE (GPT-2 networks), fixed confidence.

NEW CODE (not from the ICPE repository). For each delta, loads the model trained with
`main.py --delta <delta>` (run from
runs_delta/delta<delta>/), runs the inference rollout (icpe_inference.sample_trajectories,
Algorithm 1 lines 13-15) on fresh bandits in chunks, and saves:
  <out>.csv : per delta - average stopping time, 95% CI, fraction stopped before N,
              accuracy, 95% CI, target 1 - delta, number of evaluation episodes
  <out>.npz : per delta - stopping time and correctness of every evaluation episode
  <out>.png : (a) average stopping time vs delta, (b) accuracy vs target 1 - delta
95% CIs are normal-approximation intervals over the evaluation episodes.

Usage (from this folder, inside the `bai` conda environment):
  python plot_tau_vs_delta.py --K 4 --deltas 0.1 0.01 0.001 0.0001 0.00001
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from icpe_inference import load, sample_trajectories


def evaluate(path: str, episodes: int, chunk: int, device: str):
    agent = load(path, device)
    taus, correct = [], []
    done = 0
    while done < episodes:
        B = min(chunk, episodes - done)
        traj = sample_trajectories(agent, B, 1)
        taus.append(traj.stopping_times.detach().cpu().numpy().flatten())
        correct.append(traj.rew_opt_actions[..., -1].detach().cpu().numpy().flatten())
        done += B
    return np.concatenate(taus), np.concatenate(correct)


def mean_ci(x: np.ndarray):
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    p = argparse.ArgumentParser(description='Expected stopping time vs delta (GPT-2 ICPE, fixed confidence)')
    p.add_argument('--K', type=int, default=4)
    p.add_argument('--deltas', nargs='+', default=['0.1', '0.01', '0.001', '0.0001', '0.00001'],
                   help='delta values, written exactly as in the run folder names')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--runs_dir', default='runs_delta')
    p.add_argument('--episodes', type=int, default=300000,
                   help='evaluation episodes per delta (about 3/delta are needed to check delta = 1e-5)')
    p.add_argument('--chunk', type=int, default=1000, help='bandits evaluated per batch')
    p.add_argument('--eval_seed', type=int, default=0)
    p.add_argument('--out', default='tau_vs_delta_gpt2')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    K = args.K
    rows, raw = [], {}
    for d in args.deltas:
        delta = float(d)
        path = os.path.join(args.runs_dir, f'delta{d}', 'models', 'sequential',
                            f'DeterministicBanditSequential_{K}_{K}', f'icedqn_{args.steps}_{args.seed}.pkl')
        if not os.path.exists(path):
            print(f'delta={d}: missing {path}, skipped')
            continue
        np.random.seed(args.eval_seed)
        torch.manual_seed(args.eval_seed)
        tau, correct = evaluate(path, args.episodes, args.chunk, device)
        raw[f'tau_delta{d}'] = tau
        raw[f'correct_delta{d}'] = correct
        tau_m, tau_ci = mean_ci(tau)
        acc_m, acc_ci = mean_ci(correct)
        rows.append({'delta': delta, 'avg_tau': tau_m, 'tau_ci95': tau_ci,
                     'frac_stopped_early': float((tau < K).mean()),
                     'accuracy': acc_m, 'accuracy_ci95': acc_ci, 'errors': int((correct == 0).sum()),
                     'target_accuracy': 1 - delta, 'episodes': len(tau)})
        print(f'delta={d:>8}  avg tau {tau_m:5.3f} +- {tau_ci:.3f} (max {K})  '
              f'stopped early {100 * (tau < K).mean():5.1f}%  accuracy {acc_m:.5f} '
              f'({int((correct == 0).sum())} errors / {len(tau)})  target {1 - delta:.5f}', flush=True)

    if not rows:
        print('No models found.')
        return
    df = pd.DataFrame(rows).sort_values('delta')
    df.to_csv(f'{args.out}.csv', index=False)
    np.savez_compressed(f'{args.out}.npz', **raw)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].errorbar(df.delta, df.avg_tau, yerr=df.tau_ci95, marker='o', capsize=3, label='ICPE (GPT-2)')
    ax[0].axhline(K, color='k', ls='--', lw=1, label=rf'$\tau = N = {K}$')
    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'$\delta$')
    ax[0].set_ylabel(r'expected stopping time $\mathbb{E}[\tau]$')
    ax[0].legend()
    ax[1].errorbar(df.delta, df.accuracy, yerr=df.accuracy_ci95, marker='o', capsize=3, label='ICPE (GPT-2)')
    ax[1].plot(df.delta, df.target_accuracy, 'k--', lw=1, marker='x', label=r'target $1-\delta$')
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'$\delta$')
    ax[1].set_ylabel(r'$P(\hat H_\tau = H^\star)$')
    ax[1].legend(loc='lower left')
    fig.suptitle(rf'Deterministic bandits, fixed confidence, K = N = {K} (seed {args.seed})')
    fig.tight_layout()
    fig.savefig(f'{args.out}.png', dpi=150)
    print(f'Saved {args.out}.png, {args.out}.csv and {args.out}.npz')


if __name__ == '__main__':
    main()
