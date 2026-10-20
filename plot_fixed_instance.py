"""
Plots for eval_fixed_instance.py (NEW CODE, not from the ICPE repository).

Reads <dir>/summary.csv and <dir>/environment.json and saves
  <dir>/tau_vs_log_delta.png        average stopping time vs log(1/delta), one line per algorithm (95% CI)
  <dir>/fixed_instance_panels.png   (a) average stopping time, (b) empirical error probability with the
                                    target delta, (c) average wall time per trajectory

Usage: python plot_fixed_instance.py [--dir runs_delta_sweep/fixed_instance]
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLE = {'ICPE': ('#1a54a6', 'o'), 'TaS': ('#9467bd', '^'), 'TTPS': ('#8c564b', 'x'), 'Uniform': ('#ff7f0e', 'v')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='runs_delta_sweep/fixed_instance')
    a = p.parse_args()
    df = pd.read_csv(os.path.join(a.dir, 'summary.csv'))
    with open(os.path.join(a.dir, 'environment.json')) as f:
        env = json.load(f)
    algos = [x for x in STYLE if x in set(df.algo)]
    n = int(df.num_traj.max())
    title = (f"Fixed instance μ = {np.round(env['means'], 3).tolist()} (K = {env['num_arms']}), "
             f"{n} trajectories per point")

    def tau_panel(ax):
        for algo in algos:
            d = df[df.algo == algo].sort_values('log_inv_delta')
            c, m = STYLE[algo]
            ax.errorbar(d.log_inv_delta, d.avg_tau, yerr=d.tau_ci95, color=c, marker=m, capsize=3, label=algo)
        ax.set_xlabel(r'$\log(1/\delta)$')
        ax.set_ylabel(r'average stopping time $\mathbb{E}[\tau]$')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()

    fig, ax = plt.subplots(figsize=(6, 4.2))
    tau_panel(ax)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, 'tau_vs_log_delta.png'), dpi=150)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    tau_panel(axs[0])
    for algo in algos:
        d = df[df.algo == algo].sort_values('log_inv_delta')
        c, m = STYLE[algo]
        axs[1].errorbar(d.log_inv_delta, d.error_prob, yerr=d.error_ci95, color=c, marker=m, capsize=3, label=algo)
        axs[2].plot(d.log_inv_delta, d.avg_wall_ms, color=c, marker=m, label=algo)
    xs = np.sort(df.log_inv_delta.unique())
    axs[1].plot(xs, np.exp(-xs), 'k--', lw=1, label=r'target $\delta$')
    axs[1].set_ylim(bottom=-0.005, top=max(0.15, float(df.error_prob.max()) + 0.02))
    axs[1].set_xlabel(r'$\log(1/\delta)$')
    axs[1].set_ylabel(r'empirical error probability $P(\hat H_\tau \neq H^\star)$')
    axs[2].set_xlabel(r'$\log(1/\delta)$')
    axs[2].set_ylabel('average time per trajectory (ms)')
    for ax in axs[1:]:
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, 'fixed_instance_panels.png'), dpi=150)
    print(f"Saved {a.dir}/tau_vs_log_delta.png and {a.dir}/fixed_instance_panels.png")


if __name__ == '__main__':
    main()
