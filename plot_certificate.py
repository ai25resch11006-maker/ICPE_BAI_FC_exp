"""
Plots of one certified-training run (NEW CODE, not from the ICPE repository).

Reads, from the run's model folder, training_log_<seed>.csv (one row per training step) and
certificate_<seed>.csv / .json (one row per test epoch of Proposition B.12), and saves certificate_plots.png:
  (a) accuracy of the training episodes: p_hat of each step (moving average) and the p_hat used by the
      lambda update, with the training target 1 - delta
  (b) stopping time of the training episodes: average and maximum per step
  (c) the Proposition B.12 test: X_t, running average and boundary per test epoch, with 1 - delta'
      (training stops at the first epoch T where the running average reaches the boundary)
  (d) the cost per pull c = 1/lambda

Usage: python plot_certificate.py runs_certify/dp0.1      (the run folder, i.e. where train.log is)
"""
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'runs_certify/dp0.1'
    model_dir = glob.glob(os.path.join(run, 'models', 'sequential', '*'))[0]
    tr = pd.read_csv(glob.glob(os.path.join(model_dir, 'training_log_*.csv'))[0])
    cert_csv = glob.glob(os.path.join(model_dir, 'certificate_*.csv'))
    ce = pd.read_csv(cert_csv[0]) if cert_csv else None
    js = glob.glob(os.path.join(model_dir, 'certificate_*.json'))
    summary = json.load(open(js[0])) if js else {}

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    ma = lambda s, w=50: s.rolling(w, min_periods=1).mean()

    ax = axs[0, 0]
    ax.plot(tr.training_step, tr.p_hat_used, lw=0.6, alpha=0.35, color='tab:orange', label=r'$\hat p$ used by the $\lambda$ update')
    ax.plot(tr.training_step, ma(tr.p_hat_step), lw=1.6, color='tab:blue', label=r'$\hat p$ of each step (50-step moving average)')
    if 'delta_train' in summary:
        ax.axhline(1 - summary['delta_train'], color='k', ls='--', lw=1, label=r'training target $1-\delta$')
    ax.set_title('(a) accuracy of the training episodes'); ax.set_xlabel('training step'); ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(tr.training_step, ma(tr.avg_tau), lw=1, label='average stopping time (50-step moving average)')
    ax.plot(tr.training_step, tr.max_tau, lw=0.6, alpha=0.6, label='maximum stopping time')
    ax.set_title('(b) stopping time of the training episodes'); ax.set_xlabel('training step'); ax.legend(fontsize=8)

    ax = axs[1, 0]
    if ce is not None and len(ce):
        ax.plot(ce.test_epoch, ce.X_t, lw=0.8, alpha=0.5, label=r'$X_t$ (accuracy on fresh rollouts)')
        ax.plot(ce.test_epoch, ce.running_average, lw=1.5, label='running average')
        ax.plot(ce.test_epoch, ce.boundary.clip(upper=1.0), 'k-', lw=1, label='Prop. B.12 boundary (clipped at 1)')
        if 'delta_prime' in summary:
            ax.axhline(1 - summary['delta_prime'], color='r', ls='--', lw=1, label=r"certified level $1-\delta'$")
        if summary.get('certified'):
            ax.axvline(summary['T'], color='g', ls=':', lw=1.5, label=f"T = {summary['T']} (step {summary['stopping_training_step']})")
        lo = min(ce.X_t.min(), 1 - summary.get('delta_prime', 0.1)) - 0.02
        ax.set_ylim(lo, 1.005)
    ax.set_title('(c) Proposition B.12 test'); ax.set_xlabel('test epoch t'); ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.plot(tr.training_step, tr.cost_c, lw=1)
    ax.set_title(r'(d) cost per pull $c = 1/\lambda$'); ax.set_xlabel('training step')

    for a in axs.ravel():
        a.grid(True, linestyle='--', alpha=0.3)
    status = ('certified' if summary.get('certified') else 'not certified (yet)') if summary else 'running'
    fig.suptitle(f"{run}: delta' = {summary.get('delta_prime', '?')}, delta = {summary.get('delta_train', '?')} -- {status}",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(model_dir, 'certificate_plots.png')
    fig.savefig(out, dpi=140)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
