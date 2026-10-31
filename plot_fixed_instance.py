"""
Plots for eval_fixed_instance.py (NEW CODE, not from the ICPE repository).

Reads <dir>/summary.csv and <dir>/environment.json and saves
  <dir>/tau_vs_log_delta.png        average stopping time vs log(1/delta), one line per algorithm (95% CI),
                                    with three lower bounds T*_S(mu) kl(delta, 1 - delta) of the fixed instance:
                                    S = all Gaussian bandits (no gap information), S = bandits whose best arm
                                    beats every other arm by >= gap (the class of Thm. B.25), and S = bandits
                                    whose arms all differ by >= gap (the prior used in the code)
  <dir>/lower_bounds.json            the three T*_S(mu) and optimal weights (cached; recomputed if mu changes)
  <dir>/fixed_instance_panels.png   (a) average stopping time, (b) empirical error probability with the
                                    target delta, (c) average wall time per trajectory

Usage: python plot_fixed_instance.py [--dir runs_delta_sweep/fixed_instance]
"""
import argparse
import itertools
import json
import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def characteristic_time(means: np.ndarray, sigma: float) -> tuple:
    """T*(mu) for Gaussian BAI with known variance sigma^2 (Garivier and Kaufmann, 2016).

    T*(mu)^-1 = max_w min_{a != a*} (w_* w_a / (w_* + w_a)) Delta_a^2 / (2 sigma^2). At the optimum,
    with x_a = w_a / w_*, every pair has the same value y = c_a x_a / (1 + x_a), c_a = Delta_a^2 / (2 sigma^2),
    and sum_a x_a^2 = 1; y is found by bisection. Returns T* and the optimal weights w*.
    """
    best = int(np.argmax(means))
    others = [a for a in range(len(means)) if a != best]
    c = (means[best] - means[others]) ** 2 / (2 * sigma ** 2)
    lo, hi = 0.0, c.min()
    for _ in range(200):
        y = (lo + hi) / 2
        if np.sum((y / (c - y)) ** 2) > 1:
            hi = y
        else:
            lo = y
    x = y / (c - y)
    w = np.zeros(len(means))
    w[best] = 1 / (1 + x.sum())
    w[others] = x * w[best]
    return 1 / (w[best] * y), w


def kl_bernoulli(p: float, q: float) -> float:
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def lower_bound(means: np.ndarray, sigma: float, deltas: np.ndarray) -> np.ndarray:
    """E[tau] >= T*(mu) kl(delta, 1 - delta) for delta < 1/2 (0 for delta >= 1/2)."""
    T, _ = characteristic_time(means, sigma)
    return np.array([T * kl_bernoulli(d, 1 - d) if d < 0.5 else 0.0 for d in deltas])


STYLE = {'ICPE': ('#1a54a6', 'o'), 'TaS': ('#9467bd', '^'), 'TTPS': ('#8c564b', 'x'), 'Uniform': ('#ff7f0e', 'v')}


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) regression of y with weights w (pool adjacent violators)."""
    blocks = []                                        # [value, weight, length]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.array([v for v, _, n in blocks for _ in range(n)])


def _closest_alternative_all(mu, w, gap, best):
    """min of sum_a w_a (mu_a - lambda_a)^2 over lambda whose arms all differ by >= gap and whose best arm
    is not `best`: for each order of the arms the constraints are linear, and substituting
    nu_i = lambda_(i) - i*gap turns them into an isotonic regression."""
    K, val = len(mu), np.inf
    shift = gap * np.arange(K)
    for perm in itertools.permutations(range(K)):      # perm = arms in increasing order of lambda
        if perm[-1] == best:
            continue
        p = np.array(perm)
        lam = _pava(mu[p] - shift, w[p] + 1e-15) + shift
        val = min(val, float(np.sum(w[p] * (mu[p] - lam) ** 2)))
    return val


def _closest_alternative_top2(mu, w, gap, best):
    """min of sum_a w_a (mu_a - lambda_a)^2 over lambda whose best arm b != `best` beats every other arm by
    >= gap: for fixed lambda_b = x the others are lowered to min(mu_c, x - gap); convex in x (ternary search)."""
    val = np.inf
    for b in range(len(mu)):
        if b == best:
            continue
        others = [c for c in range(len(mu)) if c != b]
        g = lambda x: w[b] * (mu[b] - x) ** 2 + sum(w[c] * max(0.0, mu[c] - (x - gap)) ** 2 for c in others)
        lo, hi = mu.min() - 2.0, mu.max() + 2.0
        for _ in range(80):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if g(m1) < g(m2):
                hi = m2
            else:
                lo = m1
        val = min(val, g((lo + hi) / 2))
    return val


def structured_characteristic_time(means: np.ndarray, sigma: float, gap: float, mode: str,
                                   iters: int = 800, seed: int = 0) -> tuple:
    """T*_S(mu) for Gaussian BAI over a restricted class S (Garivier and Kaufmann, 2016, with structure):

        T*_S(mu)^-1 = max_w  min_{lambda in S, best(lambda) != best(mu)}  sum_a w_a (mu_a - lambda_a)^2 / (2 sigma^2)

    mode='top2': S = bandits whose best arm beats every other arm by >= gap (Thm. B.25; gap = 0 gives the usual T*);
    mode='all' : S = bandits whose arms all differ by >= gap (the prior of the code: minimum_gap_sampling).
    The objective is concave in w; it is maximised by exponentiated (super)gradient ascent from several starts.
    Any algorithm that is delta-correct on every bandit of S needs E[tau] >= T*_S(mu) kl(delta, 1 - delta).
    """
    mu = np.asarray(means, float)
    K, best = len(mu), int(np.argmax(mu))
    inner = _closest_alternative_all if mode == 'all' else _closest_alternative_top2
    F = lambda w: inner(mu, w, gap, best) / (2 * sigma ** 2)
    rng = np.random.default_rng(seed)
    best_val, best_w = -1.0, None
    for w in [np.ones(K) / K] + [rng.dirichlet(np.ones(K)) for _ in range(3)]:
        w = w.copy()
        for k in range(1, iters + 1):
            f = F(w)
            if f > best_val:
                best_val, best_w = f, w.copy()
            eps = 1e-6
            gr = np.array([(F(w + eps * np.eye(K)[a]) - f) / eps for a in range(K)])
            w = w * np.exp(0.5 / math.sqrt(k) * gr / (np.abs(gr).max() + 1e-12))
            w /= w.sum()
    return 1.0 / best_val, best_w


def characteristic_times(means: np.ndarray, sigma: float, gap: float, cache: str = None) -> dict:
    """The three T*_S(mu) used in the plots and the results table (cached in `cache` as JSON)."""
    key = {'means': [round(float(m), 10) for m in means], 'sigma': sigma, 'gap': gap}
    if cache and os.path.exists(cache):
        with open(cache) as f:
            c = json.load(f)
        if c.get('key') == key:
            return c['T']
    T0, w0 = characteristic_time(np.asarray(means, float), sigma)
    T1, w1 = structured_characteristic_time(means, sigma, gap, 'top2')
    T2, w2 = structured_characteristic_time(means, sigma, gap, 'all')
    T = {'none': T0, 'top2': T1, 'all': T2}
    if cache:
        with open(cache, 'w') as f:
            json.dump({'key': key, 'T': T, 'weights': {'none': list(map(float, w0)), 'top2': list(map(float, w1)),
                       'all': list(map(float, w2))}}, f, indent=2)
    return T


LB_LABELS = {'none': 'Lower bound, no gap information', 'top2': 'Lower bound, top-two gap ≥ 0.4',
             'all': 'Lower bound, all gaps ≥ 0.4'}


def lower_bounds(T: dict, deltas: np.ndarray) -> dict:
    """E[tau] >= T*_S kl(delta, 1 - delta) for delta < 1/2 (0 otherwise), for each class S."""
    kl = np.array([kl_bernoulli(d, 1 - d) if d < 0.5 else 0.0 for d in deltas])
    return {k: v * kl for k, v in T.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='runs_delta_sweep/fixed_instance')
    p.add_argument('--gap', type=float, default=0.4, help='known minimum gap used by the structured lower bounds')
    a = p.parse_args()
    df = pd.read_csv(os.path.join(a.dir, 'summary.csv'))
    with open(os.path.join(a.dir, 'environment.json')) as f:
        env = json.load(f)
    algos = [x for x in STYLE if x in set(df.algo)]
    n = int(df.num_traj.max())
    title = (f"Fixed instance μ = {np.round(env['means'], 3).tolist()} (K = {env['num_arms']}), "
             f"{n} trajectories per point")

    means, sigma = np.array(env['means']), float(env['stds'][0])
    T = characteristic_times(means, sigma, a.gap, cache=os.path.join(a.dir, 'lower_bounds.json'))
    print('T*_S(mu): ' + ', '.join(f'{k} = {v:.3f}' for k, v in T.items()))
    xs_lb = np.linspace(float(df.log_inv_delta.min()), float(df.log_inv_delta.max()), 200)   # only the evaluated range
    lbs = lower_bounds(T, np.exp(-xs_lb))
    lb_style = {'none': dict(color='0.55', ls=':', label=rf'lower bound, no gap info ($T^*$ = {T["none"]:.1f})'),
                'top2': dict(color='k', ls='--', label=rf'lower bound, top-two gap $\geq${a.gap:g} ($T^*$ = {T["top2"]:.2f})'),
                'all':  dict(color='k', ls='-.', label=rf'lower bound, all gaps $\geq${a.gap:g} ($T^*$ = {T["all"]:.2f})')}

    def tau_panel(ax):
        for k in ['none', 'top2', 'all']:
            ax.plot(xs_lb, lbs[k], lw=1.2, **lb_style[k])
        for algo in algos:
            d = df[df.algo == algo].sort_values('log_inv_delta')
            c, m = STYLE[algo]
            ax.errorbar(d.log_inv_delta, d.avg_tau, yerr=d.tau_ci95, color=c, marker=m, capsize=3, label=algo)
        ax.set_xlabel(r'$\log(1/\delta)$')
        ax.set_ylabel(r'average stopping time $\mathbb{E}[\tau]$')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=7)

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
