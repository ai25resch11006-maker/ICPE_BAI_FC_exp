"""
Baselines for stochastic Gaussian BAI, fixed confidence: TaS, TTPS, Uniform.

NEW CODE. A one-trajectory, pure-NumPy port of the authors' baselines (branch neurips_submission,
commit 0260c4f):
  icpe/agents/bandit/baselines/tas.py      -> TaS.sample_trajectories
  icpe/agents/bandit/baselines/ttps.py     -> TTPS.sample_trajectories (beta = 0.65, epsilon = 0.3)
  icpe/agents/bandit/baselines/uniform.py  -> Uniform.sample_trajectories
  icpe/agents/bandit/base_agent.py         -> should_stop
  icpe/agents/bandit/utils.py              -> stopping_rule, compute_gaps

Differences from the repository:
  * No horizon cap: a trajectory runs until the stopping rule fires (`max_pulls` is only a safety
    guard against an infinite loop; reaching it is reported as hit_guard = 1).
  * Common random numbers: the k-th pull of arm a always returns mu_a + sigma * Z[a, k], where Z[a, .]
    comes from a generator seeded with (seed, a). Every algorithm run with the same seed therefore sees
    the same rewards for the same (arm, pull number). The algorithms' own randomness (TaS / TTPS / Uniform
    sampling) uses a separate generator seeded with (seed, 10_000).
  * Empirical means use the pulls actually made. The repository's TTPS strategy and the final MLE guess
    in HistoryBuilder.build both read the cumulative statistics one slot too far, which adds a phantom
    zero-reward pull of arm 0; this is not reproduced.

Stopping time convention (as in the repository): tau = number of pulls + 1 (the stop counts as a step).
"""
from __future__ import annotations
import numpy as np

STOP_SIGMA = 0.5          # should_stop: sigma = cfg.environment.bandit_env.std (0.5)
STOP_MINIMUM_GAP = 0.4    # should_stop: stopping_rule(..., 0.4)
TTPS_BETA = 0.65
TTPS_EPSILON = 0.3
TAS_MINIMUM_GAP = 0.0     # TaS loaded without minimum_gap in the authors' notebook
ALGO_SEED_OFFSET = 10_000


class CRNRewards:
    """Rewards with common random numbers: one Gaussian stream per arm, seeded with (seed, arm)."""

    def __init__(self, means: np.ndarray, stds: np.ndarray, seed: int):
        self.means = np.asarray(means, dtype=np.float64)
        self.stds = np.asarray(stds, dtype=np.float64)
        self.rngs = [np.random.RandomState([seed, a]) for a in range(len(self.means))]

    def pull(self, a: int) -> float:
        return float(self.means[a] + self.stds[a] * self.rngs[a].normal())


def compute_gaps(mu: np.ndarray, minimum_gap: float = 0.0) -> np.ndarray:
    """utils.compute_gaps for one trajectory: best arm gets (top1 - top2), others (top1 - mu_a + minimum_gap)."""
    top = np.sort(mu)
    best = np.isclose(mu, top[-1])
    gaps = top[-1] - mu + minimum_gap
    gaps[best] = top[-1] - top[-2]
    return gaps


def glr_should_stop(mu: np.ndarray, counts: np.ndarray, delta: float, t: int,
                    sigma: float = STOP_SIGMA, minimum_gap: float = STOP_MINIMUM_GAP) -> bool:
    """utils.stopping_rule for one trajectory (formula copied as written in the repository)."""
    best = np.isclose(mu, mu.max())
    if best.sum() >= 2:
        return False
    gaps = compute_gaps(mu, minimum_gap)
    w = counts / counts.sum()
    num = w[best] * w[~best] * gaps[~best] ** 2
    den = w[best] + w[~best] * 2 * sigma ** 2
    T = (num / den).min()
    return t * T >= np.log((1 + np.log(t)) / delta)


def run_baseline(algo: str, means: np.ndarray, stds: np.ndarray, delta: float, seed: int,
                 max_pulls: int = 10_000_000) -> dict:
    """One trajectory of TaS, TTPS or Uniform on a fixed bandit, until the GLR stopping rule fires.

    Step h (h = number of pulls so far) follows the repository's loop: choose an arm with the sampling
    rule; if h >= K and the stopping rule fires on the data of the h pulls (with t = h - 1, as
    should_stop(states, h - 1) passes it), take the stop action instead.
    """
    K = len(means)
    rewards = CRNRewards(means, stds, seed)
    rng = np.random.RandomState([seed, ALGO_SEED_OFFSET])
    counts = np.zeros(K)
    sums = np.zeros(K)
    h = 0
    stopped = False
    while h < max_pulls:
        mu = sums / np.maximum(counts, 1)

        # sampling rule
        if algo == 'Uniform':
            a = rng.randint(K)
        elif h < K:                                   # TaS and TTPS pull every arm once first
            a = h
        elif algo == 'TaS':
            gaps = np.maximum(compute_gaps(mu, TAS_MINIMUM_GAP), 1e-12)
            probs = 1 / gaps ** 2
            a = rng.choice(K, p=probs / probs.sum())
            undersampled = counts < np.sqrt(h - 1) - K / 2   # forced exploration
            if undersampled.any():
                a = int(np.argmax(undersampled))
        elif algo == 'TTPS':
            order = np.argsort(mu)
            strategy = np.zeros(K)
            strategy[order[-1]] = TTPS_BETA
            strategy[order[-2]] = 1 - TTPS_BETA
            strategy = (1 - TTPS_EPSILON) * strategy + TTPS_EPSILON / K
            a = rng.choice(K, p=strategy)
        else:
            raise ValueError(f'unknown algorithm {algo}')

        # stopping rule (overrides the sampled arm)
        if h >= K and glr_should_stop(mu, counts, delta, t=h - 1):
            stopped = True
            break

        r = rewards.pull(int(a))
        counts[a] += 1
        sums[a] += r
        h += 1

    mu = sums / np.maximum(counts, 1)
    guess = int(np.argmax(mu))
    return {'tau': h + 1 if stopped else h, 'num_pulls': h, 'stopped': int(stopped),
            'hit_guard': int(not stopped), 'guess': guess, 'correct': int(guess == int(np.argmax(means)))}
