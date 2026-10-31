"""
Training-time certification and stopping, Proposition B.12 (App. B.1.6 of the ICPE paper).
NEW CODE (the released repository trains for a fixed number of steps and has no such test).

Train for the target delta (lambda update, eq. 11) and certify a looser delta' > delta: after every
`every` training steps (one test "epoch" t), freeze the networks and run K fresh greedy rollouts on new
bandits M ~ P (never used for training). X_t = their accuracy, i.e. the fraction whose answer
argmax_H I_phi(H | D_tau) is the best arm. Stop training at

    T = inf{ t >= 1 :  (1/t) sum_{s<=t} X_s  >=  (1 - delta')
                       + (1/t) sqrt( 2 (1 + (1-delta')/K t) ln( sqrt(1 + (1-delta')/K t) / eta ) ) }.

Proposition B.12: if, with probability >= 1 - xi, the accuracy p_t is eventually non-decreasing and
reaches 1 - delta, then P(T < inf, p_T >= 1 - delta') >= P(T < inf) - (eta + xi), i.e. the returned
networks are delta'-correct (on average over the prior) with confidence about 1 - eta - xi.

The test starts at a fixed training step (`start`), chosen before training, so the running average does
not include the first, untrained epochs.
"""
from __future__ import annotations
import csv
import json
import math
import os
import numpy as np
import torch
from icpe_inference import sample_trajectories


def boundary(t: int, K: int, delta_prime: float, eta: float) -> float:
    """Right-hand side of the stopping rule in Proposition B.12 at test epoch t."""
    a = (1 - delta_prime) / K * t
    return (1 - delta_prime) + math.sqrt(2 * (1 + a) * math.log(math.sqrt(1 + a) / eta)) / t


@torch.no_grad()
def evaluate_accuracy(agent, K: int, chunk: int = 250) -> dict:
    """X_t: accuracy of the frozen greedy policy on K fresh bandits M ~ P (one trajectory each),
    with the average and maximum stopping time of these K test trajectories."""
    correct, taus = 0.0, []
    for start in range(0, K, chunk):
        n = min(chunk, K - start)
        with torch.amp.autocast(agent.device, enabled=agent.use_amp):
            traj = sample_trajectories(agent, n, 1).to_numpy()
        tau = traj.stopping_time.astype(np.int64)                              # (n, 1)
        c = np.take_along_axis(traj.rew_opt_actions, (tau - 1)[..., None], axis=-1)[..., 0]
        correct += float(c.sum())
        taus.append(tau.ravel())
    taus = np.concatenate(taus)
    return {'accuracy': correct / K, 'avg_tau': float(taus.mean()), 'max_tau': int(taus.max())}


class TrainingCertificate:
    """Running state of the Proposition B.12 test."""

    def __init__(self, delta_prime: float, eta: float, K: int):
        self.delta_prime, self.eta, self.K = delta_prime, eta, K
        self.t, self.sum_x = 0, 0.0
        self.rows = []

    def update(self, x: float, step: int, avg_tau: float = float('nan'), max_tau: int = -1):
        """Add the accuracy X_t of one test epoch; return (crossed, running average, boundary)."""
        self.t += 1
        self.sum_x += x
        avg = self.sum_x / self.t
        b = boundary(self.t, self.K, self.delta_prime, self.eta)
        crossed = avg >= b
        self.rows.append({'training_step': step, 'test_epoch': self.t, 'X_t': x,
                          'running_average': avg, 'boundary': b, 'crossed': int(crossed),
                          'test_avg_tau': avg_tau, 'test_max_tau': max_tau})
        return crossed, avg, b

    def save_history(self, path: str):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)

    def save_summary(self, path: str, certified: bool, delta: float, model_file: str | None):
        last = self.rows[-1] if self.rows else {}
        x = np.array([r['X_t'] for r in self.rows]) if self.rows else np.zeros(0)
        ma = np.convolve(x, np.ones(20) / 20, mode='valid') if len(x) >= 20 else x
        with open(path, 'w') as f:
            json.dump({'certified': certified, 'delta_prime': self.delta_prime, 'delta_train': delta,
                       'eta': self.eta, 'rollouts_per_epoch': self.K,
                       'T': self.t if certified else None,       # stopping time of the test (test epochs)
                       'test_epochs': self.t,
                       'stopping_training_step': last.get('training_step'),
                       # monotonicity check of p_t (Prop. B.12 assumes it is eventually non-decreasing):
                       # X_t is noisy, so it is checked on a 20-epoch moving average
                       'X_t_moving_avg_first': float(ma[0]) if len(ma) else None,
                       'X_t_moving_avg_last': float(ma[-1]) if len(ma) else None,
                       'X_t_moving_avg_decreases': int((np.diff(ma) < 0).sum()) if len(ma) > 1 else 0,
                       'X_t_moving_avg_steps': max(len(ma) - 1, 0),
                       'test_avg_tau_last': last.get('test_avg_tau'), 'test_max_tau_last': last.get('test_max_tau'),
                       'running_average': last.get('running_average'), 'boundary': last.get('boundary'),
                       'model_file': model_file}, f, indent=2)
