"""
Fixed-instance comparison of TaS, TTPS, Uniform and ICPE across delta (stochastic Gaussian BAI, fixed confidence).

NEW CODE (not from the ICPE repository).

1. One instance mu = (mu_1, ..., mu_K) is drawn once from the training prior P (minimum_gap_sampling with
   max_value = 0.4K, minimum_gap = 1/K, std 0.5, as in experiment_stochastic_fc.sh) after
   np.random.seed(--env_seed). It is saved in <out>/environment.json and reused on later calls, so the
   instance is the same for every delta and every algorithm.
2. For every delta = exp(-x), x in --logs, and every algorithm, it plays --num_traj trajectories on that
   instance. Trajectory j uses seed = --seed_start + j - 1 for every algorithm and every delta, with common
   random numbers: the k-th pull of arm a returns the same reward in every run with that seed
   (baselines.CRNRewards).
     TaS, TTPS, Uniform : baselines.run_baseline, no horizon cap (run until the GLR stopping rule fires).
     ICPE               : the model trained with `main.py --delta exp(-x)`, greedy a_t = argmax Q, until a_stop.
                          ICPE's transformers have exactly N input positions (N = the model's training
                          horizon), so a trajectory that reaches N pulls without a_stop cannot continue;
                          it is recorded with hit_cap = 1. Train with an N that is never reached.
3. For every trajectory it records tau (= pulls + 1 if stopped, the repository's convention), the number of
   pulls, whether it stopped, whether the final guess is the best arm, and the wall-clock time.

Outputs in --out:
  environment.json                      the fixed instance and the run settings
  log<x>/<algo>.csv                     one row per trajectory
  summary.csv                           one row per (delta, algorithm): average tau with 95% CI, empirical
                                        error probability with 95% CI and rule-of-three upper bound,
                                        average wall time per trajectory, cap / guard hits

Usage (from the repo folder, `bai` environment; ICPE needs a GPU):
  python eval_fixed_instance.py                                   # all algorithms, x = 0 2 4 6 8 10
  python eval_fixed_instance.py --algos TaS TTPS Uniform          # baselines only (CPU, no models needed)
"""
import argparse
import json
import math
import os
import time
import numpy as np
import pandas as pd
from baselines import CRNRewards, run_baseline

ALGOS = ['TaS', 'TTPS', 'Uniform', 'ICPE']
ICPE_PATH = ('runs_delta_sweep/log{x}/models/sequential/'
             'StochasticGaussianBanditFixedConfidenceSequential_{N}_{K}/icpe_{steps}_{seed}.pkl')


def fixed_instance(K: int, env_seed: int, out: str) -> dict:
    """Draw the instance once from P and store it; later calls reuse the stored instance."""
    path = os.path.join(out, 'environment.json')
    if os.path.exists(path):
        with open(path) as f:
            env = json.load(f)
        assert env['num_arms'] == K, f'{path} holds a K = {env["num_arms"]} instance, not K = {K}'
        return env
    from icpe_setup import create_config
    from icpe_sampling import sample_env
    cfg = create_config(**{'environment.num_actions': K,
                           'environment.bandit_env.max_value': float('%.6g' % (0.4 * K)),
                           'environment.bandit_env.minimum_gap': float('%.6g' % (1 / K))})
    np.random.seed(env_seed)
    e = sample_env(cfg.environment)
    env = {'env_seed': env_seed, 'num_arms': K, 'means': e.means.tolist(), 'stds': e.std.tolist(),
           'best_arm': int(np.argmax(e.means))}
    os.makedirs(out, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(env, f, indent=2)
    return env


# ---------------------------------------------------------------------------
# ICPE: one greedy trajectory until a_stop (or the model's N positions run out)
# ---------------------------------------------------------------------------
def make_icpe_runner(model_path: str):
    import torch
    from icpe_inference import load
    from icpe_setup import TransformerState
    from icpe_sampling import GaussianBanditEnv, VecBanditEnv, sample_actions

    class CRNGaussianBanditEnv(GaussianBanditEnv):
        def __init__(self, horizon, means, stds, crn: CRNRewards):
            super().__init__(horizon=horizon, means=means, stds=stds)
            self.crn = crn

        def _step(self, action: int) -> float:
            return self.crn.pull(int(action))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = load(model_path, device)
    K, N, stop_action = agent.base_num_actions, agent.horizon, agent.num_actions - 1
    sync = (lambda: torch.cuda.synchronize()) if device == 'cuda' else (lambda: None)

    with torch.inference_mode():   # warm-up so CUDA initialisation is not timed
        dummy = torch.zeros(1, N, agent.state_dim, device=device)
        sample_actions(agent, TransformerState(state=dummy, timestep=torch.zeros(1, device=device)), greedy=True)
        agent.i_net.forward(dummy, torch.zeros(1, device=device))
    sync()

    @torch.inference_mode()
    def run(means, stds, seed):
        torch.manual_seed(seed)
        env = CRNGaussianBanditEnv(N, np.asarray(means), np.asarray(stds), CRNRewards(means, stds, seed))
        envs = VecBanditEnv([env])
        states, _, _ = envs.reset()
        states = states.to_tensor(device)
        t_now = torch.zeros(1, device=device)
        stopped = False
        for _ in range(N):
            seq = states.to_transformer_input(K, t_now)
            actions, _, _ = sample_actions(agent, seq, greedy=True)
            a = int(actions.item())
            if a == stop_action:
                stopped = True
            nxt, dones, already = envs.step([a])
            states = nxt.to_tensor(device)
            t_now = t_now + (1 - torch.from_numpy(already).float().to(device))
            if stopped or bool(dones[0]):
                break
        seq = states.to_transformer_input(K, t_now)
        guess = int(agent.i_net.forward(seq.state, seq.timestep).argmax(-1).item())
        pulls = int(env.current_step)
        return {'tau': pulls + 1 if stopped else pulls, 'num_pulls': pulls, 'stopped': int(stopped),
                'hit_cap': int(not stopped), 'guess': guess, 'correct': int(guess == int(np.argmax(means)))}

    return run, sync, N


def summarise(df: pd.DataFrame) -> dict:
    n = len(df)
    tau, err, ms = df.tau.to_numpy(float), 1 - df.correct.to_numpy(float), df.wall_ms.to_numpy(float)
    p = err.mean()
    return {'num_traj': n,
            'avg_tau': tau.mean(), 'tau_ci95': 1.96 * tau.std(ddof=1) / math.sqrt(n), 'max_tau': int(tau.max()),
            'error_prob': p, 'error_ci95': 1.96 * math.sqrt(p * (1 - p) / n),
            'error_upper95_if_zero': 3 / n if p == 0 else np.nan,      # rule of three
            'avg_wall_ms': ms.mean(), 'std_wall_ms': ms.std(ddof=1),
            'not_stopped': int((df.stopped == 0).sum())}


def main():
    p = argparse.ArgumentParser(description='Fixed-instance comparison of TaS, TTPS, Uniform and ICPE across delta')
    p.add_argument('--K', type=int, default=4)
    p.add_argument('--env_seed', type=int, default=0, help='seed used once to draw the fixed instance from P')
    p.add_argument('--num_traj', type=int, default=500, help='trajectories per (delta, algorithm)')
    p.add_argument('--seed_start', type=int, default=1, help='trajectory j uses seed seed_start + j - 1')
    p.add_argument('--logs', nargs='+', default=['0', '2', '4', '6', '8', '10'], help='x values, delta = exp(-x)')
    p.add_argument('--algos', nargs='+', default=ALGOS, choices=ALGOS)
    p.add_argument('--icpe_path', default=ICPE_PATH, help='model path template with {x}, {N}, {K}, {steps}, {seed}')
    p.add_argument('--icpe_N', type=int, default=100, help='horizon N the ICPE models were trained with')
    p.add_argument('--steps', type=int, default=30000, help='ICPE checkpoint (training iteration)')
    p.add_argument('--train_seed', type=int, default=1, help='training seed of the ICPE models')
    p.add_argument('--max_pulls', type=int, default=10_000_000, help='safety guard for the baselines only')
    p.add_argument('--out', default='runs_delta_sweep/fixed_instance')
    a = p.parse_args()

    env = fixed_instance(a.K, a.env_seed, a.out)
    means, stds = np.array(env['means']), np.array(env['stds'])
    print(f"Fixed instance (env_seed {env['env_seed']}): means {np.round(means, 4)}, best arm {env['best_arm']}")
    seeds = list(range(a.seed_start, a.seed_start + a.num_traj))

    summary_path = os.path.join(a.out, 'summary.csv')
    rows = pd.read_csv(summary_path).to_dict('records') if os.path.exists(summary_path) else []

    for x in a.logs:
        delta = math.exp(-float(x))
        d = os.path.join(a.out, f'log{x}')
        os.makedirs(d, exist_ok=True)
        for algo in a.algos:
            if algo == 'ICPE':
                path = a.icpe_path.format(x=x, N=a.icpe_N, K=a.K, steps=a.steps, seed=a.train_seed)
                if not os.path.exists(path):
                    print(f'log(1/delta)={x}: ICPE model missing ({path}), skipped')
                    continue
                run, sync, N = make_icpe_runner(path)
            else:
                run = lambda m, s, seed, algo=algo: run_baseline(algo, m, s, delta, seed, a.max_pulls)
                sync = lambda: None

            recs = []
            for j, seed in enumerate(seeds, start=1):
                sync()
                t0 = time.perf_counter()
                r = run(means, stds, seed)
                sync()
                recs.append({'traj_id': j, 'seed': seed, **r, 'wall_ms': (time.perf_counter() - t0) * 1000})
            df = pd.DataFrame(recs)
            df.to_csv(os.path.join(d, f'{algo}.csv'), index=False)

            s = {'log_inv_delta': float(x), 'delta': delta, 'algo': algo, **summarise(df)}
            rows = [r for r in rows if not (r['log_inv_delta'] == s['log_inv_delta'] and r['algo'] == algo)] + [s]
            pd.DataFrame(rows).sort_values(['log_inv_delta', 'algo']).to_csv(summary_path, index=False)
            print(f"log(1/delta)={x:>3} {algo:8s} avg tau {s['avg_tau']:8.2f} +- {s['tau_ci95']:.2f} "
                  f"(max {s['max_tau']}), error {s['error_prob']:.4f}, {s['avg_wall_ms']:.2f} ms/traj, "
                  f"not stopped {s['not_stopped']}", flush=True)
    print(f'Saved {summary_path}')


if __name__ == '__main__':
    main()
