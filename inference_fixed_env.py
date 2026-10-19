"""
ICPE - inference (Algorithm 1, lines 14-15) on ONE fixed environment, one trajectory at a time.

NEW CODE (not from the ICPE repository). The rollout and the final guess are those of
icpe_inference.sample_trajectories (greedy a_t = argmax_a Q_theta(D_t, a) until a_stop or N pulls, then
H_hat = argmax_H I_phi(H | D)), run for a single trajectory at a time so that each trajectory has its own
seed and its own wall-clock time.

Fixed environment: drawn once from the training distribution P (icpe_sampling.sample_env, i.e.
minimum_gap_sampling with the model's configuration) after np.random.seed(--env_seed).

Trajectory i (i = 1, ..., --num_traj) uses seed = --seed_start + i - 1 (default seeds 1, ..., 5000):
np.random.seed(seed), torch.manual_seed(seed), and the environment's reward noise is drawn from its own
generator seeded with `seed` (GaussianBanditEnv.seed).

Saved in --out:
  environment.json  the fixed environment (means, stds, best arm, env_seed) and the run settings
  trajectories.csv  one row per trajectory:
    traj_id     1, ..., num_traj
    seed        seed of the trajectory
    stop_step   round (1-based) at which the STOP action was taken = num_pulls + 1;
                -1 if the trajectory reached N pulls without taking STOP
    num_pulls   number of arm pulls before stopping (or N)
    stopped     1 if STOP was taken, 0 if the trajectory reached N pulls
    wall_ms     wall-clock time of the trajectory in milliseconds (environment creation, rollout and
                final guess; GPU synchronised before and after)
    correct     1 if the predicted arm argmax I_phi is the best arm, else 0

Usage (from this folder, inside the `bai` conda environment):
  python inference_fixed_env.py --model <path to icpe_<step>_<seed>.pkl>
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from icpe_inference import load
from icpe_setup import TransformerState
from icpe_sampling import sample_env, sample_actions, GaussianBanditEnv, VecBanditEnv


def sync(device: str):
    if device == 'cuda':
        torch.cuda.synchronize()


@torch.inference_mode()
def run_trajectory(agent, means: np.ndarray, stds: np.ndarray, seed: int, device: str) -> dict:
    """One greedy trajectory on the fixed environment (as icpe_inference.sample_trajectories with B = N = 1)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    K = agent.base_num_actions
    stop_action = agent.num_actions - 1

    sync(device)
    t0 = time.perf_counter()

    env = GaussianBanditEnv(horizon=agent.cfg.environment.horizon, means=means, stds=stds)
    env.seed(seed)
    envs = VecBanditEnv([env])
    states, dones, stopped = envs.reset()
    states = states.to_tensor(agent.device)
    traj_current_timestep = torch.zeros(1, device=agent.device, dtype=torch.float32)

    stop_step = -1
    for h in range(agent.horizon):                                             # line 14
        seq_state = states.to_transformer_input(K, traj_current_timestep)
        actions, qvalues, probs = sample_actions(agent, seq_state, greedy=True)
        actions = actions.tolist()
        if actions[0] == stop_action and stop_step == -1:
            stop_step = h + 1

        next_states, dones, stopped = envs.step(actions)
        next_states = next_states.to_tensor(agent.device)
        dones = torch.from_numpy(dones).to(agent.device)
        stopped = torch.from_numpy(stopped).to(agent.device)

        states = next_states
        traj_current_timestep = traj_current_timestep + (1 - stopped.float())
        if torch.all(dones):
            break

    # Line 15: H_hat = argmax_H I_phi(H | D)
    seq_state = states.to_transformer_input(K, traj_current_timestep)
    log_probs = agent.i_net.forward(seq_state.state, seq_state.timestep)
    predicted = int(log_probs.argmax(-1).item())

    sync(device)
    wall_ms = (time.perf_counter() - t0) * 1000

    return {'seed': seed,
            'stop_step': stop_step,
            'num_pulls': int(env.current_step),
            'stopped': int(stop_step != -1),
            'wall_ms': wall_ms,
            'correct': int(predicted == int(np.argmax(means)))}


def main():
    p = argparse.ArgumentParser(description='ICPE inference on one fixed environment, one seed per trajectory')
    p.add_argument('--model', required=True, help='path to a saved model icpe_<step>_<seed>.pkl')
    p.add_argument('--env_seed', type=int, default=0, help='seed used to draw the fixed environment from P')
    p.add_argument('--num_traj', type=int, default=5000, help='number of trajectories')
    p.add_argument('--seed_start', type=int, default=1, help='trajectory i uses seed seed_start + i - 1')
    p.add_argument('--out', default=None, help='output folder (default: inference_env<env_seed> next to the model)')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = load(args.model, device)
    cfg = agent.cfg

    # The fixed environment, drawn from P
    np.random.seed(args.env_seed)
    env0 = sample_env(cfg.environment)
    means, stds = env0.means, env0.std

    out = args.out or os.path.join(os.path.dirname(args.model), f'inference_env{args.env_seed}')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'environment.json'), 'w') as f:
        json.dump({'env_seed': args.env_seed,
                   'means': means.tolist(),
                   'stds': stds.tolist(),
                   'best_arm': int(np.argmax(means)),
                   'num_arms': int(cfg.environment.num_actions),
                   'horizon_N': int(cfg.environment.horizon),
                   'confidence_level': float(cfg.agent.confidence_level),
                   'model': os.path.abspath(args.model),
                   'num_traj': args.num_traj,
                   'seeds': [args.seed_start, args.seed_start + args.num_traj - 1],
                   'device': device}, f, indent=2)
    print(f'Fixed environment (env_seed {args.env_seed}): means {np.round(means, 4)}, best arm {int(np.argmax(means))}')

    # GPU warm-up (CUDA initialisation), so that it is not counted in the first trajectory's time;
    # it uses no random numbers and does not change the results
    with torch.inference_mode():
        dummy = torch.zeros(1, agent.horizon, agent.state_dim, device=agent.device)
        sample_actions(agent, TransformerState(state=dummy, timestep=torch.zeros(1, device=agent.device)), greedy=True)
        agent.i_net.forward(dummy, torch.zeros(1, device=agent.device))
    sync(device)

    rows = []
    for i in range(1, args.num_traj + 1):
        r = run_trajectory(agent, means, stds, args.seed_start + i - 1, device)
        rows.append({'traj_id': i, **r})
        if i % 500 == 0:
            df = pd.DataFrame(rows)
            print(f'{i}/{args.num_traj}: accuracy {df.correct.mean():.4f}, '
                  f'mean num_pulls {df.num_pulls.mean():.2f}, mean wall {df.wall_ms.mean():.1f} ms', flush=True)

    df = pd.DataFrame(rows)[['traj_id', 'seed', 'stop_step', 'num_pulls', 'stopped', 'wall_ms', 'correct']]
    df.to_csv(os.path.join(out, 'trajectories.csv'), index=False)
    print(f'Saved {out}/trajectories.csv and {out}/environment.json')


if __name__ == '__main__':
    main()
