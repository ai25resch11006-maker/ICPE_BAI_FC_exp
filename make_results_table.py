"""
Results table for the fixed-instance delta sweep (NEW CODE, not from the ICPE repository).

Reads <dir>/summary.csv and <dir>/environment.json (written by eval_fixed_instance.py) and saves one row
per (delta, algorithm):
  delta, log(1/delta), K, algorithm, trajectories, average stopping time (95% CI),
  average wall-clock time per trajectory (ms), accuracy (trajectories whose final guess is the true best
  arm / all trajectories), empirical error probability (1 - accuracy), and the number of trajectories that
  never reached a_stop (every trajectory runs until a_stop; this count checks that all of them did, i.e.
  that ICPE never ran out of its N input positions; the baselines have no limit)
plus, for every delta, a row with the lower bound T*(mu) kl(delta, 1 - delta) of the fixed instance
as <dir>/results_table.csv (numbers) and <dir>/results_table.md (formatted, for reading or pasting).

Usage: python make_results_table.py [--dir runs_delta_sweep/fixed_instance]
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from plot_fixed_instance import lower_bound

ORDER = ['ICPE', 'TaS', 'TTPS', 'Uniform']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='runs_delta_sweep/fixed_instance')
    a = p.parse_args()
    df = pd.read_csv(os.path.join(a.dir, 'summary.csv'))
    with open(os.path.join(a.dir, 'environment.json')) as f:
        env = json.load(f)
    K = env['num_arms']

    df['algo_rank'] = df.algo.map({x: i for i, x in enumerate(ORDER)})
    df = df.sort_values(['log_inv_delta', 'algo_rank'])
    LB = 'Lower bound T*(μ)·kl(δ,1−δ)'
    table = pd.DataFrame({
        'delta': df.delta,
        'log_inv_delta': df.log_inv_delta,
        'K': K,
        'algorithm': df.algo,
        'trajectories': df.num_traj,
        'avg_stopping_time': df.avg_tau,
        'stopping_time_ci95': df.tau_ci95,
        'avg_wall_clock_ms': df.avg_wall_ms,
        'correct': np.rint((1 - df.error_prob) * df.num_traj),
        'accuracy': 1 - df.error_prob,
        'error_probability': df.error_prob,
        'never_reached_a_stop': df.not_stopped,
    })
    xs = np.sort(df.log_inv_delta.unique())
    lbs = lower_bound(np.array(env['means']), float(env['stds'][0]), np.exp(-xs))
    lb_rows = pd.DataFrame({'delta': np.exp(-xs), 'log_inv_delta': xs, 'K': K, 'algorithm': LB,
                            'avg_stopping_time': lbs})
    table = pd.concat([table, lb_rows], ignore_index=True)
    table['rank'] = table.algorithm.map({x: i for i, x in enumerate(ORDER + [LB])})
    table = table.sort_values(['log_inv_delta', 'rank']).drop(columns='rank')
    table.to_csv(os.path.join(a.dir, 'results_table.csv'), index=False)

    head = ['δ', 'log(1/δ)', 'K', 'Algorithm', 'Trajectories', 'Avg stopping time (± 95% CI)',
            'Avg wall-clock (ms/traj)', 'Accuracy (correct / total)', 'Error probability',
            'Never reached a_stop']
    lines = [f"Fixed instance μ = {np.round(env['means'], 4).tolist()} (best arm {env['best_arm']}), "
             f"env_seed {env['env_seed']}; trajectory j uses seed j for every algorithm and δ.", '',
             '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for _, r in table.iterrows():
        x = r.log_inv_delta
        dlt = 'e^0 = 1' if x == 0 else f'e^-{x:g} = {r.delta:.3g}'
        if r.algorithm == LB:
            lines.append('| ' + ' | '.join([dlt, f'{x:g}', str(K), LB, '—', f'{r.avg_stopping_time:.2f}',
                                            '—', '—', '—', '—']) + ' |')
            continue
        err = f'{r.error_probability:.4f}' if r.error_probability > 0 else f'0 (< {3 / r.trajectories:.4f})'
        lines.append('| ' + ' | '.join([
            dlt, f'{x:g}', str(K), r.algorithm,
            str(int(r.trajectories)), f'{r.avg_stopping_time:.2f} ± {r.stopping_time_ci95:.2f}',
            f'{r.avg_wall_clock_ms:.2f}', f'{r.accuracy:.4f} ({int(r.correct)} / {int(r.trajectories)})', err,
            str(int(r.never_reached_a_stop))]) + ' |')
    with open(os.path.join(a.dir, 'results_table.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"\nSaved {a.dir}/results_table.csv and {a.dir}/results_table.md")


if __name__ == '__main__':
    main()
