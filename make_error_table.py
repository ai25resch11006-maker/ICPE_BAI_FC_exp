"""
Error counts for the fixed-instance delta sweep (NEW CODE, not from the ICPE repository).

Reads the per-trajectory files <dir>/log<x>/<algo>.csv written by eval_fixed_instance.py and saves
  <dir>/error_table.csv / .md        per delta and algorithm: allowed error delta, number of trajectories
                                     with a wrong final guess, empirical error probability
  <dir>/error_trajectories.csv       one row per wrong trajectory: delta, algorithm, seed, tau, guess

Usage: python make_error_table.py [--dir runs_delta_sweep/fixed_instance]
"""
import argparse
import glob
import json
import math
import os
import pandas as pd

ORDER = ['ICPE', 'TaS', 'TTPS', 'Uniform']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='runs_delta_sweep/fixed_instance')
    a = p.parse_args()
    with open(os.path.join(a.dir, 'environment.json')) as f:
        best = json.load(f)['best_arm']

    rows, wrong = [], []
    for d in sorted(glob.glob(os.path.join(a.dir, 'log*')), key=lambda s: float(s.split('log')[-1])):
        x = float(os.path.basename(d)[3:])
        for algo in ORDER:
            path = os.path.join(d, f'{algo}.csv')
            if not os.path.exists(path):
                continue
            t = pd.read_csv(path)
            err = t[t.correct == 0]
            rows.append({'log_inv_delta': x, 'allowed_error_delta': math.exp(-x), 'algorithm': algo,
                         'trajectories': len(t), 'errors': len(err), 'error_probability': len(err) / len(t)})
            for _, r in err.iterrows():
                wrong.append({'log_inv_delta': x, 'delta': math.exp(-x), 'algorithm': algo, 'seed': int(r.seed),
                              'tau': int(r.tau), 'guess': int(r.guess), 'best_arm': best})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(a.dir, 'error_table.csv'), index=False)
    pd.DataFrame(wrong, columns=['log_inv_delta', 'delta', 'algorithm', 'seed', 'tau', 'guess', 'best_arm']) \
        .to_csv(os.path.join(a.dir, 'error_trajectories.csv'), index=False)

    algos = [x for x in ORDER if x in set(table.algorithm)]
    head = ['δ', 'Allowed error δ'] + algos
    lines = [f'Wrong trajectories (final guess ≠ best arm {best}) out of the trajectories run, per δ and algorithm.', '',
             '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for x, g in table.groupby('log_inv_delta'):
        cells = []
        for algo in algos:
            r = g[g.algorithm == algo]
            cells.append('—' if r.empty else
                         f'{int(r.errors.iloc[0])} / {int(r.trajectories.iloc[0])} = {r.error_probability.iloc[0]:.4f}')
        lines.append('| ' + ' | '.join(['e^0 = 1' if x == 0 else f'e^-{x:g}', f'{math.exp(-x):.3g}'] + cells) + ' |')
    with open(os.path.join(a.dir, 'error_table.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nSaved {a.dir}/error_table.csv, error_table.md and error_trajectories.csv ({len(wrong)} wrong trajectories)')


if __name__ == '__main__':
    main()
