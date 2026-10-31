"""
Results table as an image (NEW CODE, not from the ICPE repository).

Reads <dir>/summary.csv and <dir>/environment.json (written by eval_fixed_instance.py) and saves
<dir>/results_table.png: one row per (delta, algorithm) with the allowed error delta, empirical error
(wrong / trajectories), maximum and average stopping time, average wall-clock time per trajectory, and the
number of trajectories that never reached a_stop.
  red    : empirical error > delta  (not delta-correct on this instance)
  orange : trajectories that never reached a_stop (stopped by the horizon N instead)

Usage: python make_results_png.py [--dir runs_delta_sweep/fixed_instance]
"""
import argparse
import json
import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER = ['ICPE', 'TaS', 'TTPS', 'Uniform']
RED, ORANGE, GREY = '#f8d0d0', '#fde3c8', '#f2f2f2'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='runs_delta_sweep/fixed_instance')
    a = p.parse_args()
    df = pd.read_csv(os.path.join(a.dir, 'summary.csv'))
    with open(os.path.join(a.dir, 'environment.json')) as f:
        env = json.load(f)
    df['rank'] = df.algo.map({x: i for i, x in enumerate(ORDER)})
    df = df.sort_values(['log_inv_delta', 'rank'])

    head = ['δ', 'Algorithm', 'Empirical error\n(wrong / total)', 'Max stopping\ntime', 'Avg stopping\ntime (± 95% CI)',
            'Avg wall-clock\n(ms / trajectory)', 'Never reached\na_stop']
    rows, colors = [], []
    for _, r in df.iterrows():
        n = int(r.num_traj)
        wrong = int(round(r.error_prob * n))
        x = r.log_inv_delta
        dlt = '1' if x == 0 else f'e^-{x:g} = {math.exp(-x):.3g}'
        err = f'{wrong} / {n} = {r.error_prob:.4f}'
        rows.append([dlt, r.algo, err, f'{int(r.max_tau)}', f'{r.avg_tau:.2f} ± {r.tau_ci95:.2f}',
                     f'{r.avg_wall_ms:.1f}', f'{int(r.not_stopped)}'])
        base = 'white' if int(x) % 4 == 2 else GREY          # alternate shading per delta
        c = [base] * len(head)
        if r.error_prob > r.delta:
            c[2] = RED
        if r.not_stopped > 0:
            c[6] = ORANGE
        colors.append(c)

    fig_h = 0.30 * (len(rows) + 2)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.axis('off')
    t = ax.table(cellText=rows, colLabels=head, cellColours=colors, colColours=['#dfe7f1'] * len(head),
                 cellLoc='center', loc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.45)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor('#b8c2cc')
        if i == 0:
            cell.set_text_props(weight='bold')
            cell.set_height(cell.get_height() * 1.6)
        elif j == 2 and colors[i - 1][2] == RED:
            cell.set_text_props(color='#b00000', weight='bold')
        elif j == 6 and colors[i - 1][6] == ORANGE:
            cell.set_text_props(color='#b35900', weight='bold')
        if i > 0 and j == 1 and rows[i - 1][1] == 'ICPE':
            cell.set_text_props(weight='bold')
    mu = np.round(env['means'], 3).tolist()
    ax.set_title(f'Fixed instance μ = {mu} (K = {env["num_arms"]}), best arm {env["best_arm"]}; '
                 f'trajectory j uses seed j for every algorithm and δ', fontsize=10, pad=14)
    fig.text(0.5, 0.012, 'red: empirical error > δ (not δ-correct)     orange: trajectories that hit the horizon '
             'without a_stop',
             ha='center', fontsize=8, color='#444')
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    out = os.path.join(a.dir, 'results_table.png')
    fig.savefig(out, dpi=170, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
