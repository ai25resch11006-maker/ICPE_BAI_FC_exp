"""
Summary of the certified-training runs (NEW CODE, not from the ICPE repository).

For every run runs_certify/dp<delta'>/ it collects
  training  : certified or not, T (test epochs), stopping step, wall-clock time of the run (from the launcher
              logs) and seconds per training step, average stopping time and accuracy p_hat of the training
              episodes over the last 500 steps, final cost c = 1/lambda, average stopping time of the last test epoch
  inference : the ICPE row of runs_certify/fixed_instance/summary.csv at delta = delta' (fixed instance, 500
              seeded trajectories): error, average and maximum stopping time, wall-clock ms per trajectory,
              and TaS's average stopping time at the same delta for comparison
and saves runs_certify/certification_summary.csv / .md / .png (not certified in red).

Usage: python make_certify_summary.py [--root runs_certify]
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_times(root: str) -> dict:
    """delta' -> (start, end) of its longest START..END pair in the launcher logs (skips failed attempts)."""
    events = {}
    pat = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (START|END) delta'=([0-9.]+)")
    for log in glob.glob(os.path.join(root, 'launcher*.log')):
        for line in open(log, errors='ignore'):
            m = pat.match(line)
            if m:
                events.setdefault(m.group(3), []).append((dt.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S'), m.group(2)))
    out = {}
    for dp, ev in events.items():
        ev.sort()
        best = None
        for i, (t0, kind) in enumerate(ev):
            if kind != 'START':
                continue
            end = next((t1 for t1, k in ev[i + 1:] if k == 'END'), None)
            if end and (best is None or end - t0 > best[1] - best[0]):
                best = (t0, end)
        if best:
            out[dp] = best
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='runs_certify')
    a = p.parse_args()
    times = run_times(a.root)
    fi_path = os.path.join(a.root, 'fixed_instance', 'summary.csv')
    fi = pd.read_csv(fi_path) if os.path.exists(fi_path) else None

    rows = []
    for run in sorted(glob.glob(os.path.join(a.root, 'dp*')), key=lambda d: -float(d.split('dp')[-1])):
        dp = run.split('dp')[-1]
        js = glob.glob(os.path.join(run, 'models', 'sequential', '*', 'certificate_1.json'))
        if not js:
            continue
        c = json.load(open(js[0]))
        mdir = os.path.dirname(js[0])
        tr = pd.read_csv(os.path.join(mdir, 'training_log_1.csv'))
        last = tr.tail(500)
        horizon = int(mdir.rstrip('/').split('_')[-2])
        steps = int(tr.training_step.max())
        t0t1 = times.get(dp)
        hours = (t0t1[1] - t0t1[0]).total_seconds() / 3600 if t0t1 else float('nan')
        row = {'delta_prime': float(dp), 'delta_train': round(c['delta_train'], 6), 'N': horizon,
               'certified': c['certified'], 'T_test_epochs': c['T'], 'stopping_step': c['stopping_training_step'],
               'training_hours': hours, 'sec_per_training_step': hours * 3600 / steps if steps else float('nan'),
               'train_avg_tau_last500': last.avg_tau.mean(), 'train_p_hat_last500': last.p_hat_step.mean(),
               'final_cost_c': tr.cost_c.iloc[-1], 'test_avg_tau_last_epoch': c.get('test_avg_tau_last'),
               'running_average': c['running_average'], 'boundary': c['boundary']}
        if fi is not None:
            sel = fi[(fi.delta - float(dp)).abs() < 1e-6]
            icpe, tas = sel[sel.algo == 'ICPE'], sel[sel.algo == 'TaS']
            if len(icpe):
                r = icpe.iloc[0]
                row.update({'inf_icpe_error': r.error_prob, 'inf_icpe_avg_tau': r.avg_tau, 'inf_icpe_max_tau': int(r.max_tau),
                            'inf_icpe_ms_per_traj': r.avg_wall_ms, 'inf_tas_avg_tau': tas.avg_tau.iloc[0] if len(tas) else float('nan')})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.root, 'certification_summary.csv'), index=False)

    f = lambda v, fmt: '—' if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, fmt)
    head = ["δ′ (train δ)", 'N', 'Certified', 'T / stop step', 'Training\nwall-clock', 's / training\nstep',
            'Training avg τ\n(last 500 steps)', 'Training\naccuracy p̂', 'Final c', 'Test avg τ\n(last epoch)',
            'Inference: ICPE\nerror', 'Inference: ICPE\navg τ (max)', 'Inference: ICPE\nms / trajectory', 'Inference: TaS\navg τ']
    cells, colors = [], []
    for _, r in df.iterrows():
        cert = bool(r.certified)
        cells.append([f"{r.delta_prime:g} ({r.delta_train:g})", str(int(r.N)), 'yes' if cert else 'no',
                      f"{int(r.T_test_epochs)} / {int(r.stopping_step):,}" if cert else f"— / {int(r.stopping_step):,}",
                      f(r.training_hours, '.1f') + ' h', f(r.sec_per_training_step, '.1f'),
                      f(r.train_avg_tau_last500, '.1f'), f(r.train_p_hat_last500, '.4f'), f(r.final_cost_c, '.4f'),
                      f(r.test_avg_tau_last_epoch, '.1f'),
                      f(r.get('inf_icpe_error', float('nan')), '.4f'),
                      (f(r.get('inf_icpe_avg_tau', float('nan')), '.2f') +
                       (f" ({int(r.inf_icpe_max_tau)})" if 'inf_icpe_max_tau' in r and not pd.isna(r.inf_icpe_max_tau) else '')),
                      f(r.get('inf_icpe_ms_per_traj', float('nan')), '.1f'), f(r.get('inf_tas_avg_tau', float('nan')), '.2f')])
        colors.append(['white'] * len(head))
        if not cert:
            colors[-1][2] = '#f8d0d0'
    lines = ['| ' + ' | '.join(h.replace('\n', ' ') for h in head) + ' |', '|' + '---|' * len(head)]
    lines += ['| ' + ' | '.join(c) + ' |' for c in cells]
    with open(os.path.join(a.root, 'certification_summary.md'), 'w') as fh:
        fh.write('Certified training (Prop. B.12, eta = 0.1, K = 1000 test rollouts every 10 steps from step 10,000) '
                 'and inference on the fixed instance (500 seeded trajectories).\n\n' + '\n'.join(lines) + '\n')

    fig, ax = plt.subplots(figsize=(17, 0.5 * (len(cells) + 2) + 0.6))
    ax.axis('off')
    t = ax.table(cellText=cells, colLabels=head, cellColours=colors, colColours=['#dfe7f1'] * len(head),
                 cellLoc='center', loc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 2.0)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor('#b8c2cc')
        if i == 0:
            cell.set_text_props(weight='bold')
            cell.set_height(cell.get_height() * 1.4)
        elif j == 2 and cells[i - 1][2] == 'no':
            cell.set_text_props(color='#b00000', weight='bold')
    ax.set_title('Certified training (Prop. B.12) and inference on the fixed instance', fontsize=10, pad=8)
    fig.text(0.5, 0.02, 'training: wall-clock of the whole run incl. test epochs; avg τ / p̂ over the last 500 training steps '
             '(100 episodes each).   inference: fixed instance, 500 trajectories, seeds 1-500.', ha='center', fontsize=7.5, color='#444')
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(a.root, 'certification_summary.png'), dpi=170, bbox_inches='tight')
    print('\n'.join(lines))
    print(f"\nSaved {a.root}/certification_summary.csv, .md and .png")


if __name__ == '__main__':
    main()
