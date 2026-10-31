"""
Diagnostic figure: the beta-charge front of the simple controllers (smoothed dual threshold and
integral control, data/revision/simple_sweep) against the learned policies (final_cycling), on
the five evaluation patients, cycling 5 x 60 s. Left: PD-block beta vs total charge; right: PD-block
beta vs PD/healthy charge ratio (state selectivity).

  python scripts/plot_rev_front.py
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG_DIR = os.environ.get('CLDBS_FIG_DIR', os.path.join(ROOT, 'figures', 'revision'))
from analyze_rev_select import per_seed, C, INK, INK2  # noqa: E402

SEEDS = [42, 123, 456, 789, 1024]


def load(d):
    R = {}
    for p in glob.glob(os.path.join(d, 'cycling_*_w100*_seed*.json')):
        m = re.match(r'cycling_(.+)_w100((?:_sm10)?)_seed(\d+)\.json', os.path.basename(p))
        if m and int(m.group(3)) in SEEDS:
            R.setdefault(m.group(1) + m.group(2), {})[int(m.group(3))] = json.load(open(p))
    return R


def summarise(R, U, Cd):
    out = {}
    for lab, runs in R.items():
        rs = [per_seed(runs[s], U[s], Cd[s]) for s in SEEDS if s in runs]
        if len(rs) >= 4 and lab not in ('unstim', 'cdbs'):
            out[lab] = {'pd_beta': np.mean([r['pd_beta'] for r in rs]), 'charge': np.mean([r['charge_pct'] for r in rs]),
                        'ratio': min(np.median([r['ratio'] for r in rs]), 1e4), 'h_off': np.mean([r['h_off'] for r in rs]), 'n': len(rs)}
    return out


def main():
    fc = load(os.path.join(ROOT, 'data/revision/final_cycling')); U, Cd = fc['unstim'], fc['cdbs']
    sw = load(os.path.join(ROOT, 'data/revision/simple_sweep'))
    F, S = summarise(fc, U, Cd), summarise(sw, U, Cd)
    hb = np.mean([per_seed(U[s], U[s], Cd[s])['h_beta_unstim'] for s in SEEDS])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    groups = [('dual threshold, smoothed, 212 µA setting', lambda k: k.startswith('dual') and k.endswith('_sm10') and '_212' in k, C['aqua'], 'o'),
              ('dual threshold, smoothed, cDBS setting', lambda k: k.startswith('dual') and k.endswith('_sm10') and re.match(r'dual_\d+_\d+_sm10$', k), C['aqua'], 's'),
              ('integral, smoothed', lambda k: k.startswith('integral') and k.endswith('_sm10'), C['yellow'], '^'),
              ('dual / integral on raw β', lambda k: (k.startswith('dual') or k.startswith('integral')) and not k.endswith('_sm10'), INK2, 'x')]
    learned = [('SNN teacher RT4_p202', 'RT4_p202', C['orange'], 'o'), ('student xn_s0.03_w500', 'xn_s0.03_w500', C['orange'], 'D'),
               ('student on XyloSim', 'xylosim_xn_s0.03_w500', C['violet'], 'D'), ('student xn_s0.03_w5000', 'xn_s0.03_w5000', C['orange'], 'v'),
               ('MLP p202', 'mlp_mean80_p202', C['blue'], 'o'), ('GRU p202', 'gru_raw80_p202', C['blue'], 's'),
               ('constant 212', 'constant_180_0.06_212', INK, 'P'), ('true-state switching 250', 'oracle_180_0.06_250', INK, '*')]
    for ax, key, xlab, log in ((axes[0], 'charge', 'total charge, % cDBS', False), (axes[1], 'ratio', 'PD/healthy charge ratio (median)', True)):
        for name, sel, col, mk in groups:
            pts = [(v[key], v['pd_beta']) for k, v in S.items() if sel(k)]
            if pts:
                ax.plot(*zip(*pts), mk, color=col, ms=6, mfc='none' if mk != 'x' else col, label=name)
        for name, lab, col, mk in learned:
            if lab in F:
                ax.plot(F[lab][key], F[lab]['pd_beta'], mk, color=col, ms=9 if mk == '*' else 7, label=name)
        ax.axhline(hb, color=C['blue'], lw=0.8, ls='--'); ax.text(0.01, hb, 'healthy unstim β', fontsize=7, color=C['blue'], va='bottom', transform=ax.get_yaxis_transform())
        ax.set_xlabel(xlab); ax.set_ylabel('PD-block β (mean over 5 patients)')
        if log:
            ax.set_xscale('log')
    axes[0].legend(fontsize=6.5, loc='upper right')
    fig.suptitle('β–charge front: simple controllers (sweeps) vs learned policies, cycling 5 × 60 s, patients 42–1024', fontsize=9, color=INK)
    path = os.path.join(FIG_DIR, 'diagnostics', 'fig_front.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight'); print('wrote', path)


if __name__ == '__main__':
    main()
