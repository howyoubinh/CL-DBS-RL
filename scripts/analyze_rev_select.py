"""
Teacher selection on five fresh patients (seeds 2048, 3141, 5678, 8192, 9999), not used for training
or for the earlier screening of the teachers.

Reads data/revision/select_round4/cycling_<controller>_w100_seed<N>.json (run_rev_eval.py cycling
output on the selection patients) and prints, per controller, PD-block beta reduction vs
unstimulated, PD-block beta, healthy unstimulated beta, total charge % cDBS, median PD/healthy
charge ratio and the share of healthy windows with stimulation off. Applies the pre-stated rule:

  pick the RT4 teacher with the highest median PD/healthy charge ratio among those whose
  PD-block beta is within 10% of the healthy unstimulated beta.

Writes <dir>/summary.json and <dir>/fig_selection.png (diagnostic figure).

  python scripts/analyze_rev_select.py [--dir data/revision/select_round4] [--prefix RT4_]
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'violet': '#4a3aa7'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 150})


def load(d):
    R = {}
    for p in glob.glob(os.path.join(d, 'cycling_*_w100_seed*.json')):
        m = re.match(r'cycling_(.+)_w100_seed(\d+)\.json', os.path.basename(p))
        R.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(p))
    return R


def per_seed(d, u, cd):
    b = np.asarray(d['lfp']); ub = np.asarray(u['lfp']); pd = np.asarray(d['pd_state']) == 1
    f, pw, a = (np.asarray(d[k]) for k in ('freq', 'pw', 'amp'))
    q = f * pw * a
    qc = np.asarray(cd['freq']) * np.asarray(cd['pw']) * np.asarray(cd['amp'])
    return dict(pd_red=100 * (1 - b[pd].mean() / ub[pd].mean()), pd_beta=float(b[pd].mean()),
                h_beta_unstim=float(ub[~pd].mean()), h_beta=float(b[~pd].mean()),
                charge_pct=100 * q.sum() / qc.sum(), pd_charge_pct=100 * q[pd].mean() / qc[pd].mean(),
                h_charge_pct=100 * q[~pd].mean() / qc[~pd].mean(),
                ratio=float(q[pd].mean() / max(q[~pd].mean(), 1e-9)),
                h_off=100 * float(((f == 0) | (a == 0))[~pd].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/revision/select_round4')
    ap.add_argument('--prefix', default='RT4_', help='controllers eligible for selection')
    ap.add_argument('--tol', type=float, default=0.10)
    args = ap.parse_args()
    R = load(args.dir)
    U, Cd = R['unstim'], R['cdbs']
    out = {}
    print(f'{"controller":26s} {"PDred%":>7s} {"PDbeta":>7s} {"Hbeta_u":>8s} {"charge%":>8s} {"PDchg%":>7s} {"Hchg%":>6s} {"ratio_med":>9s} {"Hoff%":>6s}')
    for lab in sorted(R):
        if lab == 'unstim':
            continue
        rows = [per_seed(R[lab][s], U[s], Cd[s]) for s in sorted(R[lab]) if s in U and s in Cd]
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        agg['ratio_median'] = float(np.median([r['ratio'] for r in rows]))
        agg['seeds'] = {s: r for s, r in zip(sorted(R[lab]), rows)}
        out[lab] = agg
        print(f'{lab:26s} {agg["pd_red"]:7.1f} {agg["pd_beta"]:7.0f} {agg["h_beta_unstim"]:8.0f} {agg["charge_pct"]:8.1f} '
              f'{agg["pd_charge_pct"]:7.1f} {agg["h_charge_pct"]:6.1f} {agg["ratio_median"]:9.1f} {agg["h_off"]:6.0f}')
    elig = {k: v for k, v in out.items() if k.startswith(args.prefix)}
    ok = {k: v for k, v in elig.items() if v['pd_beta'] <= v['h_beta_unstim'] * (1 + args.tol)}
    print(f'\nrule: highest median PD/healthy charge ratio among {args.prefix}* with PD-block beta <= healthy unstim beta x {1 + args.tol:.2f}')
    for k, v in elig.items():
        print(f'  {k}: PD beta {v["pd_beta"]:.0f} vs limit {v["h_beta_unstim"] * (1 + args.tol):.0f} -> '
              f'{"eligible" if k in ok else "NOT eligible"}; ratio median {v["ratio_median"]:.1f}')
    pool = ok or elig or out
    sel = max(pool, key=lambda k: pool[k]['ratio_median'] if not k.startswith('oracle') else -1)
    print(f'selected: {sel}' + ('' if ok else '  (no teacher met the beta criterion; best ratio among all)'))
    json.dump({'selected': sel, 'eligible': sorted(ok), 'rule': f'max median ratio, PD beta <= H unstim beta * {1 + args.tol}',
               'controllers': out}, open(os.path.join(args.dir, 'summary.json'), 'w'), indent=1)

    # diagnostic figure: per-seed PD beta (vs healthy unstim), charge, ratio for each controller
    labs = [k for k in out if k != 'cdbs']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(labs))
    for i, (key, ylab) in enumerate((('pd_beta', 'PD-block β (mean over cycling run)'), ('charge_pct', 'total charge, % cDBS'),
                                     ('ratio', 'PD/healthy charge ratio'))):
        ax = axes[i]
        for j, lab in enumerate(labs):
            vals = [r[key] for r in out[lab]['seeds'].values()]
            col = C['orange'] if lab.startswith(args.prefix) else INK2
            ax.plot(np.full(len(vals), j) + np.linspace(-0.15, 0.15, len(vals)), vals, 'o', ms=5, color=col, mfc='none' if lab != sel else col)
            ax.plot([j - 0.25, j + 0.25], [np.median(vals)] * 2, color=col, lw=1.6)
        if key == 'pd_beta':
            hb = np.mean([v['h_beta_unstim'] for v in out.values()])
            ax.axhline(hb, color=C['blue'], lw=1, ls='--'); ax.text(0.02, hb, 'healthy unstim β', color=C['blue'], fontsize=7, va='bottom', transform=ax.get_yaxis_transform())
            ax.axhline(hb * (1 + args.tol), color=C['blue'], lw=0.8, ls=':')
        if key == 'ratio':
            ax.set_yscale('log')
        ax.set_xticks(x); ax.set_xticklabels([l.replace('constant_180_0.06_', 'const ').replace('oracle_180_0.06_', 'oracle ') for l in labs], rotation=30, ha='right', fontsize=7)
        ax.set_ylabel(ylab, fontsize=8)
    fig.suptitle(f'Teacher selection on fresh patients {sorted(U)} (filled = selected: {sel}; bar = median)', fontsize=9, color=INK)
    fig.savefig(os.path.join(args.dir, 'fig_selection.png'), bbox_inches='tight')
    print('wrote', os.path.join(args.dir, 'summary.json'), 'and fig_selection.png')


if __name__ == '__main__':
    main()
