"""
Final tables: every condition on the same seeded patients.

cycling  data/revision/final_cycling  (5 x 60 s, 10 patients): PD-block beta and reduction vs
         unstimulated, healthy-block beta, total charge % cDBS, PD/healthy charge ratio, healthy
         windows off, each as mean [min-max] over patients.
acute    data/revision/final_acute (40 s PD, 10 patients): beta reduction vs unstimulated, charge
         % cDBS, final setting.

Writes <dir>/summary.json for both and $CLDBS_FIG_DIR/diagnostics/fig_cycling_summary.png (diagnostic figure;
default figures/revision).

  python scripts/analyze_rev_final.py
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG_DIR = os.environ.get('CLDBS_FIG_DIR', os.path.join(ROOT, 'figures', 'revision'))
from analyze_rev_select import per_seed, C, INK, INK2  # noqa: E402

GROUPS = [('SNN teacher (RT4)', r'^RT4_p\d+$'), ('SNN teacher replicates', r'^RT4_p\d+_s\d$'), ('SNN delta 0', r'^RT4d0'),
          ('Xylo-native student fp32', r'^xn_'), ('student on XyloSim', r'^xylosim_'),
          ('MLP mean80', r'^mlp_mean80'), ('MLP mean16', r'^mlp_mean16'), ('GRU', r'^gru_'), ('LSTM', r'^lstm_'),
          ('published teacher / ANN / GRU', r'^published'), ('constant', r'^constant'), ('cDBS', r'^cdbs$'),
          ('dual threshold', r'^dual'), ('integral', r'^integral'), ('oracle', r'^oracle')]


def load(d, proto):
    R = {}
    for p in glob.glob(os.path.join(d, f'{proto}_*_w100_seed*.json')):
        m = re.match(rf'{proto}_(.+)_w100_seed(\d+)\.json', os.path.basename(p))
        R.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(p))
    return R


def group_of(lab):
    for g, pat in GROUPS:
        if re.search(pat, lab):
            return g
    return 'other'


def fmt(v, w=5, d=0):
    return f'{np.mean(v):{w}.{d}f} [{np.min(v):.{d}f}-{np.max(v):.{d}f}]'


def cycling(d):
    R = load(d, 'cycling'); U, Cd = R['unstim'], R['cdbs']
    out = {}
    order = sorted(R, key=lambda k: ([g for g, _ in GROUPS].index(group_of(k)) if group_of(k) != 'other' else 99, k))
    print(f'\n== CYCLING (5 x 60 s), mean [min-max] over patients\n{"condition":30s} {"n":>2s} {"PD beta":>16s} {"PD red %":>16s} {"H beta":>16s} {"charge %cDBS":>18s} {"PD/H ratio":>18s} {"H off %":>14s}')
    for lab in order:
        if lab == 'unstim':
            continue
        rs = [per_seed(R[lab][s], U[s], Cd[s]) for s in sorted(R[lab]) if s in U]
        v = {k: [r[k] for r in rs] for k in rs[0]}
        out[lab] = {'group': group_of(lab), 'n': len(rs), **{k: float(np.mean(x)) for k, x in v.items()},
                    'ratio_median': float(np.median(v['ratio'])), 'per_seed': {s: r for s, r in zip(sorted(R[lab]), rs)}}
        ratio = np.minimum(v['ratio'], 9999)
        print(f'{lab:30s} {len(rs):2d} {fmt(v["pd_beta"]):>16s} {fmt(v["pd_red"], d=1):>16s} {fmt(v["h_beta"]):>16s} {fmt(v["charge_pct"], d=1):>18s} '
              f'{np.median(ratio):7.0f} [{np.min(ratio):.0f}-{np.max(ratio):.0f}]'.ljust(19) + f' {fmt(v["h_off"]):>14s}')
    json.dump(out, open(os.path.join(d, 'summary.json'), 'w'), indent=1)
    return out


def acute(d):
    R = load(d, 'acute'); U, Cd = R['unstim'], R['cdbs']
    out = {}
    order = sorted(R, key=lambda k: ([g for g, _ in GROUPS].index(group_of(k)) if group_of(k) != 'other' else 99, k))
    print(f'\n== ACUTE (40 s PD), mean [min-max] over patients\n{"condition":30s} {"n":>2s} {"beta":>16s} {"reduction %":>18s} {"charge %cDBS":>18s} {"final f/pw/amp":>18s}')
    for lab in order:
        if lab == 'unstim':
            continue
        red, chg, beta, fin = [], [], [], []
        for s in sorted(R[lab]):
            if s not in U:
                continue
            dd = R[lab][s]; b = np.asarray(dd['lfp']); u = np.asarray(U[s]['lfp'])
            q = np.asarray(dd['freq']) * np.asarray(dd['pw']) * np.asarray(dd['amp']); qc = np.asarray(Cd[s]['freq']) * np.asarray(Cd[s]['pw']) * np.asarray(Cd[s]['amp'])
            red.append(100 * (1 - b.mean() / u.mean())); chg.append(100 * q.sum() / qc.sum()); beta.append(b.mean())
            fin.append((dd['freq'][-1], dd['pw'][-1], dd['amp'][-1]))
        fin = np.array(fin)
        out[lab] = {'group': group_of(lab), 'n': len(red), 'beta': float(np.mean(beta)), 'reduction': float(np.mean(red)), 'reduction_sd': float(np.std(red)),
                    'charge_pct': float(np.mean(chg)), 'per_seed': {'reduction': red, 'charge_pct': chg}}
        print(f'{lab:30s} {len(red):2d} {fmt(beta):>16s} {fmt(red, d=1):>18s} {fmt(chg, d=1):>18s} {fin[:, 0].mean():6.0f}/{fin[:, 1].mean():.2f}/{fin[:, 2].mean():.0f}')
    json.dump(out, open(os.path.join(d, 'summary.json'), 'w'), indent=1)
    return out


def figure(cy, path):
    labs = [k for k in cy if cy[k]['group'] not in ('oracle', 'SNN teacher replicates', 'SNN delta 0') and not k.startswith('constant_130')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(labs))
    gcol = {'SNN teacher (RT4)': C['orange'], 'Xylo-native student fp32': C['orange'], 'student on XyloSim': C['violet'], 'MLP mean80': C['blue'],
            'MLP mean16': C['aqua'], 'GRU': C['yellow'], 'LSTM': C['violet'], 'published teacher / ANN / GRU': INK2}
    for ax, key, ylab in zip(axes, ('pd_beta', 'charge_pct', 'ratio'), ('PD-block β', 'total charge, % cDBS', 'PD/healthy charge ratio')):
        for j, lab in enumerate(labs):
            vals = [r[key] for r in cy[lab]['per_seed'].values()]
            col = gcol.get(cy[lab]['group'], INK2)
            ax.plot(np.full(len(vals), j) + np.linspace(-0.2, 0.2, len(vals)), vals, 'o', ms=3.5, color=col, mfc='none')
            ax.plot([j - 0.3, j + 0.3], [np.median(vals)] * 2, color=col, lw=1.8)
        if key == 'pd_beta':
            hb = np.mean([v['h_beta_unstim'] for v in cy.values()]); ax.axhline(hb, color=C['blue'], lw=0.8, ls='--')
            ax.text(0.01, hb, 'healthy unstim β', fontsize=7, color=C['blue'], va='bottom', transform=ax.get_yaxis_transform())
        if key == 'ratio':
            ax.set_yscale('log'); ax.axhline(1, color=INK, lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([l.replace('constant_180_0.06_', 'const ').replace('_180_0.06_250', '') for l in labs], rotation=70, ha='right', fontsize=6.5)
        ax.set_ylabel(ylab, fontsize=8)
    fig.suptitle('Final cycling, 10 patients: one circle per patient, bar = median', fontsize=9, color=INK)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight'); print('wrote', path)


if __name__ == '__main__':
    cy = cycling(os.path.join(ROOT, 'data/revision/final_cycling'))
    acute(os.path.join(ROOT, 'data/revision/final_acute'))
    figure(cy, os.path.join(FIG_DIR, 'diagnostics', 'fig_cycling_summary.png'))
