"""
Matched-baseline screening: MLP / GRU / LSTM controllers trained in the same environment as the
RT4 SNN teachers, cycling on the evaluation + selection patients, against the SNN teachers, constant
stimulation and the oracle. References come from eval_round4 (seeds 42..1024) and select_round4
(2048..9999). Prints one row per controller with the per-patient PD/healthy charge ratio, writes
<eval-dir>/summary.json and <eval-dir>/fig_baselines.png (diagnostic figure).

  python scripts/analyze_rev_baselines.py [--eval-dir data/revision/eval_baselines_round4]
      [--extra-dir data/revision/eval_round4b]   # replicate SNN teachers when screened
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

from analyze_rev_select import per_seed, C, INK, INK2  # noqa: E402

ARCH_COL = {'RT4': C['orange'], 'mlp_mean80': C['blue'], 'mlp_mean16': C['aqua'], 'gru': C['yellow'], 'lstm': C['violet']}


def load(d, into=None):
    R = {} if into is None else into
    for p in glob.glob(os.path.join(d, 'cycling_*_w100_seed*.json')):
        m = re.match(r'cycling_(.+)_w100_seed(\d+)\.json', os.path.basename(p))
        R.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(p))
    return R


def arch_of(lab):
    for k in ('mlp_mean80', 'mlp_mean16', 'gru', 'lstm', 'RT4'):
        if lab.startswith(k):
            return k
    return 'ref'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', default='data/revision/eval_baselines_round4')
    ap.add_argument('--ref-dirs', nargs='+', default=['data/revision/eval_round4', 'data/revision/select_round4'])
    ap.add_argument('--extra-dir', default=None)
    args = ap.parse_args()
    ref = {}
    for d in args.ref_dirs:
        load(d, ref)
    U, Cd = ref['unstim'], ref['cdbs']
    B = load(args.eval_dir)
    if args.extra_dir and os.path.isdir(args.extra_dir):
        load(args.extra_dir, B)
    labs = [k for k in ('RT4_p101', 'RT4_p202', 'RT4_p303') if k in ref] + sorted(B, key=lambda k: (arch_of(k), k)) + \
           [k for k in ('constant_180_0.06_212', 'oracle_180_0.06_250') if k in ref]
    out = {}
    print(f'{"controller":30s} {"n":>2s} {"PDbeta":>7s} {"PDred%":>7s} {"charge%":>8s} {"ratio_med":>9s} {"Hoff%":>6s}  per-patient ratio')
    for lab in labs:
        runs = B.get(lab) or ref.get(lab)
        rs = {s: per_seed(runs[s], U[s], Cd[s]) for s in sorted(runs) if s in U}
        a = {k: float(np.mean([r[k] for r in rs.values()])) for k in next(iter(rs.values()))}
        a['ratio_median'] = float(np.median([r['ratio'] for r in rs.values()])); a['n'] = len(rs); a['seeds'] = rs
        out[lab] = a
        print(f'{lab:30s} {len(rs):2d} {a["pd_beta"]:7.0f} {a["pd_red"]:7.1f} {a["charge_pct"]:8.1f} {a["ratio_median"]:9.1f} {a["h_off"]:6.0f}  '
              + ' '.join(f'{r["ratio"]:.0f}' for r in rs.values()))
    json.dump(out, open(os.path.join(args.eval_dir, 'summary.json'), 'w'), indent=1)

    show = [l for l in labs if not l.startswith('oracle')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(show))
    for ax, key, ylab in zip(axes, ('pd_beta', 'charge_pct', 'ratio'), ('PD-block β', 'total charge, % cDBS', 'PD/healthy charge ratio')):
        for j, lab in enumerate(show):
            vals = [r[key] for r in out[lab]['seeds'].values()]
            col = ARCH_COL.get(arch_of(lab), INK2)
            ax.plot(np.full(len(vals), j) + np.linspace(-0.2, 0.2, len(vals)), vals, 'o', ms=4, color=col, mfc='none')
            ax.plot([j - 0.3, j + 0.3], [np.median(vals)] * 2, color=col, lw=1.8)
        if key == 'pd_beta':
            hb = np.mean([v['h_beta_unstim'] for v in out.values()])
            ax.axhline(hb, color=C['blue'], lw=0.8, ls='--'); ax.text(0.01, hb, 'healthy unstim β', fontsize=7, color=C['blue'], va='bottom', transform=ax.get_yaxis_transform())
        if key == 'ratio':
            ax.set_yscale('log'); ax.axhline(1, color=INK, lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([l.replace('constant_180_0.06_', 'const ') for l in show], rotation=60, ha='right', fontsize=7)
        ax.set_ylabel(ylab, fontsize=8)
    fig.suptitle('Matched baselines vs SNN teachers in cycling (one circle per patient, bar = median; colour = architecture)', fontsize=9, color=INK)
    fig.savefig(os.path.join(args.eval_dir, 'fig_baselines.png'), bbox_inches='tight')
    print('wrote', os.path.join(args.eval_dir, 'summary.json'), 'and fig_baselines.png')


if __name__ == '__main__':
    main()
