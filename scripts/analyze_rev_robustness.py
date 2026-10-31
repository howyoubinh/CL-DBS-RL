"""
Robustness and block-length tables for the fixed policies.

robustness  data/revision/final_robustness: cycling_<ctrl>_w100_<pert>_seed<N>.json, pert in
            pd{S} (disease severity), drop{P} (spike loss), ins{R} (added spikes), delay{K}
            (observation delay in windows). Each (controller, perturbation) is scored against the
            unstimulated run under the same perturbation (same seed) and cDBS from final_cycling.
blocks      data/revision/final_blocks: suffixes blk30 / blk120 / jit0.5.

Prints one block per perturbation; writes <dir>/summary.json and
$CLDBS_FIG_DIR/diagnostics/fig_robustness.png / fig_blocks.png (diagnostic figures; default figures/revision).

  python scripts/analyze_rev_robustness.py [robustness|blocks|all]
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

SHORT = {'RT4_p202': 'SNN teacher', 'xn_s0.03_w500': 'student fp32', 'xylosim_xn_s0.03_w500': 'student XyloSim',
         'mlp_mean80_p202': 'MLP', 'gru_raw80_p202': 'GRU', 'constant_180_0.06_212': 'constant 212',
         'dual_230_290': 'dual thr', 'integral_290_0.5_130_0.06': 'integral',
         'dual_250_300_180_0.06_212_sm10': 'dual thr (sm, 212)', 'integral_290_10_180_0.06_sm10': 'integral (sm)',
         'oracle_180_0.06_250': 'oracle', 'cdbs': 'cDBS', 'unstim': 'unstim'}
COL = {'SNN teacher': C['orange'], 'student fp32': C['orange'], 'student XyloSim': C['violet'], 'MLP': C['blue'], 'GRU': C['yellow'],
       'constant 212': INK2, 'dual thr': C['aqua'], 'integral': C['aqua'], 'dual thr (sm, 212)': C['aqua'], 'integral (sm)': C['yellow'], 'oracle': INK}


def load(d):
    """{(ctrl, pert): {seed: data}}; pert '' for the unperturbed runs."""
    R = {}
    for p in glob.glob(os.path.join(d, 'cycling_*_w100*_seed*.json')):
        m = re.match(r'cycling_(.+)_w100((?:_[a-z]+[0-9.]+)*)_seed(\d+)\.json', os.path.basename(p))
        ctrl, pert = m.group(1), m.group(2).lstrip('_')
        if '_sm10' in '_' + pert:            # controller smoothing is part of the controller, not the perturbation
            pert = '_'.join(x for x in pert.split('_') if x != 'sm10'); ctrl += '_sm10'
        R.setdefault((ctrl, pert), {})[int(m.group(3))] = json.load(open(p))
    return R


def table(d, title, fig_path):
    R = load(d)
    base = load(os.path.join(ROOT, 'data/revision/final_cycling'))
    perts = sorted({pt for (_, pt) in R if pt}, key=lambda x: (re.sub(r'[0-9.]+', '', x), float(re.sub(r'[^0-9.]', '', x) or 0)))
    ctrls = [c for c in SHORT if any((c, pt) in R for pt in perts)]
    out = {}
    for pt in perts:
        U = R.get(('unstim', pt)) or base[('unstim', '')]       # unstim under the same perturbation when it exists
        Cd = R.get(('cdbs', pt)) or base[('cdbs', '')]          # cDBS under the same protocol (block lengths change the run length)
        print(f'\n== {title}: {pt}\n{"controller":18s} {"n":>2s} {"PDbeta":>7s} {"PDred%":>7s} {"Hbeta":>6s} {"charge%":>8s} {"ratio_med":>9s} {"Hoff%":>6s}')
        for c in ctrls:
            if (c, pt) not in R or c == 'unstim':
                continue
            rs = [per_seed(R[(c, pt)][s], U[s], Cd[s]) for s in sorted(R[(c, pt)]) if s in U and s in Cd]
            if not rs:
                continue
            a = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}; a['ratio_median'] = float(np.median([r['ratio'] for r in rs])); a['n'] = len(rs)
            out.setdefault(pt, {})[c] = a
            print(f'{SHORT[c]:18s} {len(rs):2d} {a["pd_beta"]:7.0f} {a["pd_red"]:7.1f} {a["h_beta"]:6.0f} {a["charge_pct"]:8.1f} {min(a["ratio_median"], 9999):9.0f} {a["h_off"]:6.0f}')
    # unperturbed reference from final_cycling (same 5 patients)
    seeds5 = [42, 123, 456, 789, 1024]
    for c in ctrls:
        if (c, '') in base and c != 'unstim':
            rs = [per_seed(base[(c, '')][s], base[('unstim', '')][s], base[('cdbs', '')][s]) for s in seeds5 if s in base[(c, '')]]
            a = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}; a['ratio_median'] = float(np.median([r['ratio'] for r in rs])); a['n'] = len(rs)
            out.setdefault('none', {})[c] = a
    json.dump(out, open(os.path.join(d, 'summary.json'), 'w'), indent=1)

    fams = sorted({re.sub(r'[0-9.]+', '', pt) for pt in perts})
    fig, axes = plt.subplots(2, len(fams), figsize=(3.6 * len(fams), 6.5), squeeze=False)
    for j, fam in enumerate(fams):
        pts = ['none'] + [pt for pt in perts if re.sub(r'[0-9.]+', '', pt) == fam]
        xl = ['none'] + [re.sub(r'[^0-9.]', '', pt) for pt in pts[1:]]
        for c in ctrls:
            if c in ('unstim', 'cdbs'):
                continue
            ys = [[out.get(pt, {}).get(c, {}).get(k, np.nan) for pt in pts] for k in ('pd_red', 'ratio_median')]
            for i, y in enumerate(ys):
                axes[i, j].plot(range(len(pts)), np.minimum(y, 9999), 'o-', ms=4, lw=1.4, color=COL.get(SHORT[c], INK2), label=SHORT[c] if (i, j) == (0, 0) else None)
        axes[0, j].set_title({'pd': 'disease severity (× PD)', 'drop': 'spike loss (fraction)', 'ins': 'added spikes (Hz / neuron)', 'delay': 'observation delay (windows)', 'blk': 'block length (s)', 'jit': 'block jitter'}.get(fam, fam), fontsize=8)
        for i in range(2):
            axes[i, j].set_xticks(range(len(pts))); axes[i, j].set_xticklabels(xl, fontsize=7)
        axes[1, j].set_yscale('log')
    axes[0, 0].set_ylabel('PD β reduction, %'); axes[1, 0].set_ylabel('PD/healthy charge ratio (median)')
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle(title, fontsize=9, color=INK)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True); fig.savefig(fig_path, bbox_inches='tight'); print('wrote', fig_path)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('robustness', 'all'):
        table(os.path.join(ROOT, 'data/revision/final_robustness'), 'Robustness of fixed policies (cycling, 5 patients)', os.path.join(FIG_DIR, 'diagnostics', 'fig_robustness.png'))
    if what in ('blocks', 'all'):
        table(os.path.join(ROOT, 'data/revision/final_blocks'), 'Block lengths (cycling, 5 patients)', os.path.join(FIG_DIR, 'diagnostics', 'fig_blocks.png'))
