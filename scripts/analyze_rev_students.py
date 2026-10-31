"""
Distilled-student screening: fidelity vs sparsity vs closed-loop behaviour.

Reads models/revision/students_round4/<student>/config.json (rho, lam, held-out agreement, hidden
firing rate) and data/revision/eval_students_round4/cycling_<student>_w100_seed<N>.json, with the
teacher and references taken from --ref-dir (eval_round4). Prints one row per student and writes
<eval-dir>/summary.json and <eval-dir>/fig_students.png (diagnostic figure: student hidden rate vs
PD/healthy charge ratio, PD-block beta and total charge, teacher as the reference point).

  python scripts/analyze_rev_students.py [--students models/revision/students_round4]
      [--eval-dir data/revision/eval_students_round4] [--ref-dir data/revision/eval_round4]
      [--teacher RT4_p202]
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


def load_cycling(d, lab=None):
    R = {}
    for p in glob.glob(os.path.join(d, 'cycling_*_w100_seed*.json')):
        m = re.match(r'cycling_(.+)_w100_seed(\d+)\.json', os.path.basename(p))
        if lab is None or m.group(1) == lab:
            R.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(p))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--students', default='models/revision/students_round4')
    ap.add_argument('--eval-dir', default='data/revision/eval_students_round4')
    ap.add_argument('--ref-dir', default='data/revision/eval_round4')
    ap.add_argument('--teacher', default='RT4_p202')
    args = ap.parse_args()
    ref = load_cycling(args.ref_dir)
    U, Cd = ref['unstim'], ref['cdbs']
    S = load_cycling(args.eval_dir)
    rows = {}

    def summarise(runs):
        rs = [per_seed(runs[s], U[s], Cd[s]) for s in sorted(runs) if s in U]
        agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        agg['ratio_median'] = float(np.median([r['ratio'] for r in rs])); agg['n_seeds'] = len(rs)
        return agg

    for lab in [args.teacher, 'constant_180_0.06_212', 'oracle_180_0.06_250']:
        if lab in ref:
            rows[lab] = dict(summarise(ref[lab]), rho=None, lam=None, agree=None, rate=None)
    for p in sorted(glob.glob(os.path.join(args.students, '*/config.json'))):
        cfg = json.load(open(p)); name = cfg['run_name']
        row = dict(rho=cfg['rho'], lam=cfg['lam'], init=cfg.get('init_from_teacher', False),
                   agree=cfg['final']['val_agree_joint'], agree_heads=cfg['final']['val_agree_heads'],
                   rate=cfg['final']['hidden_rate'], teacher_rate=cfg.get('teacher_hidden_rate'))
        if name in S:
            row.update(summarise(S[name]))
        rows[name] = row
    print(f'{"student":22s} {"rho":>6s} {"lam":>6s} {"agree":>6s} {"rate":>6s} {"PDbeta":>7s} {"PDred%":>7s} {"charge%":>8s} {"ratio_med":>9s} {"Hoff%":>6s}')
    for k, v in rows.items():
        f = lambda x, w='7.1f': (f'{x:{w}}' if x is not None else ' ' * int(w.split('.')[0]))
        print(f'{k:22s} {f(v.get("rho"), "6.3f")} {f(v.get("lam"), "6.0f")} {f(v.get("agree"), "6.2f")} {f(v.get("rate"), "6.3f")} '
              f'{f(v.get("pd_beta"), "7.0f")} {f(v.get("pd_red"))} {f(v.get("charge_pct"), "8.1f")} {f(v.get("ratio_median"), "9.1f")} {f(v.get("h_off"), "6.0f")}')
    json.dump(rows, open(os.path.join(args.eval_dir, 'summary.json'), 'w'), indent=1)

    st = {k: v for k, v in rows.items() if v.get('rho') is not None and 'pd_beta' in v}
    if not st:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    lams = sorted({v['lam'] for v in st.values()})
    cols = dict(zip(lams, [C['blue'], C['aqua'], C['yellow'], C['orange'], C['violet']]))
    mk = {0.015: 'o', 0.03: 's', 0.05: '^'}
    t = rows.get(args.teacher, {})
    for ax, key, ylab in zip(axes, ('ratio_median', 'pd_beta', 'charge_pct'),
                             ('PD/healthy charge ratio (median over patients)', 'PD-block β', 'total charge, % cDBS')):
        for k, v in st.items():
            ax.plot(v['rate'], v[key], mk.get(v['rho'], 'o'), ms=7, color=cols[v['lam']], mfc='none' if v.get('init') else cols[v['lam']])
        if t:
            ax.axhline(t[key], color=INK, lw=0.8, ls='--'); ax.text(0.98, t[key], 'teacher', ha='right', va='bottom', fontsize=7, color=INK, transform=ax.get_yaxis_transform())
        if key == 'pd_beta':
            hb = np.mean([v['h_beta_unstim'] for v in st.values()]); ax.axhline(hb, color=C['blue'], lw=0.8, ls=':')
        if key == 'ratio_median':
            ax.set_yscale('log')
        ax.set_xlabel('student hidden firing rate (spikes / neuron / step)'); ax.set_ylabel(ylab, fontsize=8)
    for lam in lams:
        axes[0].plot([], [], 'o', color=cols[lam], label=f'λ = {lam:g}')
    for rho, m in mk.items():
        axes[0].plot([], [], m, color=INK2, mfc='none', label=f'ρ = {rho}')
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle('Distilled students: sparsity vs closed-loop behaviour (open = init from teacher; dashed = teacher)', fontsize=9, color=INK)
    fig.savefig(os.path.join(args.eval_dir, 'fig_students.png'), bbox_inches='tight')
    print('wrote', os.path.join(args.eval_dir, 'summary.json'), 'and fig_students.png')


if __name__ == '__main__':
    main()
