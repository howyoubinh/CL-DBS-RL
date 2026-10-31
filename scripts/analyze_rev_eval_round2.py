"""
Screen the second set of candidate SNN teachers, arms A / B / C (scripts/rev_manifest.yaml entries
teachers_round2 and eval_round2; outputs of scripts/run_rev_eval.py).

Every comparison is within a window length (beta per window is on a different scale at 100 ms
and 1 s) and paired by evaluation patient (seed).

  acute    PD only, 40 s. beta reduction vs unstimulated (same patient, same window) and charge
           as a percentage of cDBS; paired test of each controller against constant stimulation.
  cycling  5 x 60 s blocks alternating healthy/PD. beta reduction in PD blocks vs unstimulated
           PD blocks, charge vs cDBS, and adaptation = charge rate in PD blocks / healthy blocks
           (1.0 means the controller does not respond to the disease state).

  python scripts/analyze_rev_eval_round2.py [data/revision/eval_round2]
"""

import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

DIR = sys.argv[1] if len(sys.argv) > 1 else 'data/revision/eval_round2'
REFS = ('unstim', 'cdbs', 'constant_180_0.06_212', 'constant_180_0.06_250')


def load(protocol):
    out = {}
    for p in glob.glob(os.path.join(DIR, f'{protocol}_*_seed*.json')):
        m = re.match(rf'{protocol}_(.+)_w(\d+)_seed(\d+)\.json', os.path.basename(p))
        if m:
            out.setdefault((int(m.group(2)), m.group(1)), {})[int(m.group(3))] = json.load(open(p))
    return out


def rate(d, mask=None):
    """Charge per second (freq * amp * pw summed over windows / duration)."""
    f, a, w = (np.asarray(d[k], float) for k in ('freq', 'amp', 'pw'))
    q = f * a * w * d['window_ms'] / 1000.0
    if mask is None:
        return q.sum() / (len(q) * d['window_ms'] / 1000.0)
    return q[mask].sum() / (mask.sum() * d['window_ms'] / 1000.0)


def mean_sem(x):
    x = np.asarray(x, float)
    return f'{x.mean():6.1f} ± {stats.sem(x):4.1f}' if len(x) > 1 else f'{x.mean():6.1f}'


def arm(label):
    return label.split('_p')[0] if re.match(r'[ABC]_p\d+$', label) else label


summary = {}
for protocol in ('acute', 'cycling'):
    R = load(protocol)
    summary[protocol] = {}
    for win in sorted({w for w, _ in R}):
        labels = sorted(l for w, l in R if w == win)
        if 'unstim' not in labels or 'cdbs' not in labels:
            continue
        U, C = R[(win, 'unstim')], R[(win, 'cdbs')]
        print(f'\n=== {protocol.upper()}  window {win} ms ===')
        if protocol == 'acute':
            print(f'  {"controller":24s} {"n":>2s}  {"beta red. %":>13s}  {"charge % cDBS":>13s}  vs const250 (pts, p)')
        else:
            print(f'  {"controller":24s} {"n":>2s}  {"PD beta red. %":>14s}  {"charge % cDBS":>13s}  {"PD/healthy rate":>15s}')
        rows = {}
        for lab in [l for l in labels if l not in ('unstim',)]:
            D = R[(win, lab)]
            seeds = sorted(set(D) & set(U) & set(C))
            if protocol == 'acute':
                red = [100 * (1 - np.mean(D[s]['lfp']) / np.mean(U[s]['lfp'])) for s in seeds]
                chg = [100 * rate(D[s]) / rate(C[s]) for s in seeds]
                row = {'n': len(seeds), 'beta_reduction': red, 'charge_pct_cdbs': chg, 'seeds': seeds}
                cmp = ''
                K = R.get((win, 'constant_180_0.06_250'))
                if K and lab not in REFS:
                    ss = [s for s in seeds if s in K]
                    kr = [100 * (1 - np.mean(K[s]['lfp']) / np.mean(U[s]['lfp'])) for s in ss]
                    mine = [r for s, r in zip(seeds, red) if s in K]
                    if len(ss) > 1:
                        t, p = stats.ttest_rel(mine, kr)
                        cmp = f'{np.mean(np.subtract(mine, kr)):+6.1f}, p={p:.2g}'
                        row['vs_const250'] = {'diff': float(np.mean(np.subtract(mine, kr))), 'p': float(p)}
                print(f'  {lab:24s} {len(seeds):2d}  {mean_sem(red):>13s}  {mean_sem(chg):>13s}  {cmp}')
            else:
                red, chg, adapt = [], [], []
                for s in seeds:
                    pd_ = np.asarray(D[s]['pd_state']) == 1
                    upd = np.asarray(U[s]['pd_state']) == 1
                    red.append(100 * (1 - np.asarray(D[s]['lfp'])[pd_].mean() / np.asarray(U[s]['lfp'])[upd].mean()))
                    chg.append(100 * rate(D[s]) / rate(C[s]))
                    h = rate(D[s], ~pd_)
                    adapt.append(rate(D[s], pd_) / h if h > 0 else np.inf)
                row = {'n': len(seeds), 'pd_beta_reduction': red, 'charge_pct_cdbs': chg,
                       'pd_over_healthy_rate': adapt, 'seeds': seeds}
                print(f'  {lab:24s} {len(seeds):2d}  {mean_sem(red):>14s}  {mean_sem(chg):>13s}  '
                      f'{np.mean(adapt):15.2f}')
            rows[lab] = row
        # per-arm means over training patients
        arms = sorted({arm(l) for l in rows if re.match(r'[ABC]_p\d+$', l)})
        for a in arms:
            members = [l for l in rows if arm(l) == a]
            key = 'beta_reduction' if protocol == 'acute' else 'pd_beta_reduction'
            per = [np.mean(rows[l][key]) for l in members]
            chg = [np.mean(rows[l]['charge_pct_cdbs']) for l in members]
            print(f'  arm {a} ({len(members)} patients): beta red. {np.mean(per):.1f}% '
                  f'[{min(per):.1f}, {max(per):.1f}], charge {np.mean(chg):.1f}% of cDBS')
        summary[protocol][str(win)] = rows

with open(os.path.join(DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
print(f'\nwrote {DIR}/summary.json')
