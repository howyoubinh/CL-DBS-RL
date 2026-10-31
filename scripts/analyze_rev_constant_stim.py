"""
Aggregate the SNN vs constant-stimulation comparison runs (scripts/run_rev_constant_stim.py).

Reports, per protocol:
  acute    mean GPi alpha-beta biomarker per condition, reduction vs Unstimulated
           (mean, SEM, 95% CI over seeds), delivered charge, and paired tests
           SNN vs Constant.
  cycling  PD-block and healthy-block biomarker, charge reduction vs cDBS, the SNN's
           parameters by state, and paired tests SNN vs Constant (seeded runs only;
           unseeded runs are summarized without pairing).

Charge proxy matches run_on_off_experiment.py: sum(freq * amp * pw * 0.1) per step.

Usage: python scripts/analyze_rev_constant_stim.py [data/revision/constant_stim]
"""

import glob
import json
import os
import sys

import numpy as np
from scipy import stats

DIR = sys.argv[1] if len(sys.argv) > 1 else 'data/revision/constant_stim'


def load(pattern):
    out = {}
    for p in sorted(glob.glob(os.path.join(DIR, pattern))):
        d = json.load(open(p))
        out.setdefault(d['condition'], {})[d['seed']] = d
    return out


def charge(d, mask=None):
    f, a, w = (np.asarray(d[k]) for k in ('freq', 'amp', 'pw'))
    q = f * a * w * 0.1
    return float(q[mask].sum() if mask is not None else q.sum())


def ci95(x):
    x = np.asarray(x)
    if len(x) < 2:
        return (np.nan, np.nan)
    h = stats.sem(x) * stats.t.ppf(0.975, len(x) - 1)
    return (x.mean() - h, x.mean() + h)


def paired(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2:
        return {}
    t, p = stats.ttest_rel(a, b)
    d = a - b
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
    try:
        w = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w = np.nan
    print(f'    {label}: mean diff {d.mean():+.2f}, t({len(a)-1})={t:.2f}, p={p:.3g}, '
          f'Wilcoxon p={w:.3g}, dz={dz:.2f}')
    return {'mean_diff': float(d.mean()), 't': float(t), 'p': float(p), 'wilcoxon_p': float(w), 'dz': float(dz)}


summary = {}

# ---------------------------------------------------------------- acute
A = load('acute_*_seed*.json')
if 'Unstimulated' in A:
    seeds = sorted(A['Unstimulated'])
    print(f'\n=== ACUTE (PD only, 40 steps, n={len(seeds)} seeds) ===')
    base = {s: np.mean(A['Unstimulated'][s]['lfp']) for s in seeds}
    summary['acute'] = {'n': len(seeds), 'unstim_beta': float(np.mean(list(base.values())))}
    red = {}
    for c in [c for c in ('SNN', 'Constant', 'ConstantMaxAmp') if c in A]:
        ss = [s for s in seeds if s in A[c]]
        r = [100 * (base[s] - np.mean(A[c][s]['lfp'])) / base[s] for s in ss]
        q = [charge(A[c][s]) for s in ss]
        red[c] = dict(zip(ss, r))
        lo, hi = ci95(r)
        print(f'  {c:15s} reduction {np.mean(r):6.2f}% +/- {stats.sem(r):.2f} SEM '
              f'(95% CI [{lo:.1f}, {hi:.1f}]), charge {np.mean(q):8.1f}')
        summary['acute'][c] = {'reduction_mean': float(np.mean(r)), 'reduction_sem': float(stats.sem(r)),
                               'ci95': [float(lo), float(hi)], 'charge_mean': float(np.mean(q)), 'n': len(ss)}
    for c in ('Constant', 'ConstantMaxAmp'):
        if 'SNN' in red and c in red:
            ss = sorted(set(red['SNN']) & set(red[c]))
            summary['acute'][f'SNN_vs_{c}'] = paired([red['SNN'][s] for s in ss],
                                                     [red[c][s] for s in ss], f'SNN - {c} (reduction, pts)')

# ---------------------------------------------------------------- cycling
for tag, pattern in (('cycling_seeded', 'cycling_*_seed*.json'), ('cycling_unseeded', 'cycling_*_unseeded.json')):
    C = load(pattern)
    if 'cDBS' not in C:
        continue
    print(f'\n=== {tag.upper()} (500 steps, 100-step blocks, starting healthy) ===')
    summary[tag] = {}
    rows = {}
    for c in [c for c in ('SNN', 'Constant', 'cDBS', 'Unstimulated') if c in C]:
        rows[c] = {}
        for s, d in C[c].items():
            pd_ = np.asarray(d['pd_state']) == 1
            lfp = np.asarray(d['lfp'])
            ref = C['cDBS'].get(s)
            red = 100 * (1 - charge(d) / charge(ref)) if ref is not None else np.nan
            rows[c][s] = {'beta_pd': lfp[pd_].mean(), 'beta_healthy': lfp[~pd_].mean(), 'charge_red': red}
        v = rows[c].values()
        bp, bh, cr = (np.array([x[k] for x in v]) for k in ('beta_pd', 'beta_healthy', 'charge_red'))
        print(f'  {c:13s} beta PD {bp.mean():7.1f}  beta healthy {bh.mean():7.1f}  '
              f'charge vs cDBS {cr.mean():6.1f}%  (n={len(bp)})')
        summary[tag][c] = {'beta_pd': float(bp.mean()), 'beta_healthy': float(bh.mean()),
                           'charge_reduction_vs_cdbs': float(cr.mean()), 'n': int(len(bp))}
    if 'SNN' in C:
        f = np.array([C['SNN'][s]['freq'] for s in C['SNN']])
        w = np.array([C['SNN'][s]['pw'] for s in C['SNN']])
        a = np.array([C['SNN'][s]['amp'] for s in C['SNN']])
        st = np.array([C['SNN'][s]['pd_state'] for s in C['SNN']])
        late = np.arange(f.shape[1]) >= 20  # skip the initial ramp
        for state, name in ((1, 'PD'), (0, 'healthy')):
            m = (st == state) & late
            print(f'    SNN params in {name:7s} blocks (step>=20): {f[m].mean():6.1f} Hz, '
                  f'{w[m].mean():.3f} ms, {a[m].mean():6.1f} uA')
            summary[tag][f'SNN_params_{name}'] = [float(f[m].mean()), float(w[m].mean()), float(a[m].mean())]
    if tag == 'cycling_seeded' and 'SNN' in rows and 'Constant' in rows:
        ss = sorted(set(rows['SNN']) & set(rows['Constant']))
        for k, lab in (('beta_pd', 'beta in PD blocks'), ('charge_red', 'charge reduction (pts)')):
            summary[tag][f'SNN_vs_Constant_{k}'] = paired([rows['SNN'][s][k] for s in ss],
                                                         [rows['Constant'][s][k] for s in ss], f'SNN - Constant, {lab}')

with open(os.path.join(DIR, 'summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=2)
print(f'\nsummary written to {os.path.join(DIR, "summary.json")}')
