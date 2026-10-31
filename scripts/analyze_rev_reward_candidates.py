"""
Offline reward verification: before any training, check on already-recorded
simulator runs that a candidate reward actually pays for adaptation (stimulate in PD, off in
healthy). No simulation, no training.

Data: data/revision/obs (8 patients x PD/healthy x {unstim, c212, c250, cdbs, random settings held
1 s}, 60 s, beta per 100 ms window) and data/revision/headroom (cycling 5 x 60 s, unstim / constant /
cDBS / oracle, 100 ms windows, 5 evaluation patients).

Candidate reward (same form as MousePDEnvAdaptive.calc_reward with delta = 1, tau_reward = 3000):
    r = -slope * (b - tau)        if b > tau
    r = 3000 * (1 - E)            otherwise
  b      raw   : beta of the current 100 ms window (the trained reward)
         roll5 : mean beta over the last 5 windows (0.5 s)
         roll10: mean beta over the last 10 windows (1 s)
  E      rms250: RMS pulse-train current / 250 (the trained reward)
         rms67 : RMS / 67.1, the true maximum over the action space (180 Hz, 0.4 ms, 250 uA)
         charge: freq * pw * amp / its maximum over the action space
  slope  30 (trained) or 3
  tau    per beta signal, calibrated on the TRAINING patients' unstimulated runs only (101, 202, 303):
         the midpoint between the healthy 99th and PD 1st percentile of that signal; also the
         trained values 150 and 290 for raw.

Criteria, all on the 5 EVALUATION patients (42, 123, 456, 789, 1024) unless noted:
  C1 healthy: off beats stimulation. R(off, healthy) vs R(s, healthy) for every setting s; reported
     as the reward margin of off over constant 212 / 250 / cDBS (steady 60 s runs) and the share of
     the random settings with >= 5% of cDBS charge that beat off.
  C2 PD: stimulating beats off. From the steady 60 s runs (off, 212, 250, cDBS): the margin of the
     best stimulated setting over off, and that setting's beta reduction. (Random 1 s holds are
     NOT used here: after stimulation stops, PD beta stays suppressed for ~0.5 s and recovers over
     ~1 s, so an "off" hold after a stimulated one mostly measures washout. The top-5% random
     settings are still reported, as a pointer to what an intermittent policy could exploit.)
  C3 cycling: the oracle beats every constant, replayed on the recorded cycling runs (so it includes
     the transients at block switches and the smoothing lag).
  C4 learnability: C3's margin (oracle - best constant, per window) divided by the per-window reward
     standard deviation of the oracle run, a rough signal-to-noise for a learner.

Steady-state reward of a random setting uses its 1 s hold: windows 2-10 (the first is the switch);
for smoothed signals the rolling mean is taken circularly over those 9 windows (an approximation of
holding the setting longer).

  python scripts/analyze_rev_reward_candidates.py
"""

import glob
import itertools
import json
import os
import re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OBS = os.path.join(ROOT, 'data/revision/obs')
HEAD = os.path.join(ROOT, 'data/revision/headroom')
OUT = os.path.join(ROOT, 'data/revision/reward_check')
TRAIN, EVAL = (101, 202, 303), (42, 123, 456, 789, 1024)
SIGNALS = {'raw': 1, 'roll5': 5, 'roll10': 10}
E_RMS_MAX = 250.0 * np.sqrt(180 * 0.4 / 1000)
Q_MAX = 180 * 0.4 * 250.0
CDBS_Q = 130 * 0.3 * 300.0
SETTLE = 10


def energy(stim, kind):
    f, pw, a = stim[..., 0], stim[..., 1], stim[..., 2]
    rms = a * np.sqrt(f * pw / 1000)
    if kind == 'rms250':
        return np.minimum(rms / 250.0, 1.0)
    if kind == 'rms67':
        return rms / E_RMS_MAX
    return f * pw * a / Q_MAX


def roll(b, k):
    """Causal rolling mean; the first k-1 windows use what is available."""
    if k == 1:
        return b
    c = np.cumsum(np.r_[0.0, b])
    i = np.arange(1, len(b) + 1)
    lo = np.maximum(0, i - k)
    return (c[i] - c[lo]) / (i - lo)


def circ_roll(b, k):
    k = min(k, len(b))
    return np.array([np.mean(np.take(b, range(j - k + 1, j + 1), mode='wrap')) for j in range(len(b))])


def reward(bs, e, tau, slope):
    return np.where(bs > tau, -slope * (bs - tau), 3000.0 * (1.0 - e))


def load_obs():
    runs = {}
    for p in glob.glob(os.path.join(OBS, 'obs_*_seed*.npz')):
        m = re.match(r'obs_(\w+?)_(pd|healthy)_seed(\d+)\.npz', os.path.basename(p))
        d = np.load(p)
        runs[(m.group(1), int(m.group(2) == 'pd'), int(m.group(3)))] = {'beta': d['beta'], 'stim': d['stim']}
    return runs


def load_cycling():
    out = {}
    for p in glob.glob(os.path.join(HEAD, 'cycling_*_w100_seed*.json')):
        m = re.match(r'cycling_(.+)_w100_seed(\d+)\.json', os.path.basename(p))
        d = json.load(open(p))
        out.setdefault(m.group(1), {})[int(m.group(2))] = {
            'beta': np.asarray(d['lfp'], float), 'pd': np.asarray(d['pd_state'], int),
            'stim': np.stack([np.asarray(d[k], float) for k in ('freq', 'pw', 'amp')], 1)}
    return out


def calibrate(runs, k):
    h = np.concatenate([roll(runs[('unstim', 0, s)]['beta'][SETTLE:], k)[k - 1:] for s in TRAIN])
    p = np.concatenate([roll(runs[('unstim', 1, s)]['beta'][SETTLE:], k)[k - 1:] for s in TRAIN])
    return float((np.percentile(h, 99) + np.percentile(p, 1)) / 2), float(np.percentile(h, 99)), float(np.percentile(p, 1))


def steady(runs, cond, pd, seeds, k, ekind, tau, slope):
    """Mean per-window reward of a fixed 60 s condition, and its mean raw beta."""
    r, b = [], []
    for s in seeds:
        d = runs[(cond, pd, s)]
        beta, stim = d['beta'][SETTLE:], d['stim'][SETTLE:]
        bs = roll(beta, k)[k - 1:]
        r.append(reward(bs, energy(stim[k - 1:], ekind), tau, slope).mean())
        b.append(beta.mean())
    return float(np.mean(r)), float(np.mean(b))


def random_holds(runs, pd, seeds, k, ekind, tau, slope):
    """Per 1 s hold of the random condition: (setting, mean reward, mean raw beta)."""
    rows = []
    for s in seeds:
        d = runs[('random', pd, s)]
        beta, stim = d['beta'], d['stim']
        for h0 in range(SETTLE, len(beta), 10):
            b = beta[h0 + 1:h0 + 10]
            st = stim[h0]
            bs = circ_roll(b, k) if k > 1 else b
            rows.append((st, float(reward(bs, energy(st, ekind), tau, slope).mean()), float(b.mean())))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load_obs()
    cyc = load_cycling()
    unstim_pd_beta = np.mean([runs[('unstim', 1, s)]['beta'][SETTLE:].mean() for s in EVAL])

    cands = []
    for sig, k in SIGNALS.items():
        tau_c, h99, p1 = calibrate(runs, k)
        print(f'{sig:6s} calibrated tau {tau_c:.0f} (training patients: healthy p99 {h99:.0f}, PD p1 {p1:.0f})')
        taus = [tau_c] + ([150.0, 290.0] if sig == 'raw' else [])
        for tau, ekind, slope in itertools.product(taus, ('rms250', 'rms67', 'charge'), (30.0, 3.0)):
            cands.append((sig, k, tau, ekind, slope))

    results = []
    for sig, k, tau, ekind, slope in cands:
        res = {'signal': sig, 'tau': tau, 'energy': ekind, 'slope': slope}
        # C1 healthy
        r_off_h, _ = steady(runs, 'unstim', 0, EVAL, k, ekind, tau, slope)
        steady_h = {c: steady(runs, c, 0, EVAL, k, ekind, tau, slope)[0] for c in ('c212', 'c250', 'cdbs')}
        holds_h = random_holds(runs, 0, EVAL, k, ekind, tau, slope)
        strong = [r for st, r, _ in holds_h if st[0] * st[1] * st[2] >= 0.05 * CDBS_Q]
        res['C1'] = {'off': r_off_h, 'margin_vs': {c: r_off_h - v for c, v in steady_h.items()},
                     'share_strong_settings_beating_off': float(np.mean(np.array(strong) > r_off_h))}
        # C2 PD
        cand_pd = [(c, *steady(runs, c, 1, EVAL, k, ekind, tau, slope)) for c in ('unstim', 'c212', 'c250', 'cdbs')]
        holds_p = random_holds(runs, 1, EVAL, k, ekind, tau, slope)
        # rank random settings by reward, average the top 5% to damp the winner's curse
        holds_p.sort(key=lambda x: -x[1])
        top = holds_p[:max(1, len(holds_p) // 20)]
        best_steady = max(cand_pd[1:], key=lambda x: x[1])
        res['C2'] = {
            'pd_reward': {c: r for c, r, _ in cand_pd},
            'best_steady': best_steady[0],
            'margin_stim_over_off': best_steady[1] - cand_pd[0][1],
            'best_steady_beta_reduction_pct': float(100 * (1 - best_steady[2] / cand_pd[0][2])),
            'top5pct_random': {'reward': float(np.mean([r for _, r, _ in top])),
                               'beta_reduction_pct': float(100 * (1 - np.mean([b for _, _, b in top]) / unstim_pd_beta)),
                               'charge_pct_cdbs': float(100 * np.mean([st[0] * st[1] * st[2] for st, _, _ in top]) / CDBS_Q),
                               'off_share': float(np.mean([st[2] == 0 or st[0] == 0 for st, _, _ in top]))}}
        # C3 / C4 cycling replay
        cyc_r = {}
        for lab, D in cyc.items():
            per = []
            for s, d in D.items():
                bs = roll(d['beta'], k)
                per.append(reward(bs, energy(d['stim'], ekind), tau, slope))
            cyc_r[lab] = (float(np.mean([p.mean() for p in per])), float(np.mean([p.std() for p in per])))
        consts = {l: v for l, v in cyc_r.items() if l.startswith(('constant', 'cdbs', 'unstim'))}
        best_c = max(consts, key=lambda l: consts[l][0])
        o = cyc_r['oracle_180_0.06_250']
        res['C3'] = {lab: v[0] for lab, v in cyc_r.items()}
        res['C3_margin'] = o[0] - consts[best_c][0]
        res['C3_best_constant'] = best_c
        res['C4_snr'] = res['C3_margin'] / o[1] if o[1] > 0 else float('nan')
        res['pass'] = bool(min(res['C1']['margin_vs'].values()) > 0 and res['C3_margin'] > 0
                           and res['C2']['margin_stim_over_off'] > 0)
        results.append(res)

    print(f'\n{"signal":6s} {"tau":>5s} {"energy":7s} {"slope":>5s} | {"C1 off - best steady stim":>25s} {"strong>off":>10s} | '
          f'{"C2 PD: stim-off margin / best / beta red":>40s} | {"C3 oracle - best const":>22s} {"C4 SNR":>6s} | pass')
    for r in results:
        c1 = min(r['C1']['margin_vs'].values())
        c2 = r['C2']
        print(f'{r["signal"]:6s} {r["tau"]:5.0f} {r["energy"]:7s} {r["slope"]:5.0f} | {c1:25.0f} '
              f'{100 * r["C1"]["share_strong_settings_beating_off"]:9.0f}% | '
              f'{c2["margin_stim_over_off"]:17.0f} {c2["best_steady"]:>6s} {c2["best_steady_beta_reduction_pct"]:12.1f}% | '
              f'{r["C3_margin"]:14.0f} ({r["C3_best_constant"][:12]}) {r["C4_snr"]:6.2f} | {"YES" if r["pass"] else "no"}')
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nsaved {os.path.join(OUT, "summary.json")}')


if __name__ == '__main__':
    main()
