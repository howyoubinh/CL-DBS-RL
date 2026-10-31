"""
Headroom check, analysis (scripts/rev_manifest.yaml entry headroom_oracle):
how much can state-dependent switching save in the cycling protocol, and does the training
reward prefer it?

Per window length and controller (paired by patient, 5 x 60 s blocks, healthy first):
  PD beta red.     beta reduction in PD blocks vs the unstimulated run's PD blocks
  healthy beta     median beta in healthy blocks; % of healthy windows above tau
  charge           % of cDBS charge
  onset lag        oracle only: windows into a PD block until beta first falls below tau
                   (median over PD blocks); the price of switching off in healthy blocks
  reward           mean per-window reward of the training environment (MousePDEnvAdaptive.calc_reward,
                   delta = 1, tau_reward = 3000) computed from the recorded beta and settings, split
                   into healthy and PD blocks, for the window's tau (100 ms: 150 and 290; 1 s: 477)
                   and for three energy terms:
                     rms250  the training reward: RMS of the pulse train / 250 (as trained)
                     rms67   RMS / 67, the true maximum RMS over the action space
                     charge  charge rate / maximum charge rate over the action space
                   E = amp * sqrt(freq * pw / 1000) for the monophasic train in creatdbs().

Also checks that reference conditions rerun here reproduce data/revision/eval_round2 exactly.

  python scripts/analyze_rev_headroom.py [data/revision/headroom]
"""

import glob
import json
import os
import re
import sys

import numpy as np

DIR = sys.argv[1] if len(sys.argv) > 1 else 'data/revision/headroom'
ROUND2 = 'data/revision/eval_round2'
TAUS = {100: (150.0, 290.0), 1000: (477.0,)}
MAX = {'freq': 180.0, 'pw': 0.4, 'amp': 250.0}
E_RMS_MAX = MAX['amp'] * np.sqrt(MAX['freq'] * MAX['pw'] / 1000)          # ~67.1
Q_MAX = MAX['freq'] * MAX['pw'] * MAX['amp']


def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, 'cycling_*_seed*.json')):
        m = re.match(r'cycling_(.+)_w(\d+)_seed(\d+)\.json', os.path.basename(p))
        out.setdefault((int(m.group(2)), m.group(1)), {})[int(m.group(3))] = json.load(open(p))
    return out


def arrays(d):
    return {k: np.asarray(d[k], float) for k in ('lfp', 'freq', 'pw', 'amp', 'pd_state')}


def reward(a, tau, energy):
    f, pw, amp, beta = a['freq'], a['pw'], a['amp'], a['lfp']
    if energy == 'rms250':
        e = np.minimum(amp * np.sqrt(f * pw / 1000) / 250.0, 1.0)
    elif energy == 'rms67':
        e = amp * np.sqrt(f * pw / 1000) / E_RMS_MAX
    else:
        e = f * pw * amp / Q_MAX
    return np.where(beta > tau, -30.0 * (beta - tau), 3000.0 * (1.0 - e))


def onset_lag(a, tau):
    """Windows from the start of each PD block until beta first drops below tau."""
    pd = a['pd_state'].astype(int)
    starts = [i for i in range(len(pd)) if pd[i] == 1 and (i == 0 or pd[i - 1] == 0)]
    lags = []
    for s in starts:
        e = s
        while e < len(pd) and pd[e] == 1:
            e += 1
        below = np.nonzero(a['lfp'][s:e] <= tau)[0]
        lags.append(int(below[0]) if len(below) else e - s)
    return lags


def main():
    R = load(DIR)
    R2 = load(ROUND2) if os.path.isdir(ROUND2) else {}
    summary = {}
    for win in sorted({w for w, _ in R}):
        labels = sorted(l for w, l in R if w == win)
        U, C = R[(win, 'unstim')], R[(win, 'cdbs')]
        seeds = sorted(U)
        print(f'\n=== CYCLING, window {win} ms, patients {seeds} ===')
        for lab in labels:   # reproducibility against eval_round2
            if (win, lab) in R2:
                same = all(np.array_equal(R[(win, lab)][s]['lfp'], R2[(win, lab)][s]['lfp'])
                           for s in seeds if s in R2[(win, lab)])
                print(f'  reproduces eval_round2: {lab:28s} {same}')
        taus = TAUS[win]
        hdr = (f'  {"controller":26s} {"PD beta red%":>12s} {"healthy beta":>12s} '
               + ''.join(f'{"H>" + str(int(t)) + "%":>8s}' for t in taus) + f' {"charge%":>8s}')
        print(hdr)
        rows = {}
        for lab in labels:
            D = R[(win, lab)]
            red, hb, chg, above = [], [], [], {t: [] for t in taus}
            rew = {(t, e, blk): [] for t in taus for e in ('rms250', 'rms67', 'charge') for blk in ('healthy', 'pd')}
            lags = {t: [] for t in taus}
            for s in seeds:
                a, u, c = arrays(D[s]), arrays(U[s]), arrays(C[s])
                pd = a['pd_state'] == 1
                red.append(100 * (1 - a['lfp'][pd].mean() / u['lfp'][pd].mean()))
                hb.append(float(np.median(a['lfp'][~pd])))
                q = a['freq'] * a['pw'] * a['amp']
                chg.append(100 * q.sum() / (c['freq'] * c['pw'] * c['amp']).sum())
                for t in taus:
                    above[t].append(100 * (a['lfp'][~pd] > t).mean())
                    lags[t] += onset_lag(a, t)
                    for e in ('rms250', 'rms67', 'charge'):
                        r = reward(a, t, e)
                        rew[(t, e, 'healthy')].append(r[~pd].mean())
                        rew[(t, e, 'pd')].append(r[pd].mean())
            row = {'pd_beta_reduction': red, 'healthy_beta_median': hb, 'charge_pct_cdbs': chg,
                   'healthy_above_tau_pct': {str(t): v for t, v in above.items()},
                   'pd_onset_lag_windows_median': {str(t): float(np.median(v)) for t, v in lags.items()},
                   'reward': {f'{t:g}/{e}/{blk}': float(np.mean(v)) for (t, e, blk), v in rew.items()}}
            rows[lab] = row
            print(f'  {lab:26s} {np.mean(red):12.1f} {np.mean(hb):12.0f} '
                  + ''.join(f'{np.mean(above[t]):8.0f}' for t in taus) + f' {np.mean(chg):8.1f}'
                  + ('   onset lag (windows): ' + ', '.join(f'tau {t:g}: {np.median(lags[t]):.0f}' for t in taus)
                     if lab.startswith('oracle') else ''))
        for t in taus:
            for e in ('rms250', 'rms67', 'charge'):
                print(f'\n  mean reward per window, tau {t:g}, energy {e}  (healthy blocks | PD blocks | all)')
                for lab in labels:
                    h, p = rows[lab]['reward'][f'{t:g}/{e}/healthy'], rows[lab]['reward'][f'{t:g}/{e}/pd']
                    print(f'    {lab:26s} {h:9.0f} | {p:9.0f} | {(h + p) / 2:9.0f}')
        summary[str(win)] = rows
    with open(os.path.join(DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'\nsaved {os.path.join(DIR, "summary.json")}')


if __name__ == '__main__':
    main()
