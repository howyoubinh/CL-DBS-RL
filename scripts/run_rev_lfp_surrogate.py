"""
Does an LFP-like signal carry the same pathological alpha-beta signature as the point-process
biomarker the controller is rewarded on?

For each patient and condition the network is simulated as consecutive 1 s windows (state carried,
fixed seed). Per window we compute:
  biomarker  the simulator's GPi alpha-beta biomarker (multi-taper point-process estimate)
  lfp_gpi    7-35 Hz power of a GPi LFP surrogate: summed GPi spikes at 1 kHz convolved with a
             causal exponential kernel (tau = 5 ms), i.e. a population-rate proxy at the biomarker site
  lfp_stn    7-35 Hz power of an STN-site surrogate (DBS leads sit in STN; LFP mostly reflects
             synaptic input): GPe->STN inhibitory and cortex->STN excitatory input spikes convolved
             with causal synaptic kernels (GABA tau 10 ms, AMPA tau 5 ms), summed as current
             magnitudes
Band power is Welch (256 ms Hann segments, 50% overlap), absolute and relative (7-35 / 1-100 Hz).

Conditions: healthy and PD unstimulated, PD with constant 180 Hz / 0.06 ms / 212 uA, PD with cDBS.

  python scripts/run_rev_lfp_surrogate.py --seeds 42 123 456 --windows 10 --out data/revision/lfp_surrogate_obs
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
from scipy import signal, stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

CONDITIONS = {
    'healthy': (0, {'freq': 0, 'pw': 0, 'amp': 0}),
    'pd': (1, {'freq': 0, 'pw': 0, 'amp': 0}),
    'pd_constant': (1, {'freq': 180.0, 'pw': 0.06, 'amp': 212.0}),
    'pd_cdbs': (1, {'freq': 130.0, 'pw': 0.3, 'amp': 300.0}),
}
FS = 1000.0


def kernel(tau_ms, length_ms=60):
    t = np.arange(length_ms)
    k = np.exp(-t / tau_ms)
    return k / k.sum()


def pop_signal(neurons, n_bins):
    x = np.zeros(n_bins)
    for nrn in neurons:
        idx = np.floor(np.asarray(nrn['times']).ravel() * FS).astype(int)
        np.add.at(x, idx[(idx >= 0) & (idx < n_bins)], 1)
    return x


def band(x):
    f, p = signal.welch(x - x.mean(), fs=FS, nperseg=256, noverlap=128, window='hann')
    b = (f >= 7) & (f <= 35)
    tot = (f >= 1) & (f <= 100)
    absolute = np.trapz(p[b], f[b])
    return float(absolute), float(absolute / np.trapz(p[tot], f[tot]))


def run(job):
    seed, cond, n_windows, settle = job
    import src.simulation.simulate_network_optimized as S
    pd, dbs = CONDITIONS[cond]
    states, rows = {}, []
    for w in range(n_windows + settle):
        r = S.simulate_network_opt(0, pd, 0, n=10, tmax=1000, dt=0.01, PW=dbs['pw'], amplitude=dbs['amp'],
                                   dbs_freq=dbs['freq'], states=states, seed=seed)
        states = r['states']
        if w < settle:
            continue
        n_bins = int(FS)
        gpi = pop_signal(r['GPi_APs'], n_bins)
        gpe = pop_signal(r['GPe_APs'], n_bins)
        cor = pop_signal(r['Cor_APs'][:10], n_bins)            # excitatory cortical neurons (first 10)
        lfp_gpi = np.convolve(gpi, kernel(5.0))[:n_bins]
        lfp_stn = np.convolve(gpe, kernel(10.0))[:n_bins] + np.convolve(cor, kernel(5.0))[:n_bins]
        a_g, r_g = band(lfp_gpi)
        a_s, r_s = band(lfp_stn)
        rows.append({'biomarker': float(r['gpi_alpha_beta_area']), 'lfp_gpi': a_g, 'lfp_gpi_rel': r_g,
                     'lfp_stn': a_s, 'lfp_stn_rel': r_s})
    return seed, cond, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--windows', type=int, default=10)
    ap.add_argument('--settle', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/revision/lfp_surrogate_obs'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(s, c, args.windows, args.settle) for s in args.seeds for c in CONDITIONS]
    with mp.Pool(len(jobs)) as pool:
        res = pool.map(run, jobs)
    data = {f'{s}': {} for s in args.seeds}
    for s, c, rows in res:
        data[str(s)][c] = rows
    with open(os.path.join(args.out, 'e6_windows.json'), 'w') as f:
        json.dump(data, f)

    def col(key, conds):
        return np.array([r[key] for s in data for c in conds for r in data[s][c]])
    allc = list(CONDITIONS)
    bio = col('biomarker', allc)
    print(f'windows: {len(bio)} ({len(args.seeds)} patients x {len(allc)} conditions x {args.windows})')
    for key in ('lfp_gpi', 'lfp_gpi_rel', 'lfp_stn', 'lfp_stn_rel'):
        rho, p = stats.spearmanr(bio, col(key, allc))
        h, d = col(key, ['healthy']), col(key, ['pd'])
        auc = float((d[:, None] > h[None, :]).mean())
        red = {c: 100 * (1 - np.median(col(key, [c])) / np.median(d)) for c in ('pd_constant', 'pd_cdbs')}
        print(f'{key:12s} Spearman rho vs biomarker {rho:+.2f} (p={p:.1e}) | healthy-vs-PD AUC {auc:.2f} | '
              f'reduction vs PD: constant {red["pd_constant"]:+.0f}%, cDBS {red["pd_cdbs"]:+.0f}%')
    red_b = {c: 100 * (1 - np.median(col('biomarker', [c])) / np.median(col('biomarker', ['pd']))) for c in ('pd_constant', 'pd_cdbs')}
    hb, db = col('biomarker', ['healthy']), col('biomarker', ['pd'])
    print(f'{"biomarker":12s} (reference)                         | healthy-vs-PD AUC {(db[:, None] > hb[None, :]).mean():.2f} | '
          f'reduction vs PD: constant {red_b["pd_constant"]:+.0f}%, cDBS {red_b["pd_cdbs"]:+.0f}%')


if __name__ == '__main__':
    main()
