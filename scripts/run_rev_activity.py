"""
Firing, bursting and synchrony of the simulated CBGT network, healthy vs PD, from simulator
output: which burst and synchrony features distinguish the pathological state.

For each patient (seed) and state, the unstimulated network is simulated for --seconds of
continuous activity (1 s simulator calls with state carried over, fixed seed). Saved per run:
all recorded spike times (find_spike_times, -20 mV rising crossings), the GPi alpha-beta biomarker
per 1 s window, and the first second of membrane potential for one neuron per population
(captured from the arrays the simulator passes to find_spike_times; the simulator is unchanged).

Metrics (computed per population, after --settle seconds):
  rate        mean firing rate (Hz)
  cv_isi      coefficient of variation of inter-spike intervals
  burst_frac  fraction of spikes in bursts; a burst is >= 3 consecutive spikes whose ISIs are all
              <= 10 ms (a simple max-interval rule)
  sync_within mean pairwise Pearson correlation of 5 ms spike counts within the population
  sync_stn_gpe mean pairwise correlation between STN and GPe neurons (5 ms counts)

  python scripts/run_rev_activity.py --seeds 42 123 456 --seconds 10 --out data/revision/activity
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

POPS = ('TH', 'STN', 'GPe', 'GPi', 'StrD2', 'StrD1', 'Cor')
KEYS = ('TH_APs', 'STN_APs', 'GPe_APs', 'GPi_APs', 'Striat_APs_indr', 'Striat_APs_dr', 'Cor_APs')
BIN = 0.005
BURST_ISI = 0.010


def simulate(job):
    seed, pd, seconds = job
    import src.simulation.simulate_network_optimized as S
    captured = []
    orig = S.find_spike_times

    def capture(voltage, time, num_neurons):  # sees each population's full voltage array
        captured.append(np.asarray(voltage[0, ::10], dtype=np.float32))  # neuron 0, 0.1 ms resolution
        return orig(voltage, time, num_neurons)

    S.find_spike_times = capture
    spikes = {k: [[] for _ in range(20 if k == 'Cor_APs' else 10)] for k in KEYS}
    beta, states, vtrace = [], {}, None
    for w in range(seconds):
        captured.clear()
        r = S.simulate_network_opt(0, pd, 0, n=10, tmax=1000, dt=0.01, PW=0.0, amplitude=0.0, dbs_freq=0,
                                   states=states, seed=seed)
        states = r['states']
        beta.append(float(r['gpi_alpha_beta_area']))
        for k in KEYS:
            for i, neuron in enumerate(r[k]):
                spikes[k][i].extend((np.asarray(neuron['times']).ravel() + w).tolist())
        if w == 0:
            vtrace = {p: captured[i].tolist() for i, p in enumerate(POPS)}
    S.find_spike_times = orig
    return seed, pd, {'spikes': spikes, 'beta': beta, 'vtrace_first_s': vtrace, 'seconds': seconds}


def counts(trains, t0, t1):
    edges = np.arange(t0, t1 + 1e-9, BIN)
    return np.stack([np.histogram(np.asarray(s), bins=edges)[0] for s in trains])


def mean_corr(a, b=None):
    if b is None:
        c = np.corrcoef(a)
        iu = np.triu_indices(len(a), 1)
        v = c[iu]
    else:
        c = np.corrcoef(np.vstack([a, b]))[:len(a), len(a):]
        v = c.ravel()
    return float(np.nanmean(v))


def metrics(run, settle):
    T = run['seconds']
    out = {}
    for p, k in zip(POPS, KEYS):
        trains = [np.asarray([t for t in s if t >= settle]) for s in run['spikes'][k]]
        n_sp = sum(len(s) for s in trains)
        isis = [np.diff(s) for s in trains if len(s) > 2]
        cv = float(np.nanmean([i.std() / i.mean() for i in isis if i.mean() > 0])) if isis else float('nan')
        in_burst = 0
        for s in trains:
            short = np.diff(s) <= BURST_ISI
            run_len = 0
            for flag in np.append(short, False):
                if flag:
                    run_len += 1
                else:
                    if run_len >= 2:          # >= 2 short ISIs = >= 3 spikes
                        in_burst += run_len + 1
                    run_len = 0
        c = counts(trains, settle, T)
        out[p] = {'rate': n_sp / (len(trains) * (T - settle)), 'cv_isi': cv,
                  'burst_frac': in_burst / n_sp if n_sp else float('nan'),
                  'sync_within': mean_corr(c) if c.sum() else float('nan')}
    stn = counts([np.asarray([t for t in s if t >= settle]) for s in run['spikes']['STN_APs']], settle, T)
    gpe = counts([np.asarray([t for t in s if t >= settle]) for s in run['spikes']['GPe_APs']], settle, T)
    out['sync_stn_gpe'] = mean_corr(stn, gpe)
    out['beta_median'] = float(np.median(run['beta'][settle:]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--seconds', type=int, default=10)
    ap.add_argument('--settle', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/revision/activity'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [(s, pd, args.seconds) for s in args.seeds for pd in (0, 1)]
    with mp.Pool(len(jobs)) as pool:
        runs = pool.map(simulate, jobs)
    summary = {}
    for seed, pd, run in runs:
        tag = f'seed{seed}_{"pd" if pd else "healthy"}'
        with open(os.path.join(args.out, f'{tag}.json'), 'w') as f:
            json.dump(run, f)
        summary[tag] = metrics(run, args.settle)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)

    print(f'{"pop":6s} {"metric":12s} {"healthy (mean over seeds)":>26s} {"PD":>10s}')
    for p in POPS:
        for m in ('rate', 'cv_isi', 'burst_frac', 'sync_within'):
            h = np.nanmean([summary[f'seed{s}_healthy'][p][m] for s in args.seeds])
            d = np.nanmean([summary[f'seed{s}_pd'][p][m] for s in args.seeds])
            print(f'{p:6s} {m:12s} {h:26.3f} {d:10.3f}')
    for m in ('sync_stn_gpe', 'beta_median'):
        h = np.mean([summary[f'seed{s}_healthy'][m] for s in args.seeds])
        d = np.mean([summary[f'seed{s}_pd'][m] for s in args.seeds])
        print(f'{"":6s} {m:12s} {h:26.3f} {d:10.3f}')


if __name__ == '__main__':
    main()
