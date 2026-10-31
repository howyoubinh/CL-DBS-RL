"""
Calibrate the reward threshold tau_beta for a given observation window on the TRAINING patients
only, so the evaluation patients stay unseen (used for the 1 s-window arm B of teachers_round2).

For each patient: GPi alpha-beta biomarker per window with no stimulation (healthy and PD), and in
PD under constant stimulation and cDBS (to check the threshold is reachable). Proposed
tau_beta = midpoint between the highest healthy and the lowest PD unstimulated window across the
training patients.

  python scripts/run_rev_calibrate_tau.py --window-ms 1000 --patients 101 202 303 \
      --out data/revision/calibration/tau_1000ms.json
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

CONDITIONS = {
    'healthy_unstim': (0, {'freq': 0, 'pw': 0, 'amp': 0}),
    'pd_unstim': (1, {'freq': 0, 'pw': 0, 'amp': 0}),
    'pd_constant': (1, {'freq': 180.0, 'pw': 0.06, 'amp': 212.0}),
    'pd_cdbs': (1, {'freq': 130.0, 'pw': 0.3, 'amp': 300.0}),
}


def run(job):
    patient, cond, window_ms, n, settle = job
    from src.simulation.sim_wrapper import SimulationWrapper
    pd, dbs = CONDITIONS[cond]
    sim = SimulationWrapper(n_neurons=10, t_step=window_ms, dt=0.01, seed=patient)
    sim.set_pd_state(pd)
    sim.reset()
    vals = []
    for _ in range(n + settle):
        sim.set_pd_state(pd)
        vals.append(float(sim.step(dbs)['lfp']))
    return patient, cond, vals[settle:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window-ms', type=int, required=True)
    ap.add_argument('--patients', type=int, nargs='+', required=True)
    ap.add_argument('--windows', type=int, default=20)
    ap.add_argument('--settle', type=int, default=2)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    jobs = [(p, c, args.window_ms, args.windows, args.settle) for p in args.patients for c in CONDITIONS]
    with mp.Pool(len(jobs)) as pool:
        res = pool.map(run, jobs)
    data = {f'{p}': {} for p in args.patients}
    for p, c, v in res:
        data[str(p)][c] = v
    hmax = max(max(d['healthy_unstim']) for d in data.values())
    pmin = min(min(d['pd_unstim']) for d in data.values())
    tau = 0.5 * (hmax + pmin)
    print(f'window {args.window_ms} ms, training patients {args.patients}')
    for p, d in data.items():
        print(f'  patient {p}: ' + '  '.join(f'{c} median {np.median(v):.0f} [{min(v):.0f}, {max(v):.0f}]' for c, v in d.items()))
    print(f'  highest healthy window {hmax:.0f}, lowest PD window {pmin:.0f} -> proposed tau_beta {tau:.0f}'
          + ('' if hmax < pmin else '  (WARNING: overlap, no clean threshold)'))
    for c in ('pd_constant', 'pd_cdbs'):
        frac = np.mean([v < tau for d in data.values() for v in d[c]])
        print(f'  {c}: {100 * frac:.0f}% of PD windows below the proposed threshold')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'window_ms': args.window_ms, 'patients': args.patients, 'windows': args.windows,
                   'settle': args.settle, 'highest_healthy': hmax, 'lowest_pd': pmin,
                   'proposed_tau_beta': tau, 'data': data}, f, indent=1)


if __name__ == '__main__':
    main()
