"""
Constant open-loop stimulation at the SNN's operating point.

In the published cycling experiment the trained SNN settles at ~180 Hz, 0.06 ms,
~212 uA in both healthy and parkinsonian blocks. This script tests whether fixed
stimulation at that point reproduces the SNN's beta suppression and charge, i.e.
whether the learned policy adds anything beyond finding the operating point.

Two protocols, each run for one (condition, seed) per invocation so jobs can be
parallelized:

  acute    PD state only, 40 steps (4 s), warmup 0 -- same protocol as the efficacy
           experiment (run_exp1_therapeutic_efficacy.py). Metric: mean per-window GPi
           alpha-beta biomarker (obs['lfp']), reduction vs Unstimulated.
  cycling  500 steps alternating 100 healthy / 100 PD steps (10 s blocks), starting
           healthy -- same protocol as run_on_off_experiment.py, but seeded so every
           condition sees the same network instance. With --unseeded it instead
           replicates the published run exactly: no seed, so the simulator redraws
           connectivity and random conductances on every 100 ms call
           (simulate_network_optimized.py, CTX_BG_TH_network_CL); --seed is then only
           a repeat index used in the output filename.

The acute protocol reuses
quant_fidelity_harness.run_experiment (a copy of the initial-submission efficacy harness with a
controller hook); the cycling protocol reuses run_on_off_experiment's SNN
interface and mirrors its loop.

Usage:
  python scripts/run_rev_constant_stim.py --protocol acute --condition Constant \
      --seed 42 --out data/revision/constant_stim
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'scripts'))

from src.simulation.sim_wrapper import SimulationWrapper  # noqa: E402

TEACHER = os.path.join(ROOT, 'models/final_rockpool_16ch_curriculum/teacher.pth')

# SNN steady-state operating point in the published cycling run
# (on_off_results_06-01-2026_20-31-09.csv, steps >= 20): 179.7 Hz, 0.060 ms, 211.9 uA.
CONSTANT_POINTS = {
    'Constant': {'freq': 180.0, 'pw': 0.06, 'amp': 212.0},
    'ConstantMaxAmp': {'freq': 180.0, 'pw': 0.06, 'amp': 250.0},
}
CDBS = {'freq': 130.0, 'pw': 0.3, 'amp': 300.0}
OFF = {'freq': 0.0, 'pw': 0.0, 'amp': 0.0}


class ConstantController:
    """Open-loop controller: returns the same absolute stimulation every step."""

    def __init__(self, point):
        self.point = dict(point)

    def get_action(self, spike_data):
        return dict(self.point)


def run_acute(condition, seed, num_steps=40):
    import quant_fidelity_harness as qh

    if condition == 'Unstimulated':
        res = qh.run_experiment('Unstimulated', num_steps, seed=seed, warmup_steps=0)
    elif condition == 'SNN':
        res = qh.run_experiment('SNN', num_steps, model_path=TEACHER, seed=seed, warmup_steps=0)
    elif condition in CONSTANT_POINTS:
        pt = CONSTANT_POINTS[condition]
        res = qh.run_experiment('SNN', num_steps, seed=seed, warmup_steps=0,
                                controller_factory=lambda: ConstantController(pt))
    else:
        raise ValueError(condition)
    return {
        'lfp': [float(x) for x in res['lfp']],
        'freq': [float(x) for x in res['stim']['freq']],
        'pw': [float(x) for x in res['stim']['pw']],
        'amp': [float(x) for x in res['stim']['amp']],
    }


def run_cycling(condition, seed, num_steps=500, unseeded=False):
    sim = SimulationWrapper(n_neurons=10, t_step=100, dt=0.01, seed=None if unseeded else seed)
    obs = sim.reset()

    snn = None
    if condition == 'SNN':
        import run_on_off_experiment as oo
        snn = oo.SoftwareSNNInterface(TEACHER)

    # Initial state and leap values match run_on_off_experiment.py / gym_pd.py
    freq, pw, amp = 40.0, 0.3, 300.0
    leap = {'freq': 5.0, 'pw': 0.1, 'amp': 5.0}
    out = {'lfp': [], 'freq': [], 'pw': [], 'amp': [], 'pd_state': []}

    for i in range(num_steps):
        pd_state = 0 if (i % 200) < 100 else 1
        if snn is not None and i > 0 and pd_state != out['pd_state'][-1]:
            snn.model[1].reset()  # same block-transition reset as run_on_off_experiment.py
        sim.set_pd_state(pd_state)

        if condition == 'Unstimulated':
            freq, pw, amp = OFF['freq'], OFF['pw'], OFF['amp']
        elif condition == 'cDBS':
            freq, pw, amp = CDBS['freq'], CDBS['pw'], CDBS['amp']
        elif condition in CONSTANT_POINTS:
            pt = CONSTANT_POINTS[condition]
            freq, pw, amp = pt['freq'], pt['pw'], pt['amp']
        elif condition == 'SNN':
            inc = snn.get_action(obs['spike_matrix'])
            freq = max(0.0, min(180.0, freq + inc['freq'] * leap['freq']))
            pw = max(0.06, min(0.4, pw + inc['pw'] * leap['pw']))
            amp = max(0.0, min(250.0, amp + inc['amp'] * leap['amp']))
        else:
            raise ValueError(condition)

        obs = sim.step({'freq': freq, 'pw': pw, 'amp': amp})
        out['lfp'].append(float(obs['lfp']))
        out['freq'].append(float(freq))
        out['pw'].append(float(pw))
        out['amp'].append(float(amp))
        out['pd_state'].append(pd_state)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--protocol', choices=['acute', 'cycling'], required=True)
    ap.add_argument('--condition', required=True,
                    help='Unstimulated | SNN | cDBS (cycling only) | ' + ' | '.join(CONSTANT_POINTS))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--num-steps', type=int, default=None)
    ap.add_argument('--unseeded', action='store_true',
                    help='cycling only: replicate the published unseeded run (seed = repeat index)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/revision/constant_stim'))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = f'rep{args.seed}_unseeded' if args.unseeded else f'seed{args.seed}'
    path = os.path.join(args.out, f'{args.protocol}_{args.condition}_{tag}.json')
    if args.protocol == 'acute':
        res = run_acute(args.condition, args.seed, args.num_steps or 40)
    else:
        res = run_cycling(args.condition, args.seed, args.num_steps or 500, unseeded=args.unseeded)
    res.update({'protocol': args.protocol, 'condition': args.condition, 'seed': args.seed,
                'unseeded': bool(args.unseeded)})
    with open(path, 'w') as f:
        json.dump(res, f)
    print(f'saved {path}')


if __name__ == '__main__':
    main()
