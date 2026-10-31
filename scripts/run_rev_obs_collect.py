"""
Observability check, data collection: can the disease state (PD vs healthy) be read from the 80
recorded neurons while stimulation is on?

The controller acts while stimulating, and stimulation suppresses the GPi biomarker in both
states, so separability without stimulation does not settle whether a stimulating policy can tell
when to back off. This script records the 80-neuron observation under several stimulation
conditions, for both states, on a fixed network (seeded patient), continuously in 100 ms simulator
calls exactly as the controller sees them (SimulationWrapper, state carried over).

Conditions:
  unstim              no stimulation
  c212 / c250         constant 180 Hz / 0.06 ms / 212 or 250 uA (the learned operating points)
  cdbs                130 Hz / 0.3 ms / 300 uA
  random              piecewise-constant random settings, redrawn every --hold-s seconds
                      (freq 0-180 Hz in 5 Hz steps, pw 0.06-0.4 ms, amp 0-250 uA; off with p=0.2),
                      i.e. the range an exploring policy visits

Saved per job (npz): counts (windows, 100, 80) uint8 spike counts in 1 ms bins (constructor_hash
population order: TH, STN, GPe, GPi, Str-D2, Str-D1, Cortex x 20), beta (windows,) GPi alpha-beta
area per 100 ms window, stim (windows, 3) freq/pw/amp, and the metadata.

  python scripts/run_rev_obs_collect.py --seed 101 --pd 1 --condition c250 --out data/revision/obs
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.environment.gym_pd_rev import count_bin  # noqa: E402
from src.simulation.sim_wrapper import SimulationWrapper  # noqa: E402

FIXED = {'unstim': (0.0, 0.0, 0.0), 'c212': (180.0, 0.06, 212.0), 'c250': (180.0, 0.06, 250.0),
         'cdbs': (130.0, 0.3, 300.0)}


def random_setting(rng):
    if rng.random() < 0.2:
        return 0.0, 0.0, 0.0
    return float(5 * rng.integers(0, 37)), float(rng.uniform(0.06, 0.4)), float(rng.uniform(0.0, 250.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, required=True, help='patient (network) seed')
    ap.add_argument('--pd', type=int, choices=[0, 1], required=True)
    ap.add_argument('--condition', choices=list(FIXED) + ['random'], required=True)
    ap.add_argument('--seconds', type=float, default=60.0)
    ap.add_argument('--hold-s', type=float, default=1.0, help='random condition: seconds per setting')
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/revision/obs'))
    args = ap.parse_args()

    n_win = int(round(args.seconds / 0.1))
    hold = int(round(args.hold_s / 0.1))
    rng = np.random.default_rng(1_000_003 * args.seed + 10 * args.pd + 7)
    sim = SimulationWrapper(n_neurons=10, t_step=100, dt=0.01, seed=args.seed)
    sim.pd_state = args.pd
    sim.reset()   # one unstimulated 100 ms call to prime the state (not recorded)

    counts = np.zeros((n_win, 100, 80), np.uint8)
    beta = np.zeros(n_win)
    stim = np.zeros((n_win, 3))
    setting = FIXED.get(args.condition)
    for w in range(n_win):
        if args.condition == 'random' and w % hold == 0:
            setting = random_setting(rng)
        f, pw, a = setting
        obs = sim.step({'freq': f, 'pw': pw, 'amp': a})
        c = count_bin(obs['raw_vars'], 100, 0.1)
        assert c.shape == (100, 80) and c.max() < 256, c.shape
        counts[w] = c
        beta[w] = float(obs['lfp'])
        stim[w] = setting

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'obs_{args.condition}_{"pd" if args.pd else "healthy"}_seed{args.seed}.npz')
    np.savez_compressed(path, counts=counts, beta=beta, stim=stim,
                        meta=json.dumps({'seed': args.seed, 'pd': args.pd, 'condition': args.condition,
                                         'seconds': args.seconds, 'hold_s': args.hold_s, 'window_ms': 100}))
    rates = counts.sum(1).mean(0) / 0.1
    print(f'saved {path}: {n_win} windows, beta median {np.median(beta):.0f}, '
          f'mean rate {rates.mean():.1f} Hz, zero-beta windows {(beta == 0).sum()}')


if __name__ == '__main__':
    main()
