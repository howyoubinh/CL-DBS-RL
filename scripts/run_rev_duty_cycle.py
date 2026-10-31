"""
Duty-cycle loophole check for the smoothed-beta reward (--env round3): that reward averages beta
over the last 1 s, and PD beta stays suppressed for ~0.5 s after stimulation stops. Could an
intermittent pattern score higher than steady stimulation on the reward while leaving raw beta
pathological in the off phases? Simulates fixed on/off patterns in PD on a seeded patient and
records beta per 100 ms window and the setting applied.

Patterns (amplitude A in uA at 180 Hz / 0.06 ms; cycle = on_windows + off_windows, in 100 ms):
  steady         always on
  d5_5           0.5 s on, 0.5 s off
  d3_7, d7_3     0.3 s on / 0.7 s off, and the reverse
  d2_8           0.2 s on / 0.8 s off
  d10_10         1 s on, 1 s off
  d1_1           alternate every window

  python scripts/run_rev_duty_cycle.py --seed 42 --amp 212 --pattern d5_5 --out data/revision/duty
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.simulation.sim_wrapper import SimulationWrapper  # noqa: E402

PATTERNS = {'steady': (1, 0), 'd5_5': (5, 5), 'd3_7': (3, 7), 'd7_3': (7, 3), 'd2_8': (2, 8),
            'd10_10': (10, 10), 'd1_1': (1, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--amp', type=float, required=True)
    ap.add_argument('--pattern', choices=list(PATTERNS), required=True)
    ap.add_argument('--seconds', type=float, default=30.0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/revision/duty'))
    args = ap.parse_args()

    on, off = PATTERNS[args.pattern]
    n = int(round(args.seconds / 0.1))
    sim = SimulationWrapper(n_neurons=10, t_step=100, dt=0.01, seed=args.seed)
    sim.pd_state = 1
    sim.reset()
    beta, stim = np.zeros(n), np.zeros((n, 3))
    for w in range(n):
        is_on = off == 0 or (w % (on + off)) < on
        setting = (180.0, 0.06, args.amp) if is_on else (0.0, 0.0, 0.0)
        obs = sim.step({'freq': setting[0], 'pw': setting[1], 'amp': setting[2]})
        beta[w], stim[w] = float(obs['lfp']), setting
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'duty_{args.pattern}_a{args.amp:g}_seed{args.seed}.npz')
    np.savez_compressed(path, beta=beta, stim=stim,
                        meta=json.dumps({'seed': args.seed, 'amp': args.amp, 'pattern': args.pattern,
                                         'on_windows': on, 'off_windows': off, 'seconds': args.seconds}))
    print(f'saved {path}: beta median {np.median(beta[10:]):.0f}')


if __name__ == '__main__':
    main()
