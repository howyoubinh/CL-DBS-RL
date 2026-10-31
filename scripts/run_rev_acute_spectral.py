"""
Acute-protocol recording with spectra for the deployed student (data for the supplementary
spectral figure).

For each patient and condition, runs the 40 s acute protocol (sustained parkinsonian state, 400
decisions of 100 ms) with the same harness components as run_rev_eval.py and records, per window,
the biomarker, the stimulation parameters, the 80-channel spike matrix and the simulator's
multi-taper GPi spectrum (gpi_S, gpi_f). Output: <out>/acute_<label>_seed<N>.npz.

  python scripts/run_rev_acute_spectral.py --seeds 42 123 456 789 1024 2048 3141 5678 8192 9999 \
      --controllers unstim xylosim:models/revision/students_round4/xn_s0.03_w500 "constant:180,0.06,212"
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from run_rev_eval import make_controller  # noqa: E402
from src.simulation.sim_wrapper import SimulationWrapper  # noqa: E402


def run(spec, seed, duration_s, out_dir):
    ctrl, win, binning, label = make_controller(spec)
    win = win or 100
    path = os.path.join(out_dir, f'acute_{label}_seed{seed}.npz')
    if os.path.exists(path):
        print('exists', path); return
    sim = SimulationWrapper(n_neurons=10, t_step=win, dt=0.01, seed=seed)
    sim.pd_state = 1
    obs = sim.reset()
    n = int(round(duration_s / (win / 1000)))
    lfp, freq, pw, amp, spikes, S, F = [], [], [], [], [], [], []
    for _ in range(n):
        if hasattr(ctrl, 'observe'):
            ctrl.observe(float(obs['lfp']))
        a = ctrl.get_action(np.asarray(obs['spike_matrix'], dtype=np.float64))
        obs = sim.step(a)
        lfp.append(float(obs['lfp'])); freq.append(a['freq']); pw.append(a['pw']); amp.append(a['amp'])
        spikes.append(np.asarray(obs['spike_matrix'], dtype=np.uint8))
        s = obs['raw_vars'].get('gpi_S'); f = obs['raw_vars'].get('gpi_f')
        S.append(np.asarray(s, dtype=np.float32) if s is not None else np.zeros(0, np.float32))
        F.append(np.asarray(f, dtype=np.float32) if f is not None else np.zeros(0, np.float32))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(path, lfp=np.array(lfp), freq=np.array(freq), pw=np.array(pw), amp=np.array(amp),
                        spikes=np.stack(spikes), gpi_S=np.array(S, dtype=object), gpi_f=np.array(F, dtype=object),
                        label=label, seed=seed, window_ms=win)
    print('saved', path, f'mean beta {np.mean(lfp):.0f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024, 2048, 3141, 5678, 8192, 9999])
    ap.add_argument('--controllers', nargs='+', default=['unstim', 'xylosim:models/revision/students_round4/xn_s0.03_w500', 'constant:180,0.06,212'])
    ap.add_argument('--duration-s', type=float, default=40.0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/revision/final_acute_spectral'))
    args = ap.parse_args()
    for seed in args.seeds:
        for spec in args.controllers:
            run(spec, seed, args.duration_s, args.out)


if __name__ == '__main__':
    main()
