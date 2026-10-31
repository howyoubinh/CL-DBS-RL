"""
Xylo HDK measurement of the deployed student: latency, power, energy and chip-vs-simulator fidelity.

For each core clock requested: loads <run>/xylo/hw_config.json on the XyloAudio 3 HDK (Vmem
readout, state reset per window), runs the recorded test windows, and reports
  - latency per 100-step window (host round trip, mean / median / max),
  - mean power per channel (Io, AnalogLogic, DigitalLogic = SNN core) during inference and idle,
  - energy per inference (core power x latency) and energy per 100 ms decision window
    (active energy + idle energy for the rest of the window when latency < 100 ms),
  - fidelity: chip Q (sum of output Vmem) vs bit-accurate XyloSim and vs the float32 expected
    decisions from the package.
Writes <run>/xylo/hdk_<timestamp>.json and prints a table.

  python scripts/run_rev_xylo_hdk.py --run models/revision/students_round4/xn_s0.03_w500 \
      --windows data/revision/mcu_package/snn_reference.npz --clocks 6.25 12.5 25 50 --n 100
  python scripts/run_rev_xylo_hdk.py --run ... --sim-only     # smoke test without the HDK

Requires samna + rockpool with the HDK connected (not needed for --sim-only).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def heads(q):
    return np.stack([q[..., 3 * j:3 * j + 3].argmax(-1) for j in range(3)], -1)


def load_spec(hw):
    return {k: (np.array(v) if isinstance(v, list) else v) for k, v in json.load(open(hw)).items()}


def make_config(spec):
    import rockpool.devices.xylo.syns65302 as xa3
    cfg = xa3.config_from_specification(**spec)
    return cfg[0] if isinstance(cfg, tuple) else cfg


def run_xylosim(cfg, dt, X, carry=False):
    """Bit-accurate simulator reference. carry=False: state reset before every window (the
    evaluation convention). carry=True: windows evolved back to back with state carried, which is
    what the chip does here, since XyloSamna.reset_state() is a no-op in this Rockpool version."""
    from rockpool.devices.xylo.syns65302 import XyloSim
    q = []
    mod = XyloSim.from_config(cfg, dt=dt, output_mode='Vmem'); mod.reset_state()
    for x in X:
        if not carry:
            mod = XyloSim.from_config(cfg, dt=dt, output_mode='Vmem'); mod.reset_state()
        out, _, _ = mod(x)
        q.append(np.asarray(out).sum(0))
    return np.stack(q)


def power_by_channel(events):
    import samna
    ch = samna.xyloAudio3.MeasurementChannels
    out = {}
    for name in ('Io', 'AnalogLogic', 'DigitalLogic'):
        vals = [e.value for e in events if e.channel == int(getattr(ch, name))]
        out[name] = float(np.mean(vals)) if vals else float('nan')
        out[name + '_n'] = len(vals)
    return out


def run_hdk(cfg, dt, X, clock_mhz, warmup=5, power_hz=100.0, idle_s=3.0, settle_s=30.0, reset_per_window=False):
    from rockpool.devices.xylo import find_xylo_hdks
    from rockpool.devices.xylo.syns65302 import xa3_devkit_utils as xa3utils, XyloSamna
    # The chip needs time after a large configuration is loaded (16k hidden synapses) before its
    # timestep register answers; the default 3 s read timeout then fails and wedges the board.
    if not getattr(xa3utils.get_current_timestep, '_patched', False):
        _orig = xa3utils.get_current_timestep
        def _patched(*a, **k):
            k.setdefault('timeout', 30.0); return _orig(*a, **k)
        _patched._patched = True; xa3utils.get_current_timestep = _patched
    hdks, _, _ = find_xylo_hdks()
    if not hdks:
        raise RuntimeError('No Xylo HDK found')
    hdk = hdks[0]
    xa3utils.set_xylo_core_clock_freq(hdk, clock_mhz)
    mod = XyloSamna(hdk, cfg, dt=dt, output_mode='Vmem', power_frequency=power_hz)
    time.sleep(settle_s)

    def reset():
        if reset_per_window and hasattr(mod, 'reset_state'):
            mod.reset_state()

    for x in X[:warmup]:
        reset(); mod(x, record_power=False)
    mod._power_buf.get_events()
    q, lat = [], []
    for x in X:
        reset()
        t0 = time.perf_counter(); out, _, _ = mod(x, record_power=False); lat.append(time.perf_counter() - t0)
        q.append(np.asarray(out).sum(0))
    p_inf = power_by_channel(mod._power_buf.get_events())
    time.sleep(0.5); mod._power_buf.get_events(); time.sleep(idle_s)
    p_idle = power_by_channel(mod._power_buf.get_events())
    return np.stack(q), np.array(lat), p_inf, p_idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--windows', default=os.path.join(ROOT, 'data/revision/mcu_package/snn_reference.npz'))
    ap.add_argument('--clocks', type=float, nargs='+', default=[6.25, 12.5, 25.0, 50.0], help='core clock MHz')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--sim-only', action='store_true')
    ap.add_argument('--settle', type=float, default=30.0, help='seconds to wait after loading the configuration')
    ap.add_argument('--reset-per-window', action='store_true', help='call reset_state() on the chip before every window (slow if it reloads the configuration)')
    args = ap.parse_args()
    hw = os.path.join(args.run, 'xylo', 'hw_config.json')
    spec = load_spec(hw); dt = float(spec.get('dt', 10e-3)); cfg = make_config(spec)
    d = np.load(args.windows)
    X16 = d['X16'][:args.n]                              # (n, 100, 16) pooled means (multiples of 0.2)
    X = [np.round(x * 5).astype(int) for x in X16]        # integer sum-pool counts for the chip
    q_fp = d['Q'][:args.n]                                # float32 reference (numpy)
    print(f'{args.run}: {len(X)} windows; clocks {args.clocks} MHz; sim-only={args.sim_only}')
    q_sim = run_xylosim(cfg, dt, X)                       # reset per window (evaluation convention)
    q_simc = run_xylosim(cfg, dt, X, carry=True)          # state carried across windows (what the chip does)
    a_sim = (heads(q_sim) == heads(q_fp)).all(1).mean(); a_simc = (heads(q_simc) == heads(q_fp)).all(1).mean()
    print(f'XyloSim vs float32: joint action agreement {a_sim:.2f} (reset per window), {a_simc:.2f} (state carried)')
    res = {'run': args.run, 'hw_config': hw, 'n_windows': len(X), 'when': datetime.now().isoformat(timespec='seconds'),
           'xylosim_vs_fp32_agreement': float(a_sim), 'xylosim_carried_vs_fp32_agreement': float(a_simc), 'chip_state': 'carried across windows (reset_state is a no-op in rockpool 3.0.1)', 'clocks': {}}
    if args.sim_only:
        print('sim-only: no HDK measurement'); json.dump(res, open(os.path.join(args.run, 'xylo', 'hdk_simonly.json'), 'w'), indent=1); return
    print(f'{"clock MHz":>9s} {"lat mean":>9s} {"lat med":>8s} {"lat max":>8s} {"core mW":>8s} {"io mW":>7s} {"afe mW":>7s} {"core idle":>9s} {"E/inf uJ":>9s} {"E/100ms uJ":>10s} {"=sim":>5s} {"=fp32":>6s}')
    for clk in args.clocks:
        q_chip, lat, p_inf, p_idle = run_hdk(cfg, dt, X, clk, settle_s=args.settle, reset_per_window=args.reset_per_window)
        core_w, core_idle_w = p_inf['DigitalLogic'], p_idle['DigitalLogic']
        lat_m = float(lat.mean())
        e_inf = core_w * lat_m
        e_win = e_inf + core_idle_w * max(0.0, 0.1 - lat_m) if lat_m <= 0.1 else e_inf   # per 100 ms decision window
        same_sim = float((heads(q_chip) == heads(q_simc)).all(1).mean()); same_fp = float((heads(q_chip) == heads(q_fp)).all(1).mean())
        exact = float(np.mean(np.abs(q_chip - q_simc) < 1e-6))
        corr = [float(np.corrcoef(q_chip[:, j], q_simc[:, j])[0, 1]) for j in range(9)] if len(X) > 2 else []
        np.savez(os.path.join(args.run, 'xylo', f'hdk_q_{clk:g}MHz_{datetime.now().strftime("%Y%m%d_%H%M")}.npz'), q_chip=q_chip, q_sim_reset=q_sim, q_sim_carried=q_simc, q_fp32=q_fp, latency=lat)
        res['clocks'][str(clk)] = {'latency_s': {'mean': lat_m, 'median': float(np.median(lat)), 'max': float(lat.max())}, 'q_corr_chip_vs_xylosim_carried': corr,
                                   'power_W_inference': p_inf, 'power_W_idle': p_idle, 'energy_per_inference_J': e_inf,
                                   'energy_per_100ms_window_J': e_win, 'real_time': bool(lat_m <= 0.1),
                                   'chip_vs_xylosim_agreement': same_sim, 'chip_vs_xylosim_exact_q_fraction': exact, 'chip_vs_fp32_agreement': same_fp}
        print(f'{clk:9.2f} {lat_m * 1e3:8.1f}ms {np.median(lat) * 1e3:7.1f}ms {lat.max() * 1e3:7.1f}ms {core_w * 1e3:8.3f} {p_inf["Io"] * 1e3:7.3f} {p_inf["AnalogLogic"] * 1e3:7.3f} '
              f'{core_idle_w * 1e3:9.3f} {e_inf * 1e6:9.1f} {e_win * 1e6:10.1f} {same_sim:5.2f} {same_fp:6.2f}' + (f'  Q corr min/med {min(corr):.3f}/{np.median(corr):.3f}' if corr else ''))
    out = os.path.join(args.run, 'xylo', f'hdk_{datetime.now().strftime("%Y%m%d_%H%M")}.json')
    json.dump(res, open(out, 'w'), indent=1); print('wrote', out)


if __name__ == '__main__':
    main()
