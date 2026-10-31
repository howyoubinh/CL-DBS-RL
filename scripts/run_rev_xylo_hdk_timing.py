"""
Chip-side timing of the deployed student on the XyloAudio 3 HDK: accelerated-time mode with
state monitoring switched off (Spike readout, no recording), so the only traffic after the input
events is the end-of-window handshake. This isolates the chip's processing time from the Vmem
streaming that dominates the Vmem-readout measurement of run_rev_xylo_hdk.py.

Per clock: loads the configuration, settles, runs --n windows back to back (state carried, as on
the chip in continuous operation), reports mean/median/min latency per 100-step window and the
mean core/IO/AFE power during the run and at idle. Writes <run>/xylo/hdk_timing_<stamp>.json.

  python scripts/run_rev_xylo_hdk_timing.py --run models/revision/students_round4/xn_s0.03_w500 --clocks 6.25 25 50 --n 20
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
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from run_rev_xylo_hdk import load_spec, make_config, power_by_channel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--windows', default=os.path.join(ROOT, 'data/revision/mcu_package/snn_reference.npz'))
    ap.add_argument('--clocks', type=float, nargs='+', default=[6.25, 25.0, 50.0])
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--settle', type=float, default=30.0)
    args = ap.parse_args()
    from rockpool.devices.xylo import find_xylo_hdks
    from rockpool.devices.xylo.syns65302 import xa3_devkit_utils as u, XyloSamna
    _orig = u.get_current_timestep
    def _patched(*a, **k):
        k.setdefault('timeout', 30.0); return _orig(*a, **k)
    u.get_current_timestep = _patched
    spec = load_spec(os.path.join(args.run, 'xylo', 'hw_config.json')); dt = float(spec.get('dt', 10e-3)); cfg = make_config(spec)
    d = np.load(args.windows); X = [np.round(x * 5).astype(int) for x in d['X16'][:args.n]]
    hdks, _, _ = find_xylo_hdks(); hdk = hdks[0]
    res = {'run': args.run, 'mode': 'accelerated time, Spike readout, no state recording', 'n_windows': len(X), 'when': datetime.now().isoformat(timespec='seconds'), 'clocks': {}}
    print(f'{"clock MHz":>9s} {"lat mean":>9s} {"lat med":>8s} {"lat min":>8s} {"core mW":>8s} {"core idle":>9s} {"io mW":>7s} {"afe mW":>7s} {"E/window uJ (P_core x lat)":>26s} {"E/100ms uJ (P_core x 0.1 s)":>27s}')
    for clk in args.clocks:
        u.set_xylo_core_clock_freq(hdk, clk)
        mod = XyloSamna(hdk, cfg, dt=dt, output_mode='Spike', power_frequency=100.0)
        time.sleep(args.settle)
        for x in X[:3]:
            mod(x, record_power=False)
        mod._power_buf.get_events()
        lat = []
        for x in X:
            t0 = time.perf_counter(); mod(x, record_power=False); lat.append(time.perf_counter() - t0)
        p_inf = power_by_channel(mod._power_buf.get_events())
        time.sleep(0.5); mod._power_buf.get_events(); time.sleep(3.0)
        p_idle = power_by_channel(mod._power_buf.get_events())
        lat = np.array(lat); core = p_inf['DigitalLogic']
        res['clocks'][str(clk)] = {'latency_s': {'mean': float(lat.mean()), 'median': float(np.median(lat)), 'min': float(lat.min()), 'max': float(lat.max())},
                                   'power_W_processing': p_inf, 'power_W_idle': p_idle, 'energy_per_window_J_core_x_latency': float(core * lat.mean()), 'energy_per_100ms_J_core_x_0.1': float(core * 0.1)}
        print(f'{clk:9.2f} {lat.mean() * 1e3:8.1f}ms {np.median(lat) * 1e3:7.1f}ms {lat.min() * 1e3:7.1f}ms {core * 1e3:8.3f} {p_idle["DigitalLogic"] * 1e3:9.3f} {p_inf["Io"] * 1e3:7.3f} {p_inf["AnalogLogic"] * 1e3:7.3f} {core * lat.mean() * 1e6:26.1f} {core * 0.1 * 1e6:27.1f}')
        del mod
    out = os.path.join(args.run, 'xylo', f'hdk_timing_{datetime.now().strftime("%Y%m%d_%H%M")}.json')
    json.dump(res, open(out, 'w'), indent=1); print('wrote', out)


if __name__ == '__main__':
    main()
