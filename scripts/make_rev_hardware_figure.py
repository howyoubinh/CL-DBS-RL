"""
Hardware figure (Fig. 6) and the numbers for Table 1, from the HDK and microcontroller results:
  A  energy per 100 ms decision window: Xylo core at each clock (measured), Cortex-M7 MLP int8 /
     float32 and SNN (datasheet-estimated power x measured time), low-power M4F-class estimate
     from cycle counts, against the stimulator's energy per window at the clinical cDBS reference
     and under the learned policy.
  B  time per window against the 100 ms decision period: Xylo chip compute at each clock (fit) and the host
     protocol, MLP and SNN on the Cortex-M7.
Sources: the microcontroller and Xylo HDK benchmark notes in data/revision/mcu_results/ (2026-10-22)
and the HDK JSON files in the deployed student's xylo/ folder.

  python scripts/make_rev_hardware_figure.py  -> $CLDBS_FIG_DIR/fig_hardware_comparison.{pdf,png} (default
      figures/revision), data/revision/mcu_results/table1_numbers.json
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from make_rev_main_figures import C, panel_label, save  # noqa: E402

RUN = os.path.join(ROOT, 'models/revision/students_round4/xn_s0.03_w500/xylo')
WINDOW_S = 0.1
# --- microcontroller (STM32F746 benchmark, 2026-10-22): measured times, datasheet power 0.36 W active (109 mA @ 3.3 V)
MCU = {'P_active_W': 0.36, 'P_active_source': 'STM32F746 datasheet typical, estimate',
       'mlp_f32': {'latency_s': 1.037e-3, 'cycles': 223953, 'E_inf_J': 0.37e-3, 'E_cycle_J': 0.62e-3},
       'mlp_int8': {'latency_s': 0.311e-3, 'cycles': 67200, 'E_inf_J': 0.11e-3, 'E_cycle_J': 0.36e-3},
       'mlp_int16': {'latency_s': 1.153e-3, 'cycles': 249000},
       'snn_c': {'latency_s': 0.118, 'cycles': 25.5e6},
       'wake_s': 0.275e-3}
MCU['snn_c']['E_inf_J'] = MCU['P_active_W'] * MCU['snn_c']['latency_s']
LOWPOWER_PJ_PER_CYCLE = 33.0           # Apollo4-class Cortex-M4F, datasheet (estimate)
LOWPOWER_SLEEP_W = 10e-6 * 3.3         # ~10 uA deep sleep with retention (estimate)
# --- stimulator at the clinical cDBS reference (Methods, drawn from the device): 0.32 mW; learned policy = 10.1% of cDBS charge
STIM_CDBS_W, STIM_POLICY_FRAC = 0.32e-3, 0.101


def xylo():
    """Core power per clock (idle-dominated) and chip compute time from the monitoring-off fit."""
    timing = json.load(open(sorted(glob.glob(os.path.join(RUN, 'hdk_timing_*.json')))[-1]))
    power = {}
    for p in sorted(glob.glob(os.path.join(RUN, 'hdk_2026*.json'))):
        for clk, v in json.load(open(p))['clocks'].items():
            power[float(clk)] = v['power_W_inference']['DigitalLogic']
    lat = {float(k): v['latency_s']['mean'] for k, v in timing['clocks'].items()}
    f = np.array(sorted(lat)); y = np.array([lat[c] for c in f])
    A = np.c_[np.ones_like(f), 1 / f]; a, b = np.linalg.lstsq(A, y, rcond=None)[0]      # latency = a + b/f
    return power, lat, a, b


def main():
    power, lat, a, b = xylo()
    clocks = sorted(power)
    rows = {}
    for c in clocks:
        rows[f'xylo_{c:g}MHz'] = {'E_window_J': float(power[c] * WINDOW_S), 'P_core_W': float(power[c]), 'compute_s': float(b / c), 'real_time': bool(b / c <= WINDOW_S), 'source': 'measured (core power x 100 ms; compute from monitoring-off fit)'}
    rows['m7_mlp_int8'] = {'E_window_J': MCU['mlp_int8']['E_cycle_J'], 'E_inf_J': MCU['mlp_int8']['E_inf_J'], 'latency_s': MCU['mlp_int8']['latency_s'], 'source': 'measured time, datasheet power (estimate)'}
    rows['m7_mlp_f32'] = {'E_window_J': MCU['mlp_f32']['E_cycle_J'], 'E_inf_J': MCU['mlp_f32']['E_inf_J'], 'latency_s': MCU['mlp_f32']['latency_s'], 'source': 'measured time, datasheet power (estimate)'}
    rows['m7_snn_c'] = {'E_window_J': MCU['snn_c']['E_inf_J'], 'latency_s': MCU['snn_c']['latency_s'], 'real_time': False, 'source': 'measured time (unoptimized C), datasheet power (estimate)'}
    for k, cyc in (('m4f_mlp_int8', MCU['mlp_int8']['cycles']), ('m4f_mlp_f32', MCU['mlp_f32']['cycles'])):
        rows[k] = {'E_window_J': cyc * LOWPOWER_PJ_PER_CYCLE * 1e-12 + LOWPOWER_SLEEP_W * WINDOW_S, 'E_inf_J': cyc * LOWPOWER_PJ_PER_CYCLE * 1e-12, 'source': 'estimate: measured cycles x 33 pJ/cycle + 10 uA sleep'}
    rows['stim_cdbs'] = {'E_window_J': STIM_CDBS_W * WINDOW_S, 'source': 'clinical cDBS reference, drawn from the device (Methods)'}
    rows['stim_policy'] = {'E_window_J': STIM_CDBS_W * STIM_POLICY_FRAC * WINDOW_S, 'source': 'cDBS reference x measured charge fraction of the deployed student (10.1%)'}
    os.makedirs(os.path.join(ROOT, 'data/revision/mcu_results'), exist_ok=True)
    json.dump({'rows': rows, 'xylo_latency_fit': {'host_overhead_s': a, 'compute_s_times_MHz': b}, 'mcu': MCU}, open(os.path.join(ROOT, 'data/revision/mcu_results/table1_numbers.json'), 'w'), indent=1)
    for k, v in rows.items():
        print(f'{k:16s} {v["E_window_J"] * 1e6:9.1f} uJ/window   {v["source"]}')

    # ---------------------------------------------------------------- figure: every label >= 7 pt at 7.2 in
    TICK = 7
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.6), gridspec_kw={'wspace': 0.62, 'width_ratios': [2.0, 1.45], 'left': 0.08, 'right': 0.98, 'bottom': 0.2, 'top': 0.93})

    def group_headers(ax, groups):
        """Device name under a group of tick labels (second row of the x axis)."""
        for xc, name in groups:
            ax.annotate(name, (xc, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -30), textcoords='offset points', ha='center', va='top', fontsize=TICK, color=C['ink'])

    ax = axes[0]; panel_label(ax, 'A')
    xa = [0, 1, 2, 3.5, 4.5, 5.5, 7.0]
    items = [(f'{c:g}\nMHz', rows[f'xylo_{c:g}MHz']['E_window_J'], C['blue'], 'measured') for c in clocks if c in (6.25, 12.5, 25.0)]
    items += [('MLP\nint8', rows['m7_mlp_int8']['E_window_J'], C['ink2'], 'estimate'), ('MLP\nfloat32', rows['m7_mlp_f32']['E_window_J'], C['ink2'], 'estimate'),
              ('SNN\nin C', rows['m7_snn_c']['E_window_J'], C['ink2'], 'estimate'), ('MLP\nint8', rows['m4f_mlp_int8']['E_window_J'], C['ink2'], 'estimate')]
    for x, (name, e, col, kind) in zip(xa, items):
        ax.bar(x, e * 1e6, color=col, alpha=1.0 if kind == 'measured' else 0.45, width=0.7, hatch='' if kind == 'measured' else '//', edgecolor=col, linewidth=0.6)
    ax.axhline(rows['stim_cdbs']['E_window_J'] * 1e6, color=C['orange'], lw=1.0, ls='--')
    ax.axhline(rows['stim_policy']['E_window_J'] * 1e6, color=C['orange'], lw=1.0, ls=':')
    ax.set_yscale('log'); ax.set_ylim(1, 3e5); ax.set_xlim(-0.7, 7.7); ax.set_ylabel('Energy per 100 ms decision window (µJ)')
    ax.set_xticks(xa); ax.set_xticklabels([n for n, *_ in items], fontsize=TICK)
    [t.set_fontweight('bold') for t in ax.get_xticklabels() if '12.5' in t.get_text()]
    group_headers(ax, [(1.0, 'Xylo SNN'), (4.5, 'Cortex-M7'), (7.0, 'M4F-class')])
    handles = [Patch(facecolor=C['blue'], edgecolor=C['blue'], label='Measured'),
               Patch(facecolor=C['ink2'], edgecolor=C['ink2'], alpha=0.45, hatch='//', label='Estimate'),
               Line2D([], [], color=C['orange'], lw=1.0, ls='--', label='Stimulation, cDBS'),
               Line2D([], [], color=C['orange'], lw=1.0, ls=':', label='Stimulation,\nlearned policy')]
    ax.legend(handles=handles, loc='upper left', fontsize=TICK, handlelength=1.6, handletextpad=0.5, borderaxespad=0.1, labelspacing=0.45)

    ax = axes[1]; panel_label(ax, 'B', x=-0.62)
    # time per 100-step window, one row per configuration; left of the dashed line = within the decision period
    rows_b = [(f'Xylo core, {c:g} MHz', b / c * 1e3, C['blue'], 'o') for c in clocks] + [('Xylo host protocol', a * 1e3, C['ink2'], 's'),
              ('Cortex-M7, MLP int8', MCU['mlp_int8']['latency_s'] * 1e3, C['ink2'], 'o'), ('Cortex-M7, MLP float32', MCU['mlp_f32']['latency_s'] * 1e3, C['ink2'], 'o'),
              ('Cortex-M7, MLP int16', MCU['mlp_int16']['latency_s'] * 1e3, C['ink2'], 'o'), ('Cortex-M7, SNN in C', MCU['snn_c']['latency_s'] * 1e3, C['ink2'], 'o')]
    yb = np.arange(len(rows_b))[::-1].astype(float); yb[:4] += 0.4; yb[4] += 0.2
    for y, (name, t, col, mk) in zip(yb, rows_b):
        ax.hlines(y, 0.1, t, color=col, lw=1.0, alpha=0.5); ax.plot(t, y, mk, color=col, ms=5.5)
        if name.startswith('Xylo core'):
            ax.text(t, y + 0.28, f'{t:.0f} ms', fontsize=TICK, color=C['ink2'], ha='right', va='bottom')
    ax.axvline(100, color=C['orange'], lw=0.9, ls='--'); ax.text(118, yb[0] + 0.9, 'Decision\nperiod', fontsize=TICK, color=C['orange'], ha='left', va='center')
    ax.set_xscale('log'); ax.set_xlim(0.1, 1000); ax.set_ylim(-0.7, yb[0] + 1.9); ax.set_xlabel('Time per 100-step window (ms)')
    ax.set_yticks(yb); ax.set_yticklabels([r[0] for r in rows_b], fontsize=TICK); ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
    ax.set_xticks([0.1, 1, 10, 100, 1000]); ax.set_xticklabels(['0.1', '1', '10', '100', '1000'])
    [t.set_fontweight('bold') for t in ax.get_yticklabels() if '12.5' in t.get_text()]
    save(fig, 'fig_hardware_comparison')


if __name__ == '__main__':
    main()
