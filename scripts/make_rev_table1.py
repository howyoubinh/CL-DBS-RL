"""
Recompute the implant power budget (manuscript Table 1) from stimulation trajectories and measured
inference power, instead of by hand.

Stimulation power follows the manuscript's accounting:
  P_tissue(cDBS reference) = V^2 * f * PW / Z with the clinical reference 130 Hz, 90 us, 3.5 V, 1 kOhm
  P_drawn = P_tissue / eta_driver (eta = 0.45)
  For current-controlled stimulation V = I * Z, so TEED scales as I^2 * f * PW; each controller's
  stimulation power is the reference scaled by its TEED ratio to the simulated cDBS condition
  (130 Hz, 0.3 ms, 300 uA). Charge ratio (I * f * PW) is reported alongside.
Total = stimulation (drawn) + inference; battery life for a 5 Wh cell.

Inputs:
  --csv       an on_off_results CSV (columns <COND>_Freq/_PW/_Amp) from scripts/run_on_off_experiment.py, or
  --eval-dir  run_rev_eval.py cycling outputs (one JSON per controller and seed; seeds are averaged)
  --inference LABEL=mW ...   measured inference power per controller label

  # regression check on the 06-01-2026 run_on_off_experiment.py output (expects SNN TEED -85.6%, RNN +28.2%)
  python scripts/make_rev_table1.py --csv data/on_off_results/on_off_results_06-01-2026_20-31-09.csv \
      --controllers SNN RNN ANN --inference SNN=0.52 ANN=5703 RNN=5592
"""

import argparse
import glob
import json
import os

import numpy as np

V_REF, F_REF, PW_REF, Z, ETA, BATTERY_WH = 3.5, 130.0, 90e-6, 1000.0, 0.45, 5.0
SIM_CDBS = (130.0, 0.3, 300.0)


def ratios(f, pw, a):
    f, pw, a = (np.asarray(x, dtype=float) for x in (f, pw, a))
    cf, cpw, ca = SIM_CDBS
    teed = np.mean(a ** 2 * f * pw) / (ca ** 2 * cf * cpw)
    charge = np.mean(a * f * pw) / (ca * cf * cpw)
    return float(teed), float(charge)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv')
    ap.add_argument('--eval-dir')
    ap.add_argument('--controllers', nargs='+', required=True, help='CSV condition names or eval labels')
    ap.add_argument('--inference', nargs='*', default=[], help='LABEL=mW')
    ap.add_argument('--window-ms', type=int, default=None, help='eval-dir: restrict to one window length')
    ap.add_argument('--out', default=None, help='write JSON summary here')
    args = ap.parse_args()

    p_tissue = V_REF ** 2 * F_REF * PW_REF / Z * 1e3            # mW
    p_ref = p_tissue / ETA
    inf = {k: float(v) for k, v in (x.split('=') for x in args.inference)}
    rows = {}
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        for c in args.controllers:
            rows[c] = ratios(df[f'{c}_Freq'], df[f'{c}_PW'], df[f'{c}_Amp'])
    else:
        for c in args.controllers:
            files = glob.glob(os.path.join(args.eval_dir, f'cycling_{c}_w*_seed*.json'))
            if args.window_ms:
                files = [f for f in files if f'_w{args.window_ms}_' in f]
            per_seed = [ratios(d['freq'], d['pw'], d['amp']) for d in (json.load(open(f)) for f in files)]
            rows[c] = tuple(np.mean(per_seed, axis=0)) if per_seed else (float('nan'), float('nan'))

    print(f'cDBS reference: P_tissue {p_tissue:.3f} mW, drawn {p_ref:.3f} mW (eta {ETA})')
    print(f'{"controller":14s} {"TEED vs cDBS":>13s} {"charge vs cDBS":>15s} {"stim drawn (mW)":>16s} '
          f'{"inference (mW)":>15s} {"total (mW)":>11s} {"battery (5 Wh)":>15s}')
    summary = {'p_tissue_mw': p_tissue, 'p_ref_drawn_mw': p_ref, 'rows': {}}
    for c, (teed, charge) in rows.items():
        stim = p_ref * teed
        total = stim + inf.get(c, float('nan'))
        hours = BATTERY_WH * 1e3 / total if total == total else float('nan')
        life = f'{hours / 8760:.2f} yr' if hours >= 8760 else f'{hours:.1f} h'
        print(f'{c:14s} {100 * (teed - 1):+12.1f}% {100 * (charge - 1):+14.1f}% {stim:16.3f} '
              f'{inf.get(c, float("nan")):15.2f} {total:11.2f} {life:>15s}')
        summary['rows'][c] = {'teed_ratio': teed, 'charge_ratio': charge, 'stim_drawn_mw': stim,
                              'inference_mw': inf.get(c), 'total_mw': total, 'battery_hours': hours}
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
