#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Experiment 1 Figure: Therapeutic Efficacy (Spectrograms & PSD)

Re-plots Fig 1 from a saved NPZ WITHOUT re-running the simulation — useful for
iterating on plot visuals. To guarantee it renders identically to the
experiment script (and never drifts), it imports and calls the SAME
generate_figure1() from run_exp1_therapeutic_efficacy.py rather than keeping a
parallel copy of the plotting code.

The NPZ (written by run_exp1_therapeutic_efficacy.py) stores every plotted seed
under keys s{i}_{cond}_{field}; this script reconstructs the per-seed result
dicts and the aggregate stats, then hands them to generate_figure1 exactly as
the experiment script does.

Usage:
    python scripts/generate_exp1_figure.py data/exp1_results/exp1_data_*.npz
    python scripts/generate_exp1_figure.py data/exp1_results/exp1_data_*.npz --output-dir figures/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import argparse

# Single source of truth for the figure — same code the experiment uses.
from scripts.run_exp1_therapeutic_efficacy import generate_figure1


def _build_cond(data, prefix):
    """Reconstruct one condition's result dict from NPZ keys at `prefix`."""
    def get(field, default=None):
        key = prefix + field
        return data[key] if key in data.files else default

    spikes_raw = get('spikes')
    spikes = list(spikes_raw) if isinstance(spikes_raw, np.ndarray) else []
    mt_S = get('gpi_mt_S')
    mt_f = get('gpi_mt_f')
    return {
        'lfp': get('lfp', np.array([])),
        'stim': {
            'freq': list(get('freq', [])),
            'pw': list(get('pw', [])),
            'amp': list(get('amp', [])),
        },
        'spikes': spikes,
        'gpi_mt_S': list(mt_S) if isinstance(mt_S, np.ndarray) else None,
        'gpi_mt_f': list(mt_f) if isinstance(mt_f, np.ndarray) else None,
    }


def load_all_seeds_npz(npz_path):
    """Load every saved seed as a list of per-seed result dicts.

    Supports the multi-seed schema (s{i}_{cond}_{field} + n_seeds) and falls
    back to the legacy single-seed schema ({cond}_{field}).
    """
    data = np.load(npz_path, allow_pickle=True)
    files = set(data.files)

    if 'n_seeds' in files:
        n = int(data['n_seeds'])
        all_results = []
        for i in range(n):
            prefix_i = f's{i}_'
            conds = [k[len(prefix_i):-len('_lfp')] for k in data.files
                     if k.startswith(prefix_i) and k.endswith('_lfp')]
            seed_res = {c: _build_cond(data, f's{i}_{c}_') for c in conds}
            all_results.append(seed_res)
        return all_results

    # Legacy single-seed NPZ (unprefixed keys)
    conds = [k[:-len('_lfp')] for k in data.files if k.endswith('_lfp')]
    return [{c: _build_cond(data, f'{c}_') for c in conds}]


def main():
    parser = argparse.ArgumentParser(
        description='Re-generate Experiment 1 Figure from saved NPZ data')
    parser.add_argument('npz_path', type=str, help='Path to saved experiment NPZ file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as NPZ file)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.npz_path) or 'data/exp1_results'

    all_results = load_all_seeds_npz(args.npz_path)
    if not all_results:
        print("Error: no seeds found in NPZ.")
        return
    print(f"Loaded {len(all_results)} seed(s); conditions: {list(all_results[0].keys())}")

    # Rebuild aggregate stats exactly as run_exp1 does (mean of lfp biomarker).
    all_pd = [float(np.mean(r['Unstimulated']['lfp'])) for r in all_results
              if 'Unstimulated' in r and len(r['Unstimulated']['lfp'])]
    all_snn = [float(np.mean(r['SNN']['lfp'])) for r in all_results
               if 'SNN' in r and len(r['SNN']['lfp'])]
    aggregate_stats = None
    if all_pd and len(all_snn) == len(all_pd):
        n = len(all_pd)
        aggregate_stats = {
            'Unstimulated': {'mean': float(np.mean(all_pd)),
                             'sem': float(np.std(all_pd, ddof=1) / np.sqrt(n)) if n > 1 else 0.0},
            'SNN': {'mean': float(np.mean(all_snn)),
                    'sem': float(np.std(all_snn, ddof=1) / np.sqrt(n)) if n > 1 else 0.0},
            'n': n,
        }

    # Mirror the experiment's call: all_results only when >1 seed (else None).
    all_results_arg = all_results if len(all_results) > 1 else None
    beta_powers, _ = generate_figure1(all_results[0], output_dir,
                                      aggregate_stats=aggregate_stats,
                                      all_results=all_results_arg)

    # Summary
    print("\n" + "=" * 50)
    print("BETA POWER SUMMARY (mean of lfp biomarker)")
    print("=" * 50)
    if aggregate_stats:
        pd_m = aggregate_stats['Unstimulated']['mean']
        snn_m = aggregate_stats['SNN']['mean']
        print(f"  Unstimulated: {pd_m:.4e}")
        print(f"  SNN:          {snn_m:.4e}")
        # Mean of per-seed reductions, matching run_exp1's headline convention.
        reductions = [100 * (pd - snn) / pd for pd, snn in zip(all_pd, all_snn)]
        print(f"  Beta Reduction: {np.mean(reductions):.2f}%  (n={aggregate_stats['n']})")
    print("=" * 50)


if __name__ == "__main__":
    main()
