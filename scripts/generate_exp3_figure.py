#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Experiment 3 Figure: Hardware Benchmark Comparison
============================================================
Combines Xylo and Jetson benchmark results into publication-quality figure.

Usage:
    python scripts/generate_exp3_figure.py \
        --xylo data/exp3_results/xylo_benchmark_*.csv \
        --jetson-cpu data/exp3_results/jetson_benchmark_cpu.csv \
        --jetson-cuda data/exp3_results/jetson_benchmark_cuda.csv \
        --output figures/exp3_hardware_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob


def load_xylo_results(csv_path):
    """Load and aggregate Xylo benchmark results."""
    df = pd.read_csv(csv_path)
    
    results = {}
    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name]
        results[model_name] = {
            'latency_ms': model_df['latency_ms'].mean(),
            'latency_std': model_df['latency_ms'].std(),
            'power_mw': model_df['inf_power_snn_W'].mean() * 1000,  # Convert to mW
            'power_std': model_df['inf_power_snn_W'].std() * 1000,
            'edp': model_df['edp'].mean(),
            'edp_std': model_df['edp'].std(),
        }
    return results


def load_jetson_results(csv_path, device_name):
    """Load and aggregate Jetson benchmark results (averages the whole CSV)."""
    df = pd.read_csv(csv_path)

    return {
        device_name: {
            'latency_ms': df['latency_ms'].mean(),
            'latency_std': df['latency_ms'].std(),
            'power_mw': df['power_w'].mean() * 1000,  # Convert to mW
            'power_std': df['power_w'].std() * 1000,
            'edp': df['edp'].mean(),
            'edp_std': df['edp'].std(),
        }
    }


def load_jetson_by_type(csv_path):
    """Load the combined Jetson benchmark CSV, aggregated per model_type (ANN/RNN).

    The CSV (from benchmark_jetson_ann_rnn.py) holds both ANN and RNN rows on the
    same device (CUDA). Note: the `edp` column is actually energy per inference
    (power_w * latency_s = J), not a true Energy-Delay Product.
    """
    df = pd.read_csv(csv_path)
    out = {}
    for mtype in df['model_type'].unique():
        d = df[df['model_type'] == mtype]
        out[str(mtype).upper()] = {
            'latency_ms': d['latency_ms'].mean(),
            'latency_std': d['latency_ms'].std(),
            'power_mw': d['power_w'].mean() * 1000,  # Convert to mW
            'power_std': d['power_w'].std() * 1000,
            'edp': d['edp'].mean(),
            'edp_std': d['edp'].std(),
        }
    return out


def generate_figure(xylo_results, jetson_results, output_path):
    """Generate publication-quality hardware comparison figure."""
    
    # --- SNN / ANN / RNN figure (matches exp3_hardware_comparison_06-11) ---
    # SNN  = distilled Student on Xylo;  ANN/RNN = Jetson Orin Nano (CUDA).
    # xylo_results: {'SNN': {...}}  |  jetson_results: {'ANN': {...}, 'RNN': {...}}
    all_results = {}
    if 'SNN' in xylo_results:
        all_results['SNN'] = xylo_results['SNN']
    for m in ('ANN', 'RNN'):
        if m in jetson_results:
            all_results[m] = jetson_results[m]

    model_names = list(all_results.keys())   # ['SNN', 'ANN', 'RNN']
    n_models = len(model_names)

    # Extract data
    latencies = [all_results[m]['latency_ms'] for m in model_names]
    latency_stds = [all_results[m]['latency_std'] for m in model_names]
    powers = [all_results[m]['power_mw'] for m in model_names]
    power_stds = [all_results[m]['power_std'] for m in model_names]
    edps = [all_results[m]['edp'] for m in model_names]
    edp_stds = [all_results[m].get('edp_std', 0.0) for m in model_names]

    # Colors: green for SNN (Xylo), purple for ANN, orange for RNN (Jetson)
    BAR_COLORS = {'SNN': '#27ae60', 'ANN': '#9b59b6', 'RNN': '#e67e22'}
    colors = [BAR_COLORS.get(m, '#7f8c8d') for m in model_names]

    # Create figure
    plt.rcParams.update({
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 14
    })
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(n_models)
    bar_width = 0.6
    
    # A) Latency (log scale)
    ax1 = axes[0]
    bars1 = ax1.bar(x, latencies, bar_width, yerr=latency_stds,
                    color=colors, capsize=5, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Latency (ms)', fontsize=16, fontweight='bold')
    ax1.set_title('A) Inference Latency', fontsize=18, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=14, rotation=0)
    ax1.set_yscale('log')
    ax1.grid(True, which='major', alpha=0.3, axis='y')
    ax1.grid(True, which='minor', alpha=0.15, axis='y', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Add value labels (placed above the top of the error bar)
    for bar, val, err in zip(bars1, latencies, latency_stds):
        ax1.text(bar.get_x() + bar.get_width()/2, (val + err) * 1.10,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylim(top=max(v + e for v, e in zip(latencies, latency_stds)) * 3)
    
    # B) Power (log scale)
    ax2 = axes[1]
    bars2 = ax2.bar(x, powers, bar_width, yerr=power_stds,
                    color=colors, capsize=5, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Power (mW)', fontsize=16, fontweight='bold')
    ax2.set_title('B) Power Consumption', fontsize=18, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=14, rotation=0)
    ax2.set_yscale('log')
    ax2.grid(True, which='major', alpha=0.3, axis='y')
    ax2.grid(True, which='minor', alpha=0.15, axis='y', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    # Add value labels (placed above the top of the error bar)
    for bar, val, err in zip(bars2, powers, power_stds):
        label = f'{val:.2f}' if val < 10 else (f'{val:.1f}' if val < 100 else f'{val:.0f}')
        ax2.text(bar.get_x() + bar.get_width()/2, (val + err) * 1.10,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylim(top=max(v + e for v, e in zip(powers, power_stds)) * 3)
    
    # C) Energy per inference (log scale)
    # NOTE: `edp` is power x latency = energy per inference, in Joules (W*s).
    # This is NOT an Energy-Delay Product (which would be energy x delay = P*t^2, J*s).
    ax3 = axes[2]
    bars3 = ax3.bar(x, edps, bar_width, yerr=edp_stds, capsize=5,
                    color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Energy (J)', fontsize=16, fontweight='bold')
    ax3.set_title('C) Energy per Inference', fontsize=18, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, fontsize=14, rotation=0)
    ax3.set_yscale('log')
    ax3.grid(True, which='major', alpha=0.3, axis='y')
    ax3.grid(True, which='minor', alpha=0.15, axis='y', linewidth=0.5)
    ax3.set_axisbelow(True)
    
    # Add value labels (placed above the top of the error bar)
    for bar, val, err in zip(bars3, edps, edp_stds):
        ax3.text(bar.get_x() + bar.get_width()/2, (val + err) * 1.10,
                f'{val:.1e}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_ylim(top=max(v + e for v, e in zip(edps, edp_stds)) * 3)

    # Add legend
    from matplotlib.patches import Patch
    legend_labels = {
        'SNN': 'SNN Best Student (Xylo Audio SNN Core)',
        'ANN': 'ANN CUDA (Jetson Orin Nano)',
        'RNN': 'RNN CUDA (Jetson Orin Nano)',
    }
    legend_elements = [
        Patch(facecolor=BAR_COLORS[m], edgecolor='black', label=legend_labels[m])
        for m in model_names if m in legend_labels
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               fontsize=14, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")

    pdf_path = os.path.splitext(output_path)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")
    
    plt.close()


def print_summary(xylo_results, jetson_results):
    """Print comparison summary."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: HARDWARE BENCHMARK SUMMARY")
    print("="*80)
    
    # Get reference values
    teacher = xylo_results.get('teacher', {})
    
    print(f"\n{'Model':<30} {'Latency (ms)':<15} {'Power (mW)':<15} {'Energy/inf (J)':<15}")
    print("-"*80)
    
    for name, data in xylo_results.items():
        print(f"SNN {name.capitalize()} (Xylo)          {data['latency_ms']:<15.2f} {data['power_mw']:<15.2f} {data['edp']:<15.2e}")
    
    for name, data in jetson_results.items():
        print(f"{name:<30} {data['latency_ms']:<15.2f} {data['power_mw']:<15.2f} {data['edp']:<15.2e}")
    
    # Ratios - compare with best Jetson config (CPU)
    if teacher and 'ANN CPU (Jetson)' in jetson_results:
        jetson_cpu = jetson_results['ANN CPU (Jetson)']
        print("\n" + "-"*80)
        print("COMPARISON (Xylo Teacher vs Jetson CPU):")
        print(f"  Speed:  Jetson is {teacher['latency_ms'] / jetson_cpu['latency_ms']:.0f}× faster")
        print(f"  Power:  Xylo uses {jetson_cpu['power_mw'] / teacher['power_mw']:.0f}× less power")
        print(f"  Energy: Xylo is {jetson_cpu['edp'] / teacher['edp']:.1f}× more energy-efficient per inference")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Experiment 3 hardware comparison figure.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--xylo', type=str,
                        help='Path to Xylo benchmark CSV (or auto-detect latest)')
    parser.add_argument('--jetson', type=str,
                        help='Path to combined Jetson ANN+RNN benchmark CSV (or auto-detect latest)')
    parser.add_argument('--snn-model', type=str, default='student',
                        choices=['student', 'teacher'],
                        help='Which Xylo model is the SNN bar (default: student)')
    parser.add_argument('--output', type=str, default='data/exp3_results/exp3_hardware_comparison.png',
                        help='Output path for figure')

    args = parser.parse_args()

    # Auto-detect latest CSV files if not specified
    results_dir = 'data/exp3_results'

    if args.xylo is None:
        xylo_files = sorted(glob(os.path.join(results_dir, 'xylo_benchmark_*.csv')))
        if xylo_files:
            args.xylo = xylo_files[-1]
            print(f"Auto-detected Xylo results: {args.xylo}")
        else:
            print("Error: No Xylo benchmark CSV found. Run run_xylo_rl_benchmark.py first.")
            return

    # Auto-detect combined Jetson ANN+RNN benchmark
    if args.jetson is None:
        jetson_files = sorted(glob(os.path.join(results_dir, 'jetson_ann_rnn_benchmark_*.csv')))
        if jetson_files:
            args.jetson = jetson_files[-1]
            print(f"Auto-detected Jetson ANN+RNN results: {args.jetson}")
        else:
            print("Error: No jetson_ann_rnn_benchmark_*.csv found. Run benchmark_jetson_ann_rnn.py first.")
            return

    # Load results: SNN = chosen Xylo model; ANN/RNN split from the combined Jetson CSV
    xylo_all = load_xylo_results(args.xylo)
    xylo_results = {'SNN': xylo_all[args.snn_model]}
    jetson_results = load_jetson_by_type(args.jetson)   # {'ANN': ..., 'RNN': ...}

    # Add timestamp to output filename if using default
    if args.output == 'data/exp3_results/exp3_hardware_comparison.png':
        from datetime import datetime
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        args.output = f'data/exp3_results/exp3_hardware_comparison_{timestamp}.png'
    
    # Print summary
    print_summary(xylo_results, jetson_results)
    
    # Generate figure
    generate_figure(xylo_results, jetson_results, args.output)


if __name__ == "__main__":
    main()
