#!/usr/bin/env python
"""
Generate Experiment 2B Supplementary Figure: Energy–Efficacy Pareto Front

Creates a two-panel publication figure showing:
  (A) Heatmap of beta reduction across (target_sparsity × λ) grid
  (B) Energy–efficacy Pareto front: the best distilled students recover
      teacher-level beta suppression at a fraction of the SynOps, while
      over-sparsified models degrade (some worse than unstimulated PD).

Reference lines are the two physically meaningful levels: the Teacher
(achievable efficacy ceiling) and Unstimulated PD (no-treatment level; points
above it are worse than no stimulation). There is NO arbitrary 50% threshold.

Supports both single-seed and rigorous (multi-seed) CSV formats.

Usage:
    python scripts/generate_exp2b_figure.py data/exp2b_results/exp2b_fine_grained_rigorous_*.csv
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime


def detect_baseline(csv_path):
    """Auto-detect baseline from companion JSON, or fall back to default."""
    csv_dir = os.path.dirname(csv_path)
    csv_base = os.path.basename(csv_path)
    
    # Try to find matching baseline JSON
    # exp2b_fine_grained_rigorous_<date>_<time>.csv -> exp2b_baseline_rigorous_<date>_<time>.json
    timestamp = csv_base.split('_')[-1].replace('.csv', '')  # time component
    date_part = csv_base.split('_')[-2]  # date component
    
    for f in os.listdir(csv_dir):
        if f.startswith('exp2b_baseline') and f.endswith('.json') and timestamp in f:
            json_path = os.path.join(csv_dir, f)
            with open(json_path) as jf:
                data = json.load(jf)
            print(f"Loaded baseline from {json_path}")
            return {
                'mean': data['baseline_beta_mean'],
                'std': data.get('baseline_beta_std', 0),
                'sem': data.get('baseline_beta_sem', 0),
                'n_seeds': data.get('n_seeds', 1)
            }
    
    print("No baseline JSON found, attempting to recover from CSV data...")
    try:
        df = pd.read_csv(csv_path)
        if 'Beta_Power' in df.columns and 'Beta_Reduction_%' in df.columns and len(df) > 0:
            row = df.iloc[0]
            reduction = float(row['Beta_Reduction_%']) / 100.0
            power = float(row['Beta_Power'])
            inferred_baseline = power / (1 - reduction)
            print(f"Inferred baseline from CSV data: {inferred_baseline:.2f}")
            return {'mean': inferred_baseline, 'std': 0, 'sem': 0, 'n_seeds': 1}
    except Exception as e:
        print(f"Could not infer from CSV: {e}")
        
    print("Falling back to default baseline")
    return {'mean': 312.57, 'std': 0, 'sem': 0, 'n_seeds': 1}


def is_rigorous(df):
    """Check if the CSV has multi-seed columns."""
    return 'Beta_Power_Std' in df.columns


def create_success_heatmap(ax, df, baseline_beta):
    """
    Heatmap of beta reduction % across the hyperparameter grid.
    Diverging colormap centered at 0% (= unstimulated PD): green = therapeutic
    (reduces beta), red = worse than PD (increases beta), gray = untested.
    """
    df_numeric = df[pd.to_numeric(df['Target_Sparsity'], errors='coerce').notna()].copy()
    df_numeric['Target_Sparsity'] = df_numeric['Target_Sparsity'].astype(float)
    df_numeric['Sparsity_Weight'] = df_numeric['Sparsity_Weight'].astype(float)
    df_numeric = df_numeric[df_numeric['Sparsity_Weight'] > 0]

    if len(df_numeric) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return

    sparsities = sorted(df_numeric['Target_Sparsity'].unique())
    weights = sorted(df_numeric['Sparsity_Weight'].unique())

    reduction_matrix = np.full((len(sparsities), len(weights)), np.nan)
    for i, ts in enumerate(sparsities):
        for j, sw in enumerate(weights):
            row = df_numeric[(df_numeric['Target_Sparsity'] == ts) &
                             (df_numeric['Sparsity_Weight'] == sw)]
            if len(row) > 0:
                reduction_matrix[i, j] = row['Beta_Reduction_%'].values[0]

    # Diverging colormap (red → yellow → green)
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf',
              '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('reduction', colors)
    cmap.set_bad(color='#e0e0e0')  # Gray for missing data

    masked = np.ma.masked_invalid(reduction_matrix)

    # Symmetric range so 0% (= unstimulated PD) is the white midpoint of the
    # diverging map: green = beta reduced (therapeutic), red = beta increased.
    im = ax.imshow(masked, cmap=cmap, aspect='auto',
                   vmin=-50, vmax=50, origin='lower')

    # Annotate cells with reduction percentages only
    for i in range(len(sparsities)):
        for j in range(len(weights)):
            val = reduction_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if (val < -25 or val > 35) else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        color=color, fontsize=12, fontweight='bold')
            else:
                ax.text(j, i, '—', ha='center', va='center',
                        color='#999999', fontsize=12)

    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels([f'{int(w)}' for w in weights], fontsize=14, rotation=45, ha='right')
    ax.set_yticks(range(len(sparsities)))
    ax.set_yticklabels([f'{s}' for s in sparsities], fontsize=14)

    ax.set_xlabel('Sparsity Weight (λ)', fontsize=16)
    ax.set_ylabel('Target Sparsity', fontsize=16)
    ax.set_title('(A) Beta Reduction (%) by Target Sparsity and Weight', fontsize=18, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Beta Reduction (%)', fontsize=14)

    return im


def create_zoomed_pareto(ax, df, baseline_info):
    """
    Energy–efficacy Pareto front with error bars (if multi-seed data available).
    Points are colored by target sparsity; reference lines mark the Teacher
    (efficacy ceiling) and Unstimulated PD (no-treatment level).
    """
    baseline_beta = baseline_info['mean']
    rigorous = is_rigorous(df)

    teacher_df = df[df['Target_Sparsity'] == 'Teacher']
    student_df = df[pd.to_numeric(df['Target_Sparsity'], errors='coerce').notna()].copy()
    student_df['Target_Sparsity'] = student_df['Target_Sparsity'].astype(float)
    student_df['Sparsity_Weight'] = student_df['Sparsity_Weight'].astype(float)

    # Regularized models only (the lambda=0 distillation-only student is excluded).
    regularized_df = student_df[student_df['Sparsity_Weight'] > 0]

    # Color by target_sparsity
    target_sparsities = sorted(regularized_df['Target_Sparsity'].unique())
    norm = plt.Normalize(vmin=min(target_sparsities), vmax=max(target_sparsities))
    color_map = {ts: plt.cm.viridis(norm(ts)) for ts in target_sparsities}

    # --- Plot all regularized models (filled circles with error bars) ---
    for _, row in regularized_df.iterrows():
        ts = row['Target_Sparsity']
        x, y = row['SynOps_per_ms'], row['Beta_Power']
        
        if rigorous and pd.notna(row.get('Beta_Power_Std')):
            ax.errorbar(x, y,
                       yerr=row['Beta_Power_SEM'],
                       xerr=row.get('SynOps_per_ms_SEM', 0),
                       fmt='none', ecolor=color_map.get(ts, 'gray'),
                       elinewidth=0.8, capsize=2, alpha=0.4, zorder=2)
        
        ax.scatter(x, y,
                   c=[color_map.get(ts, 'gray')], s=70,
                   edgecolors='black', linewidth=0.6, marker='o',
                   alpha=0.8, zorder=3)

    # --- Teacher model (star) + efficacy-ceiling reference line ---
    teacher_beta = None
    if len(teacher_df) > 0:
        for _, row in teacher_df.iterrows():
            x, y = row['SynOps_per_ms'], row['Beta_Power']
            teacher_beta = y

            if rigorous and pd.notna(row.get('Beta_Power_Std')):
                ax.errorbar(x, y,
                           yerr=row['Beta_Power_SEM'],
                           xerr=row.get('SynOps_per_ms_SEM', 0),
                           fmt='none', ecolor='goldenrod',
                           elinewidth=1.2, capsize=4, alpha=0.8, zorder=9)

            ax.scatter(x, y,
                       marker='*', s=300, c='gold', edgecolors='black',
                       linewidth=1.5, zorder=10)
            ax.annotate('Teacher', xy=(x, y), xytext=(-64, 12), textcoords='offset points',
                        ha='left', va='center', fontsize=12, fontweight='bold', color='goldenrod')

    # --- Teacher efficacy-ceiling line (students at/below it match the teacher) ---
    if teacher_beta is not None:
        ax.axhline(y=teacher_beta, color='goldenrod', linestyle='--',
                   alpha=0.8, linewidth=1.5, zorder=2)
        ax.annotate('Teacher efficacy',
                    xy=(ax.get_xlim()[1] * 0.7 if ax.get_xlim()[1] > 0 else 45, teacher_beta),
                    xytext=(0, -8), textcoords='offset points',
                    fontsize=12, fontstyle='italic', color='goldenrod',
                    ha='center', va='top')

    # --- Baseline reference line (unstimulated PD) ---
    if baseline_info['std'] > 0:
        ax.axhspan(baseline_beta - baseline_info['sem'], 
                   baseline_beta + baseline_info['sem'],
                   alpha=0.1, color='red', zorder=0)
    ax.axhline(y=baseline_beta, color='#ff0000', linestyle='-', alpha=0.5,
               linewidth=2, zorder=1)
    ax.annotate('Unstimulated PD', xy=(ax.get_xlim()[1] * 0.7 if ax.get_xlim()[1] > 0 else 45, baseline_beta),
                xytext=(0, -8), textcoords='offset points',
                fontsize=12, fontstyle='italic', color='#ff0000',
                ha='center', va='top')

    # --- Colorbar for Target Sparsity ---
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Target Sparsity (%)', fontsize=14)

    # --- Error Bar Note ---
    if rigorous:
        n = baseline_info.get('n_seeds', '')
        ax.text(0.98, 0.98, f'Error bars: ±SEM (n={n})', transform=ax.transAxes,
                ha='right', va='top', fontsize=10, color='dimgray')

    ax.set_xlabel('Synaptic Operations per ms (SynOps/ms)', fontsize=16)
    ax.set_ylabel('Beta Power (α-β Band, 7–35 Hz)', fontsize=16)
    ax.set_title('(B) Energy–Efficacy Pareto Front', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.2, linewidth=0.5)

    return ax


def generate_supplementary_figure(csv_path, output_dir=None, baseline_override=None):
    """Generate the two-panel supplementary figure."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} models from {csv_path}")
    
    if is_rigorous(df):
        print(f"  Rigorous format detected (multi-seed with error bars)")
    
    # Auto-detect baseline
    if baseline_override:
        baseline_info = {'mean': baseline_override, 'std': 0, 'sem': 0, 'n_seeds': 1}
    else:
        baseline_info = detect_baseline(csv_path)
    
    print(f"  Baseline beta: {baseline_info['mean']:.2f} ± {baseline_info['std']:.2f} (n={baseline_info['n_seeds']})")

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Two-panel layout
    plt.rcParams.update({
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 10
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    create_success_heatmap(ax1, df, baseline_info['mean'])
    create_zoomed_pareto(ax2, df, baseline_info)

    plt.tight_layout(w_pad=3)

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    fig_path = os.path.join(output_dir, f'fig_supp_pareto_{timestamp}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")

    pdf_path = os.path.join(output_dir, f'fig_supp_pareto_{timestamp}.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to {pdf_path}")

    plt.close()
    return fig_path, pdf_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_exp2b_figure.py <csv_path> [output_dir] [--baseline BETA]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = None
    baseline_override = None
    
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--baseline' and i + 1 < len(args):
            baseline_override = float(args[i + 1])
            i += 2
        else:
            output_dir = args[i]
            i += 1

    generate_supplementary_figure(csv_path, output_dir, baseline_override)


if __name__ == "__main__":
    main()
