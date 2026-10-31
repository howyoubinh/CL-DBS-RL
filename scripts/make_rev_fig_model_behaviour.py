"""
Main-text Fig. 2: behavior of the simulated CBGT network, healthy vs parkinsonian, from real
simulator output (data from scripts/run_rev_activity.py): the model's typical activity patterns
and the signals the closed-loop controller receives as input.

Panels
  A  membrane potential of one STN, GPe and GPi neuron over 1 s (unstimulated), healthy vs PD,
     with the -20 mV spike-recording threshold
  B  recorded spikes of all 80 neurons over the same second, grouped by population
  C  per-population rate, synchrony (mean pairwise 5 ms spike-count correlation) and ISI CV;
     one dot per patient (n = 3), bar = mean
  D  the controller's input: the 16-channel pooled observation for one decision window, at the
     controller's 100 ms / 1 ms binning and at 1 s / 10 ms (comparison)
  E  GPi alpha-beta biomarker per 1 s window, healthy vs PD: box plot over all windows of all patients (whiskers, range)

  python scripts/make_rev_fig_model_behaviour.py [--activity data/revision/activity]
      -> $CLDBS_FIG_DIR/fig_model_behaviour.{pdf,png} (default figures/revision)
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG_DIR = os.environ.get('CLDBS_FIG_DIR', os.path.join(ROOT, 'figures', 'revision'))

HEALTHY, PD = '#2a78d6', '#eb6834'          # validated categorical slots 1-2 (dataviz palette)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
SEEDS = (42, 123, 456)
SHOW_SEED = 42
KEYS = ('TH_APs', 'STN_APs', 'GPe_APs', 'GPi_APs', 'Striat_APs_indr', 'Striat_APs_dr', 'Cor_APs')
LABELS = ('TH', 'STN', 'GPe', 'GPi', 'Str-D2', 'Str-D1', 'Cortex')
T0 = 1.0                                     # show the second second (after settling)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def load(activity, seed, pd):
    return json.load(open(os.path.join(activity, f'seed{seed}_{"pd" if pd else "healthy"}.json')))


def panel_label(ax, s, x=-0.12, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', color=INK, va='bottom')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--activity', default=os.path.join(ROOT, 'data/revision/activity'))
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    summary = json.load(open(os.path.join(args.activity, 'summary.json')))
    runs = {pd: load(args.activity, SHOW_SEED, pd) for pd in (0, 1)}

    fig = plt.figure(figsize=(5.9, 7.4))      # sized so 7 pt text stays >= 6.5 pt at the manuscript text width
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.05, 1.5, 0.95], hspace=0.62, wspace=1.1,
                  left=0.08, right=0.94, top=0.97, bottom=0.30)
    gs_bottom = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.7], wspace=1.0,
                         left=0.08, right=0.985, top=0.225, bottom=0.055)

    # A: membrane traces (first simulated second is what was captured; it starts at t = 0)
    for col, pd in enumerate((0, 1)):
        ax = fig.add_subplot(gs[0, 3 * col:3 * col + 3])
        v = runs[pd]['vtrace_first_s']
        t = np.arange(len(v['STN'])) * 1e-4
        for i, pop in enumerate(('STN', 'GPe', 'GPi')):
            ax.plot(t, np.asarray(v[pop]) - 110 * i, lw=0.5, color=PD if pd else HEALTHY)
            ax.axhline(-20 - 110 * i, lw=0.5, color=INK2, alpha=0.6)
            ax.text(-0.02, -60 - 110 * i, pop, ha='right', va='center', fontsize=7, color=INK, transform=ax.get_yaxis_transform())
        ax.set_yticks([]); ax.spines['left'].set_visible(False)
        ax.set_xlim(0, 1); ax.set_xlabel('Time (s)')
        ax.set_title(('Parkinsonian' if pd else 'Healthy') + ' (membrane potential)', color=INK, loc='left')
        if col == 0:
            panel_label(ax, 'A', x=-0.1)
        else:  # scale bar outside the traces, right edge
            ax.plot([1.015, 1.015], [-300, -250], color=INK, lw=0.8, clip_on=False, transform=ax.get_yaxis_transform())
            ax.text(1.025, -275, '50 mV', fontsize=7, va='center', color=INK, transform=ax.get_yaxis_transform(), clip_on=False)

    # B: rasters of all 80 neurons, one second
    for col, pd in enumerate((0, 1)):
        ax = fig.add_subplot(gs[1, 3 * col:3 * col + 3])
        row, ticks = 0, []
        for k, lab in zip(KEYS, LABELS):
            trains = runs[pd]['spikes'][k]
            for s in trains:
                s = np.asarray(s); s = s[(s >= T0) & (s < T0 + 1)] - T0
                ax.vlines(s, row + 0.1, row + 0.9, lw=0.5, color=PD if pd else HEALTHY)
                row += 1
            ticks.append(row - len(trains) / 2)
            ax.axhline(row, lw=0.4, color=GRID)
        ax.set_ylim(row, 0); ax.set_xlim(0, 1)
        ax.set_yticks(ticks); ax.set_yticklabels(LABELS if col == 0 else [''] * len(LABELS))
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel('Time (s)')
        ax.set_title(('Parkinsonian' if pd else 'Healthy') + ' (recorded spikes, 80 neurons)', color=INK, loc='left')
        if col == 0:
            panel_label(ax, 'B', x=-0.1)

    # C: per-population metrics, one dot per patient
    pops = ('STN', 'GPe', 'GPi', 'TH')
    metrics = (('rate', 'Firing rate (Hz)'), ('sync_within', 'Synchrony (pairwise r)'), ('cv_isi', 'ISI CV'))
    for j, (m, lab) in enumerate(metrics):
        ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
        for i, p in enumerate(pops):
            for d, (pd, colr) in enumerate(((0, HEALTHY), (1, PD))):
                vals = [summary[f'seed{s}_{"pd" if pd else "healthy"}'][p][m] for s in SEEDS]
                x = i + (d - 0.5) * 0.34
                ax.bar(x, np.mean(vals), width=0.26, color=colr, alpha=0.18, edgecolor='none')
                ax.scatter([x] * len(vals), vals, s=12, color=colr, edgecolor='white', linewidth=0.6, zorder=3)
        ax.set_xticks(range(len(pops))); ax.set_xticklabels(pops)
        ax.set_ylabel(lab); ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, lw=0.4, color=GRID); ax.set_axisbelow(True)
        if j == 0:
            panel_label(ax, 'C', x=-0.32)
        if m == 'cv_isi':  # legend above the panel, outside the data
            ax.scatter([], [], s=12, color=HEALTHY, label='Healthy'); ax.scatter([], [], s=12, color=PD, label='Parkinsonian')
            ax.legend(frameon=False, fontsize=7, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.0, borderaxespad=0.2)

    # D: the controller's 16-channel pooled input for one decision window
    for j, (win, nb, title) in enumerate(((0.1, 100, '100 ms window, 1 ms bins'),
                                          (1.0, 100, '1 s window, 10 ms bins (comparison)'))):
        ax = fig.add_subplot(gs_bottom[0, j])
        trains = [np.asarray(s) for k in KEYS for s in runs[1]['spikes'][k]]
        edges = np.linspace(T0, T0 + win, nb + 1)
        c = np.stack([np.histogram(s, bins=edges)[0] for s in trains])         # (80, nb)
        pooled = c.reshape(16, 5, nb).mean(1)                                  # AvgPool1d(5): 80 -> 16
        im = ax.imshow(pooled, aspect='auto', cmap='Oranges', vmin=0, interpolation='nearest',
                       extent=(0, win * 1000, 16, 0))
        ax.set_xlabel('Time in window (ms)'); ax.set_ylabel('Input channel')
        ax.set_yticks([0.5, 15.5]); ax.set_yticklabels(['1', '16'])
        ax.set_title(title, color=INK, loc='left', fontsize=7, pad=4)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=7, length=2); cb.outline.set_visible(False)
        if j == 1:  # one label for both colorbars
            cb.set_label('Mean spikes per bin', fontsize=7, color=INK2)
        if j == 0:
            panel_label(ax, 'D', x=-0.25)

    # E: GPi biomarker per 1 s window
    ax = fig.add_subplot(gs_bottom[0, 2])
    for d, (pd, colr, lab) in enumerate(((0, HEALTHY, 'Healthy'), (1, PD, 'Parkinsonian'))):
        vals = np.concatenate([load(args.activity, s, pd)['beta'][1:] for s in SEEDS])
        ax.boxplot(vals, positions=[d], widths=0.5, whis=(0, 100), patch_artist=True,
                   boxprops=dict(facecolor=colr, alpha=0.35, edgecolor=colr, lw=0.8), medianprops=dict(color=INK, lw=1.2),
                   whiskerprops=dict(color=colr, lw=0.8), capprops=dict(color=colr, lw=0.8))
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Healthy', 'Parkin-\nsonian'])
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(bottom=0)
    ax.set_ylabel('GPi $\\alpha$-$\\beta$ power\n(1 s windows, µV²)')
    ax.yaxis.grid(True, lw=0.4, color=GRID); ax.set_axisbelow(True)
    panel_label(ax, 'E', x=-0.42)

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'fig_model_behaviour.{ext}'), dpi=300)
    print('saved', os.path.join(FIG_DIR, 'fig_model_behaviour.{pdf,png}'))


if __name__ == '__main__':
    main()
