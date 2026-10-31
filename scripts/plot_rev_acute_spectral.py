"""
Supplementary spectral figure for the deployed student in sustained parkinsonian activity, from
the recordings of run_rev_acute_spectral.py:
  A1/A2  spectrogram of the GPi population spike train (representative patient), unstimulated and
         under the student on XyloSim; B  multi-taper GPi power spectrum, mean +/- SEM over
         patients; C  alpha-beta power over the 40 s (1 s mean), mean +/- SEM; D  the student's
         stimulation parameters, mean +/- SEM; E  per-patient mean alpha-beta power per condition.

  python scripts/plot_rev_acute_spectral.py  -> $CLDBS_FIG_DIR/Sfig_acute_spectral.{pdf,png} (default figures/revision)
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from make_rev_main_figures import C, panel_label  # noqa: E402,F401
except ImportError:  # same palette and axes style as make_rev_main_figures.py, defined locally
    C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'violet': '#4a3aa7', 'ink': '#0b0b0b', 'ink2': '#52514e', 'grid': '#e4e3df'}
    plt.rcParams.update({'font.size': 8, 'font.family': 'sans-serif', 'axes.edgecolor': C['ink2'], 'axes.labelcolor': C['ink'], 'xtick.color': C['ink2'],
                         'ytick.color': C['ink2'], 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': C['grid'],
                         'grid.linewidth': 0.5, 'axes.axisbelow': True, 'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42})

DATA = os.path.join(ROOT, 'data/revision/final_acute_spectral')
FIG_DIR = os.environ.get('CLDBS_FIG_DIR', os.path.join(ROOT, 'figures', 'revision'))
GRID = np.linspace(1, 100, 200)


def load():
    R = {}
    for p in glob.glob(os.path.join(DATA, 'acute_*_seed*.npz')):
        m = re.match(r'acute_(.+)_seed(\d+)\.npz', os.path.basename(p))
        R.setdefault(m.group(1), {})[int(m.group(2))] = np.load(p, allow_pickle=True)
    return R


def gpi_signal(spikes):
    s = spikes.reshape(-1, spikes.shape[-1])[:, 30:40].sum(1).astype(float)   # GPi neurons 30-39, 1 ms bins
    k = np.exp(-np.arange(20) / 5.0); k /= k.sum()
    return np.convolve(s, k, mode='same')


def mt_psd(d):
    out = []
    for S, f in zip(d['gpi_S'], d['gpi_f']):
        S = np.asarray(S, float); f = np.asarray(f, float)
        if S.size and S.size == f.size:
            out.append(np.interp(GRID, f, S))
    return np.mean(out, 0) if out else np.full(GRID.shape, np.nan)


def roll(b, k=10):
    c = np.cumsum(np.r_[0.0, b]); i = np.arange(1, len(b) + 1); lo = np.maximum(0, i - k); return (c[i] - c[lo]) / (i - lo)


def main():
    R = load()
    labs = {'unstim': ('Unstimulated', C['ink2']), 'xylosim_xn_s0.03_w500': ('SNN student on XyloSim', C['blue']), 'constant_180_0.06_212': ('Constant 212 µA', C['yellow'])}
    seeds = sorted(set.intersection(*[set(R[k]) for k in labs if k in R]))
    rep = seeds[0]
    fig = plt.figure(figsize=(7.2, 7.4)); gs = fig.add_gridspec(3, 2, hspace=0.58, wspace=0.35)
    for i, (k, (name, col)) in enumerate([('unstim', labs['unstim']), ('xylosim_xn_s0.03_w500', labs['xylosim_xn_s0.03_w500'])]):
        ax = fig.add_subplot(gs[0, i]); ax.text(-0.2, 1.08, 'A' + str(i + 1), transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='right', color=C['ink'])
        d = R[k][rep]; M = np.stack([np.interp(GRID, np.asarray(f, float), np.asarray(S, float)) if np.asarray(S).size else np.full(GRID.shape, np.nan) for S, f in zip(d['gpi_S'], d['gpi_f'])])
        t = np.arange(len(M)) * 0.1; m = GRID <= 60
        im = ax.pcolormesh(t, GRID[m], 10 * np.log10(M[:, m].T + 1e-9), shading='nearest', cmap='viridis', vmin=0, vmax=20)
        ax.axhline(7, color='white', lw=0.6, ls='--'); ax.axhline(35, color='white', lw=0.6, ls='--')
        ax.set_title(name, fontsize=8); ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)' if i == 0 else ''); ax.grid(False)
        if i == 1:  # one colour bar for both spectrograms (same scale), outside the right edge of A2
            cb = fig.colorbar(im, cax=ax.inset_axes([1.04, 0.0, 0.035, 1.0]), ticks=[0, 10, 20]); cb.set_label('Power (dB)', fontsize=7)
            cb.ax.tick_params(labelsize=7, length=2); cb.outline.set_visible(False)
    lab = lambda ax, L: ax.text(-0.22, 1.08, L, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='right', color=C['ink'])
    ax = fig.add_subplot(gs[1, 0]); lab(ax, 'B')
    for k, (name, col) in labs.items():
        if k not in R:
            continue
        P = np.stack([mt_psd(R[k][s]) for s in seeds]); mu, se = np.nanmean(P, 0), np.nanstd(P, 0, ddof=1) / np.sqrt(len(seeds))
        ax.plot(GRID, mu, color=col, lw=1.2, label=name); ax.fill_between(GRID, mu - se, mu + se, color=col, alpha=0.2, lw=0)
    ax.axvspan(7, 35, color=C['grid'], alpha=0.5, lw=0); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('GPi multi-taper power (µV²/Hz)'); ax.set_xlim(1, 60); ax.legend(fontsize=7, loc='upper right', handlelength=1.4)
    ax = fig.add_subplot(gs[1, 1]); lab(ax, 'C')
    for k, (name, col) in labs.items():
        if k not in R:
            continue
        B = np.stack([roll(R[k][s]['lfp']) for s in seeds]); t = np.arange(B.shape[1]) * 0.1
        mu, se = B.mean(0), B.std(0, ddof=1) / np.sqrt(len(seeds)); ax.plot(t, mu, color=col, lw=1.2); ax.fill_between(t, mu - se, mu + se, color=col, alpha=0.2, lw=0)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('GPi α–β power (1 s mean, µV²)')
    ax = fig.add_subplot(gs[2, 0]); lab(ax, 'D')
    k = 'xylosim_xn_s0.03_w500'
    for key, scale, name, col in (('freq', 180, 'Frequency / 180 Hz', C['blue']), ('pw', 0.4, 'Pulse width / 0.4 ms', C['aqua']), ('amp', 250, 'Amplitude / 250 µA', C['violet'])):
        V = np.stack([R[k][s][key] / scale for s in seeds]); t = np.arange(V.shape[1]) * 0.1
        mu, se = V.mean(0), V.std(0, ddof=1) / np.sqrt(len(seeds)); ax.plot(t, mu, color=col, lw=1.2, label=name); ax.fill_between(t, mu - se, mu + se, color=col, alpha=0.2, lw=0)
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1.0]); ax.set_xlabel('Time (s)'); ax.set_ylabel('Student stimulation (normalized)')
    ax.legend(fontsize=7, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.2, columnspacing=1.0, borderaxespad=0.0)
    ax = fig.add_subplot(gs[2, 1]); lab(ax, 'E')
    for j, (k, (name, col)) in enumerate(labs.items()):
        if k not in R:
            continue
        v = [float(np.mean(R[k][s]['lfp'])) for s in seeds]
        ax.plot(np.full(len(v), j) + np.linspace(-0.15, 0.15, len(v)), v, 'o', ms=4, color=col, mfc='none')
        ax.plot([j - 0.25, j + 0.25], [np.mean(v)] * 2, color=col, lw=1.8)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels([v[0].replace(' on XyloSim', '\n(XyloSim)') for v in labs.values()], fontsize=7); ax.set_ylabel('Mean α–β power over 40 s (µV²)')
    ax.set_xlim(-0.5, len(labs) - 0.5); lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.18 * (hi - lo))   # headroom for the note
    ax.text(0.98, 0.97, f'One circle per patient (n = {len(seeds)})\nbar = mean', transform=ax.transAxes, fontsize=7, va='top', ha='right', color=C['ink2'])
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'Sfig_acute_spectral.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(FIG_DIR, 'Sfig_acute_spectral.png'), bbox_inches='tight')
    u = [float(np.mean(R['unstim'][s]['lfp'])) for s in seeds]; st = [float(np.mean(R['xylosim_xn_s0.03_w500'][s]['lfp'])) for s in seeds]
    red = [100 * (a - b) / a for a, b in zip(u, st)]
    print(f'patients {len(seeds)}: unstim {np.mean(u):.0f}, student {np.mean(st):.0f}, reduction {np.mean(red):.1f}% [{min(red):.1f}-{max(red):.1f}]; wrote Sfig_acute_spectral')


if __name__ == '__main__':
    main()
