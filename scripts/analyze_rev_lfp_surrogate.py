"""
LFP-surrogate check: does the parkinsonian signature the controller
responds to appear in a field-potential-like signal of the recorded nucleus?

Surrogate: the GPi population spike count (10 neurons, 1 ms bins) convolved with a synaptic-like
kernel (difference of exponentials, 1 ms rise, 10 ms decay), i.e. the standard spike-to-LFP proxy;
its 7-35 Hz band power is taken per 1 s window (Welch, 500 ms segments) and compared with the
simulator's point-process biomarker averaged over the same 1 s.

Data: data/revision/obs (3 training patients, healthy and parkinsonian, unstimulated and under
constant / cDBS / random stimulation, 60 s each) and data/revision/final_acute_spectral (10
evaluation patients, 40 s sustained PD, unstimulated / student on XyloSim / constant 212 uA).

Outputs: data/revision/lfp_surrogate/summary.json and Sfig_lfp_surrogate.{pdf,png} in $CLDBS_FIG_DIR
(default figures/revision).
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'data/revision/lfp_surrogate')
FIG_DIR = os.environ.get('CLDBS_FIG_DIR', os.path.join(ROOT, 'figures', 'revision'))
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'violet': '#4a3aa7', 'ink': '#0b0b0b', 'ink2': '#52514e', 'grid': '#e4e3df'}
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': C['ink2'], 'axes.labelcolor': C['ink'], 'xtick.color': C['ink2'], 'ytick.color': C['ink2'],
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.5,
                     'axes.axisbelow': True, 'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42})
FS = 1000.0


def surrogate(spikes):
    """spikes: (n_windows, 100, 80) binary 1 ms bins -> continuous GPi LFP-like trace at 1 kHz."""
    s = spikes.reshape(-1, spikes.shape[-1])[:, 30:40].sum(1).astype(float)
    t = np.arange(0, 0.06, 1 / FS)
    k = np.exp(-t / 0.010) - np.exp(-t / 0.001); k /= k.sum()
    return np.convolve(s, k, mode='full')[:len(s)]


def band_power_1s(x, lo=7, hi=35):
    """7-35 Hz power of each non-overlapping 1 s window (Welch, 500 ms segments)."""
    n = len(x) // 1000
    out = []
    for i in range(n):
        f, p = signal.welch(x[i * 1000:(i + 1) * 1000], fs=FS, nperseg=500, noverlap=250, detrend='constant')
        out.append(float(np.trapz(p[(f >= lo) & (f <= hi)], f[(f >= lo) & (f <= hi)])))
    return np.array(out)


def beta_1s(beta):
    n = len(beta) // 10
    return beta[:n * 10].reshape(n, 10).mean(1)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    res = {'kernel': 'difference of exponentials, rise 1 ms, decay 10 ms, GPi population (10 neurons)', 'window': '1 s, Welch 500 ms segments, 7-35 Hz'}
    # ---- A: training patients, both states, stimulation conditions
    obs = {}
    for p in glob.glob(os.path.join(ROOT, 'data/revision/obs/obs_*.npz')):
        m = re.match(r'obs_(.+)_(healthy|pd)_seed(\d+)\.npz', os.path.basename(p)); d = np.load(p)
        x = surrogate(np.minimum(d['counts'], 1)); obs[(m.group(1), m.group(2), int(m.group(3)))] = (band_power_1s(x), beta_1s(d['beta']))
    conds = ['unstim', 'c212', 'c250', 'cdbs', 'random']; seeds = sorted({k[2] for k in obs})
    res['state_separation'] = {}; res['correlation_obs'] = {}
    for c in conds:
        a_s, a_b, r = [], [], []
        for s in seeds:
            sp, bp = obs[(c, 'pd', s)]; sh, bh = obs[(c, 'healthy', s)]
            a_s.append(auc(sp, sh)); a_b.append(auc(bp, bh))
            allx = np.r_[sp, sh]; ally = np.r_[bp, bh]; r.append(float(np.corrcoef(allx, ally)[0, 1]))
        res['state_separation'][c] = {'auc_surrogate': a_s, 'auc_biomarker': a_b}; res['correlation_obs'][c] = r
        print(f'{c:7s} AUC healthy-vs-PD: surrogate {np.mean(a_s):.3f} [{min(a_s):.2f}-{max(a_s):.2f}]  biomarker {np.mean(a_b):.3f} [{min(a_b):.2f}-{max(a_b):.2f}]  r(surrogate, biomarker) {np.mean(r):.2f} [{min(r):.2f}-{max(r):.2f}]')
    # ---- B: evaluation patients, sustained PD, under the student
    ac = {}
    for p in glob.glob(os.path.join(ROOT, 'data/revision/final_acute_spectral/acute_*_seed*.npz')):
        m = re.match(r'acute_(.+)_seed(\d+)\.npz', os.path.basename(p)); d = np.load(p, allow_pickle=True)
        ac[(m.group(1), int(m.group(2)))] = (band_power_1s(surrogate(d['spikes'])), beta_1s(d['lfp']))
    labs = {'unstim': 'Unstimulated', 'xylosim_xn_s0.03_w500': 'SNN student on XyloSim', 'constant_180_0.06_212': 'Constant 212 µA'}
    aseeds = sorted({k[1] for k in ac if k[0] == 'unstim'})
    res['acute'] = {}
    for lab in labs:
        red_s, red_b, r = [], [], []
        for s in aseeds:
            sp, bp = ac[(lab, s)]; su, bu = ac[('unstim', s)]
            red_s.append(100 * (1 - sp[5:].mean() / su[5:].mean())); red_b.append(100 * (1 - bp[5:].mean() / bu[5:].mean()))
            r.append(float(np.corrcoef(sp, bp)[0, 1]))
        res['acute'][lab] = {'reduction_surrogate_pct': red_s, 'reduction_biomarker_pct': red_b, 'r_within_run': r}
        if lab != 'unstim':
            print(f'{lab:24s} reduction vs unstim (after 5 s): surrogate {np.mean(red_s):.1f}% [{min(red_s):.0f}-{max(red_s):.0f}]  biomarker {np.mean(red_b):.1f}% [{min(red_b):.0f}-{max(red_b):.0f}]')
    pooled = np.concatenate([np.c_[ac[k][0], ac[k][1]] for k in ac]); res['acute_pooled_r'] = float(np.corrcoef(pooled[:, 0], pooled[:, 1])[0, 1])
    print(f'pooled over all acute runs and windows: r = {res["acute_pooled_r"]:.2f} (n = {len(pooled)} 1 s windows)')
    json.dump(res, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={'wspace': 0.5, 'width_ratios': [1.2, 1, 1]})
    ax = axes[0]; ax.text(-0.2, 1.08, 'A', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='right')
    d = np.load(os.path.join(ROOT, 'data/revision/obs/obs_unstim_pd_seed202.npz')); x = surrogate(np.minimum(d['counts'][:10], 1))
    t = np.arange(len(x)) / FS
    s = np.minimum(d['counts'][:10], 1).reshape(-1, 80)[:, 30:40].sum(1)
    ax.plot(t, s, color=C['ink2'], lw=0.5, alpha=0.5, label='GPi spike count (1 ms)'); ax.plot(t, x * 10, color=C['orange'], lw=1.0, label='LFP surrogate (×10)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Spikes / ms  ·  surrogate (a.u.)'); ax.set_xlim(0, 1); ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=1)
    ax = axes[1]; ax.text(-0.25, 1.08, 'B', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='right')
    for st, col in (('healthy', C['blue']), ('pd', C['orange'])):
        xs = np.concatenate([obs[('unstim', st, s)][0] for s in seeds]); ys = np.concatenate([obs[('unstim', st, s)][1] for s in seeds])
        ax.plot(ys, xs, 'o', ms=2.5, color=col, alpha=0.5, mec='none', label='Healthy' if st == 'healthy' else 'Parkinsonian')
    ax.set_xlabel('Biomarker: GPi α–β power (µV², 1 s mean)'); ax.set_ylabel('Surrogate 7–35 Hz power (a.u.)')
    ax.set_xlim(190, 400); ax.set_ylim(0.008, 0.05); ax.legend(fontsize=6.5, loc='upper left', markerscale=2)
    ax = axes[2]; ax.text(-0.25, 1.08, 'C', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='right')
    for j, (lab, name) in enumerate(labs.items()):
        if lab == 'unstim':
            continue
        rs, rb = res['acute'][lab]['reduction_surrogate_pct'], res['acute'][lab]['reduction_biomarker_pct']
        ax.plot(np.full(len(rs), j - 1) - 0.12 + np.linspace(-0.06, 0.06, len(rs)), rs, 'o', ms=3.5, color=C['orange'], mfc='none'); ax.plot([j - 1 - 0.22, j - 1 - 0.02], [np.mean(rs)] * 2, color=C['orange'], lw=1.8)
        ax.plot(np.full(len(rb), j - 1) + 0.12 + np.linspace(-0.06, 0.06, len(rb)), rb, 'o', ms=3.5, color=C['ink2'], mfc='none'); ax.plot([j - 1 + 0.02, j - 1 + 0.22], [np.mean(rb)] * 2, color=C['ink2'], lw=1.8)
    ax.plot([], [], 'o', color=C['orange'], mfc='none', label='LFP surrogate'); ax.plot([], [], 'o', color=C['ink2'], mfc='none', label='Biomarker'); ax.legend(fontsize=6.5, loc='lower right')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['SNN student\n(XyloSim)', 'Constant\n212 µA'], fontsize=7); ax.set_ylabel('Reduction vs unstimulated (%)'); ax.set_ylim(0, 80); ax.axhline(0, color=C['ink'], lw=0.5)
    fig.savefig(os.path.join(FIG_DIR, 'Sfig_lfp_surrogate.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(FIG_DIR, 'Sfig_lfp_surrogate.png'), bbox_inches='tight'); print('wrote Sfig_lfp_surrogate')


if __name__ == '__main__':
    main()
