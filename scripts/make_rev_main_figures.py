"""
Main-text figures from the final summaries:

  fig_adaptation.pdf   A: cycling trace (patient 42): beta of the deployed student on XyloSim vs
                       unstimulated and constant stimulation, PD blocks shaded; B: its stimulation
                       parameters; C: PD-block beta vs total charge for every condition (10 patients);
                       D: PD/healthy charge ratio.
  fig_controllers.pdf  A: beta-charge front of tuned simple controllers vs learned policies (5 patients);
                       B: non-spiking baselines per training patient and the hyperparameter sweep.
  fig_distillation.pdf A: students' hidden rate vs closed-loop behavior; B: fp32 vs XyloSim per student;
                       C: Xylo action agreement vs fp32 margin (from xylo_check).
  fig_robustness.pdf   delay, spike loss, added spikes, severity, block length: PD beta reduction and
                       PD/healthy ratio for teacher, student (fp32, XyloSim), MLP, GRU, simple controllers.
  Sfig_simple_per_patient.pdf, Sfig_students.pdf, Sfig_robustness_per_patient.pdf
                       supplementary versions showing every patient / every student.

  python scripts/make_rev_main_figures.py   -> $CLDBS_FIG_DIR/*.pdf (+ .png previews; default figures/revision)
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from analyze_rev_select import per_seed  # noqa: E402
import analyze_rev_robustness as AR  # noqa: E402

FIG_DIR = os.environ.get('CLDBS_FIG_DIR', os.path.join(ROOT, 'figures', 'revision'))
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'violet': '#4a3aa7', 'ink': '#0b0b0b', 'ink2': '#52514e', 'grid': '#e4e3df'}
plt.rcParams.update({'font.size': 8, 'font.family': 'sans-serif', 'axes.edgecolor': C['ink2'], 'axes.labelcolor': C['ink'], 'xtick.color': C['ink2'],
                     'ytick.color': C['ink2'], 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': C['grid'],
                     'grid.linewidth': 0.5, 'axes.axisbelow': True, 'legend.frameon': False, 'legend.fontsize': 7, 'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42})
J = lambda p: json.load(open(os.path.join(ROOT, p)))
SEEDS5 = [42, 123, 456, 789, 1024]


def panel_label(ax, s, x=-0.12, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='right', color=C['ink'])


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name + '.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(FIG_DIR, name + '.png'), bbox_inches='tight')
    plt.close(fig); print('wrote', name)


def load_runs(d):
    """{controller: {seed: run}} for the five-patient experiments (seeds 42-1024)."""
    R = {}
    for p in glob.glob(os.path.join(ROOT, d, 'cycling_*_w100*_seed*.json')):
        m = re.match(r'cycling_(.+)_w100((?:_sm10)?)_seed(\d+)\.json', os.path.basename(p))
        if m and int(m.group(3)) in SEEDS5:
            R.setdefault(m.group(1) + m.group(2), {})[int(m.group(3))] = J(os.path.relpath(p, ROOT))
    return R


def patients(runs, U, Cd, key):
    """Per-patient values of one metric (the summaries' means are over exactly these rows)."""
    return [per_seed(runs[s], U[s], Cd[s])[key] for s in SEEDS5 if s in runs]


def dots(ax, x, ys, col, mk, ms=2.6):
    """Individual patients as small open symbols behind the mean."""
    ax.plot(np.broadcast_to(x, np.shape(ys)), ys, mk, color=col, ms=ms, mfc='none', mew=0.5, alpha=0.55, zorder=1)


def roll(b, k=10):
    c = np.cumsum(np.r_[0.0, b]); i = np.arange(1, len(b) + 1); lo = np.maximum(0, i - k); return (c[i] - c[lo]) / (i - lo)


# ------------------------------------------------------------------ Figure: adaptation
def fig_adaptation():
    d = lambda lab: J(f'data/revision/final_cycling/cycling_{lab}_w100_seed42.json')
    st, un, co, oc = d('xylosim_xn_s0.03_w500'), d('unstim'), d('constant_180_0.06_212'), d('oracle_180_0.06_250')
    t = np.arange(len(st['lfp'])) * 0.1; pd = np.asarray(st['pd_state']) == 1
    fig = plt.figure(figsize=(6.0, 6.9)); gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.0, 1.5], hspace=0.6, wspace=0.35)
    ax = fig.add_subplot(gs[0, :]); panel_label(ax, 'A')
    for s0, s1 in zip(*[np.nonzero(np.diff(np.r_[0, pd.astype(int), 0]) == v)[0] for v in (1, -1)]):
        ax.axvspan(t[s0], t[min(s1, len(t) - 1)], color=C['orange'], alpha=0.08, lw=0)
    ax.plot(t, roll(np.asarray(un['lfp'])), color=C['ink2'], lw=1.0, label='Unstimulated')
    ax.plot(t, roll(np.asarray(co['lfp'])), color=C['yellow'], lw=1.0, label='Constant 180 Hz / 60 µs / 212 µA')
    ax.plot(t, roll(np.asarray(st['lfp'])), color=C['blue'], lw=1.3, label='SNN student on XyloSim')
    ax.set_ylabel('GPi α–β power (1 s mean, µV²)'); ax.set_xlabel('Time (s)'); ax.set_xlim(0, t[-1]); ax.set_ylim(80, 420)
    ax.legend(loc='lower left', ncol=3, bbox_to_anchor=(0, 1.0), handlelength=1.6, columnspacing=1.5)
    ax = fig.add_subplot(gs[1, :]); panel_label(ax, 'B')
    q = np.asarray(st['freq']) * np.asarray(st['pw']) * np.asarray(st['amp']) / (130 * 0.3 * 300) * 100
    for s0, s1 in zip(*[np.nonzero(np.diff(np.r_[0, pd.astype(int), 0]) == v)[0] for v in (1, -1)]):
        ax.axvspan(t[s0], t[min(s1, len(t) - 1)], color=C['orange'], alpha=0.08, lw=0)
    ax.plot(t, np.asarray(st['freq']) / 180, color=C['blue'], lw=0.9, label='Frequency / 180 Hz')
    ax.plot(t, np.asarray(st['pw']) / 0.4, color=C['aqua'], lw=0.9, label='Pulse width / 0.4 ms')
    ax.plot(t, np.asarray(st['amp']) / 250, color=C['violet'], lw=0.9, label='Amplitude / 250 µA')
    ax.plot(t, q / 100, color=C['ink'], lw=1.2, ls='--', label='Charge rate / cDBS')
    ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel('Stimulation (normalized)'); ax.set_xlabel('Time (s)'); ax.set_xlim(0, t[-1])
    ax.legend(loc='lower left', ncol=4, bbox_to_anchor=(0, 1.0), handlelength=1.3, handletextpad=0.4, columnspacing=0.8)
    cy = J('data/revision/final_cycling/summary.json')
    show = [('xylosim_xn_s0.03_w500', 'Deployed SNN student (XyloSim)', C['blue'], 'D'),
            ('constant_180_0.06_212', 'Constant 212 µA', C['yellow'], 's'), ('cdbs', 'cDBS', C['ink'], 'P'),
            ('oracle_180_0.06_250', 'True-state switching', C['ink'], '*')]
    hb = np.mean([v['h_beta_unstim'] for v in cy.values()])
    for col, key, lab, pl in ((0, 'charge_pct', 'Total charge (% of cDBS)', 'C'), (1, 'ratio', 'Charge rate ratio, PD / healthy blocks', 'D')):
        ax = fig.add_subplot(gs[2, col]); panel_label(ax, pl, x=-0.3 if col == 0 else -0.16)
        for k, name, colr, mk in show:
            if k not in cy:
                continue
            xs = np.array([min(r[key], 2e3) for r in cy[k]['per_seed'].values()]); ys = np.array([r['pd_beta'] for r in cy[k]['per_seed'].values()])
            cx = np.median(xs) if key == 'ratio' else xs.mean(); cy_ = ys.mean()     # ratios as medians, as in the text
            ax.errorbar(cx, cy_, xerr=np.clip([[cx - xs.min()], [xs.max() - cx]], 0, None), yerr=np.clip([[cy_ - ys.min()], [ys.max() - cy_]], 0, None), fmt='none', ecolor=colr, elinewidth=0.8, capsize=0, alpha=0.7, zorder=1)
            ax.plot(cx, cy_, mk, color=colr, ms=7 if mk != '*' else 10, label=name, mec='white', mew=0.6, zorder=2)
        ax.axhline(hb, color=C['blue'], lw=0.7, ls=':')
        if col == 0:  # same line in D; labeled once (no marker-free spot next to the line in D)
            ax.text(0.99, hb, 'Healthy, unstimulated', fontsize=7, color=C['blue'], va='bottom', ha='right', transform=ax.get_yaxis_transform())
        ax.set_xlabel(lab); ax.set_ylabel('PD-block α–β power (µV²)' if col == 0 else '')
        if key == 'ratio':
            ax.set_xscale('log'); ax.set_xlim(0.5, 3e3); ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(['1', '10', '100', '≥1000'])
            ax.axvline(1, color=C['ink2'], lw=0.6, ls=':', zorder=0)
            ax.annotate('', xy=(1500, 0.9), xytext=(40, 0.9), xycoords=ax.get_xaxis_transform(), arrowprops=dict(arrowstyle='->', color=C['ink2'], lw=0.7))
            ax.text(250, 0.92, 'more state-selective', transform=ax.get_xaxis_transform(), fontsize=7, color=C['ink2'], ha='center', va='bottom')
        else:
            ax.set_xlim(0, 105)
        if col == 1:
            h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.045), handletextpad=0.3, columnspacing=1.2)
    save(fig, 'fig_adaptation')


# ------------------------------------------------------------------ Figure: controllers
def whiskers(ax, xs, ys, col, cx=None, cy=None, lw=0.7, alpha=0.6):
    """Range over patients (min to max) in x and y around the summary point."""
    cx = np.mean(xs) if cx is None else cx; cy = np.mean(ys) if cy is None else cy
    ax.errorbar(cx, cy, xerr=np.clip([[cx - np.min(xs)], [np.max(xs) - cx]], 0, None), yerr=np.clip([[cy - np.min(ys)], [np.max(ys) - cy]], 0, None),
                fmt='none', ecolor=col, elinewidth=lw, capsize=0, alpha=alpha, zorder=1)


def fig_controllers():
    cy = J('data/revision/final_cycling/summary.json'); sw = J('data/revision/simple_sweep/summary.json'); bl = J('data/revision/eval_baselines_round4/summary.json')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={'wspace': 0.3, 'width_ratios': [1.0, 1.1]})
    ax = axes[0]; panel_label(ax, 'A')
    swr, fcr = load_runs('data/revision/simple_sweep'), load_runs('data/revision/final_cycling'); U, Cd = fcr['unstim'], fcr['cdbs']
    groups = [('Dual threshold, 212 µA when on', lambda k: k.startswith('dual') and k.endswith('_sm10') and '_212' in k, C['ink2'], 'o'),
              ('Dual threshold, cDBS when on', lambda k: re.match(r'dual_\d+_\d+_sm10$', k), C['ink2'], 's'),
              ('Integral amplitude control', lambda k: k.startswith('integral') and k.endswith('_sm10'), C['ink2'], '^')]
    for name, sel, col, mk in groups:
        keys = [k for k in sw if sel(k)]
        for k in keys:
            whiskers(ax, patients(swr[k], U, Cd, 'charge_pct'), patients(swr[k], U, Cd, 'pd_beta'), col, lw=0.5, alpha=0.35)
        pts = sorted((sw[k]['charge_pct'], sw[k]['pd_beta']) for k in keys)
        ax.plot(*zip(*pts), mk, color=col, ms=5, mfc='white', mew=1.0, label=name, zorder=2)
    for k, name, col, mk in (('RT4_p202', 'SNN teacher', C['orange'], 'o'), ('xylosim_xn_s0.03_w500', 'Deployed SNN student (XyloSim)', C['blue'], 'D'), ('mlp_mean80_p202', 'MLP', C['aqua'], 'P'),
                             ('gru_raw80_p202', 'GRU', C['aqua'], 'X'), ('constant_180_0.06_212', 'Constant 212 µA', C['yellow'], 's'), ('oracle_180_0.06_250', 'True-state switching', C['ink'], '*')):
        xs, ys = patients(fcr[k], U, Cd, 'charge_pct'), patients(fcr[k], U, Cd, 'pd_beta')
        ax.plot(xs, ys, mk, color=col, ms=5 if mk == '*' else 3.6, mfc=col if mk == '*' else 'none', mew=0.8, alpha=0.5 if mk == '*' else 0.75, zorder=2)
        ax.plot(np.mean(xs), np.mean(ys), mk, color=col, ms=9 if mk == '*' else 6.5, label=name, mec='white', mew=0.5, zorder=3)
    hb = np.mean([v['h_beta_unstim'] for v in cy.values()]); ax.axhline(hb, color=C['blue'], lw=0.7, ls=':'); ax.text(0.99, hb, 'Healthy, unstimulated', fontsize=7, color=C['blue'], va='bottom', ha='right', transform=ax.get_yaxis_transform())
    ax.set_xlabel('Total charge (% of cDBS)'); ax.set_ylabel('PD-block α–β power (µV²)'); ax.set_xlim(0, 36); ax.set_ylim(145, 300)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, handletextpad=0.3, columnspacing=1.0)
    ax = axes[1]; panel_label(ax, 'B')
    # one column per architecture at its default hyperparameters (symbol = training patient), plus a sweep column for MLP-80 and GRU
    cols = [('RT4', 'SNN', cy, False), ('mlp_mean80', 'MLP-80', bl, True), ('mlp_mean16', 'MLP-16', bl, False), ('gru_raw80', 'GRU', bl, True), ('lstm_raw80', 'LSTM', bl, False)]
    pmk = {'101': 'o', '202': 's', '303': '^'}; pcol = {'101': C['violet'], '202': C['yellow'], '303': C['aqua']}   # training patient: shape and fill
    x = 0; ticks, sub, heads = [], [], []
    for gi, (k, name, src, has_sweep) in enumerate(cols):
        x0 = x; col = C['orange'] if k == 'RT4' else C['aqua']
        for r in sorted(r for r in src if r.startswith(k + '_p') and 'lr' not in r):
            pat = r.split('_p')[-1][:3]; jit = {'101': -0.17, '202': 0.0, '303': 0.17}[pat] + {'_s2': 0.055, '_s3': -0.055}.get(r[-3:], 0.0)  # SNN agent seeds side by side
            ax.plot(x + jit, max(min(src[r]['ratio_median'], 2e3), 0.7), pmk[pat], mfc=pcol[pat], mec=C['ink'], ms=6, mew=1.0, alpha=0.9, zorder=2)
        ticks.append(x); sub.append('')
        if has_sweep:
            x += 0.9
            sweep = [max(src[r]['ratio_median'], 0.7) for r in src if r.startswith(k + '_p101_lr')]
            ax.plot(np.full(len(sweep), x), sweep, 'x', color=pcol['101'], ms=5.5, mew=1.4, zorder=2)
        ax.axvspan(x0 - 0.42, x + 0.42, color=C['grid'], alpha=0.6, lw=0, zorder=0)
        heads.append(((x0 + x) / 2, name)); x += 1.25
    ax.set_yscale('log'); ax.axhline(1, color=C['ink'], lw=0.6, ls=':'); ax.set_ylabel('Charge rate ratio, PD / healthy blocks (median)'); ax.set_ylim(0.6, 3e3); ax.set_xlim(-0.5, x - 0.8)
    ax.set_xticks([]); ax.grid(axis='x', visible=False)
    for xc, name in heads:
        ax.annotate(name, (xc, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -4), textcoords='offset points', ha='center', va='top', fontsize=8, color=C['ink'])
    ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(['1', '10', '100', '1000'])
    h = [plt.Line2D([], [], ls='', marker=m, mfc=pcol[p], mec=C['ink'], mew=1.0, ms=6, label=f'Training patient {p}') for p, m in pmk.items()]
    h.append(plt.Line2D([], [], ls='', marker='x', color=pcol['101'], ms=5.5, mew=1.4, label='Sweep (patient 101)'))
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, handletextpad=0.3)
    save(fig, 'fig_controllers')


# ------------------------------------------------------------------ Figure: distillation / Xylo
def fig_distillation():
    st = J('data/revision/eval_students_round4/summary.json'); xs = J('data/revision/eval_xylosim_round4/summary.json')
    esr, xsr, r4r = load_runs('data/revision/eval_students_round4'), load_runs('data/revision/eval_xylosim_round4'), load_runs('data/revision/eval_round4')
    U, Cd = r4r['unstim'], r4r['cdbs']
    fig = plt.figure(figsize=(7.2, 5.4)); gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.55, height_ratios=[1.0, 0.95])
    # A: each student's closed-loop mean (five patients), grouped by its firing-rate target; boxes over students
    students = {k: v for k, v in st.items() if v.get('rho') is not None and not (v['lam'] == 0 and k != 'scratch_s0.03_w0')}  # the three lambda = 0 runs are one student
    groups = [('Teacher', lambda k, v: False), ('None', lambda k, v: v['lam'] == 0), ('0.05', lambda k, v: v['lam'] > 0 and v['rho'] == 0.05),
              ('0.03', lambda k, v: v['lam'] > 0 and v['rho'] == 0.03), ('0.015', lambda k, v: v['lam'] > 0 and v['rho'] == 0.015)]
    rng = np.random.default_rng(0)
    for col_i, (key, ylab, log) in enumerate((('charge_pct', 'Total charge (% of cDBS)', False), ('ratio_median', 'Charge rate ratio,\nPD / healthy (median)', True))):
        ax = fig.add_subplot(gs[0, col_i]); panel_label(ax, 'A' if col_i == 0 else '', x=-0.32)
        for gi, (name, sel) in enumerate(groups):
            if name == 'Teacher':
                ax.plot(gi, st['RT4_p202'][key], 'o', color=C['orange'], ms=6, mec='white', mew=0.5); continue
            mem = [(k, v) for k, v in students.items() if sel(k, v)]
            vals = [v[key] for _, v in mem]
            if len(vals) > 1:
                ax.boxplot(vals, positions=[gi], widths=0.55, whis=(0, 100), patch_artist=True, medianprops=dict(color=C['ink'], lw=1.2),
                           boxprops=dict(facecolor=C['grid'], edgecolor=C['ink2'], lw=0.8), whiskerprops=dict(color=C['ink2'], lw=0.8), capprops=dict(color=C['ink2'], lw=0.8))
            else:
                ax.plot(gi, vals[0], 'o', color=C['ink2'], ms=5, mfc='none', mew=1.0)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=7, rotation=35, ha='right', rotation_mode='anchor'); ax.grid(axis='x', visible=False)
        ax.set_ylabel(ylab); ax.set_xlabel('Firing-rate target ρ', fontsize=7.5)
        if log:
            ax.set_yscale('log'); ax.set_ylim(10, 400); ax.set_yticks([10, 100]); ax.set_yticklabels(['10', '100'])
        else:
            ax.set_ylim(0, 25)
    # B: action agreement on recorded windows, each Xylo-native student
    ax = fig.add_subplot(gs[0, 2]); panel_label(ax, 'B', x=-0.35)
    checks = sorted(glob.glob(os.path.join(ROOT, 'models/revision/students_round4/xn_*/xylo_check.json')))
    vals = [(json.load(open(p))['xylosim'], os.path.basename(os.path.dirname(p))) for p in checks]
    for xi, key in enumerate(('agree_joint', 'agree_clear')):
        ys = [v[key] for v, _ in vals]
        for (v, name), y, dx in zip(vals, ys, np.linspace(-0.12, 0.12, len(vals))):
            dep = name == 'xn_s0.03_w500'
            ax.plot(xi + dx, y, 'D', color=C['blue'], ms=5.5 if dep else 4, mfc=C['blue'] if dep else 'none', mew=0.8)
        ax.hlines(np.median(ys), xi - 0.25, xi + 0.25, color=C['ink'], lw=1.0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['All\nwindows', 'Margin\n≥ 0.15'], fontsize=7); ax.set_xlim(-0.5, 1.5); ax.set_ylim(0.6, 1.0); ax.grid(axis='x', visible=False)
    ax.set_ylabel('Same three actions as\nfloat32 (fraction of windows)')
    # C: float32 against the Xylo simulator, per patient, for the four Xylo-native students
    pairs = sorted((k, 'xylosim_' + k) for k in st if k.startswith('xn_') and 'xylosim_' + k in xs)
    for j, (key, lab, lim) in enumerate((('pd_beta', 'PD-block α–β power (µV²)', (170, 270)), ('charge_pct', 'Total charge (% of cDBS)', (0, 32)), ('h_off', 'Healthy windows off (%)', (80, 100)))):
        ax = fig.add_subplot(gs[1, j]); panel_label(ax, 'C' if j == 0 else '', x=-0.32)
        ax.plot(lim, lim, color=C['ink2'], lw=0.7, ls='--', zorder=0)
        for a, b in pairs:
            dep = a == 'xn_s0.03_w500'
            ax.plot(patients(esr[a], U, Cd, key), patients(xsr[b], U, Cd, key), 'D', color=C['blue'], ms=5 if dep else 3.8, mfc=C['blue'] if dep else 'none', mew=0.8, alpha=1 if dep else 0.8)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('float32'); ax.set_ylabel('XyloSim'); ax.set_title(lab, fontsize=7.5, pad=4)
    h = [plt.Line2D([], [], ls='', marker='o', color=C['orange'], ms=6, label='Teacher'),
         plt.Line2D([], [], ls='', marker='D', mfc='none', color=C['blue'], ms=4.5, label='Xylo-native students'),
         plt.Line2D([], [], ls='', marker='D', color=C['blue'], ms=5.5, label='Deployed student')]
    fig.legend(handles=h, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.995), handletextpad=0.3, columnspacing=1.4)
    save(fig, 'fig_distillation')


# ------------------------------------------------------------------ Figure: robustness
def fig_robustness():
    fams = [('delay', 'Observation delay\n(windows)', ['none', 'delay1', 'delay3'], ['0', '1', '3']), ('drop', 'Spike loss\n(fraction)', ['none', 'drop0.1', 'drop0.3'], ['0', '10%', '30%']),
            ('ins', 'Added spikes\n(Hz / neuron)', ['none', 'ins5', 'ins20'], ['0', '5', '20']), ('pd', 'Disease severity\n(× full change)', ['pd0.1', 'pd0.2', 'pd0.3', 'none'], ['0.1', '0.2', '0.3', '1']),
            ('blk', 'Block length\n(s)', ['blk30', 'none', 'blk120', 'blk60_jit0.5'], ['30', '60', '120', 'R'])]
    ctrls = [('RT4_p202', 'SNN teacher', C['orange'], 'o'), ('xylosim_xn_s0.03_w500', 'Deployed SNN student (XyloSim)', C['blue'], 'D'),
             ('mlp_mean80_p202', 'MLP', C['aqua'], 'P'), ('gru_raw80_p202', 'GRU', C['aqua'], 'X'), ('constant_180_0.06_212', 'Constant 212 µA', C['yellow'], 's')]
    base, runs = AR.load(os.path.join(ROOT, 'data/revision/final_cycling')), {'rb': AR.load(os.path.join(ROOT, 'data/revision/final_robustness')), 'bk': AR.load(os.path.join(ROOT, 'data/revision/final_blocks'))}

    def per_patient(fam, pt, c, key):
        R = runs['bk' if fam == 'blk' else 'rb']
        if pt == 'none':
            src, U, Cd = base.get((c, ''), {}), base[('unstim', '')], base[('cdbs', '')]
        else:
            src, U, Cd = R.get((c, pt), {}), R.get(('unstim', pt)) or base[('unstim', '')], R.get(('cdbs', pt)) or base[('cdbs', '')]
        return np.array(patients(src, U, Cd, key))
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 5.4), gridspec_kw={'wspace': 0.35, 'hspace': 0.45, 'width_ratios': [3, 3, 3, 4, 4]})
    for j, (fam, title, pts, xl) in enumerate(fams):
        for ci, (c, name, col, mk) in enumerate(ctrls):
            off = (ci - (len(ctrls) - 1) / 2) * 0.11; xs = np.arange(len(pts)) + off
            for row, key, center in ((0, 'pd_red', np.mean), (1, 'ratio', np.median)):
                vals = [np.clip(per_patient(fam, pt, c, key), 0.7 if row else -100, 2e3) for pt in pts]
                ys = np.array([center(v) for v in vals]); ax = axes[row, j]
                nl = 3 if fam == 'blk' else len(pts)          # the random-length point is not on the 30-120 s axis
                ls = '--' if c.startswith('gru') else '-'
                ax.plot(xs[:nl], ys[:nl], ls, color=col, lw=0.8, alpha=0.8, zorder=2)
                ax.plot(xs, ys, mk, color=col, ms=4.5, mec='white', mew=0.4, zorder=3)
                if (row, j) == (0, 0):
                    ax.plot([], [], ls + mk, color=col, ms=4.5, lw=0.8, mec='white', mew=0.4, label=name)
        axes[0, j].set_title(title, fontsize=7.5, pad=5); axes[1, j].set_yscale('log'); axes[1, j].set_ylim(0.6, 3e3); axes[1, j].axhline(1, color=C['ink'], lw=0.5)
        for r in range(2):
            axes[r, j].set_xticks(range(len(pts))); axes[r, j].set_xticklabels(xl, fontsize=7); axes[r, j].set_xlim(-0.5, len(pts) - 0.5); axes[r, j].grid(axis='x', visible=False)
        axes[0, j].set_ylim(-5, 62)
        axes[0, j].text(-0.08, 1.3, 'ABCDE'[j], transform=axes[0, j].transAxes, fontsize=11, fontweight='bold', va='top', ha='right', color=C['ink'])
        if j:
            axes[0, j].set_yticklabels([]); axes[1, j].set_yticklabels([])
    axes[0, 0].set_ylabel('PD-block α–β reduction (%)'); axes[1, 0].set_ylabel('Charge rate ratio, PD / healthy')
    axes[1, 0].set_yticks([1, 10, 100, 1000]); axes[1, 0].set_yticklabels(['1', '10', '100', '1000'])
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04), handletextpad=0.4, columnspacing=1.6)
    save(fig, 'fig_robustness')


# ------------------------------------------------------------------ Supplementary: every value behind the main-text summaries
def sfig_simple_per_patient():
    """Every patient for every smoothed simple-controller setting of Fig. 4A, one panel per family."""
    sw = J('data/revision/simple_sweep/summary.json'); swr, fcr = load_runs('data/revision/simple_sweep'), load_runs('data/revision/final_cycling'); U, Cd = fcr['unstim'], fcr['cdbs']
    fams = [('Dual threshold, 212 µA when on', r'dual_(\d+)_(\d+)_180_0.06_212_sm10$', 'o', lambda m: f'{m[0]}/{m[1]}'),
            ('Dual threshold, cDBS when on', r'dual_(\d+)_(\d+)_sm10$', 's', lambda m: f'{m[0]}/{m[1]}'),
            ('Integral amplitude control', r'integral_(\d+)_(\d+)_180_0.06_sm10$', None, lambda m: f'{m[0]}, gain {m[1]}')]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.9), sharey=True, gridspec_kw={'wspace': 0.12})
    hb = np.mean([v['h_beta_unstim'] for v in J('data/revision/final_cycling/summary.json').values()])
    for ax, (title, pat, mk, lab), L in zip(axes, fams, 'ABC'):
        keys = sorted((k for k in sw if re.match(pat, k)), key=lambda k: [int(x) for x in re.match(pat, k).groups()])
        if mk is None:   # integral: color by target, symbol by gain
            tg = sorted({int(re.match(pat, k).group(1)) for k in keys}); cmap = dict(zip(tg, plt.cm.viridis(np.linspace(0.05, 0.8, len(tg)))))
            cols = [cmap[int(re.match(pat, k).group(1))] for k in keys]; mks = [{'2': 'v', '5': '^', '10': 'D'}[re.match(pat, k).group(2)] for k in keys]
        else:
            cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(keys))); mks = [mk] * len(keys)
        for k, col, mk in zip(keys, cols, mks):
            xs, ys = patients(swr[k], U, Cd, 'charge_pct'), patients(swr[k], U, Cd, 'pd_beta')
            ax.plot(xs, ys, mk, color=col, ms=4, mfc='none', mew=0.8)
            ax.plot(np.mean(xs), np.mean(ys), mk, color=col, ms=7, mec='white', mew=0.5, label=lab(re.match(pat, k).groups()))
        ax.axhline(hb, color=C['blue'], lw=0.7, ls=':')
        ax.set_title(title, fontsize=7.5, pad=4); ax.set_xlabel('Total charge (% of cDBS)'); panel_label(ax, L, x=-0.02 if L != 'A' else -0.16, y=1.16)
        ax.legend(fontsize=7, loc='lower left', handletextpad=0.2, borderaxespad=0.2, labelspacing=0.2, ncol=1, title='Bounds (µV²)' if 'Dual' in title else 'Target (µV²)', title_fontsize=7)
    axes[0].set_ylabel('PD-block α–β power (µV²)'); axes[0].set_ylim(140, 320)
    save(fig, 'Sfig_simple_per_patient')


def sfig_students():
    """Fig. 5A with every distilled student."""
    st = J('data/revision/eval_students_round4/summary.json')
    students = {k: v for k, v in st.items() if v.get('rho') is not None and not (v['lam'] == 0 and k != 'scratch_s0.03_w0')}
    groups = [('Teacher', lambda k, v: False), ('None', lambda k, v: v['lam'] == 0), ('0.05', lambda k, v: v['lam'] > 0 and v['rho'] == 0.05),
              ('0.03', lambda k, v: v['lam'] > 0 and v['rho'] == 0.03), ('0.015', lambda k, v: v['lam'] > 0 and v['rho'] == 0.015)]
    kind = lambda k: ('Xylo-native', 'D') if k.startswith('xn_') else ('Initialized from the teacher', 's') if k.startswith('init_') else ('Without linear biases', '^') if k.startswith('nb_') \
        else ('Temperature or loss variant', 'v') if k.startswith('T') else ('Standard', 'o')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={'wspace': 0.3})
    for ax, (key, ylab, log), L in zip(axes, (('charge_pct', 'Total charge (% of cDBS)', False), ('ratio_median', 'Charge rate ratio, PD / healthy (median)', True)), 'AB'):
        panel_label(ax, L, x=-0.15)
        for gi, (name, sel) in enumerate(groups):
            if name == 'Teacher':
                ax.plot(gi, st['RT4_p202'][key], 'o', color=C['orange'], ms=7, mec='white', mew=0.5); continue
            mem = sorted((k, v) for k, v in students.items() if sel(k, v))
            if len(mem) > 1:
                ax.boxplot([v[key] for _, v in mem], positions=[gi], widths=0.6, whis=(0, 100), patch_artist=True, medianprops=dict(color=C['ink'], lw=1.0),
                           boxprops=dict(facecolor=C['grid'], edgecolor=C['ink2'], lw=0.7), whiskerprops=dict(color=C['ink2'], lw=0.7), capprops=dict(color=C['ink2'], lw=0.7))
            for i, (k, v) in enumerate(mem):
                lab, mk = kind(k); dep = k == 'xn_s0.03_w500'
                ax.plot(gi - 0.22 + 0.44 * (i + 0.5) / len(mem), v[key], mk, color=C['blue'] if lab == 'Xylo-native' else C['ink2'], ms=6 if dep else 4.5,
                        mfc=C['blue'] if dep else 'white', mew=0.9, zorder=3)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups]); ax.set_xlabel('Firing-rate target ρ'); ax.set_ylabel(ylab); ax.grid(axis='x', visible=False)
        if log:
            ax.set_yscale('log'); ax.set_ylim(10, 400); ax.set_yticks([10, 100]); ax.set_yticklabels(['10', '100'])
        else:
            ax.set_ylim(0, 25)
    h = [plt.Line2D([], [], ls='', marker=m, color=c, mfc='white', mew=0.9, ms=5, label=l) for l, m, c in
         (('Standard', 'o', C['ink2']), ('Initialized from the teacher', 's', C['ink2']), ('Without linear biases', '^', C['ink2']), ('Temperature or loss variant', 'v', C['ink2']), ('Xylo-native', 'D', C['blue']))]
    h += [plt.Line2D([], [], ls='', marker='D', color=C['blue'], ms=6, label='Deployed student'), plt.Line2D([], [], ls='', marker='o', color=C['orange'], ms=7, label='Teacher')]
    fig.legend(handles=h, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.02), handletextpad=0.3, columnspacing=1.2)
    save(fig, 'Sfig_students')


def sfig_robustness_per_patient():
    """Fig. 7 with every patient."""
    fams = [('delay', 'Observation delay\n(windows)', ['none', 'delay1', 'delay3'], ['0', '1', '3']), ('drop', 'Spike loss\n(fraction)', ['none', 'drop0.1', 'drop0.3'], ['0', '10%', '30%']),
            ('ins', 'Added spikes\n(Hz / neuron)', ['none', 'ins5', 'ins20'], ['0', '5', '20']), ('pd', 'Disease severity\n(× full change)', ['pd0.1', 'pd0.2', 'pd0.3', 'none'], ['0.1', '0.2', '0.3', '1']),
            ('blk', 'Block length\n(s)', ['blk30', 'none', 'blk120', 'blk60_jit0.5'], ['30', '60', '120', 'R'])]
    ctrls = [('RT4_p202', 'SNN teacher', C['orange'], 'o'), ('xylosim_xn_s0.03_w500', 'Deployed SNN student (XyloSim)', C['blue'], 'D'),
             ('mlp_mean80_p202', 'MLP', C['aqua'], 'P'), ('gru_raw80_p202', 'GRU', C['aqua'], 'X'), ('constant_180_0.06_212', 'Constant 212 µA', C['yellow'], 's')]
    base, runs = AR.load(os.path.join(ROOT, 'data/revision/final_cycling')), {'rb': AR.load(os.path.join(ROOT, 'data/revision/final_robustness')), 'bk': AR.load(os.path.join(ROOT, 'data/revision/final_blocks'))}

    def per_patient(fam, pt, c, key):
        R = runs['bk' if fam == 'blk' else 'rb']
        if pt == 'none':
            src, U, Cd = base.get((c, ''), {}), base[('unstim', '')], base[('cdbs', '')]
        else:
            src, U, Cd = R.get((c, pt), {}), R.get(('unstim', pt)) or base[('unstim', '')], R.get(('cdbs', pt)) or base[('cdbs', '')]
        return np.array(patients(src, U, Cd, key))
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 6.2), gridspec_kw={'wspace': 0.35, 'hspace': 0.45, 'width_ratios': [3, 3, 3, 4, 4]})
    for j, (fam, title, pts, xl) in enumerate(fams):
        for ci, (c, name, col, mk) in enumerate(ctrls):
            off = (ci - (len(ctrls) - 1) / 2) * 0.15
            for row, key, center in ((0, 'pd_red', np.mean), (1, 'ratio', np.median)):
                ax = axes[row, j]
                for xi, pt in enumerate(pts):
                    v = np.clip(per_patient(fam, pt, c, key), 0.7 if row else -100, 2e3)
                    ax.plot(np.full(len(v), xi + off), v, mk, color=col, ms=3.2, mfc='none', mew=0.7, alpha=0.7, zorder=2)
                    ax.plot([xi + off - 0.09, xi + off + 0.09], [center(v)] * 2, color=C['ink'], lw=2.4, solid_capstyle='butt', zorder=3)
                    ax.plot([xi + off - 0.09, xi + off + 0.09], [center(v)] * 2, color=col, lw=1.4, solid_capstyle='butt', zorder=4)
                if (row, j) == (0, 0):
                    ax.plot([], [], mk, color=col, mfc='none', mew=0.8, ms=4.5, label=name)
        axes[0, j].set_title(title, fontsize=7.5, pad=5); axes[1, j].set_yscale('log'); axes[1, j].set_ylim(0.6, 3e3); axes[1, j].axhline(1, color=C['ink'], lw=0.5)
        for r in range(2):
            axes[r, j].set_xticks(range(len(pts))); axes[r, j].set_xticklabels(xl, fontsize=7); axes[r, j].set_xlim(-0.5, len(pts) - 0.5); axes[r, j].grid(axis='x', visible=False)
            for xi in range(len(pts) - 1):
                axes[r, j].axvline(xi + 0.5, color=C['grid'], lw=0.8)
        axes[0, j].set_ylim(-5, 65)
        axes[0, j].text(-0.08, 1.3, 'ABCDE'[j], transform=axes[0, j].transAxes, fontsize=11, fontweight='bold', va='top', ha='right', color=C['ink'])
        if j:
            axes[0, j].set_yticklabels([]); axes[1, j].set_yticklabels([])
    axes[0, 0].set_ylabel('PD-block α–β reduction (%)'); axes[1, 0].set_ylabel('Charge rate ratio, PD / healthy')
    axes[1, 0].set_yticks([1, 10, 100, 1000]); axes[1, 0].set_yticklabels(['1', '10', '100', '1000'])
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03), handletextpad=0.4, columnspacing=1.6)
    save(fig, 'Sfig_robustness_per_patient')


if __name__ == '__main__':
    fig_adaptation(); fig_controllers(); fig_distillation(); fig_robustness()
    sfig_simple_per_patient(); sfig_students(); sfig_robustness_per_patient()
