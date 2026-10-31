"""
Diagnostic figures for a training run (train_rev_teacher.py output folder), for inspecting
what the agent learned.

  <run>/fig_episodes.png   per-episode reward, final setting, final beta, split by disease state
  <run>/fig_steps.png      (runs that saved step_log.npz) per-step traces of the last episodes
                           of each curriculum block: state, setting, raw and smoothed beta, reward
  <run>/fig_policy.png     (runs that saved step_log.npz) charge rate and smoothed beta per step, PD vs healthy,
                           over the course of training (binned by episode)

  python scripts/plot_rev_training.py models/revision/teachers_round4/RT4_p202 [more runs ...]
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'violet': '#4a3aa7'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
HEALTHY, PD = C['blue'], C['orange']
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 150})


def load_results(run):
    p = os.path.join(run, 'training_results.pth')
    if not os.path.exists(p):   # run still in progress: train() checkpoints every 25 episodes
        p = os.path.join(run, 'data/results_rockpool_16ch_curriculum/intermediate_results.pth')
    try:
        return torch.load(p, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(p, map_location='cpu')


def episode_states(cfg, n):
    cyc = cfg['pd_duration'] + cfg['healthy_duration']
    return np.array([1 if (i % cyc) < cfg['pd_duration'] else 0 for i in range(n)])


def fig_episodes(run, cfg, res):
    n = len(res['episode_rewards'])
    st = episode_states(cfg, n)
    ep = np.arange(n)
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    series = [('episode_rewards', 'episode reward', None),
              (None, 'final amplitude (µA) / frequency (Hz)', None),
              ('episode_alpha_beta', 'final-window β', 290.0),
              ('episode_stim_energy', 'summed RMS current per episode', None)]
    for ax, (key, lab, hline) in zip(axes, series):
        for s, col, name in ((0, HEALTHY, 'healthy episode'), (1, PD, 'PD episode')):
            m = st == s
            if key is None:
                amp = np.array([d['amp'] for d in res['episode_dbs_params']])
                frq = np.array([d['freq'] for d in res['episode_dbs_params']])
                ax.plot(ep[m], amp[m], 'o', ms=3.5, color=col, label=f'{name}: amp')
                ax.plot(ep[m], frq[m], 's', ms=3, mfc='none', color=col, label=f'{name}: freq')
            else:
                ax.plot(ep[m], np.asarray(res[key])[m], 'o', ms=3.5, color=col, label=name)
        if hline:
            ax.axhline(hline, color=INK, lw=0.8, ls='--')
        ax.set_ylabel(lab, fontsize=8)
    axes[0].legend(loc='upper left', ncol=2)
    axes[1].legend(loc='upper left', ncol=4, fontsize=7)
    axes[-1].set_xlabel('episode')
    fig.suptitle(f'{os.path.basename(run)}: per-episode summary (starting state of each episode; '
                 f'{"state switches once inside each episode" if cfg.get("transitions") else "state fixed within episodes"})',
                 fontsize=9, color=INK)
    fig.savefig(os.path.join(run, 'fig_episodes.png'), bbox_inches='tight')
    plt.close(fig)


def fig_steps(run, cfg, rows, cols):
    col = {c: i for i, c in enumerate(cols)}
    episodes = np.unique(rows[:, col['episode']]).astype(int)
    cyc = cfg['pd_duration'] + cfg['healthy_duration']
    # last episode of each curriculum half-block, up to 6 panels
    picks = [e for e in episodes if (e + 1) % cfg['pd_duration'] == 0 or (e + 1) % cyc == 0]
    picks = (picks[-6:] if picks else list(episodes[-6:]))
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 2.3 * len(picks)), sharex=True, squeeze=False)
    for ax, e in zip(axes[:, 0], picks):
        r = rows[rows[:, col['episode']] == e]
        t = r[:, col['step']]
        pd = r[:, col['PD']]
        for s0, s1 in zip(*[np.nonzero(np.diff(np.r_[0, pd, 0]) == v)[0] for v in (1, -1)]):
            ax.axvspan(t[s0] - 0.5, t[min(s1, len(t) - 1)] + 0.5, color=PD, alpha=0.08, lw=0)
        ax.plot(t, r[:, col['beta']], color=INK2, lw=0.7, alpha=0.6, label='β (window)')
        ax.plot(t, r[:, col['beta_smooth']], color=INK, lw=1.4, label='β (1 s mean)')
        ax.axhline(cfg['tau_beta_max'], color=INK, lw=0.8, ls='--')
        ax.set_ylabel(f'ep {e}\nβ', fontsize=8)
        ax2 = ax.twinx()   # same x; separate quantity (stimulation) shown as a step trace, not a second data axis
        q = r[:, col['freq']] * r[:, col['pw']] * r[:, col['amp']] / (130 * 0.3 * 300) * 100
        ax2.step(t, q, where='mid', color=C['aqua'], lw=1.4, label='charge rate, % cDBS')
        ax2.set_ylim(0, 160)
        ax2.set_ylabel('charge rate\n% cDBS', fontsize=8, color=C['aqua'])
        ax2.grid(False)
        ax2.spines['right'].set_visible(True)
    axes[0, 0].legend(loc='upper left', ncol=2, fontsize=7)
    axes[-1, 0].set_xlabel('step within episode (100 ms); shaded = PD')
    fig.suptitle(f'{os.path.basename(run)}: per-step traces, last episode of each curriculum block', fontsize=9, color=INK)
    fig.savefig(os.path.join(run, 'fig_steps.png'), bbox_inches='tight')
    plt.close(fig)


def fig_policy(run, cfg, rows, cols):
    col = {c: i for i, c in enumerate(cols)}
    ep = rows[:, col['episode']]
    pd = rows[:, col['PD']]
    q = rows[:, col['freq']] * rows[:, col['pw']] * rows[:, col['amp']] / (130 * 0.3 * 300) * 100
    bs = rows[:, col['beta_smooth']]
    rew = rows[:, col['reward']]
    nb = max(1, int(ep.max() + 1) // 10)
    edges = np.linspace(0, ep.max() + 1, nb + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for s, colr, name in ((0, HEALTHY, 'healthy windows'), (1, PD, 'PD windows')):
        m = pd == s
        x, yq, yb, yr = [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            mm = m & (ep >= a) & (ep < b)
            if mm.sum() > 0:
                x.append((a + b) / 2); yq.append(q[mm].mean()); yb.append(bs[mm].mean()); yr.append(rew[mm].mean())
        axes[0].plot(x, yq, 'o-', color=colr, lw=1.6, ms=4, label=name)
        axes[1].plot(x, yb, 'o-', color=colr, lw=1.6, ms=4)
        axes[2].plot(x, yr, 'o-', color=colr, lw=1.6, ms=4)
    axes[0].set_ylabel('mean charge rate, % cDBS')
    axes[1].set_ylabel('mean β (1 s mean)')
    axes[1].axhline(cfg['tau_beta_max'], color=INK, lw=0.8, ls='--')
    axes[2].set_ylabel('mean reward per step')
    axes[2].set_xlabel('training episode (binned)')
    axes[0].legend()
    fig.suptitle(f'{os.path.basename(run)}: what the policy does in PD vs healthy windows as training proceeds '
                 '(adaptation = the two charge curves separating)', fontsize=9, color=INK)
    fig.savefig(os.path.join(run, 'fig_policy.png'), bbox_inches='tight')
    plt.close(fig)


def main():
    for run in sys.argv[1:]:
        cfg = json.load(open(os.path.join(run, 'config.json')))
        res = load_results(run)
        fig_episodes(run, cfg, res)
        p = os.path.join(run, 'step_log.npz')
        if os.path.exists(p):
            d = np.load(p)
            fig_steps(run, cfg, d['rows'], list(d['columns']))
            fig_policy(run, cfg, d['rows'], list(d['columns']))
        print('wrote figures in', run)


if __name__ == '__main__':
    main()
