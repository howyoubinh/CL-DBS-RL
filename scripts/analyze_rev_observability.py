"""
Observability check, analysis (data from
scripts/run_rev_obs_collect.py): can PD be told from healthy from the 80-neuron observation while
stimulation is on, per 100 ms window and per 1 s window?

Windows: the first 1 s of every run is dropped (settling). 100 ms windows are the controller's
observation; 1 s windows are 10 consecutive 100 ms windows (aligned with the random condition's
1 s settings, so each 1 s window has one setting).

Readouts, from simplest to the controller's own architecture:
  pop7       logistic regression on the 7 population mean rates
  pool16     logistic regression on the 16 pooled channels the SNN sees (mean of 5 adjacent neurons)
  neuron80   logistic regression on the 80 per-neuron rates
  mlp80      MLP (128, 128) on the 80 per-neuron rates
  stn_gpi    logistic regression on the 20 STN + GPi neuron rates (what a DBS lead near the target could
             record; the full observation includes striatum, cortex and thalamus, and the simulator's
             PD condition is set partly in striatum, upstream of the STN stimulation)
  bg_nostr   logistic regression on STN + GPe + GPi rates (30, no striatum)
  mlp_bg     MLP (128, 128) on STN + GPe + GPi rates
  stn_gpi_stim / mlp_bg_stim  the same inputs plus the current stimulation setting (freq, pw, amp,
             and RMS current): the policy is not given its own setting, but it could be
  snn        the teacher's topology (80 -> avg-pool 16 -> LIF 128 -> LIF 128 -> LIF out), output Vmem
             summed over the window, trained as a 2-class classifier on the spike trains
             (binary 1 ms bins at 100 ms; counts in 10 ms bins at 1 s, as for the 1 s-window teachers)
  snn_bg     the same SNN with every channel outside STN, GPe and GPi set to zero

Splits (AUC on held-out windows, reported per stimulation condition):
  lopo       leave one patient out over all 8 patients, trained on all conditions
  single     trained on one training patient (101, 202 or 303; all conditions), tested on the 5
             evaluation patients: the RL setting (one training patient, unseen test patients)
  unstim2stim  trained on unstimulated windows only (LOPO), tested on stimulated windows:
             does a detector learned without stimulation still work under it?
  shuffle    lopo with the state labels permuted across (patient, condition) runs: must be ~0.5

Also reported: per-population mean rates (healthy vs PD) and the single-population AUC per
condition (rank AUC of the population rate, direction-free = max(AUC, 1 - AUC)).

  python scripts/analyze_rev_observability.py [--obs data/revision/obs] [--no-snn]
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

POPS = ('TH', 'STN', 'GPe', 'GPi', 'StrD2', 'StrD1', 'Cor')
POP_SLICES = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80)]
CONDS = ('unstim', 'c212', 'c250', 'cdbs', 'random')
TRAIN_PATIENTS = (101, 202, 303)
SETTLE = 10   # 100 ms windows dropped at the start of each run


def load(obs_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(obs_dir, 'obs_*_seed*.npz'))):
        m = re.match(r'obs_(\w+?)_(pd|healthy)_seed(\d+)\.npz', os.path.basename(p))
        d = np.load(p)
        runs.append({'cond': m.group(1), 'pd': int(m.group(2) == 'pd'), 'seed': int(m.group(3)),
                     'counts': d['counts'][SETTLE:], 'beta': d['beta'][SETTLE:], 'stim': d['stim'][SETTLE:]})
    return runs


def windows(runs, win):
    """Returns X_seq (N, 100, 80) float32, rates (N, 80 + 4) Hz with the stimulation setting
    (freq, pw, amp, RMS current) appended as the last 4 columns, y, seed, cond, run_id."""
    Xs, R, y, seed, cond, rid = [], [], [], [], [], []
    for i, r in enumerate(runs):
        c = r['counts'].astype(np.float32)
        if win == 100:
            seq = np.minimum(c, 1.0)                                    # binary 1 ms bins (hash-like)
            rate = c.sum(1) / 0.1
            st = r['stim']
        else:
            n = (len(c) // 10) * 10
            c = c[:n].reshape(-1, 10, 100, 80).reshape(-1, 1000, 80)    # 1 s at 1 ms
            seq = c.reshape(-1, 100, 10, 80).sum(2)                      # counts in 10 ms bins
            rate = c.sum(1) / 1.0
            st = r['stim'][:n:10]       # one setting per 1 s (random condition holds 1 s)
        st = np.c_[st, st[:, 2] * np.sqrt(st[:, 0] * st[:, 1] / 1000)]
        Xs.append(seq); R.append(np.c_[rate, st])
        k = len(seq)
        y += [r['pd']] * k; seed += [r['seed']] * k; cond += [r['cond']] * k; rid += [i] * k
    return (np.concatenate(Xs), np.concatenate(R), np.array(y), np.array(seed), np.array(cond),
            np.array(rid))


BG = np.r_[10:40]            # STN, GPe, GPi
STN_GPI = np.r_[10:20, 30:40]


def features(rates, kind):
    if kind == 'stn_gpi_stim':
        return rates[:, np.r_[STN_GPI, 80:84]]
    if kind == 'mlp_bg_stim':
        return rates[:, np.r_[BG, 80:84]]
    rates = rates[:, :80]
    if kind == 'stn_gpi':
        return rates[:, STN_GPI]
    if kind in ('bg_nostr', 'mlp_bg'):
        return rates[:, BG]
    if kind == 'pop7':
        return np.stack([rates[:, a:b].mean(1) for a, b in POP_SLICES], 1)
    if kind == 'pool16':
        return rates.reshape(len(rates), 16, 5).mean(2)
    return rates


def sk_model(kind):
    if kind in ('mlp80', 'mlp_bg', 'mlp_bg_stim'):
        return make_pipeline(StandardScaler(), MLPClassifier((128, 128), max_iter=300, early_stopping=True,
                                                             random_state=0))
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float('nan')


def per_cond_auc(y, s, cond):
    return {c: auc(y[cond == c], s[cond == c]) for c in CONDS if (cond == c).any()}


# ---------------------------------------------------------------- SNN classifier
class SNNClassifier:
    def __init__(self, num_steps, device, epochs=20, lr=1e-3, batch=256, seed=0):
        self.num_steps, self.device, self.epochs, self.lr, self.batch, self.seed = num_steps, device, epochs, lr, batch, seed

    def _net(self):
        import torch.nn as nn
        from scripts.train_rev_teacher import AvgPoolDownsampler
        from src.models.rockpool_dqsn import RockpoolDQSN
        rock = RockpoolDQSN(16, 128, 0.95, 2, self.num_steps, self.batch, use_mempot=True, dt=10e-3)
        return nn.Sequential(AvgPoolDownsampler(5, 5), rock).to(self.device)

    def _q(self, X):
        from src.utils.rev_rl_utils import q_values
        q, _, _ = q_values(self.net, X)
        return q

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net = self._net()
        # input scale: count inputs at 1 s are standardized by their overall mean so both window
        # lengths start from comparable drive
        self.scale = float(1.0 / max(X.mean(), 1e-6)) * 0.05
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        Xt = torch.as_tensor(X)
        yt = torch.as_tensor(y, dtype=torch.long)
        for _ in range(self.epochs):
            idx = rng.permutation(len(X))
            for b in range(0, len(idx), self.batch):
                j = idx[b:b + self.batch]
                xb = Xt[j].to(self.device) * self.scale
                q = self._q(xb)
                loss = torch.nn.functional.cross_entropy(q / 100.0, yt[j].to(self.device))
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def decision_function(self, X):
        import torch
        out = []
        with torch.no_grad():
            for b in range(0, len(X), 1024):
                q = self._q(torch.as_tensor(X[b:b + 1024]).to(self.device) * self.scale)
                out.append((q[:, 1] - q[:, 0]).cpu().numpy())
        return np.concatenate(out)


def run_split(split, kind, Xseq, rates, y, seed, cond, rid, device, rng):
    """Returns per-condition AUC dict (for 'single': averaged over the 3 training patients)."""
    def fit_score(tr, te, labels):
        if kind in ('snn', 'snn_bg'):
            X = Xseq
            if kind == 'snn_bg':
                X = np.zeros_like(Xseq)
                X[:, :, BG] = Xseq[:, :, BG]
            m = SNNClassifier(Xseq.shape[1], device).fit(X[tr], labels[tr])
            return m.decision_function(X[te])
        F = features(rates, kind)
        m = sk_model(kind).fit(F[tr], labels[tr])
        return m.decision_function(F[te]) if hasattr(m, 'decision_function') else m.predict_proba(F[te])[:, 1]

    patients = sorted(set(seed))
    if split in ('lopo', 'shuffle', 'unstim2stim'):
        labels = y.copy()
        if split == 'shuffle':      # permute labels across whole runs, so within-run structure is kept
            runs = np.unique(rid)
            perm = dict(zip(runs, rng.permutation([y[rid == r][0] for r in runs])))
            labels = np.array([perm[r] for r in rid])
        score = np.full(len(y), np.nan)
        for p in patients:
            tr = seed != p
            if split == 'unstim2stim':
                tr &= cond == 'unstim'
            te = seed == p
            score[te] = fit_score(tr, te, labels)
        mask = cond != 'unstim' if split == 'unstim2stim' else np.ones(len(y), bool)
        res = per_cond_auc(labels[mask], score[mask], cond[mask])
        res['all'] = auc(labels[mask], score[mask])
        return res
    if split == 'single':
        if not (~np.isin(seed, TRAIN_PATIENTS)).any():
            return {'all': float('nan')}
        per = []
        for p in TRAIN_PATIENTS:
            tr = seed == p
            te = ~np.isin(seed, TRAIN_PATIENTS)
            s = fit_score(tr, te, y)
            r = per_cond_auc(y[te], s, cond[te]); r['all'] = auc(y[te], s)
            per.append(r)
        return {k: float(np.mean([r[k] for r in per])) for k in per[0]} | {'per_patient': per}
    raise ValueError(split)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--obs', default=os.path.join(ROOT, 'data/revision/obs'))
    ap.add_argument('--no-snn', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--kinds', nargs='+', default=None, help='subset of readouts (default: all)')
    ap.add_argument('--splits', nargs='+', default=['lopo', 'single', 'unstim2stim', 'shuffle'])
    ap.add_argument('--windows', nargs='+', type=int, default=[100, 1000])
    ap.add_argument('--tag', default='', help='summary_<tag>.json, so partial runs can go in parallel')
    args = ap.parse_args()

    runs = load(args.obs)
    print(f'{len(runs)} runs, patients {sorted({r["seed"] for r in runs})}')
    out = {'n_runs': len(runs)}

    # descriptive: population rates and beta, healthy vs PD, per condition (100 ms windows)
    desc = {}
    for c in CONDS:
        rows = {}
        for pd in (0, 1):
            rr = [r for r in runs if r['cond'] == c and r['pd'] == pd]
            if not rr:
                continue
            rate = np.concatenate([r['counts'].sum(1) / 0.1 for r in rr])
            rows['pd' if pd else 'healthy'] = {
                'beta_median': float(np.median(np.concatenate([r['beta'] for r in rr]))),
                **{p: float(rate[:, a:b].mean()) for p, (a, b) in zip(POPS, POP_SLICES)}}
        desc[c] = rows
    out['describe'] = desc
    print('\nmean rate (Hz) healthy / PD, 100 ms windows, all patients')
    print(f'{"cond":8s}' + ''.join(f'{p:>14s}' for p in POPS) + f'{"beta med":>14s}')
    for c, rows in desc.items():
        if len(rows) == 2:
            print(f'{c:8s}' + ''.join(f'{rows["healthy"][p]:6.1f}/{rows["pd"][p]:<7.1f}' for p in POPS)
                  + f'{rows["healthy"]["beta_median"]:6.0f}/{rows["pd"]["beta_median"]:<7.0f}')

    rng = np.random.default_rng(0)
    kinds = args.kinds or (['pop7', 'pool16', 'neuron80', 'mlp80', 'stn_gpi', 'bg_nostr', 'mlp_bg', 'stn_gpi_stim',
                             'mlp_bg_stim'] + ([] if args.no_snn else ['snn', 'snn_bg']))
    for win in args.windows:
        Xseq, rates, y, seed, cond, rid = windows(runs, win)
        print(f'\n===== {win} ms windows: {len(y)} ({y.mean():.2f} PD) =====')
        uni = {}
        for c in CONDS:
            m = cond == c
            uni[c] = {p: (lambda a: max(a, 1 - a))(auc(y[m], rates[m, a:b].mean(1)))
                      for p, (a, b) in zip(POPS, POP_SLICES)}
        out[f'w{win}_population_auc'] = uni
        print('single-population AUC (direction-free), all patients pooled')
        print(f'{"cond":8s}' + ''.join(f'{p:>7s}' for p in POPS))
        for c, r in uni.items():
            print(f'{c:8s}' + ''.join(f'{r[p]:7.3f}' for p in POPS))

        res = {}
        for split in args.splits:
            for kind in kinds:
                if split in ('shuffle', 'unstim2stim') and kind in ('mlp80', 'mlp_bg', 'mlp_bg_stim', 'snn', 'snn_bg'):
                    continue   # controls run on the cheap readouts only
                r = run_split(split, kind, Xseq, rates, y, seed, cond, rid, args.device, rng)
                res[f'{split}/{kind}'] = r
                print(f'  {split:12s} {kind:9s} all {r["all"]:.3f}  ' +
                      '  '.join(f'{c} {r[c]:.3f}' for c in CONDS if c in r), flush=True)
        out[f'w{win}'] = res

    path = os.path.join(args.obs, f'summary{"_" + args.tag if args.tag else ""}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
