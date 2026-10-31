"""
Supplementary tables (LaTeX), generated from the summary files:
  stab:teachers   all RT4 teachers (3 patients x 3 seeds) and the omega = 0 ablation, cycling, 10 patients
  stab:baselines  non-spiking baselines per training patient + sweep, cycling, 10 patients
  stab:students   distilled students: rho, lambda, agreement, hidden rate, closed-loop behaviour (5 patients)
  stab:simple     simple controllers on the smoothed beta (5 patients)
  stab:hardening  hardened teachers (data/revision/eval_round5) under perturbation (5 patients)
Writes supp_tables.tex and supp_table_hardening.tex (for \input in the supplementary document) to
$CLDBS_TABLE_DIR (default figures/revision/tables).
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TABLE_DIR = os.environ.get('CLDBS_TABLE_DIR', os.path.join(ROOT, 'figures', 'revision', 'tables'))
J = lambda p: json.load(open(os.path.join(ROOT, p)))
L = []
P = L.append
f0 = lambda v: f'{v:.0f}'
f1 = lambda v: f'{v:.1f}'.replace('-', '$-$')
rat = lambda v: ('$\\geq$1000' if v >= 1000 else f'{v:.0f}')


def table(label, caption, header, rows, ncol, notes=None):
    """rows: lists of cells, or ('group', text) for an italic group heading spanning all columns."""
    P('\\begin{table}[htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{5pt}\n\\caption{' + caption + '}\n\\label{' + label + '}')
    P('\\begin{tabular}{@{}l' + 'r' * (ncol - 1) + '@{}}\n\\toprule\n' + ' & '.join(header) + ' \\\\\n\\midrule')
    first = True
    for r in rows:
        if isinstance(r, tuple):
            P(('' if first else '\\addlinespace\n') + '\\multicolumn{' + str(ncol) + '}{@{}l}{\\textit{' + r[1] + '}} \\\\')
        else:
            P(' & '.join(r) + ' \\\\')
        first = False
    P('\\bottomrule\n\\end{tabular}')
    if notes:
        P('\\par\\smallskip\\footnotesize ' + notes)
    P('\\end{table}\n')


hd = lambda a, b: '\\begin{tabular}[b]{@{}c@{}}' + a + '\\\\' + b + '\\end{tabular}'
M = [hd('PD-block $\\beta$', '($\\mu V^2$)'), hd('Reduction', '(\\%)'), hd('Charge', '(\\% cDBS)'), hd('PD/healthy', 'charge-rate ratio'), hd('Healthy', 'off (\\%)')]
met = lambda v: [f0(v['pd_beta']), f1(v['pd_red']), f1(v['charge_pct']), rat(v['ratio_median']), f0(v['h_off'])]

cy = J('data/revision/final_cycling/summary.json'); rb = J('data/revision/eval_round4b/summary.json')['controllers']
rows = [('group', 'Energy-aware reward')]
for p in ('101', '202', '303'):
    for s in ('1', '2', '3'):
        v = cy[f'RT4_p{p}'] if s == '1' else rb[f'RT4_p{p}_s{s}']
        rows.append([p + ('$^{\\dagger}$' if (p, s) == ('202', '1') else ''), s] + met(v))
rows.append(('group', 'No energy term ($\\omega = 0$)'))
for p in ('101', '202', '303'):
    rows.append([p, '1'] + met(rb[f'RT4d0_p{p}']))
rows.append(('group', 'Comparison conditions'))
for k, n in (('cdbs', 'cDBS'), ('constant_180_0.06_212', 'Constant, 212~$\\mu$A'), ('oracle_180_0.06_250', 'True-state switching, 250~$\\mu$A')):
    rows.append(['\\multicolumn{2}{@{}l}{' + n + '}'] + met(cy[k]))
table('stab:teachers', 'SNN teachers in the cycling protocol on the ten evaluation patients: every teacher trained with the energy-aware reward (three training patients, three agent seeds each) and the three teachers trained without the energy term (agent seed as for seed~1). Values are means over patients, except the parkinsonian-to-healthy charge-rate ratio, which is the median. The unstimulated $\\alpha$-$\\beta$ power is 257~$\\mu V^2$ in healthy and 332~$\\mu V^2$ in parkinsonian blocks. Constant stimulation and true-state switching use 180~Hz and 60~$\\mu$s.',
      [hd('Training', 'patient'), 'Seed'] + M, rows, 7, '$^{\\dagger}$Teacher of the main text.')

bl = J('data/revision/eval_baselines_round4/summary.json')
LR = {'0.0003': '$3 \\times 10^{-4}$', '0.001': '$10^{-3}$', '0.003': '$3 \\times 10^{-3}$'}
names = {'mlp_mean80': 'MLP, rates of all 80 channels', 'mlp_mean16': 'MLP, rates of the 16 pooled channels', 'gru_raw80': 'GRU, 80-channel spike sequence', 'lstm_raw80': 'LSTM, 80-channel spike sequence'}
rows = []
for arch in names:
    rows.append(('group', names[arch]))
    for k in sorted(r for r in bl if r.startswith(arch + '_p') and 'lr' not in r):
        rows.append([k.split('_p')[-1], LR['0.001'], '128'] + met(bl[k]))
    for k in sorted((r for r in bl if r.startswith(arch + '_p101_lr')), key=lambda r: tuple(float(x) for x in r.split('_lr')[1].split('_h'))):
        lr, h = k.split('_lr')[1].split('_h'); rows.append(['101', LR[lr], h] + met(bl[k]))
table('stab:baselines', 'Non-spiking baselines trained with the SNN\'s reward, action space and curriculum, in the cycling protocol on the ten evaluation patients (columns as in Supplementary Table~\\ref{stab:teachers}). Every architecture was trained on each training patient at the default setting (learning rate $10^{-3}$, width 128). The 80-channel MLP and the GRU were also trained at the other eight combinations of learning rate and width on patient~101.',
      [hd('Training', 'patient'), hd('Learning', 'rate'), 'Width'] + M, rows, 8)

st = J('data/revision/eval_students_round4/summary.json'); xs = J('data/revision/eval_xylosim_round4/summary.json')
smet = lambda v: [f0(v['pd_beta']), f1(v['charge_pct']), rat(v['ratio_median']), f0(v['h_off'])]
t = st['RT4_p202']
rows = [['\\multicolumn{4}{@{}l}{Teacher}', f'{next(v["teacher_rate"] for v in st.values() if v.get("teacher_rate")):.2f}'] + smet(t)]
groups = (('scratch_', 'Standard: linear biases, the teacher\'s time constants'), ('init_', 'Standard, initialized from the teacher'),
          ('nb_', 'Without linear biases, with the gap term'), ('T', 'Temperature and loss variants, with linear biases and the teacher\'s time constants'),
          ('xn_', 'Xylo-native: no linear biases, bit-shift time constants, gap term'))
for pre, gname in groups:
    rows.append(('group', gname))
    ks = [k for k in st if k.startswith(pre) and st[k].get('rho') is not None and 'pd_beta' in st[k]]
    ks.sort(key=lambda k: (float(k.split('_')[0][1:]) if pre == 'T' else 0, st[k]['rho'], st[k]['lam'], k))
    seen_l0 = False
    for k in ks:
        v = st[k]
        if v['lam'] == 0:  # without the penalty rho does not enter the loss: the three lambda = 0 runs are one student
            if seen_l0:
                continue
            seen_l0 = True
        T = k.split('_')[0][1:] if pre == 'T' else '4'
        if pre == 'T' and '_m1_' in k:
            T += '$^{*}$'
        lam = f'{v["lam"]:g}' + ('$^{\\dagger}$' if k == 'xn_s0.03_w500' else '')
        rows.append(['--' if v['lam'] == 0 else f'{v["rho"]:g}', lam, T, f'{v["agree"]:.2f}', f'{v["rate"]:.3f}'] + smet(v))
        if 'xylosim_' + k in xs:
            rows.append(['\\multicolumn{5}{@{}l}{\\quad on XyloSim}'] + smet(xs['xylosim_' + k]))
table('stab:students', 'Distilled students of the main-text teacher (patient~202) in the cycling protocol on five evaluation patients. Students were distilled from 10,000 of the teacher\'s closed-loop windows for 30 epochs (standard students) or 60 epochs (all others). Agreement is the fraction of held-out windows on which the student selects all three of the teacher\'s actions; hidden rate is spikes per neuron per step, averaged over the two hidden layers; the gap term is the mean squared error between head-centered Q-values (Methods). Other columns as in Supplementary Table~\\ref{stab:teachers}.',
      ['$\\rho$', '$\\lambda$', '$T_{\\mathrm{KD}}$', 'Agreement', hd('Hidden', 'rate'), M[0], M[2], M[3], M[4]], rows, 9,
      '$^{*}$With the gap term. $^{\\dagger}$Deployed student.')

sw = J('data/revision/simple_sweep/summary.json')
G = {}
for k in sw:
    if not k.endswith('_sm10'):
        continue
    p = k.split('_')
    if k.startswith('dual'):
        g = ('Dual threshold, 180~Hz, 60~$\\mu$s, ' + p[5] + '~$\\mu$A when on') if len(p) > 4 else 'Dual threshold, cDBS when on'
        G.setdefault(g, []).append(((int(p[1]), int(p[2])), [f'{p[1]}/{p[2]}'] + met(sw[k])))
    else:
        g = f'Integral amplitude control, {p[3]}~Hz, {float(p[4]) * 1000:.0f}~$\\mu$s'
        G.setdefault(g, []).append(((int(p[1]), float(p[2])), [f'target {p[1]}, gain {p[2]}'] + met(sw[k])))
rows = []
for g in sorted(G, key=lambda g: (g.startswith('Integral'), 'cDBS' in g, g)):
    rows.append(('group', g)); rows += [r for _, r in sorted(G[g])]
table('stab:simple', 'Dual-threshold and integral amplitude controllers on the 1~s running mean of the GPi $\\alpha$-$\\beta$ estimate, cycling protocol, five evaluation patients. Bounds and targets are in $\\mu V^2$; other columns as in Supplementary Table~\\ref{stab:teachers}.',
      ['Bounds or target'] + M, rows, 6)

r5 = J('data/revision/eval_round5/summary.json')['controllers']
import glob, re, sys
sys.path.insert(0, os.path.join(ROOT, 'scripts')); from analyze_rev_select import per_seed
R = {}
for p in glob.glob(os.path.join(ROOT, 'data/revision/eval_round5/cycling_*_w100*_seed*.json')):
    m = re.match(r'cycling_(.+)_w100((?:_[a-z]+[0-9.]+)*)_seed(\d+)\.json', os.path.basename(p)); R.setdefault((m.group(1), m.group(2).lstrip('_')), {})[int(m.group(3))] = json.load(open(p))
rows = []
for c, n in (('RT4_p202', 'Main-text teacher, all 80 channels'), ('xn_s0.03_w500', 'Deployed student, float32'),
             ('RT4bg_p202', 'Teacher on STN, GPe and GPi'), ('RT4n_p202', 'Teacher on all channels, trained with 5~Hz added spikes'),
             ('RT4bgn_p202', 'Teacher on STN, GPe and GPi, trained with 5~Hz added spikes'), ('RT4bgn_p202_s2', 'The same, second agent seed'),
             ('RT4bgnd_p202', 'Teacher on STN, GPe and GPi, trained with 5~Hz added spikes and 30\\% loss')):
    rows.append(('group', n))
    for pt, pn in (('', 'None'), ('ins5', '5~Hz added'), ('ins20', '20~Hz added'), ('drop0.3', '30\\% spike loss'), ('delay3', '3-window delay')):
        if (c, pt) not in R:
            continue
        U = R.get(('unstim', pt)) or R[('unstim', '')]; Cd = R[('cdbs', '')]
        rs = [per_seed(R[(c, pt)][s], U[s], Cd[s]) for s in sorted(R[(c, pt)]) if s in U and (pt or s in (42, 123, 456, 789, 1024))]
        a = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        a['ratio_median'] = np.median([r['ratio'] for r in rs])
        rows.append([pn] + met(a))
n_main = len(L)
table('stab:hardening', 'Teachers trained on patient~202 with a restricted observation, input augmentation or both, against the main-text teacher and the deployed student, in the cycling protocol without and with observation perturbations, on five evaluation patients (columns as in Supplementary Table~\\ref{stab:teachers}).',
      ['Perturbation'] + M, rows, 6)

# The hardening table sits in its own supplementary section, after the text that cites it
L, H = L[:n_main], L[n_main:]
os.makedirs(TABLE_DIR, exist_ok=True)
for name, lines in (('supp_tables.tex', L), ('supp_table_hardening.tex', H)):
    out = os.path.join(TABLE_DIR, name)
    open(out, 'w').write('% generated by scripts/write_rev_supp_tables.py\n' + '\n'.join(lines))
    print('wrote', out, len(lines), 'lines')
