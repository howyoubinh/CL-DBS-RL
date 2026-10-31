"""
Source-data archive: one CSV per figure panel and supplementary table, written from the same
summary files the figures are drawn from.

  python scripts/write_rev_source_data.py   -> data/revision/source_data/<figure>_<panel>.csv + README.md
"""
import csv
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'data/revision/source_data')
J = lambda p: json.load(open(os.path.join(ROOT, p)))
LOG = []


def write(name, header, rows, note):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + '.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header)
        w.writerows([[c.replace('oracle_', 'true_state_') if isinstance(c, str) else c for c in r] for r in rows])
    LOG.append(f'- `{name}.csv`: {note} ({len(rows)} rows)')


def per_seed_rows(summary, labels=None):
    rows = []
    for lab, v in summary.items():
        if labels and lab not in labels:
            continue
        for s, r in (v.get('per_seed') or v.get('seeds') or {}).items():
            rows.append([lab, s, f"{r['pd_beta']:.2f}", f"{r['pd_red']:.2f}", f"{r['h_beta']:.2f}", f"{r['h_beta_unstim']:.2f}", f"{r['charge_pct']:.3f}", f"{r['pd_charge_pct']:.3f}", f"{r['h_charge_pct']:.3f}", f"{min(r['ratio'], 1e6):.3f}", f"{r['h_off']:.2f}"])
    return rows


H = ['controller', 'patient_seed', 'pd_block_beta', 'pd_beta_reduction_pct', 'healthy_block_beta', 'healthy_unstim_beta', 'charge_pct_cdbs', 'pd_charge_pct_cdbs', 'healthy_charge_pct_cdbs', 'pd_healthy_charge_ratio', 'healthy_windows_off_pct']

# Fig 3 (adaptation): A-B trace, C-D per-patient summary
d = J('data/revision/final_cycling/cycling_xylosim_xn_s0.03_w500_w100_seed42.json'); u = J('data/revision/final_cycling/cycling_unstim_w100_seed42.json'); c = J('data/revision/final_cycling/cycling_constant_180_0.06_212_w100_seed42.json')
write('fig3_AB_trace_patient42', ['window', 'time_s', 'pd_state', 'beta_unstim', 'beta_constant212', 'beta_student_xylosim', 'student_freq_hz', 'student_pw_ms', 'student_amp_uA'],
      [[i, i * 0.1, d['pd_state'][i], f"{u['lfp'][i]:.3f}", f"{c['lfp'][i]:.3f}", f"{d['lfp'][i]:.3f}", d['freq'][i], d['pw'][i], d['amp'][i]] for i in range(len(d['lfp']))], 'Fig. 3A-B, per 100 ms window')
cy = J('data/revision/final_cycling/summary.json')
write('fig3_CD_cycling_per_patient', H, per_seed_rows({k: v for k, v in cy.items() if not k.startswith('published')}), 'Fig. 3C-D, every condition, cycling, 10 patients')
# Fig 4 (controllers)
sw = J('data/revision/simple_sweep/summary.json')
write('fig4_A_simple_controllers', ['controller', 'n_patients', 'pd_block_beta_mean', 'pd_beta_reduction_pct', 'charge_pct_cdbs', 'pd_healthy_charge_ratio_median', 'healthy_windows_off_pct'],
      [[k, v['n'], f"{v['pd_beta']:.2f}", f"{v['pd_red']:.2f}", f"{v['charge_pct']:.3f}", f"{min(v['ratio_median'], 1e6):.3f}", f"{v['h_off']:.2f}"] for k, v in sorted(sw.items())], 'Fig. 4A, simple-controller sweep (5 patients; _sm10 = 1 s smoothed beta)')
bl = J('data/revision/eval_baselines_round4/summary.json')
write('fig4_B_baselines_per_patient', H, per_seed_rows(bl) + per_seed_rows(cy, ['RT4_p101', 'RT4_p202', 'RT4_p303']), 'Fig. 4B and Supplementary Table S4, baselines incl. sweep and SNN teachers, 10 patients')
# Fig 5 (distillation)
st = J('data/revision/eval_students_round4/summary.json'); xs = J('data/revision/eval_xylosim_round4/summary.json')
rows = []
for k, v in st.items():
    if v.get('rho') is None or 'pd_beta' not in v:
        continue
    rows.append([k, v['rho'], v['lam'], f"{v['agree']:.3f}", f"{v['rate']:.4f}", f"{v['pd_beta']:.2f}", f"{v['charge_pct']:.3f}", f"{min(v['ratio_median'], 1e6):.3f}", f"{v['h_off']:.2f}"])
write('fig5_A_students', ['student', 'rho', 'lambda', 'heldout_joint_agreement', 'hidden_rate', 'pd_block_beta', 'charge_pct_cdbs', 'pd_healthy_charge_ratio_median', 'healthy_windows_off_pct'], rows, 'Fig. 5A and Supplementary Table S5, students, cycling, 5 patients')
rows = []
for k, v in xs.items():
    rows.append([k, f"{v['pd_beta']:.2f}", f"{v['pd_red']:.2f}", f"{v['charge_pct']:.3f}", f"{min(v['ratio_median'], 1e6):.3f}", f"{v['h_off']:.2f}"])
write('fig5_C_fp32_vs_xylosim', ['run', 'pd_block_beta', 'pd_beta_reduction_pct', 'charge_pct_cdbs', 'pd_healthy_charge_ratio_median', 'healthy_windows_off_pct'], rows, 'Fig. 5C (means; per patient in fig5_AC_per_patient), students in float32 and on XyloSim, cycling, 5 patients')
rows = []
for p in sorted(glob.glob(os.path.join(ROOT, 'models/revision/students_round4/xn_*/xylo_check.json'))):
    v = json.load(open(p)); x = v['xylosim']
    rows.append([os.path.basename(os.path.dirname(p)), v['n'], f"{x['agree_joint']:.3f}", f"{x['agree_clear']:.3f}", f"{min(x['q_corr']):.4f}", f"{v.get('vmem_out_over_threshold_max', float('nan')):.3f}", v.get('output_resets', '')])
write('fig5_B_xylo_check', ['student', 'n_windows', 'joint_agreement', 'agreement_margin_ge_0.15', 'min_q_correlation', 'max_vmem_over_threshold', 'output_resets'], rows, 'Fig. 5B, XyloSim vs fp32 on recorded windows')
# Fig 7 (robustness) + blocks
for name, path, note in (('fig7_ABCD_robustness', 'data/revision/final_robustness/summary.json', 'Fig. 7A-D, perturbations, 5 patients'), ('fig7_E_blocks', 'data/revision/final_blocks/summary.json', 'Fig. 7E, block lengths, 5 patients')):
    rb = J(path); rows = []
    for pt, cs in rb.items():
        for cname, v in cs.items():
            rows.append([pt, cname, v['n'], f"{v['pd_beta']:.2f}", f"{v['pd_red']:.2f}", f"{v['h_beta']:.2f}", f"{v['charge_pct']:.3f}", f"{min(v['ratio_median'], 1e6):.3f}", f"{v['h_off']:.2f}"])
    write(name, ['perturbation', 'controller', 'n_patients', 'pd_block_beta', 'pd_beta_reduction_pct', 'healthy_block_beta', 'charge_pct_cdbs', 'pd_healthy_charge_ratio_median', 'healthy_windows_off_pct'], rows, note)
# Per-patient values behind the five-patient panels (Figs. 4A, 5A-B, 7), recomputed from the run records
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_rev_main_figures as MF  # noqa: E402
from analyze_rev_select import per_seed  # noqa: E402


def fmt(lab, s, r, extra=()):
    return [*extra, lab, s, f"{r['pd_beta']:.2f}", f"{r['pd_red']:.2f}", f"{r['h_beta']:.2f}", f"{r['h_beta_unstim']:.2f}", f"{r['charge_pct']:.3f}", f"{r['pd_charge_pct']:.3f}", f"{r['h_charge_pct']:.3f}", f"{min(r['ratio'], 1e6):.3f}", f"{r['h_off']:.2f}"]


fcr = MF.load_runs('data/revision/final_cycling'); swr = MF.load_runs('data/revision/simple_sweep')
write('fig4_A_per_patient', H, [fmt(k, s, per_seed(swr[k][s], fcr['unstim'][s], fcr['cdbs'][s])) for k in sorted(swr) if k in sw for s in sorted(swr[k])]
      + [fmt(k, s, per_seed(fcr[k][s], fcr['unstim'][s], fcr['cdbs'][s])) for k in ('RT4_p202', 'xylosim_xn_s0.03_w500', 'mlp_mean80_p202', 'gru_raw80_p202', 'constant_180_0.06_212', 'oracle_180_0.06_250') for s in sorted(fcr[k])],
      'Fig. 4A, every plotted controller, per patient (5 patients)')
r4r = MF.load_runs('data/revision/eval_round4'); esr = MF.load_runs('data/revision/eval_students_round4'); xsr = MF.load_runs('data/revision/eval_xylosim_round4')
rows = [fmt(k, s, per_seed(src[k][s], r4r['unstim'][s], r4r['cdbs'][s])) for src in (esr, xsr) for k in sorted(src) if k not in ('unstim', 'cdbs') for s in sorted(src[k])]
rows += [fmt('RT4_p202', s, per_seed(r4r['RT4_p202'][s], r4r['unstim'][s], r4r['cdbs'][s])) for s in sorted(r4r['RT4_p202'])]
write('fig5_AC_per_patient', H, rows, 'Fig. 5A and C, students in float32 and on XyloSim and the teacher, per patient (5 patients)')
import analyze_rev_robustness as AR  # noqa: E402
base = AR.load(os.path.join(ROOT, 'data/revision/final_cycling')); rows = []
for d in ('final_robustness', 'final_blocks'):
    R = AR.load(os.path.join(ROOT, 'data/revision', d))
    for (c, pt), runs in sorted(R.items()):
        if c in ('unstim', 'cdbs') or not pt:
            continue
        U, Cd = R.get(('unstim', pt)) or base[('unstim', '')], R.get(('cdbs', pt)) or base[('cdbs', '')]
        rows += [fmt(c, s, per_seed(runs[s], U[s], Cd[s]), (pt,)) for s in sorted(runs) if s in U and s in Cd]
for c in ('RT4_p202', 'xn_s0.03_w500', 'xylosim_xn_s0.03_w500', 'mlp_mean80_p202', 'gru_raw80_p202', 'constant_180_0.06_212'):
    rows += [fmt(c, s, per_seed(base[(c, '')][s], base[('unstim', '')][s], base[('cdbs', '')][s]), ('none',)) for s in MF.SEEDS5]
write('fig7_per_patient', ['perturbation'] + H, rows, 'Fig. 7A-E, every perturbation and controller, per patient (5 patients; none = unperturbed 60 s blocks)')
# acute
ac = {k: v for k, v in J('data/revision/final_acute/summary.json').items() if not k.startswith('published')}; rows = []
for k, v in ac.items():
    for red, chg in zip(v['per_seed']['reduction'], v['per_seed']['charge_pct']):
        rows.append([k, f'{red:.2f}', f'{chg:.3f}'])
write('results_acute_per_patient', ['controller', 'beta_reduction_pct', 'charge_pct_cdbs'], rows, 'Results, acute protocol (40 s PD), 10 patients')
# replicate and no-energy-term teachers; hardened teachers
rb = J('data/revision/eval_round4b/summary.json')['controllers']
write('tableS3_teachers_per_patient', H, per_seed_rows({k: v for k, v in rb.items() if k.startswith('RT4')}), 'Supplementary Table S3, replicate and omega-0 teachers, 10 patients')
r5 = J('data/revision/eval_round5/summary.json')['controllers']
write('tableS7_hardened_per_patient', H, per_seed_rows({k: v for k, v in r5.items() if k.startswith('RT4')}), 'Supplementary Table S7, hardened teachers, clean cycling, 10 patients')

# Fig 2 (model behaviour, unstimulated; run_rev_activity.py output): C per-population metrics, E 1 s biomarker
activity = J('data/revision/activity/summary.json')
rows = []
for key, pops in activity.items():
    seed, state = key.replace('seed', '').split('_')
    for pop, m in pops.items():
        if isinstance(m, dict):
            rows.append([seed, state, pop, f"{m['rate']:.3f}", f"{m['sync_within']:.4f}", f"{m['cv_isi']:.4f}", f"{m['burst_frac']:.4f}"])
write('fig2_C_population_metrics', ['patient_seed', 'state', 'population', 'rate_hz', 'synchrony_pairwise_r', 'cv_isi', 'burst_fraction'], rows, 'Fig. 2C (and burst fraction in Results), unstimulated, 3 evaluation patients')
rows = []
for seed in (42, 123, 456):
    for state in ('healthy', 'pd'):
        for w, b in enumerate(J(f'data/revision/activity/seed{seed}_{state}.json')['beta'][1:], start=1):
            rows.append([seed, state, w, f'{b:.2f}'])
write('fig2_E_biomarker_1s', ['patient_seed', 'state', 'window_s', 'gpi_alpha_beta_power_uV2'], rows, 'Fig. 2E, GPi alpha-beta power per 1 s window (first second dropped), 3 patients; spike times and membrane traces for Fig. 2A-B, D are in data/revision/activity/*.json')

hw = os.path.join(ROOT, 'data/revision/mcu_results/table1_numbers.json')
if os.path.exists(hw):
    t = json.load(open(hw)); rows = [[k, f"{v['E_window_J'] * 1e6:.2f}", v.get('P_core_W', ''), v.get('latency_s', v.get('compute_s', '')), v['source']] for k, v in t['rows'].items()]
    write('fig6_table1_hardware', ['item', 'energy_uJ_per_100ms_window', 'core_power_W', 'latency_or_compute_s', 'basis'], rows, 'Fig. 6 and Table 1, energy per decision window (measured / estimated as stated)')
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, 'models/revision/students_round4/xn_s0.03_w500/xylo/hdk_*.json'))):
        d = json.load(open(p))
        for clk, v in d['clocks'].items():
            lat = v['latency_s']; pw = v.get('power_W_inference') or v.get('power_W_processing'); pi = v['power_W_idle']
            rows.append([os.path.basename(p), d.get('mode', 'Vmem readout'), clk, d['n_windows'], f"{lat['mean']:.4f}", f"{pw['DigitalLogic']:.6f}", f"{pi['DigitalLogic']:.6f}", f"{pw['Io']:.6f}", f"{pw['AnalogLogic']:.6f}", v.get('chip_vs_xylosim_agreement', ''), v.get('chip_vs_fp32_agreement', '')])
    write('fig6_xylo_hdk_runs', ['file', 'mode', 'clock_MHz', 'n_windows', 'latency_mean_s', 'core_W_processing', 'core_W_idle', 'io_W', 'afe_W', 'chip_vs_xylosim_agreement', 'chip_vs_fp32_agreement'], rows, 'Fig. 6B and Results (core power per clock), every HDK run')
with open(os.path.join(OUT, 'README.md'), 'w') as f:
    f.write('# Source data\n\nGenerated by `scripts/write_rev_source_data.py` from the summary files under `data/revision/`. '
            'Per-window closed-loop records (biomarker, parameters, disease state per 100 ms decision) for every controller, patient and protocol are the JSON files in '
            '`data/revision/final_cycling`, `final_acute`, `final_robustness`, `final_blocks`, `simple_sweep`, `eval_*`.\n\n' + '\n'.join(LOG) + '\n')
print('\n'.join(LOG)); print('wrote', OUT)
