"""
Xylo pre-check: does a model with a non-spiking Q readout (large output threshold) survive the
Xylo deployment path?

Builds the XyloAudio 3 deployment graph (quant_fidelity_lib) with the model's output
threshold, maps + channel-quantizes it, runs bit-accurate XyloSim
in Vmem mode on recorded windows (state reset before each window, as in run_rev_eval.py)
and compares the per-head argmax of the summed output membrane with the fp32 model. Reports:
  - quantized output threshold (16-bit clamp) and the max |Vmem_out| seen -> output spikes?
  - per-head and joint action agreement fp32 vs XyloSim, split by the fp32 margin
  - the same for the deployment graph in fp32 (no quantization) to separate the two effects
Writes <run>/xylo_check.json and the quantized spec to <run>/xylo/hw_config.json.

  python scripts/run_rev_xylo_check.py --run models/revision/students_round4/xn_s0.03_w500 \
      --windows data/revision/distill/RT4_p202/windows.npz --n 300
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from scripts.quant_fidelity_lib import (build_corrected_deployment_graph, load_fp32_model,  # noqa: E402
                                        sum_pool_80_to_16, avg_pool_80_to_16, make_xylosim_factory)


def heads(q):
    return np.stack([q[..., 3 * j:3 * j + 3].argmax(-1) for j in range(3)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--windows', default='data/revision/distill/RT4_p202/windows.npz')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.run, 'config.json')))
    thr = float(cfg.get('out_threshold', 1.0))
    ckpt = os.path.join(args.run, 'model.pth')
    d = np.load(args.windows)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(d['X']), size=min(args.n, len(d['X'])), replace=False)
    X = d['X'][idx].astype(np.float32)          # (n, 100, 80)

    # fp32 reference (the model as evaluated: avg-pool input, reset per window)
    from scripts.train_rev_teacher import XYLO_TAUS
    taus = (XYLO_TAUS['mem'], XYLO_TAUS['syn']) if cfg.get('taus') == 'xylo' else (100e-3, 50e-3)
    fp = load_fp32_model(ckpt, 'cpu', thr, *taus)
    q_fp = []
    with torch.no_grad():
        for x in X:
            fp.reset_state() if hasattr(fp, 'reset_state') else None
            from src.utils import rev_rl_utils as R
            R.reset_state(fp)
            _, mem, _, _, _ = fp(torch.as_tensor(avg_pool_80_to_16(x)).unsqueeze(0))
            q_fp.append(mem.sum(1)[0].numpy())
    q_fp = np.stack(q_fp)

    # Export threshold for the output layer. channel_quantize scales each output neuron's weights to
    # max 127 and its threshold by the same factor; if the scaled threshold exceeds 32767 it instead
    # scales by 32767/threshold, which with threshold 1e6 rounds every output weight to zero. Use the
    # largest threshold that keeps the weight scaling (per neuron: 32767/127 * max|w_out|), checked
    # against the largest per-step |Vmem_out| the fp32 model produces so the output still never spikes.
    w_out = fp.model[4].weight.data.abs().max(0).values.numpy()       # (9,) max |w| into each output
    thr_export = float(0.95 * 32767 / 127 * w_out.min()) if thr > 100 else thr
    vstep = []
    with torch.no_grad():
        for x in X[:100]:
            R.reset_state(fp)
            _, mem, _, _, _ = fp(torch.as_tensor(avg_pool_80_to_16(x)).unsqueeze(0))
            vstep.append(float(mem.abs().max()))
    print(f'export output threshold {thr_export:.1f} (fp32 units); fp32 max per-step |Vmem_out| {max(vstep):.1f} '
          f'-> {"ok, never spikes" if max(vstep) < thr_export else "WARNING: output would spike"}')

    # deployment graph in fp32 (no quantization): sum-pool integer input, biases folded
    net = build_corrected_deployment_graph(ckpt, 'cpu', thr_export, *taus)
    q_ex = []
    with torch.no_grad():
        for x in X:
            net.reset_state()
            out, _, rec = net(torch.as_tensor(sum_pool_80_to_16(x), dtype=torch.float32).unsqueeze(0), record=True)
            vm = rec['5_LIFTorch']['vmem'][0] if '5_LIFTorch' in rec else list(rec.values())[-1]['vmem'][0]
            q_ex.append(vm.sum(0).numpy())
    q_ex = np.stack(q_ex)

    # map + quantize
    from rockpool.devices.xylo import syns65302 as xa3
    from rockpool.transform import quantize_methods as q
    spec = xa3.mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')
    spec['dt'] = 10e-3
    qs = q.channel_quantize(**spec)
    for k in ['dt', 'aliases']:
        if k in spec:
            qs[k] = spec[k]
    for k in ['dash_mem', 'dash_mem_out', 'dash_syn', 'dash_syn_2', 'dash_syn_out']:
        if k in qs:
            qs[k] = np.abs(qs[k]).astype(np.uint8)
    os.makedirs(os.path.join(args.run, 'xylo'), exist_ok=True)
    hw = os.path.join(args.run, 'xylo', 'hw_config.json')
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in qs.items()}, open(hw, 'w'))
    thr_out_q = np.asarray(qs['threshold_out']).ravel()
    print(f'output threshold export {thr_export:g} -> quantized {thr_out_q.min()}..{thr_out_q.max()} (16-bit clamp 32767); '
          f'hidden thresholds {np.asarray(qs["threshold"]).min()}..{np.asarray(qs["threshold"]).max()}')

    factory = make_xylosim_factory(hw)
    q_sim, vpos = [], np.zeros(9)
    resets = 0        # output spike = reset: Vmem drops by >= half its threshold in one step
    for x in X:
        mod = factory()
        inp = sum_pool_80_to_16(x).astype(int)
        out, _, rec = mod(inp, record=True)
        vm = np.asarray(out)                     # (T, 9) Vmem_out
        q_sim.append(vm.sum(0)); vpos = np.maximum(vpos, vm.max(0))
        resets += int((np.diff(vm, axis=0) < -0.5 * thr_out_q[None, :]).sum())
    q_sim = np.stack(q_sim)
    ratio = vpos / thr_out_q
    print(f'XyloSim: per-neuron max Vmem_out / threshold_out: max {ratio.max():.2f} (neuron {ratio.argmax()}); '
          f'reset events {resets}; weights_out nonzero {np.count_nonzero(np.asarray(qs["weights_out"]))}/{np.asarray(qs["weights_out"]).size}')

    res = {'run': args.run, 'n': int(len(X)), 'out_threshold': thr, 'out_threshold_export': thr_export, 'vmem_step_max_fp32': max(vstep), 'threshold_out_quant': thr_out_q.tolist(),
           'vmem_out_over_threshold_max': float(ratio.max()), 'output_resets': resets}
    a_fp = heads(q_fp)
    from scripts.quant_fidelity_lib import fit_calibration, apply_calibration
    half = len(X) // 2
    calib = fit_calibration(q_sim[:half], q_fp[:half])
    q_cal = np.stack([apply_calibration(qq, calib) for qq in q_sim])
    a_fp, q_fp_all = heads(q_fp), q_fp
    for name, qq in (('exact_graph', q_ex), ('xylosim', q_sim), ('xylosim_cal', q_cal[half:])):
        if name == 'xylosim_cal':
            q_fp = q_fp_all[half:]; a_fp = heads(q_fp)
        else:
            q_fp = q_fp_all; a_fp = heads(q_fp)
        a = heads(qq)
        per = (a == a_fp).mean(0); joint = float((a == a_fp).all(1).mean())
        gap = np.stack([np.sort(q_fp[:, 3 * j:3 * j + 3], 1)[:, 2] - np.sort(q_fp[:, 3 * j:3 * j + 3], 1)[:, 1] for j in range(3)], 1)
        clear = (a == a_fp)[gap >= 0.15].mean() if (gap >= 0.15).any() else float('nan')
        corr = [float(np.corrcoef(q_fp[:, j], qq[:, j])[0, 1]) for j in range(9)]
        print(f'{name:12s}: agreement freq/pw/amp {per[0]:.2f}/{per[1]:.2f}/{per[2]:.2f} joint {joint:.2f}; '
              f'where fp32 gap >= 0.15: {clear:.2f}; Q corr per neuron min {min(corr):.3f} median {np.median(corr):.3f}')
        res[name] = {'agree_heads': per.tolist(), 'agree_joint': joint, 'agree_clear': float(clear), 'q_corr': corr}
    json.dump(res, open(os.path.join(args.run, 'xylo_check.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
