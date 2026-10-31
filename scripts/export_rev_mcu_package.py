"""
Microcontroller benchmark package: one small network as C headers in
float32 and int8, 100 test input vectors with expected outputs, a numpy reference, and a C test
harness. Primary network: the MLP controller (80 -> 128 -> 128 -> 9, ReLU, time-averaged 80-channel
input), trained in the same environment as the SNN teachers (models/revision/baselines_round4/mlp_mean80_p202).
Optional second arm: the deployed SNN student (16 -> 128 -> 128 -> 9 LIF, 100 steps of 10 ms),
float32 only, with the exact update equations.

  python scripts/export_rev_mcu_package.py [--out data/revision/mcu_package]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from src.models.rev_baselines import build  # noqa: E402

MLP_RUN = os.path.join(ROOT, 'models/revision/baselines_round4/mlp_mean80_p202')
SNN_RUN = os.path.join(ROOT, 'models/revision/students_round4/xn_s0.03_w500')
WINDOWS = os.path.join(ROOT, 'data/revision/distill/RT4_p202/windows.npz')


def c_array(name, a, ctype):
    a = np.asarray(a)
    dims = ''.join(f'[{d}]' for d in a.shape)
    fmt = (lambda v: f'{v:.8e}f') if ctype == 'float' else (lambda v: str(int(v)))
    body = ', '.join(fmt(v) for v in a.ravel())
    return f'static const {ctype} {name}{dims} = {{{body}}};\n'


def round_half_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quant_sym(w, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = float(np.max(np.abs(w))) / qmax
    return np.clip(round_half_away(w / s), -qmax - 1, qmax).astype(np.int32), s


def mlp_float(x, P):
    h1 = np.maximum(x @ P['W1'].T + P['b1'], 0.0)
    h2 = np.maximum(h1 @ P['W2'].T + P['b2'], 0.0)
    return h2 @ P['W3'].T + P['b3']


def mlp_int(xq, Q, bits):
    """Integer pipeline: intN x intN -> int32/int64 accumulate, integer bias, ReLU, requantise with a
    float multiplier (round half away from zero, clip to intN). Final layer returns float Q values."""
    lo, hi = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    acc1 = xq.astype(np.int64) @ Q['W1q'].T.astype(np.int64) + Q['b1q']
    q1 = np.clip(round_half_away(np.maximum(acc1, 0) * Q['M1']), lo, hi)
    acc2 = q1.astype(np.int64) @ Q['W2q'].T.astype(np.int64) + Q['b2q']
    q2 = np.clip(round_half_away(np.maximum(acc2, 0) * Q['M2']), lo, hi)
    acc3 = q2.astype(np.int64) @ Q['W3q'].T.astype(np.int64) + Q['b3q']
    return acc3 * Q['M3']


def quantise_mlp(P, xa, bits):
    """Per-tensor symmetric weights, calibrated input/activation scales (99.99th percentile / qmax)."""
    qmax = 2 ** (bits - 1) - 1
    h1 = np.maximum(xa @ P['W1'].T + P['b1'], 0); h2 = np.maximum(h1 @ P['W2'].T + P['b2'], 0)
    s_x = float(np.percentile(xa, 99.99)) / qmax; s_a1 = float(np.percentile(h1, 99.99)) / qmax; s_a2 = float(np.percentile(h2, 99.99)) / qmax
    W1q, s_w1 = quant_sym(P['W1'], bits); W2q, s_w2 = quant_sym(P['W2'], bits); W3q, s_w3 = quant_sym(P['W3'], bits)
    return {'W1q': W1q, 'W2q': W2q, 'W3q': W3q,
            'b1q': round_half_away(P['b1'] / (s_x * s_w1)).astype(np.int64), 'b2q': round_half_away(P['b2'] / (s_a1 * s_w2)).astype(np.int64),
            'b3q': round_half_away(P['b3'] / (s_a2 * s_w3)).astype(np.int64),
            'M1': s_x * s_w1 / s_a1, 'M2': s_a1 * s_w2 / s_a2, 'M3': s_a2 * s_w3, 's_x': s_x, 's_w': [s_w1, s_w2, s_w3], 's_a': [s_a1, s_a2]}


def write_int_header(path, Q, bits, tag):
    ct, bt = (f'int{bits}_t', 'int32_t' if bits == 8 else 'int64_t')
    with open(path, 'w') as f:
        f.write(f'/* MLP controller, int{bits} symmetric per-tensor quantisation (see README for the arithmetic).\n'
                f'   input: xq = clip(round(x / {tag}_S_X)); weight scales s_w {Q["s_w"][0]:.8e}, {Q["s_w"][1]:.8e}, {Q["s_w"][2]:.8e};\n'
                f'   activation scales {Q["s_a"][0]:.8e}, {Q["s_a"][1]:.8e} (99.99th percentile / {2 ** (bits - 1) - 1}, calibrated on 10,000 windows).\n'
                f'   layer l: acc = sum_i xq[i]*Wq[j][i] + bq[j] ({bt}); hidden: q = clip(round_half_away(max(acc,0) * M), int{bits});\n'
                f'   output: y = acc3 * M3 (float). */\n#pragma once\n#include <stdint.h>\n#define MLP_N_IN 80\n#define MLP_N_HID 128\n#define MLP_N_OUT 9\n')
        f.write(f'static const float {tag}_S_X = {Q["s_x"]:.10e}f;\nstatic const float {tag}_M1 = {Q["M1"]:.10e}f;\n'
                f'static const float {tag}_M2 = {Q["M2"]:.10e}f;\nstatic const float {tag}_M3 = {Q["M3"]:.10e}f;\n')
        f.write(c_array(f'{tag}_W1q', Q['W1q'], ct)); f.write(c_array(f'{tag}_b1q', Q['b1q'], bt))
        f.write(c_array(f'{tag}_W2q', Q['W2q'], ct)); f.write(c_array(f'{tag}_b2q', Q['b2q'], bt))
        f.write(c_array(f'{tag}_W3q', Q['W3q'], ct)); f.write(c_array(f'{tag}_b3q', Q['b3q'], bt))


def heads(q):
    return np.stack([q[..., 3 * j:3 * j + 3].argmax(-1) for j in range(3)], -1)


def snn_float(X16, S, record=False):
    """Rockpool LIFTorch dynamics, feedforward 16 -> 128 -> 128 -> 9, dt 10 ms, 100 steps, state zero
    at the start of every window. Per step and layer: isyn = (isyn + W x) * beta; vmem = alpha * vmem + isyn;
    hidden layers: n_spk = floor(vmem / thr) if vmem >= thr else 0 (multiple spikes per step allowed),
    vmem -= n_spk * thr; output layer: non-spiking, Q = sum over steps of vmem."""
    T = X16.shape[0]
    a, b = S['alpha'], S['beta']
    isyn = [np.zeros(128), np.zeros(128), np.zeros(9)]; vmem = [np.zeros(128), np.zeros(128), np.zeros(9)]
    Ws = [S['W0'], S['W2'], S['W4']]; q = np.zeros(9); nsp = [0, 0]
    for t in range(T):
        x = X16[t]
        for l in range(3):
            isyn[l] = (isyn[l] + Ws[l] @ x) * b
            vmem[l] = a * vmem[l] + isyn[l]
            if l < 2:
                s = np.where(vmem[l] >= 1.0, np.floor(vmem[l]), 0.0)
                vmem[l] = vmem[l] - s; x = s; nsp[l] += s.sum()
            else:
                q += vmem[l]
    return q, nsp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/revision/mcu_package'))
    ap.add_argument('--n', type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    d = np.load(WINDOWS); X = d['X']; PD = d['PD']
    idx = np.concatenate([rng.choice(np.nonzero(PD == 1)[0], args.n // 2, replace=False), rng.choice(np.nonzero(PD == 0)[0], args.n - args.n // 2, replace=False)])
    Xw = X[idx].astype(np.float32)                       # (n, 100, 80) binary 1 ms bins

    # ---------------- MLP
    cfg = json.load(open(os.path.join(MLP_RUN, 'config.json')))
    net = build(cfg['arch'], cfg['input'], cfg['hidden']); net.load_state_dict(torch.load(os.path.join(MLP_RUN, 'model.pth'), map_location='cpu')); net.eval()
    P = {'W1': net.net[0].weight.detach().numpy().astype(np.float32), 'b1': net.net[0].bias.detach().numpy().astype(np.float32),
         'W2': net.net[2].weight.detach().numpy().astype(np.float32), 'b2': net.net[2].bias.detach().numpy().astype(np.float32),
         'W3': net.net[4].weight.detach().numpy().astype(np.float32), 'b3': net.net[4].bias.detach().numpy().astype(np.float32)}
    x = Xw.mean(1)                                       # (n, 80) in [0, 1]
    with torch.no_grad():
        y_torch = net(torch.as_tensor(Xw)).numpy()
    y_f = mlp_float(x.astype(np.float64), {k: v.astype(np.float64) for k, v in P.items()})
    assert np.abs(y_f - y_torch).max() < 1e-3, np.abs(y_f - y_torch).max()
    xa = X.astype(np.float32).mean(1)
    Q8, Q16 = quantise_mlp(P, xa, 8), quantise_mlp(P, xa, 16)
    xq8 = np.clip(round_half_away(x / Q8['s_x']), -128, 127).astype(np.int64); xq16 = np.clip(round_half_away(x / Q16['s_x']), -32768, 32767).astype(np.int64)
    y_q8, y_q16 = mlp_int(xq8, Q8, 8), mlp_int(xq16, Q16, 16)
    agree8 = (heads(y_q8) == heads(y_f)).all(1).mean(); agree16 = (heads(y_q16) == heads(y_f)).all(1).mean()
    with open(os.path.join(args.out, 'mlp_weights_f32.h'), 'w') as f:
        f.write('/* MLP controller 80 -> 128 -> 128 -> 9 (ReLU after layers 1 and 2), float32.\n'
                '   y = W3 relu(W2 relu(W1 x + b1) + b2) + b3; x = mean firing (0..1) of 80 channels over the 100 ms window.\n'
                '   Decision: argmax of y[0:3] (frequency), y[3:6] (pulse width), y[6:9] (amplitude) -> {-1, 0, +1} each. */\n'
                '#pragma once\n#define MLP_N_IN 80\n#define MLP_N_HID 128\n#define MLP_N_OUT 9\n')
        for k in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            f.write(c_array(f'MLP_{k}', P[k], 'float'))
    write_int_header(os.path.join(args.out, 'mlp_weights_int8.h'), Q8, 8, 'MLP8')
    write_int_header(os.path.join(args.out, 'mlp_weights_int16.h'), Q16, 16, 'MLP16')
    with open(os.path.join(args.out, 'mlp_test_vectors.h'), 'w') as f:
        f.write(f'/* {args.n} test windows (first {args.n // 2} PD, then healthy). MLP_X: mean firing per channel (float);\n'
                '   MLP_Xq8 / MLP_Xq16: quantised inputs; MLP_Y_F32: expected float32 outputs; MLP_Y_INT8 / MLP_Y_INT16: expected\n'
                '   outputs of the integer pipelines; MLP_ACT_*: expected per-head argmax (0, 1, 2). */\n#pragma once\n#include <stdint.h>\n'
                f'#define MLP_N_TEST {args.n}\n')
        f.write(c_array('MLP_X', x, 'float')); f.write(c_array('MLP_Xq8', xq8, 'int8_t')); f.write(c_array('MLP_Xq16', xq16, 'int16_t'))
        f.write(c_array('MLP_Y_F32', y_f.astype(np.float32), 'float')); f.write(c_array('MLP_Y_INT8', y_q8.astype(np.float32), 'float')); f.write(c_array('MLP_Y_INT16', y_q16.astype(np.float32), 'float'))
        f.write(c_array('MLP_ACT_F32', heads(y_f), 'int8_t')); f.write(c_array('MLP_ACT_INT8', heads(y_q8), 'int8_t')); f.write(c_array('MLP_ACT_INT16', heads(y_q16), 'int8_t'))
    try:
        torch.onnx.export(net.net, torch.zeros(1, 80), os.path.join(args.out, 'mlp_f32.onnx'), input_names=['mean_rate_80'], output_names=['q_values_9'], opset_version=13)
        print('wrote mlp_f32.onnx')
    except Exception as e:
        print('ONNX export skipped:', e)
    np.savez(os.path.join(args.out, 'mlp_reference.npz'), **P, x=x, y_f32=y_f, y_int8=y_q8, y_int16=y_q16, idx=idx)

    # ---------------- SNN student (optional arm)
    scfg = json.load(open(os.path.join(SNN_RUN, 'config.json')))
    sd = torch.load(os.path.join(SNN_RUN, 'model.pth'), map_location='cpu')
    key = lambda k: next(v for kk, v in sd.items() if kk.endswith(k))
    S = {'W0': key('0_LinearTorch.weight').numpy().T.astype(np.float64), 'W2': key('2_LinearTorch.weight').numpy().T.astype(np.float64),
         'W4': key('4_LinearTorch.weight').numpy().T.astype(np.float64), 'alpha': 1 - 2 ** -3, 'beta': 1 - 2 ** -2}
    # check the numpy reference against the fp32 model on the same windows
    from scripts.train_rev_teacher import make_net
    from src.utils import rev_rl_utils as R
    snn = make_net(100, False, scfg['out_threshold'], 'cpu', scfg['linear_bias'], scfg['taus']); snn[1].q_scale = scfg['q_scale']
    snn.load_state_dict(sd); snn.eval()
    X16 = Xw.reshape(len(Xw), 100, 16, 5).mean(3)       # avg-pool 80 -> 16 channels per 1 ms bin
    # The SNN is simulated at 10 ms per step but the observation has 100 bins of 1 ms; as in the original
    # pipeline, the 100 bins are fed as 100 steps (time dilated x10).
    q_np, nsp = zip(*[snn_float(X16[i], S) for i in range(len(X16))]); q_np = np.stack(q_np)
    with torch.no_grad():
        R.reset_state(snn); _, mem, _, _, rec = snn(torch.as_tensor(Xw)); q_t = mem.sum(1).numpy()
    corr = min(np.corrcoef(q_np[:, j], q_t[:, j])[0, 1] for j in range(9)); agree_snn = (heads(q_np) == heads(q_t)).all(1).mean()
    print(f'SNN numpy reference vs fp32 model: min Q corr {corr:.5f}, joint action agreement {agree_snn:.2f}, max |dQ| {np.abs(q_np - q_t).max():.3g}')
    with open(os.path.join(args.out, 'snn_weights_f32.h'), 'w') as f:
        f.write('/* Deployed SNN student (Xylo-native): 16 -> 128 -> 128 -> 9 leaky integrate-and-fire, feedforward, no biases.\n'
                '   Per window: 100 steps; input per step x16 = mean of 5 neighbouring channels of the 80-channel 1 ms spike bins (0..1).\n'
                '   State zero at the start of each window. Per step, layer by layer (W is [out][in]: W0[128][16], W2[128][128], W4[9][128]):\n'
                '     isyn[j] = (isyn[j] + sum_i W[j][i] * x[i]) * BETA;  vmem[j] = ALPHA * vmem[j] + isyn[j];\n'
                '     hidden layers: n = (vmem >= 1) ? floor(vmem) : 0;  vmem -= n;  x_next = n;\n'
                '     output layer: no spikes; Q += vmem (sum over the 100 steps).\n'
                '   Decision: argmax of Q[0:3], Q[3:6], Q[6:9]. */\n#pragma once\n'
                f'#define SNN_N_IN 16\n#define SNN_N_HID 128\n#define SNN_N_OUT 9\n#define SNN_N_STEPS 100\n'
                f'static const float SNN_ALPHA = {S["alpha"]:.8e}f;\nstatic const float SNN_BETA = {S["beta"]:.8e}f;\n')
        f.write(c_array('SNN_W0', S['W0'], 'float')); f.write(c_array('SNN_W2', S['W2'], 'float')); f.write(c_array('SNN_W4', S['W4'], 'float'))
    with open(os.path.join(args.out, 'snn_test_vectors.h'), 'w') as f:
        f.write(f'/* {args.n} test windows: SNN_X16c[n][100][16] pooled input COUNTS (0..5; the network input is x = count / 5), SNN_Q expected Q (float32, numpy reference),\n'
                '   SNN_ACT expected per-head argmax, SNN_SPIKES expected hidden spike count per window (layer 1 + layer 2). */\n#pragma once\n'
                f'#define SNN_N_TEST {args.n}\n#include <stdint.h>\n')
        f.write(c_array('SNN_X16c', round_half_away(X16 * 5).astype(np.int64), 'uint8_t')); f.write(c_array('SNN_Q', q_np.astype(np.float32), 'float'))
        f.write(c_array('SNN_ACT', heads(q_np), 'int8_t')); f.write(c_array('SNN_SPIKES', np.array([sum(n) for n in nsp]), 'int32_t'))
    np.savez(os.path.join(args.out, 'snn_reference.npz'), W0=S['W0'], W2=S['W2'], W4=S['W4'], alpha=S['alpha'], beta=S['beta'], X16=X16, Q=q_np, Q_torch=q_t)

    ops = {'mlp_macs': 80 * 128 + 128 * 128 + 128 * 9, 'snn_dense_macs_per_window': 100 * (16 * 128 + 128 * 128 + 128 * 9),
           'snn_mean_hidden_spikes_per_window': float(np.mean([sum(n) for n in nsp])),
           'snn_event_synops_per_window': float(np.mean([n[0] * 128 + n[1] * 9 for n in nsp]) + 100 * 16 * 128)}
    json.dump({'mlp': {'run': MLP_RUN, 'int8_vs_f32_joint_action_agreement': float(agree8), 'int16_vs_f32_joint_action_agreement': float(agree16)},
               'snn': {'run': SNN_RUN, 'numpy_vs_torch_min_q_corr': float(corr), 'joint_action_agreement': float(agree_snn)},
               'ops': ops, 'windows': idx.tolist()}, open(os.path.join(args.out, 'package_info.json'), 'w'), indent=1)
    print(f'MLP vs float32 joint action agreement on the {args.n} test windows: int8 {agree8:.2f}, int16 {agree16:.2f}; ops {ops}')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
