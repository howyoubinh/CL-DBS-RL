#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deployment graph and controllers for running a spiking student on the XyloAudio 3.

The float32 student takes the 16-channel AvgPool1d(5) average of the 80 recorded channels
(values k/5, k = 0..5) and has biases on its linear layers. The XyloAudio 3 digital core takes
integer input events and carries biases on its neurons. The deployment graph built here maps
the float32 network onto that form without retraining:

1. Input. The 80 channels are sum-pooled into 16 integer counts (0..5), and the first linear
   layer is scaled by 1/POOL_SIZE. Since (W0/POOL) @ (POOL * x_avg) = W0 @ x_avg, the graph
   computes the same first-layer input as the float32 model.

2. Biases. Each linear-layer bias is folded into the bias of the following LIF neurons. A
   linear bias enters the synaptic current and reaches the membrane through the synaptic DC
   gain 1/(1 - exp(-dt/tau_syn)), whereas a neuron bias adds to the membrane directly, so the
   folded value is scaled by that gain. Students trained without linear biases have nothing to
   fold, and the mapping is exact.

3. Optional output calibration. A host-side affine map per output neuron, q_cal = s*q + o,
   can be fit once against the float32 network and applied before the decode. The decode runs
   on the host, so the calibration needs no change on the chip.

Decode: the output membrane is summed over the window, and the argmax within each of the three
heads gives the increment in {-1, 0, +1} for frequency, pulse width and amplitude. Decisions
near ties are sensitive to quantization, so closed-loop efficacy is the main fidelity measure
alongside per-window agreement.

make_xylosim_factory builds fresh bit-accurate XyloSim modules from a hw_config.json (samna
and xylosim must be installed; no board is needed). All controllers keep their recurrent state
across decision windows and reset only at episode start, like the deployed chip.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

from src.models.rockpool_dqsn import RockpoolDQSN

# --- Architecture / deployment constants (must match training & export) -------
N_OBS_RAW = 80          # raw CBGT channels (10 neurons x 8 populations)
POOL_SIZE = 5           # 80 -> 16 pooling kernel/stride
N_OBS = N_OBS_RAW // POOL_SIZE  # 16
N_HIDDEN = 128
N_ACT = 9               # 3 heads x 3 options
BETA = 0.95
DT = 10e-3
TAU_MEM = 100e-3
TAU_SYN = 50e-3
# DC gain of the synaptic filter for a constant input (folds a linear bias,
# which enters i_syn, into an equivalent membrane-side Xylo neuron bias).
SYNAPTIC_DC_GAIN = 1.0 / (1.0 - np.exp(-DT / TAU_SYN))

# Closed-loop DBS integration (initial-submission controller defaults).
DEFAULT_LEAP = {"freq": 5.0, "pw": 0.1, "amp": 5.0}
DEFAULT_INIT_DBS = {"freq": 40.0, "pw": 0.3, "amp": 300.0}
DBS_BOUNDS = {"freq": (0.0, 180.0), "pw": (0.06, 0.4), "amp": (0.0, 250.0)}


# --- Pooling front ends --------------------------------------------------------
def avg_pool_80_to_16(obs80):
    """Fractional AvgPool1d(5) front end (fp32 model input). (T,80)->(T,16)."""
    o = np.asarray(obs80, dtype=np.float32)
    return o.reshape(o.shape[0], N_OBS, POOL_SIZE).mean(axis=2).astype(np.float32)


def sum_pool_80_to_16(obs80):
    """Integer sum-pool front end (Xylo input, equal to POOL_SIZE * the average). (T,80)->(T,16)."""
    o = np.asarray(obs80, dtype=np.float32)
    return o.reshape(o.shape[0], N_OBS, POOL_SIZE).sum(axis=2).astype(np.float32)


# --- WTA decode ----------------------------------------------------------------
def decode_heads(q9):
    """argmax per head over a 9-vector of accumulated membrane -> increments."""
    q = np.asarray(q9).reshape(-1)
    return (
        int(q[0:3].argmax()) - 1,
        int(q[3:6].argmax()) - 1,
        int(q[6:9].argmax()) - 1,
    )


# --- Output calibration (host-side per-neuron affine) --------------------------
def fit_calibration(q_quant, q_fp32):
    """Least-squares per-output-neuron affine mapping quant Q -> fp32 Q.

    Args:
        q_quant, q_fp32: (N, 9) arrays of accumulated output membrane collected
            on the SAME observations.
    Returns:
        dict with 'slope' (9,) and 'offset' (9,).
    """
    q_quant = np.asarray(q_quant, dtype=float)
    q_fp32 = np.asarray(q_fp32, dtype=float)
    slope = np.zeros(N_ACT)
    offset = np.zeros(N_ACT)
    for j in range(N_ACT):
        slope[j], offset[j] = np.polyfit(q_quant[:, j], q_fp32[:, j], 1)
    return {"slope": slope.tolist(), "offset": offset.tolist()}


def apply_calibration(q9, calib):
    if calib is None:
        return np.asarray(q9, dtype=float).reshape(-1)
    s = np.asarray(calib["slope"], dtype=float)
    o = np.asarray(calib["offset"], dtype=float)
    return s * np.asarray(q9, dtype=float).reshape(-1) + o


def save_calibration(calib, path):
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)


def load_calibration(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# --- Checkpoint loading --------------------------------------------------------
def load_student_state_dict(ckpt_path, device="cpu"):
    """Load a student checkpoint into a bare RockpoolDQSN state_dict.

    New-format checkpoints are saved from nn.Sequential(downsampler, RockpoolDQSN)
    so weights carry a leading '1.' prefix (index-0 AvgPool is parameter-free).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if any(k.startswith("1.") for k in sd):
        sd = {k[2:]: v for k, v in sd.items() if k.startswith("1.")}
    return sd


def load_fp32_model(ckpt_path, device="cpu", out_threshold=1.0, tau_mem=TAU_MEM, tau_syn=TAU_SYN):
    """fp32 student (bare RockpoolDQSN, accumulated-membrane decode).

    out_threshold: output-layer LIF threshold (1.0 for the initial-submission models; the
    revised models use a non-spiking readout, config.json 'out_threshold')."""
    sd = load_student_state_dict(ckpt_path, device)
    model = RockpoolDQSN(N_OBS, N_HIDDEN, BETA, N_ACT, 100, 1,
                         dt=DT, use_mempot=True, out_threshold=out_threshold,
                         linear_bias=any(k.endswith('LinearTorch.bias') for k in sd),
                         tau_mem=tau_mem, tau_syn=tau_syn).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def build_corrected_deployment_graph(ckpt_path, device="cpu", out_threshold=1.0, tau_mem=TAU_MEM, tau_syn=TAU_SYN):
    """Rockpool Sequential for export to the XyloAudio 3 (the deployment graph).

    Layer-1 weights are divided by POOL_SIZE (paired with the integer sum-pool input); each
    linear bias is folded into the next LIF neuron bias times SYNAPTIC_DC_GAIN; the linear
    layers have has_bias=False (Xylo carries biases on neurons, not on linear layers).
    """
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.parameters import Constant

    src = load_fp32_model(ckpt_path, device, out_threshold, tau_mem, tau_syn).model
    zb = lambda n: torch.zeros(n)   # bias-free (Xylo-native) checkpoints: nothing to fold
    W0, b0 = src[0].weight.data.clone(), (src[0].bias.data.clone() if src[0].bias is not None else zb(N_HIDDEN))
    W2, b2 = src[2].weight.data.clone(), (src[2].bias.data.clone() if src[2].bias is not None else zb(N_HIDDEN))
    W4, b4 = src[4].weight.data.clone(), (src[4].bias.data.clone() if src[4].bias is not None else zb(N_ACT))

    def lif(n, bias_vec, threshold=1.0):
        return LIFTorch((n,), dt=DT, tau_mem=Constant(float(tau_mem)),
                        tau_syn=Constant(float(tau_syn)), threshold=Constant(float(threshold)),
                        bias=(bias_vec * SYNAPTIC_DC_GAIN).numpy().astype(np.float32))

    net = Sequential(
        LinearTorch((N_OBS, N_HIDDEN), has_bias=False), lif(N_HIDDEN, b0),
        LinearTorch((N_HIDDEN, N_HIDDEN), has_bias=False), lif(N_HIDDEN, b2),
        LinearTorch((N_HIDDEN, N_ACT), has_bias=False), lif(N_ACT, b4, out_threshold),
    )
    net[0].weight.data = W0 / POOL_SIZE
    net[2].weight.data = W2
    net[4].weight.data = W4
    return net


def make_xylosim_factory(hw_config_path, dt=DT):
    """Return a 0-arg factory that builds a fresh bit-accurate XyloSim (Vmem mode).

    A factory (not a single instance) lets each episode start from clean state.
    """
    import rockpool.devices.xylo.syns65302 as xa3
    from rockpool.devices.xylo.syns65302 import XyloSim

    with open(hw_config_path) as f:
        spec = {k: (np.array(v) if isinstance(v, list) else v)
                for k, v in json.load(f).items()}
    cfg = xa3.config_from_specification(**spec)
    cfg = cfg[0] if isinstance(cfg, tuple) else cfg

    def factory():
        mod = XyloSim.from_config(cfg, dt=spec.get("dt", dt), output_mode="Vmem")
        mod.reset_state()
        return mod

    return factory


def make_xylosamna_factory(hw_config_path, dt=DT, clock_freq=6.25):
    """Return a 0-arg factory that builds a fresh XyloSamna on the XyloAudio 3
    development kit (Vmem readout). Requires samna and a connected board. Uses the same
    config_from_specification and output_mode='Vmem' as the XyloSim factory, so the chip
    and XyloSim share the deployed configuration and the host-side decode.
    """
    import rockpool.devices.xylo.syns65302 as xa3
    from rockpool.devices.xylo import find_xylo_hdks
    from rockpool.devices.xylo.syns65302 import xa3_devkit_utils as xa3utils
    from rockpool.devices.xylo.syns65302 import XyloSamna

    with open(hw_config_path) as f:
        spec = {k: (np.array(v) if isinstance(v, list) else v)
                for k, v in json.load(f).items()}
    cfg = xa3.config_from_specification(**spec)
    cfg = cfg[0] if isinstance(cfg, tuple) else cfg

    hdks, _, _ = find_xylo_hdks()
    if not hdks:
        raise RuntimeError("No Xylo HDK found (connect the device for backend='chip').")
    hdk = hdks[0]
    xa3utils.set_xylo_core_clock_freq(hdk, clock_freq)

    def factory():
        # A fresh XyloSamna re-applies the configuration (clears chip state).
        return XyloSamna(hdk, cfg, dt=spec.get("dt", dt),
                         output_mode="Vmem", power_frequency=20.0)

    return factory


# --- Controllers (interface: get_action(spike_matrix_80) -> DBS setting dict) ---
class _DBSController:
    """Shared DBS-parameter integration (leap + safety clip), stateful.

    decode_mode selects how the (T, 9) output-membrane window is reduced to the
    9-vector that the WTA decode acts on:
      'sum' (default) -- accumulate over the window;
      't0'            -- the membrane at timestep 0 (what the initial-submission
                         decoder, src/utils/action_utils_rockpool.py, reads).
    """

    def __init__(self, leap=None, init_dbs=None, decode_mode="sum"):
        self.leap = dict(leap) if leap else dict(DEFAULT_LEAP)
        self.init_dbs = dict(init_dbs) if init_dbs else dict(DEFAULT_INIT_DBS)
        self.decode_mode = decode_mode
        self.last_q_raw = None       # reduced output membrane (pre-calibration)
        self.last_increments = None  # decoded {-1,0,+1}^3
        self.reset()

    def reset(self):
        self.freq = self.init_dbs["freq"]
        self.pw = self.init_dbs["pw"]
        self.amp = self.init_dbs["amp"]
        self._reset_state()

    def _reduce(self, vmem_TxN):
        """Reduce a (T, 9) output-membrane window to the 9-vector for decoding."""
        v = np.asarray(vmem_TxN)
        return v[0] if self.decode_mode == "t0" else v.sum(axis=0)

    def _reset_state(self):
        raise NotImplementedError

    def _q_raw(self, obs80):
        """Return the reduced 9-vector of output membrane for this window."""
        raise NotImplementedError

    def _calibrate(self, q_raw):
        return q_raw

    def get_action(self, obs80):
        q_raw = self._q_raw(obs80)
        inc = decode_heads(self._calibrate(q_raw))
        self.last_q_raw = q_raw
        self.last_increments = inc
        lo_f, hi_f = DBS_BOUNDS["freq"]; lo_p, hi_p = DBS_BOUNDS["pw"]; lo_a, hi_a = DBS_BOUNDS["amp"]
        self.freq = float(np.clip(self.freq + inc[0] * self.leap["freq"], lo_f, hi_f))
        self.pw = float(np.clip(self.pw + inc[1] * self.leap["pw"], lo_p, hi_p))
        self.amp = float(np.clip(self.amp + inc[2] * self.leap["amp"], lo_a, hi_a))
        return {"freq": self.freq, "pw": self.pw, "amp": self.amp}


class Fp32Controller(_DBSController):
    """M_fp32: float student, AvgPool input. decode_mode='sum' (default) or 't0'."""

    def __init__(self, ckpt_path, leap=None, init_dbs=None, device="cpu", decode_mode="sum"):
        self.device = device
        self.model = load_fp32_model(ckpt_path, device)
        super().__init__(leap, init_dbs, decode_mode=decode_mode)

    def _reset_state(self):
        self.model.reset()

    def _q_raw(self, obs80):
        x = torch.from_numpy(avg_pool_80_to_16(obs80)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, mem, _, _, _ = self.model(x)        # mem: (1, T, 9)
        return self._reduce(mem[0].cpu().numpy())  # (T,9) -> 9 per decode_mode


class XyloSimController(_DBSController):
    """M_quant: bit-accurate XyloSim, integer sum-pool input, calibrated decode.

    Subclass `_make_factory`/`_call` to retarget the same input/decode pipeline at
    a different Xylo backend (see XyloChipController).
    """

    def __init__(self, hw_config_path, calibration=None, leap=None, init_dbs=None,
                 decode_mode="sum", **kw):
        self._factory = self._make_factory(hw_config_path, **kw)
        self.calibration = calibration
        self.mod = None
        super().__init__(leap, init_dbs, decode_mode=decode_mode)

    def _make_factory(self, hw_config_path, **kw):
        return make_xylosim_factory(hw_config_path)

    def _reset_state(self):
        self.mod = self._factory()

    def _call(self, x):
        out, _, _ = self.mod(x)
        return out

    def _q_raw(self, obs80):
        vmem = self._call(sum_pool_80_to_16(obs80))
        return self._reduce(vmem)                   # (T,9) -> 9 per decode_mode

    def _calibrate(self, q_raw):
        return apply_calibration(q_raw, self.calibration)


class XyloChipController(XyloSimController):
    """M_chip: the XyloAudio 3 through XyloSamna (Vmem readout). Requires the board and samna.

    Same input front end, decode and calibration as XyloSimController; only the backend
    module differs. XyloSim is a bit-accurate model of the chip's digital core.
    """

    def _make_factory(self, hw_config_path, clock_freq=6.25):
        return make_xylosamna_factory(hw_config_path, clock_freq=clock_freq)

    def _call(self, x):
        out, _, _ = self.mod(x, record_power=False)
        return out


# --- Calibration data collection ----------------------------------------------
def collect_calibration_pairs(ckpt_path, hw_config_path, seeds, num_steps,
                              leap=None, init_dbs=None):
    """Run fp32-driven closed loops; collect paired (quant_Q, fp32_Q) per step.

    fp32 drives the simulator (on-distribution pathological states); the quant
    model replays the SAME observations with state carried, so the two Q-vectors
    are aligned for the affine fit.  Returns (q_quant (N,9), q_fp32 (N,9)).
    """
    from src.simulation.sim_wrapper import SimulationWrapper

    qf_all, qq_all = [], []
    for seed in seeds:
        fp = Fp32Controller(ckpt_path, leap, init_dbs)
        quant = XyloSimController(hw_config_path, None, leap, init_dbs)
        sim = SimulationWrapper(n_neurons=10, t_step=100, dt=0.01, pd=1, seed=seed)
        sim.set_pd_state(1)
        obs = sim.reset()
        for _ in range(num_steps):
            o80 = obs["spike_matrix"]
            action = fp.get_action(o80)            # fp32 drives; sets fp.last_q_raw
            quant.get_action(o80)                   # replay; sets quant.last_q_raw
            qf_all.append(fp.last_q_raw)
            qq_all.append(quant.last_q_raw)
            obs = sim.step(action)
    return np.array(qq_all), np.array(qf_all)
