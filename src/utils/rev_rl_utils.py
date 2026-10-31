"""
Revised Q-function, action selection and optimization step for the spiking DQN.

Q(s) is defined as in the Methods: for each sample independently, the output-layer membrane
potential summed over the observation's time steps, computed from a freshly reset state with
batch-first input. The same function is used for acting, for the training targets and for
evaluation, so the policy that acts is the policy being trained.

This module differs from the initial-submission pipeline (action_utils_rockpool.py,
optimization_utils_rockpool.py) in three design choices:
  1. Action decoding. The output membrane potential has shape (batch=1, time, 9). The
     initial-submission decoder sums axis 0 and takes element [0], which reads the potential
     at the first time step of the window. This decoder sums the potential over the window.
  2. Training input layout. The initial-submission optimization step permutes each minibatch
     to (time, batch, channels). Rockpool modules are batch-first, so with that layout the
     network integrates across the samples of the minibatch. This step keeps the batch-first
     layout, so each sample is integrated over its own time steps.
  3. State reset. RockpoolDQSN.reset() calls layer.reset(), which Rockpool's LIF layers do not
     define, so the network state carries over from call to call. This module resets the state
     with Sequential.reset_state() before computing Q.

Everything else matches the initial-submission optimization step: per-head (freq, pw, amp)
Q-values, per-head max over the target network for the Bellman target, SmoothL1 loss, gradient
value clipping at 100, and the optional sparsity penalty.
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn

from src.environment.replay_memory import Transition
from src.utils.action_utils_rockpool import flatten_action

HEADS = ('freq', 'pw', 'amp')  # output neurons 0-2, 3-5, 6-8 (same order as flatten_action)


def _rockpool(net):
    return net[1] if isinstance(net, nn.Sequential) else net


def reset_state(net):
    """Reset every LIF layer's state (RockpoolDQSN.reset() leaves it unchanged)."""
    _rockpool(net).model.reset_state()


def q_values(net, x):
    """x: (batch, time, channels), batch-first. Returns (Q of shape (batch, 9), sparsity_penalty,
    current_sparsity). State is reset first, so each call depends only on x."""
    reset_state(net)
    spk, mem, sparsity_penalty, current_sparsity, _ = net(x)
    # Rockpool returns the recorded Vmem history on the CPU; bring Q back to the input's device.
    return mem.sum(1).to(x.device) / getattr(_rockpool(net), 'q_scale', 1.0), sparsity_penalty, current_sparsity


def decode(q_row):
    """Per-head argmax of one row of Q (9,) -> {'freq','pw','amp'} in {-1, 0, 1}."""
    return {h: int(torch.argmax(q_row[3 * j:3 * j + 3])) - 1 for j, h in enumerate(HEADS)}


def select_action_rev(policy_net, target_net, state, env, steps_done, device, eps_start, eps_end, eps_decay):
    """Drop-in replacement for action_utils_rockpool.select_action (same signature and returns)."""
    eps = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    if random.random() > eps:
        with torch.no_grad():
            q, _, _ = q_values(policy_net, state)
        return decode(q[0]), [0]
    return env.action_space.sample(), [0]


def optimize_model_rev(policy_net, target_net, optimizer, memory, device, batch_size, gamma):
    """Drop-in replacement for optimization_utils_rockpool.optimize_model."""
    if len(memory) < batch_size:
        return None, 0
    batch = Transition(*zip(*memory.sample(batch_size)))
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    state_batch = torch.cat(batch.state).to(device)          # (B, T, 80), batch-first, no permute
    reward_batch = torch.cat(batch.reward).to(device)

    q, sparsity_penalty, current_sparsity = q_values(policy_net, state_batch)
    state_action_values = q.gather(1, flatten_action(batch.action, device).to(device))   # (B, 3)

    next_state_values = torch.zeros(batch_size, 3, device=device)
    next_states = [s for s in batch.next_state if s is not None]
    if next_states:
        with torch.no_grad():
            qn, _, _ = q_values(target_net, torch.cat(next_states).to(device))
            next_state_values[non_final_mask] = torch.stack(
                [qn[:, 3 * j:3 * j + 3].max(1).values for j in range(3)], dim=1)

    expected = next_state_values * gamma + reward_batch.unsqueeze(1)
    loss = nn.SmoothL1Loss()(state_action_values, expected) + sparsity_penalty
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss.item(), float(current_sparsity)


POPS = {'th': (0, 10), 'stn': (10, 20), 'gpe': (20, 30), 'gpi': (30, 40), 'strd2': (40, 50), 'strd1': (50, 60), 'cor': (60, 80)}


def obs_mask(keep):
    """Boolean (80,) mask of the channels to keep; keep = iterable of POPS names or None (all)."""
    if not keep:
        return None
    m = np.zeros(80, dtype=bool)
    for k in keep:
        a, b = POPS[k]
        m[a:b] = True
    return m


def apply_obs_mask(spikes, mask):
    """Zero the channels outside the mask (channels beyond 80, e.g. setting channels, are kept)."""
    if mask is None:
        return spikes
    spikes = np.array(spikes, dtype=np.float32, copy=True)
    spikes[:, :80][:, ~mask] = 0.0
    return spikes


class CorrectedSNNController:
    """Closed-loop evaluation controller using the revised Q (reset per window, summed readout).

    get_action(spike_matrix_80) -> absolute {'freq','pw','amp'}, applying the per-head increments
    with the training environment's leap values and clipping (gym_pd.MousePDEnv.step). Same
    interface as quant_fidelity_harness controllers, so it can be passed as controller_factory.
    """

    def __init__(self, model_path, device='cpu', leap=None, init_dbs=None, num_steps=100, setting_input=False,
                 out_threshold=1.0, q_scale=1.0, linear_bias=True, taus=None, obs_keep=None):
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from scripts.train_rev_teacher import make_net
        self.setting_input, self.num_steps = setting_input, num_steps
        self.net = make_net(num_steps, setting_input, out_threshold, device, linear_bias, taus)
        self.net[1].q_scale = float(q_scale)
        sd = torch.load(model_path, map_location=device)
        self.net.load_state_dict(sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd)
        self.net.eval()
        self.device = device
        self.leap = leap or {'freq': 5.0, 'pw': 0.1, 'amp': 5.0}
        init = init_dbs or {'freq': 40.0, 'pw': 0.3, 'amp': 300.0}
        self.freq, self.pw, self.amp = init['freq'], init['pw'], init['amp']
        self.applied = (0.0, 0.0, 0.0)   # setting that produced the window about to be decoded (first: unstimulated)
        self.mask = obs_mask(obs_keep)   # deployment-side channel restriction (e.g. basal ganglia only)

    def increments(self, spike_matrix):
        spike_matrix = apply_obs_mask(spike_matrix, self.mask)
        if self.setting_input:   # the setting that produced this window = the controller's current output
            from src.environment.gym_pd_round3 import setting_channels
            spike_matrix = np.concatenate([np.asarray(spike_matrix, dtype=np.float32),
                                           setting_channels(self.applied, self.num_steps)], 1)
        x = torch.as_tensor(spike_matrix, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q, _, _ = q_values(self.net, x)
        return decode(q[0])

    def get_action(self, spike_matrix):
        inc = self.increments(spike_matrix)
        self.freq = max(0.0, min(180.0, self.freq + inc['freq'] * self.leap['freq']))
        self.pw = max(0.06, min(0.4, self.pw + inc['pw'] * self.leap['pw']))
        self.amp = max(0.0, min(250.0, self.amp + inc['amp'] * self.leap['amp']))
        self.applied = (self.freq, self.pw, self.amp)
        return {'freq': self.freq, 'pw': self.pw, 'amp': self.amp}


class XyloSimSNNController(CorrectedSNNController):
    """The same closed-loop controller with the decision made by bit-accurate XyloSim on the
    channel-quantized network (hw_config.json written by scripts/run_rev_xylo_check.py): integer
    sum-pool input, fresh chip state per window, Q = sum_t Vmem_out, no host-side calibration.
    Decoding is argmax per head, so q_scale is irrelevant."""

    def __init__(self, run_dir, **kw):
        import os, json, sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from scripts.quant_fidelity_lib import make_xylosim_factory
        cfg = json.load(open(os.path.join(run_dir, 'config.json')))
        hw = os.path.join(run_dir, 'xylo', 'hw_config.json')
        if not os.path.exists(hw):
            raise FileNotFoundError(f'{hw}: run scripts/run_rev_xylo_check.py --run {run_dir} first')
        super().__init__(os.path.join(run_dir, 'model.pth'), setting_input=False, out_threshold=cfg.get('out_threshold', 1.0),
                         q_scale=cfg.get('q_scale', 1.0), linear_bias=cfg.get('linear_bias', True), taus=cfg.get('taus'),
                         obs_keep=cfg.get('obs_keep'), **kw)
        self.factory = make_xylosim_factory(hw)

    def increments(self, spike_matrix):
        from scripts.quant_fidelity_lib import sum_pool_80_to_16
        mod = self.factory()
        spike_matrix = apply_obs_mask(spike_matrix, self.mask)
        out, _, _ = mod(sum_pool_80_to_16(np.asarray(spike_matrix, dtype=np.float32)).astype(int))
        q = torch.as_tensor(np.asarray(out).sum(0), dtype=torch.float32)
        return decode(q)
