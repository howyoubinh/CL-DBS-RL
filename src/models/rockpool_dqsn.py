"""Rockpool-based Deep Q Spiking Network (DQSN) for Xylo-compatible DBS control.

A three-layer leaky integrate-and-fire network built with Rockpool's ``LIFTorch``
modules. The forward pass returns both the output spike train and the membrane
potential history so the agent can act on either, and computes an optional
KL-divergence sparsity penalty on the hidden-layer firing rates.
"""

import torch
import torch.nn as nn
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential
from rockpool.parameters import Constant


class RockpoolDQSN(nn.Module):
    def __init__(self, n_observations, num_hidden, beta, n_actions, num_steps, batch_size,
                 input_width=80, use_mempot=False, target_sparsity=1.0, sparsity_weight=0.0, dt=10e-3):
        super(RockpoolDQSN, self).__init__()
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.use_mempot = use_mempot
        self.dt = dt
        self.target_sparsity = target_sparsity
        self.sparsity_weight = sparsity_weight
        self.beta = beta

        tau_mem = Constant(100e-3)
        tau_syn = Constant(50e-3)
        threshold = Constant(1.)
        bias = Constant(0.)

        self.model = Sequential(
            LinearTorch(shape=(n_observations, num_hidden), has_bias=True),
            LIFTorch(shape=(num_hidden,), dt=self.dt, tau_mem=tau_mem, tau_syn=tau_syn, threshold=threshold, bias=bias),
            LinearTorch(shape=(num_hidden, num_hidden), has_bias=True),
            LIFTorch(shape=(num_hidden,), dt=self.dt, tau_mem=tau_mem, tau_syn=tau_syn, threshold=threshold, bias=bias),
            LinearTorch(shape=(num_hidden, n_actions), has_bias=True),
            LIFTorch(shape=(n_actions,), dt=self.dt, tau_mem=tau_mem, tau_syn=tau_syn, threshold=threshold, bias=bias)
        )

    def forward(self, x, record=False):
        # record=True keeps the spike/membrane history on the computation graph.
        spk_out, _, rec, = self.model(x, record=True)

        # rec is keyed by layer name strings. Layer 5 is the output LIF.
        mem_out_history = rec['5_LIFTorch']['vmem']
        spk1 = rec['1_LIFTorch']['spikes']
        spk2 = rec['3_LIFTorch']['spikes']

        # Optional KL-divergence sparsity penalty on hidden-layer firing rates.
        total_sparsity_penalty = torch.tensor(0.0, device=x.device)
        avg_rate_1 = spk1.mean()
        avg_rate_2 = spk2.mean()

        if self.sparsity_weight > 0:
            target = torch.tensor(self.target_sparsity, device=x.device)
            penalty1 = self.kl_divergence(target, avg_rate_1)
            penalty2 = self.kl_divergence(target, avg_rate_2)
            total_sparsity_penalty = self.sparsity_weight * (penalty1 + penalty2)

        global_avg_rate = (avg_rate_1 + avg_rate_2) / 2

        return spk_out, mem_out_history, total_sparsity_penalty, global_avg_rate, rec

    def reset(self):
        for layer in self.model:
            if hasattr(layer, 'reset'):
                layer.reset()

    def kl_divergence(self, target, actual):
        eps = 1e-4
        actual = actual.clamp(eps, 1-eps)
        target = target.clamp(eps, 1-eps)
        return target * torch.log(target / actual) + (1 - target) * torch.log((1 - target) / (1 - actual))

    def get_spike_count(self):
        return [0]
