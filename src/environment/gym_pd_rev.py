"""
Environment identical to MousePDEnvAdaptive (the reward the published teacher was
trained with) except for one safety change and an optional observation binning.

When the multi-taper estimate fails on a 100 ms window (too few or too synchronized
GPi spikes), the simulator reports alpha-beta power as exactly 0.0
(simulate_network_optimized.make_Spectrum). Treated literally, that reads as perfect
suppression and earns the maximum reward. Here a failed window is treated as missing:
the reward uses the last valid value from the same episode. The failure is rare
(about 5 in 21,000 windows in the run_rev_constant_stim.py runs), so this guards against an exploit rather
than changing typical training.

Optional observation binning (binning='count'): spike counts per bin over the actual window
(window_ms / num_steps per bin). The original constructor_hash() bins spike times over a
hard-coded 0-0.1 s range, which matches 100 ms windows; at 1 s it places every later
spike into the last bins. binning='hash' (default) keeps the original behaviour exactly.

The threshold tau_beta_max, the window (TMAX, ms) and the patient seed are constructor
arguments; scripts/train_rev_teacher.py exposes them as flags.
"""

import math

import numpy as np

from src.environment.gym_pd import MousePDEnvAdaptive

# Same population order as data_processing.constructor_hash (TH, STN, GPe, GPi, Str-D2, Str-D1, Cortex)
POPULATIONS = ('TH_APs', 'STN_APs', 'GPe_APs', 'GPi_APs', 'Striat_APs_indr', 'Striat_APs_dr', 'Cor_APs')


def count_bin(sim_vars, num_steps, window_s):
    """(num_steps, n_neurons) spike counts per bin over [0, window_s); spike times are in seconds."""
    edges = np.linspace(0.0, window_s, num_steps + 1)
    cols = [np.histogram(np.asarray(neuron['times']).ravel(), bins=edges)[0]
            for key in POPULATIONS for neuron in sim_vars[key]]
    return np.stack(cols, axis=1).astype(np.float64)


class MousePDEnvAdaptiveRev(MousePDEnvAdaptive):

    def __init__(self, *args, binning='hash', **kwargs):
        assert binning in ('hash', 'count'), binning
        self.binning = binning
        super().__init__(*args, **kwargs)
        self._last_valid_beta = None
        self.n_missing_beta = 0  # failed estimates over the env's lifetime
        self.n_windows = 0

    def reset(self, *args, **kwargs):
        self._last_valid_beta = None
        return super().reset(*args, **kwargs)

    def set_data(self, sim_vars):
        super().set_data(sim_vars)
        if self.binning == 'count':
            self.spikes = count_bin(sim_vars, self.num_steps, self.TMAX / 1000.0)
        self.n_windows += 1
        beta = self.gpi_alpha_beta_area
        if beta == 0.0 or not math.isfinite(beta):
            self.n_missing_beta += 1
            if self._last_valid_beta is not None:
                self.gpi_alpha_beta_area = self._last_valid_beta
        else:
            self._last_valid_beta = beta
