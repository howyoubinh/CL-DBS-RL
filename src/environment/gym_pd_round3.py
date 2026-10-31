"""
Training environment with a smoothed-beta reward (selected with --env round3 in
scripts/train_rev_teacher.py). Subclass of MousePDEnvAdaptiveRev; the published reward structure
is kept (hinge above tau, energy-scaled reward below it), with the two changes the offline check
(scripts/analyze_rev_reward_candidates.py) showed are needed for adaptation to pay, plus two
optional curriculum changes (run names: R = reward changes only, RT = reward and curriculum
changes).

Reward changes (always applied):
  beta_smooth_windows  the hinge is applied to the mean biomarker over the last k windows of the
                       episode (k = 10 at 100 ms = 1 s) instead of the current window. The
                       observation is unchanged (one 100 ms window). Healthy beta crosses any
                       usable threshold window to window; averaged over 1 s it does not.
  energy_norm          'max': the energy term is RMS pulse-train current / its maximum over the
                       action space (180 Hz, 0.4 ms, 250 uA; ~67 uA RMS), computed with the
                       simulator's own creatdbs(); '250': the published constant (which made
                       narrow-pulse stimulation cost ~10% of the reward).

Curriculum changes (optional, used by the RT teachers):
  transitions          the disease state switches once per episode at a random step in
                       [switch_min, max_steps - switch_min]; the episode's starting state is
                       whatever the training loop set with set_pd_state(). Gives the agent
                       onsets and offsets to learn from (the published curriculum never changes
                       state inside an episode).
  random_start         each episode starts from a random stimulation setting on the action grid
                       (off with probability p_off) instead of 40 Hz / 0.3 ms / 300 uA, so
                       ramping up from "off" in PD is practised.

Everything else (window, binning, missing-beta handling, leaps, clipping, episode length,
thresh_time_req) is inherited.

Optional additions used with the non-spiking Q readout (in the teachers_round3 runs, with a spiking
readout, the summed output reached its ceiling and the decoded actions did not depend on the state):
  setting_input        the observation gets 3 extra channels (freq / 180, pw / 0.4, amp / 250) as
                       deterministic, evenly spaced spike trains over the window's num_steps bins,
                       carrying the stimulation setting that produced the window (zeros for the
                       unstimulated priming window at reset). The controller always knows its own
                       output; without it the value of an incremental action is unidentifiable.
  reward_scale         every reward is divided by this (3000 puts the per-step reward in
                       [-(beta - tau) / 100, 1]) so the Q-values fit the readout's range.

step_log records every training step (episode, step, state, setting, beta, smoothed beta, E,
reward) so a run's behaviour can be inspected afterwards; scripts/train_rev_teacher.py saves it
next to the model.
"""

import numpy as np

from src.environment.gym_pd_rev import MousePDEnvAdaptiveRev
from src.simulation.simulate_network_optimized import creatdbs

MAX_SETTING = (180.0, 0.4, 250.0)   # freq Hz, pw ms, amp uA: the action space's corners
N_SETTING_CHANNELS = 3


def setting_channels(setting, num_steps):
    """(num_steps, 3) binary matrix: for each of freq / 180, pw / 0.4, amp / 250 (clipped to [0, 1]),
    round(num_steps * value) spikes spread evenly over the window. Deterministic, so the same
    setting always gives the same channels (on chip these would be the stimulator's own output,
    rate-coded)."""
    out = np.zeros((num_steps, N_SETTING_CHANNELS))
    for j, (v, vmax) in enumerate(zip(setting, MAX_SETTING)):
        p = min(max(v / vmax, 0.0), 1.0)
        out[:, j] = np.diff(np.floor(np.arange(num_steps + 1) * p)) > 0
    return out
REWARD_SCALE = 30.0                 # hinge slope, as published


def max_rms_current(tmax_ms, dt_ms):
    """RMS of the simulator's pulse train at the action space's maximum setting."""
    f, pw, a = MAX_SETTING
    idbs = creatdbs(f, tmax_ms, dt_ms, pw, a)
    return float(np.sqrt(np.mean(np.square(idbs))))


class MousePDEnvRound3(MousePDEnvAdaptiveRev):

    def __init__(self, *args, beta_smooth_windows=10, energy_norm='max', transitions=False,
                 switch_min=10, random_start=False, p_off=0.25, rng_seed=0, setting_input=False,
                 reward_scale=1.0, obs_keep=None, obs_noise_hz=0.0, obs_drop=0.0, **kwargs):
        self._in_reset = False
        self.setting_input = setting_input
        super().__init__(*args, **kwargs)
        self.reward_scale = float(reward_scale)
        assert beta_smooth_windows >= 1
        self.beta_smooth_windows = int(beta_smooth_windows)
        self.energy_norm = energy_norm
        self.E_max = max_rms_current(self.TMAX, self.DT) if energy_norm == 'max' else float(energy_norm)
        self.transitions = transitions
        self.switch_min = switch_min
        self.random_start = random_start
        self.p_off = p_off
        from src.utils.rev_rl_utils import obs_mask
        self.obs_keep = list(obs_keep) if obs_keep else None
        self.obs_mask = obs_mask(self.obs_keep)          # channels outside are zeroed (lead-realistic observation)
        self.obs_noise_hz = float(obs_noise_hz)          # Poisson background spikes added per neuron (training augmentation)
        self.obs_drop = float(obs_drop)                  # fraction of recorded spikes dropped (training augmentation)
        self._noise_rng = np.random.default_rng(1_000_003 + rng_seed)
        self.rng = np.random.default_rng(rng_seed)
        self._beta_hist = []
        self.beta_smooth = None
        self.switch_at = None
        self.pd_trace = []
        self.episode_index = -1
        self.step_log = []   # rows: episode, step, PD, freq, pw, amp, beta, beta_smooth, E, reward

    # ---------------------------------------------------------------- episode structure
    def reset(self, *args, **kwargs):
        self._beta_hist = []
        self.pd_trace = []
        self.switch_at = None
        self.episode_index += 1
        self._in_reset = True
        try:
            out = super().reset(*args, **kwargs)
        finally:
            self._in_reset = False        # one unstimulated priming window; sets 40/0.3/300
        if self.transitions:
            lo, hi = self.switch_min, self.max_steps - self.switch_min
            self.switch_at = int(self.rng.integers(lo, hi + 1)) if hi >= lo else None
        if self.random_start:
            if self.rng.random() < self.p_off:
                self.freq, self.pw, self.amp = 0.0, 0.06, 0.0
            else:
                self.freq = float(5 * self.rng.integers(1, 37))           # 5..180
                self.pw = float(np.round(self.rng.choice(np.r_[0.06, np.arange(0.1, 0.41, 0.1)]), 2))
                self.amp = float(5 * self.rng.integers(0, 51))            # 0..250
        return out

    def step(self, action):
        if self.switch_at is not None and self.step_count + 1 == self.switch_at:
            self.PD = 1 - self.PD
        self.pd_trace.append(self.PD)
        out = super().step(action)
        self.step_log.append((self.episode_index, self.step_count, self.PD, self.freq, self.pw, self.amp,
                              self.gpi_alpha_beta_area, self.beta_smooth, self.E, self.reward))
        return out

    # ---------------------------------------------------------------- reward
    def window_setting(self):
        """(freq, pw, amp) applied during the window just simulated."""
        if self._in_reset or not hasattr(self, 'stim_set'):
            return 0.0, 0.0, 0.0
        return self.stim_set['freq'], self.stim_set['pw'], self.stim_set['amp']

    def set_data(self, sim_vars):
        super().set_data(sim_vars)                  # handles missing beta (uses the last valid value)
        if self.obs_drop > 0:
            self.spikes = self.spikes * (self._noise_rng.random(self.spikes.shape) >= self.obs_drop)
        if self.obs_noise_hz > 0:                   # 1 ms bins, binary ('hash') observation
            noise = self._noise_rng.random(self.spikes.shape) < self.obs_noise_hz / 1000.0
            self.spikes = np.maximum(self.spikes, noise.astype(self.spikes.dtype))
        if self.obs_mask is not None:
            from src.utils.rev_rl_utils import apply_obs_mask
            self.spikes = apply_obs_mask(self.spikes, self.obs_mask)
        if self.setting_input:
            self.spikes = np.concatenate([self.spikes, setting_channels(self.window_setting(), self.num_steps)], 1)
        self._beta_hist.append(float(self.gpi_alpha_beta_area))
        self.beta_smooth = float(np.mean(self._beta_hist[-self.beta_smooth_windows:]))

    def under_threshold(self):
        return self.beta_smooth < self.tau_beta_max

    def calc_reward(self):
        e = min(self.E / self.E_max, 1.0)
        error = max(0.0, self.beta_smooth - self.tau_beta_max)
        alpha = min(self.delta, 1.0)
        if error > 0:
            step_reward = -(error * REWARD_SCALE)
        else:
            step_reward = self.tau_reward * ((1.0 - alpha) + alpha * (1.0 - e))
        self.reward = float(step_reward) / self.reward_scale
