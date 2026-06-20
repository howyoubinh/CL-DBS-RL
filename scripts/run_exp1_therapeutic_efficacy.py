#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 1: Baseline Therapeutic Efficacy
============================================
Generates data for Fig 1: Spectrograms & PSD

Goal: Prove the SNN works in the pathological (PD) state by comparing:
- Control Group: PD simulation with stimulation OFF
- Experimental Group: PD simulation with SNN closed-loop control

Output Metrics:
- Power Spectral Density (PSD) using multi-taper method (Chronux-style)
- Beta band power (7-35 Hz) reduction percentage
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal.windows import dpss
from scipy.interpolate import interp1d
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from datetime import datetime

from src.simulation.sim_wrapper import SimulationWrapper
from src.simulation.simulate_network_optimized import mtspectrumpt
from src.models.rockpool_dqsn import RockpoolDQSN
from src.utils.action_utils_rockpool import get_action_dict


# --- Model Loading ---
class AvgPoolDownsampler(nn.Module):
    def __init__(self, kernel_size=5, stride=5):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        B, T, C = x.shape
        x_reshaped = x.reshape(B * T, 1, C) 
        out = self.pool(x_reshaped)
        return out.reshape(B, T, -1)


class SNNController:
    """Software SNN Controller for closed-loop DBS."""
    def __init__(self, model_path, device='cpu', leap=None, init_dbs=None):
        self.device = device
        
        # Architecture must match training
        self.rockpool_model = RockpoolDQSN(16, 128, 0.95, 9, 100, 1, dt=10e-3, use_mempot=True)
        self.downsampler = AvgPoolDownsampler(kernel_size=5, stride=5)
        self.model = nn.Sequential(self.downsampler, self.rockpool_model).to(device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Loaded SNN model from {model_path}")
        
        self.init_dbs = init_dbs if init_dbs is not None else {'freq': 40.0, 'pw': 0.3, 'amp': 300.0}

        self.freq = self.init_dbs['freq']
        self.pw = self.init_dbs['pw']
        self.amp = self.init_dbs['amp']

        self.leap = leap if leap is not None else {'freq': 5.0, 'pw': 0.1, 'amp': 5.0}
        print(f"Initial DBS: freq={self.freq}, pw={self.pw}, amp={self.amp}")
        print(f"Leap values: {self.leap}")
        
    def reset(self):
        """Reset DBS parameters to initial state."""
        self.freq = self.init_dbs['freq']
        self.pw = self.init_dbs['pw']
        self.amp = self.init_dbs['amp']
        self.rockpool_model.reset()
        
    def get_action(self, spike_data):
        """Get DBS parameters from SNN policy."""
        with torch.no_grad():
            x = torch.from_numpy(spike_data.astype(np.float32)).unsqueeze(0).to(self.device)
            spk_out, mem_out, _, _, _ = self.model(x)
            action_list = get_action_dict(self.model, spk_out, mem_out)
            increments = action_list[0]
            
            # Update DBS parameters with deployment safety bounds
            self.freq = np.clip(self.freq + increments['freq'] * self.leap['freq'], 0, 180)
            self.pw = np.clip(self.pw + increments['pw'] * self.leap['pw'], 0.06, 0.4)
            self.amp = np.clip(self.amp + increments['amp'] * self.leap['amp'], 0, 250)
            
            return {'freq': self.freq, 'pw': self.pw, 'amp': self.amp}


def spike_matrix_to_lfp(spike_matrices, gpi_indices=None):
    """
    Convert spike matrices to a continuous LFP-like signal for spectral analysis.
    Uses GPi population spikes convolved with an exponential kernel.
    
    Args:
        spike_matrices: List of spike matrices [T, N_neurons] per timestep
        gpi_indices: Indices for GPi neurons (default: neurons 30-39)
    
    Returns:
        Continuous LFP-like signal array
    """
    if not spike_matrices:
        return None
    
    if gpi_indices is None:
        # GPi neurons are typically indices 30-39 (4th population of 10)
        gpi_indices = slice(30, 40)
    
    # Concatenate all spike matrices along time axis
    all_spikes = np.concatenate(spike_matrices, axis=0)  # [Total_T, N_neurons]
    
    # Extract GPi spikes and sum across neurons
    if all_spikes.shape[1] > 40:
        gpi_spikes = all_spikes[:, gpi_indices].sum(axis=1)
    else:
        # If fewer neurons, just use all of them
        gpi_spikes = all_spikes.sum(axis=1)
    
    # Convolve with exponential kernel to create smooth LFP-like signal
    kernel_len = 20  # 20ms kernel
    kernel = np.exp(-np.arange(kernel_len) / 5)
    kernel = kernel / kernel.sum()
    
    lfp = np.convolve(gpi_spikes, kernel, mode='same')
    return lfp


def compute_spectrogram_scipy(spike_matrices, fs=1000, gpi_indices=None):
    """
    Compute spectrogram using scipy's spectrogram function on binned spike data.
    More robust than multi-taper for sparse data.
    
    Args:
        spike_matrices: List of spike matrices per timestep
        fs: Sampling frequency (1000 Hz = 1ms bins)
        gpi_indices: Indices for GPi neurons
        
    Returns:
        t, f, Sxx (time, frequencies, spectrogram)
    """
    lfp = spike_matrix_to_lfp(spike_matrices, gpi_indices)
    
    if lfp is None or len(lfp) < 64:
        return None, None, None
    
    # Compute spectrogram with appropriate window size
    nperseg = min(256, len(lfp) // 4)
    if nperseg < 32:
        nperseg = min(32, len(lfp))
    
    f, t, Sxx = signal.spectrogram(lfp, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

    # |Sxx|: harmless for live data (already real, non-negative), but required
    # when spikes are reconstructed from an NPZ (can come back complex-typed).
    Sxx = np.abs(Sxx).astype(float)
    return t, f, 10 * np.log10(Sxx + 1e-10)


def compute_psd_scipy(spike_matrices, fs=1000, gpi_indices=None):
    """
    Compute PSD using scipy's Welch method on binned spike data.
    
    Args:
        spike_matrices: List of spike matrices per timestep
        fs: Sampling frequency (1000 Hz = 1ms bins)
        gpi_indices: Indices for GPi neurons
    
    Returns:
        frequencies, psd
    """
    lfp = spike_matrix_to_lfp(spike_matrices, gpi_indices)
    
    if lfp is None or len(lfp) < 64:
        return None, None
    
    # Compute PSD using Welch's method
    nperseg = min(512, len(lfp) // 2)
    if nperseg < 32:
        nperseg = min(32, len(lfp))
    
    freq, psd = signal.welch(lfp, fs=fs, nperseg=nperseg)
    return freq, psd


def extract_beta_power(freq, psd, beta_range=(7, 35)):
    """Extract integrated power in the alpha-beta band (7-35 Hz)."""
    if freq is None or psd is None:
        return np.nan
    beta_mask = (freq >= beta_range[0]) & (freq <= beta_range[1])
    if beta_mask.sum() == 0:
        return np.mean(psd) if len(psd) > 0 else np.nan
    return np.trapz(psd[beta_mask], freq[beta_mask])


# Common frequency grid for averaging multi-taper spectra. The simulator's
# point-process multi-taper grid (getfgrid) length varies per 100 ms window,
# so every per-step PSD is interpolated onto this shared grid before averaging.
MT_FREQ_GRID = np.linspace(1, 100, 200)


def compute_psd_multitaper(result, freq_grid=MT_FREQ_GRID):
    """Average the simulator's per-step multi-taper GPi PSDs onto a common grid.

    Uses gpi_S / gpi_f collected from obs['raw_vars'] (make_Spectrum -> mtspectrumpt,
    DPSS tapers) during the recording.

    Returns (freq_grid, mean_psd) or (None, None) if no valid spectra.
    """
    s_list = result.get('gpi_mt_S')
    f_list = result.get('gpi_mt_f')
    if not s_list or not f_list:
        return None, None

    interp_psds = []
    for S, f in zip(s_list, f_list):
        S = np.asarray(S, dtype=float)
        f = np.asarray(f, dtype=float)
        # Skip degenerate windows: make_Spectrum returns empty arrays when DPSS
        # fails on sparse/silent spike trains (e.g. successfully treated state).
        if S.size == 0 or f.size == 0 or S.size != f.size:
            continue
        interp_psds.append(np.interp(freq_grid, f, S))

    if not interp_psds:
        return None, None
    return freq_grid, np.mean(interp_psds, axis=0)


# Finer common grid for the high-resolution multi-taper (diagnostic) PSD.
HIGHRES_FREQ_GRID = np.linspace(1, 100, 1000)


def compute_psd_multitaper_highres(result, t_step_s=0.1, NW=4, K=7, fmax=100):
    """High-resolution multi-taper GPi PSD over the full recording.

    Unlike compute_psd_multitaper (which averages independent 100 ms-window spectra
    at NW/T = 3/0.1s ~ 30 Hz resolution), this stitches the raw GPi point process
    across all windows into one continuous spike train for a single multi-taper
    estimate, giving NW/T ~ 1 Hz resolution over the full duration. Diagnostic only;
    the RL beta biomarker remains the per-window estimate.

    Returns (freq, psd) on the simulator's native grid, or (None, None).
    """
    windows = result.get('gpi_spike_times')
    if not windows:
        return None, None

    n_ch = len(windows[0])
    # Concatenate per-channel spike times onto a continuous timeline: window w's
    # times (in [0, t_step_s]) are shifted by w * t_step_s.
    cat = [[] for _ in range(n_ch)]
    for w, win in enumerate(windows):
        offset = w * t_step_s
        for ch in range(min(n_ch, len(win))):
            ts = win[ch].get('times', [])
            if len(ts):
                cat[ch].extend((np.asarray(ts, dtype=float) + offset).tolist())

    # Plain list of dicts with Python-list 'times' == exactly the format
    # find_spike_times/make_Spectrum feed mtspectrumpt (numpy arrays break the
    # truthiness check in minmaxsptimes).
    data = [{'times': c} for c in cat]
    if sum(len(d['times']) for d in data) == 0:
        return None, None

    params = {
        'Fs': 1 / (0.01e-3),   # 1e5 Hz, identical to the simulator
        'fpass': [1, fmax],
        'tapers': [NW, K],     # NW/T sets resolution (4 over ~4 s -> ~1 Hz)
        'trialave': 1,
    }
    try:
        S, f = mtspectrumpt(data, params)
    except Exception:
        return None, None
    return np.asarray(f, dtype=float), np.asarray(S, dtype=float)


def run_experiment(condition, num_steps, model_path=None, device='cpu', leap=None, init_dbs=None, seed=None, warmup_steps=0):
    """
    Run a single experimental condition.

    Args:
        condition: 'Unstimulated' (control) or 'SNN' (experimental)
        num_steps: Number of simulation steps
        model_path: Path to trained SNN model (required for 'SNN' condition)
        device: Torch device
        leap: Dict with leap values {'freq', 'pw', 'amp'} for SNN controller
        init_dbs: Dict with initial DBS values {'freq', 'pw', 'amp'} for SNN controller
        seed: Random seed for simulation stochasticity
        warmup_steps: Steps to run before recording, letting network reach steady state

    Returns:
        Dictionary with LFP history, DBS parameters, and raw spike times
    """
    print(f"\n{'='*50}")
    print(f"Running Condition: {condition}" + (f" (seed={seed})" if seed is not None else ""))
    print(f"{'='*50}")

    sim = SimulationWrapper(n_neurons=10, t_step=100, dt=0.01, pd=1, seed=seed)
    sim.set_pd_state(1)  # Ensure PD state
    obs = sim.reset()

    # Warmup: let network reach steady state before recording
    controller = None
    if condition == 'SNN':
        if model_path is None:
            raise ValueError("model_path required for SNN condition")
        controller = SNNController(model_path, device, leap=leap, init_dbs=init_dbs)

    for _ in range(warmup_steps):
        if condition == 'Unstimulated':
            action = {'freq': 0, 'pw': 0, 'amp': 0}
        elif condition == 'SNN':
            action = controller.get_action(obs['spike_matrix'])
        obs = sim.step(action)
    
    lfp_history = []
    stim_history = {'freq': [], 'pw': [], 'amp': []}
    spike_matrix_history = []
    gpi_spike_times_history = []
    stn_spike_times_history = []
    gpi_mt_S_history = []   # per-step multi-taper GPi PSD (for Panel B)
    gpi_mt_f_history = []   # per-step multi-taper frequency grid

    for i in tqdm(range(num_steps), desc=f"{condition}"):
        if condition == 'Unstimulated':
            action = {'freq': 0, 'pw': 0, 'amp': 0}
        elif condition == 'SNN':
            action = controller.get_action(obs['spike_matrix'])
        
        obs = sim.step(action)
        
        lfp_history.append(obs['lfp'])
        stim_history['freq'].append(action['freq'])
        stim_history['pw'].append(action['pw'])
        stim_history['amp'].append(action['amp'])
        spike_matrix_history.append(obs['spike_matrix'])
        
        # Collect raw spike times from raw_vars
        if 'raw_vars' in obs:
            gpi_spike_times_history.append(obs['raw_vars']['GPi_APs'])
            stn_spike_times_history.append(obs['raw_vars']['STN_APs'])
            # Multi-taper PSD curve + freq grid for this 100 ms window
            gpi_mt_S_history.append(obs['raw_vars'].get('gpi_S'))
            gpi_mt_f_history.append(obs['raw_vars'].get('gpi_f'))

    return {
        'lfp': np.array(lfp_history),
        'stim': stim_history,
        'spikes': spike_matrix_history,
        'gpi_spike_times': gpi_spike_times_history,
        'stn_spike_times': stn_spike_times_history,
        'gpi_mt_S': gpi_mt_S_history,
        'gpi_mt_f': gpi_mt_f_history
    }


def generate_figure1(results, output_dir, aggregate_stats=None, all_results=None):
    """Generate Figure 1: Spectrograms & PSD comparison using scipy Welch method.

    results:         single-seed data dict — always used for Panels A1/A2.
    aggregate_stats: {'Unstimulated': {'mean', 'sem'}, 'SNN': {'mean', 'sem'}, 'n': int}
                     If provided, Panel E shows mean ± SEM bars.
    all_results:     list of per-seed result dicts.
                     If provided (len > 1), Panels B/C/D show all seeds as light traces
                     with bold mean ± SEM; Panel E adds individual-seed scatter points.
    """
    os.makedirs(output_dir, exist_ok=True)
    multi_seed = all_results is not None and len(all_results) > 1

    def _order(r):
        out = {}
        if 'Unstimulated' in r: out['Unstimulated'] = r['Unstimulated']
        if 'SNN' in r:          out['SNN'] = r['SNN']
        for k, v in r.items():
            if k not in out: out[k] = v
        return out

    results = _order(results)
    cond_names = list(results.keys())
    if multi_seed:
        all_results = [_order(r) for r in all_results]

    cond_colors = {'Unstimulated': '#4ecdc4', 'SNN': '#ff6b6b'}

    plt.rcParams.update({
        'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'axes.labelsize': 16,  'axes.titlesize': 18,
        'legend.fontsize': 14
    })

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    beta_powers = {}  # always computed from first/single seed (used as Panel E fallback)

    # --- Precompute all-seed PSDs + per-seed beta powers (reused by B and E) ---
    all_psds = {}
    all_seed_betas = {}
    psd_freq_ref = None
    if multi_seed:
        all_psds = {c: [] for c in cond_names}
        all_seed_betas = {c: [] for c in cond_names}
        for sr in all_results:
            for cond in cond_names:
                if cond in sr and sr[cond]['spikes']:
                    f, p = compute_psd_multitaper(sr[cond])
                    if f is not None:
                        if psd_freq_ref is None:
                            psd_freq_ref = f
                        all_psds[cond].append(p)
                        all_seed_betas[cond].append(extract_beta_power(f, p))

    # ------------------------------------------------------------------ #
    # Panels A1, A2 — Spectrograms (always single/first seed)
    # ------------------------------------------------------------------ #
    spectrograms = {}
    for cond, data in results.items():
        if data['spikes'] and len(data['spikes']) > 0:
            t, f, spec_db = compute_spectrogram_scipy(data['spikes'])
            if t is not None and f is not None:
                fm = f <= 50
                spectrograms[cond] = {'t': t, 'f': f[fm], 'spec_db': spec_db[fm, :]}

    if spectrograms:
        vmin = min(s['spec_db'].min() for s in spectrograms.values())
        vmax = max(s['spec_db'].max() for s in spectrograms.values())
    else:
        vmin, vmax = -80, -20

    for idx, (cond, data) in enumerate(results.items()):
        ax = axes[0, idx]
        if cond in spectrograms:
            sd = spectrograms[cond]
            im = ax.pcolormesh(sd['t'], sd['f'], sd['spec_db'],
                               shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
            ax.axhspan(0, 7, color='gray', alpha=0.4, zorder=2)
            ax.axhspan(35, sd['f'].max(), color='gray', alpha=0.4, zorder=2)
            ax.axhline(y=7,  color='white', linestyle='--', alpha=0.9, linewidth=1.5, zorder=3)
            ax.axhline(y=35, color='white', linestyle='--', alpha=0.9, linewidth=1.5, zorder=3)
            ax.set_ylabel('Frequency (Hz)', fontsize=16)
            ax.set_xlabel('Time (s)', fontsize=16)
            ax.set_title(f'A{idx+1}) GPi Spectrogram: {cond}', fontsize=18)
            plt.colorbar(im, ax=ax, label='Power (dB)')
        else:
            ax.text(0.5, 0.5, 'No spike data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=16)
            ax.set_title(f'A{idx+1}) GPi Spectrogram: {cond}', fontsize=18)

    # ------------------------------------------------------------------ #
    # Panel B — PSD Comparison
    # ------------------------------------------------------------------ #
    ax = axes[0, 2]
    if multi_seed and psd_freq_ref is not None:
        fm = psd_freq_ref <= 50
        for cond in cond_names:
            psds = all_psds.get(cond, [])
            if not psds: continue
            color = cond_colors.get(cond, 'C0')
            mean_p = np.mean(psds, axis=0)
            sem_p  = np.std(psds, axis=0, ddof=1) / np.sqrt(len(psds))
            ax.semilogy(psd_freq_ref[fm], mean_p[fm], color=color,
                        linewidth=2.5, label=cond)
            ax.fill_between(psd_freq_ref[fm],
                            np.maximum(mean_p[fm] - sem_p[fm], 1e-20),
                            mean_p[fm] + sem_p[fm],
                            color=color, alpha=0.25)
        # also populate beta_powers from first seed for Panel E fallback
        for cond, data in results.items():
            if data['spikes']:
                f, p = compute_psd_multitaper(data)
                if f is not None:
                    beta_powers[cond] = extract_beta_power(f, p)
    else:
        for cond, data in results.items():
            if data['spikes']:
                f, p = compute_psd_multitaper(data)
                if f is not None:
                    fm = f <= 50
                    ax.semilogy(f[fm], p[fm], label=cond, linewidth=2,
                                color=cond_colors.get(cond, 'C0'))
                    beta_powers[cond] = extract_beta_power(f, p)
    ax.axvspan(0,  7,  color='gray', alpha=0.35, zorder=2)
    ax.axvspan(35, 50, color='gray', alpha=0.35, zorder=2)
    ax.axvline(x=7,  color='gray', linestyle='--', alpha=0.8, linewidth=1.2, zorder=3)
    ax.axvline(x=35, color='gray', linestyle='--', alpha=0.8, linewidth=1.2, zorder=3)
    ax.set_xlabel('Frequency (Hz)', fontsize=16)
    ax.set_ylabel('Power Spectral Density (log)', fontsize=16)
    ax.set_title('B) PSD Comparison (Multi-taper)', fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Panel C — Beta Power Over Time
    # ------------------------------------------------------------------ #
    ax = axes[1, 0]
    if multi_seed:
        for cond in cond_names:
            color = cond_colors.get(cond, 'C0')
            lfps = [r[cond]['lfp'] for r in all_results if cond in r]
            if not lfps: continue
            min_len = min(len(l) for l in lfps)
            arr = np.array([l[:min_len] for l in lfps])
            t_c = np.arange(min_len) * 0.1
            mean_l = np.mean(arr, axis=0)
            sem_l  = np.std(arr, axis=0, ddof=1) / np.sqrt(len(arr))
            ax.plot(t_c, mean_l, color=color, linewidth=2.5, label=cond)
            ax.fill_between(t_c, mean_l - sem_l, mean_l + sem_l, color=color, alpha=0.25)
    else:
        for cond, data in results.items():
            t_c = np.arange(len(data['lfp'])) * 0.1
            ax.plot(t_c, data['lfp'], label=cond, alpha=0.8, linewidth=2,
                    color=cond_colors.get(cond, 'C0'))
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel('GPi Beta Power', fontsize=16)
    ax.set_title('C) Beta Power Over Time', fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Panel D — DBS Parameters (SNN only)
    # ------------------------------------------------------------------ #
    ax  = axes[1, 1]
    ax2 = ax.twinx()
    has_snn = 'SNN' in results or (multi_seed and any('SNN' in r for r in all_results))
    if has_snn:
        if multi_seed and any('SNN' in r for r in all_results):
            snn_results = [r for r in all_results if 'SNN' in r]
            all_freqs = np.array([r['SNN']['stim']['freq'] for r in snn_results])
            all_amps  = np.array([r['SNN']['stim']['amp']  for r in snn_results])
            all_pws   = np.array([r['SNN']['stim']['pw']   for r in snn_results])
            t_d = np.arange(all_freqs.shape[1]) * 0.1
            # bold mean ± SEM for freq and amp
            for arr, color, lbl in [(all_freqs, 'blue', 'Frequency (Hz)'),
                                     (all_amps,  'orange', 'Amplitude (µA/cm²)')]:
                mv = np.mean(arr, axis=0)
                sv = np.std(arr, axis=0, ddof=1) / np.sqrt(len(snn_results))
                ax.plot(t_d, mv, color=color, linewidth=2.5, label=lbl)
                ax.fill_between(t_d, mv - sv, mv + sv, color=color, alpha=0.2)
            # PW: mean only (stochastic — no SEM fill)
            mean_pw = np.mean(all_pws, axis=0)
            ax2.plot(t_d, mean_pw, color='green', linewidth=2.5,
                     linestyle='--', label='Pulse Width (ms)')
        else:
            stim = results['SNN']['stim']
            t_d  = np.arange(len(stim['freq'])) * 0.1
            ax.plot(t_d, stim['freq'], label='Frequency (Hz)',      color='blue',   linewidth=2)
            ax.plot(t_d, stim['amp'],  label='Amplitude (µA/cm²)',  color='orange', linewidth=2)
            ax2.plot(t_d, stim['pw'],  label='Pulse Width (ms)',     color='green',  linewidth=2, linestyle='--')
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Freq / Amp', fontsize=16)
        ax2.set_ylabel('Pulse Width (ms)', fontsize=16, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12,
                  loc='lower right', bbox_to_anchor=(1, 0.1))
        ax.set_title('D) SNN-Controlled DBS Parameters', fontsize=18)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No SNN data', ha='center', va='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_title('D) SNN-Controlled DBS Parameters', fontsize=18)

    # ------------------------------------------------------------------ #
    # Panel E — Beta Power Bar Chart
    # ------------------------------------------------------------------ #
    ax = axes[1, 2]
    if aggregate_stats and beta_powers:
        n_seeds   = aggregate_stats.get('n', '')
        plot_means = {c: aggregate_stats[c]['mean'] for c in beta_powers if c in aggregate_stats}
        plot_sems  = {c: aggregate_stats[c]['sem']  for c in beta_powers if c in aggregate_stats}
        colors = [cond_colors.get(c, 'C0') for c in plot_means]
        bars = ax.bar(plot_means.keys(), plot_means.values(),
                      color=colors, edgecolor='black', linewidth=1.5,
                      yerr=list(plot_sems.values()), capsize=6,
                      error_kw={'elinewidth': 2, 'ecolor': 'black'})
        ax.set_ylabel('Integrated Alpha-Beta Power (7-35 Hz)', fontsize=16)
        ax.set_title(f'E) Beta Band Power\n(mean ± SEM, n={n_seeds})', fontsize=18)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(plot_means.values()) * 1.15)
        for bar, (cond, val) in zip(bars, plot_means.items()):
            sem = plot_sems[cond]
            ax.text(bar.get_x() + bar.get_width() / 2, val + sem + 0.02 * max(plot_means.values()),
                    f'{val:.2e}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    elif beta_powers:
        colors = [cond_colors.get(c, 'C0') for c in beta_powers]
        bars = ax.bar(beta_powers.keys(), beta_powers.values(),
                      color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Integrated Alpha-Beta Power (7-35 Hz)', fontsize=16)
        ax.set_title('E) Beta Band Power Comparison', fontsize=18)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(beta_powers.values()) + 0.005)
        for bar, val in zip(bars, beta_powers.values()):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01 * max(beta_powers.values()),
                        f'{val:.2e}', ha='center', va='bottom',
                        fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No PSD data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_title('E) Beta Band Power Comparison', fontsize=18)

    plt.tight_layout()

    timestamp = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    fig_path = os.path.join(output_dir, f'fig1_therapeutic_efficacy_{timestamp}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    pdf_path = os.path.join(output_dir, f'fig1_therapeutic_efficacy_{timestamp}.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to {pdf_path}")
    plt.close()

    return beta_powers, timestamp


def generate_highres_psd_diagnostic(all_results, output_dir, timestamp,
                                    cond_names=('Unstimulated', 'SNN')):
    """Diagnostic figure: high-resolution multi-taper PSD + difference spectrum.

    Left panel: PD vs SNN mean +/- SEM. Right panel: PD - SNN difference spectrum,
    isolating the oscillatory component removed by stimulation.
    """
    if not all_results or len(all_results) < 1:
        return

    cond_colors = {'Unstimulated': '#4ecdc4', 'SNN': '#ff6b6b'}
    grid = HIGHRES_FREQ_GRID

    # Per-seed PSDs interpolated onto the common high-res grid.
    psds = {c: [] for c in cond_names}
    for sr in all_results:
        for cond in cond_names:
            if cond in sr:
                f, p = compute_psd_multitaper_highres(sr[cond])
                if f is not None and p is not None and f.size > 1:
                    psds[cond].append(np.interp(grid, f, p))

    # Per-seed difference (paired PD - SNN), only where both conditions exist.
    diffs = []
    for sr in all_results:
        if all(c in sr for c in cond_names):
            f0, p0 = compute_psd_multitaper_highres(sr[cond_names[0]])
            f1, p1 = compute_psd_multitaper_highres(sr[cond_names[1]])
            if f0 is not None and f1 is not None and p0 is not None and p1 is not None:
                diffs.append(np.interp(grid, f0, p0) - np.interp(grid, f1, p1))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))
    fm = grid <= 50

    # --- Left: high-res PSD comparison ---
    for cond in cond_names:
        arr = psds.get(cond, [])
        if not arr:
            continue
        arr = np.asarray(arr)
        mean_p = arr.mean(axis=0)
        sem_p = arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(mean_p)
        color = cond_colors.get(cond, 'C0')
        axL.semilogy(grid[fm], mean_p[fm], color=color, lw=2.5, label=cond)
        axL.fill_between(grid[fm], np.maximum(mean_p[fm] - sem_p[fm], 1e-20),
                         mean_p[fm] + sem_p[fm], color=color, alpha=0.25)
    for x in (7, 35):
        axL.axvline(x, color='gray', ls='--', alpha=0.8, lw=1.2)
    axL.axvspan(0, 7, color='gray', alpha=0.3)
    axL.axvspan(35, 50, color='gray', alpha=0.3)
    axL.set_xlabel('Frequency (Hz)', fontsize=14)
    axL.set_ylabel('Power Spectral Density (log)', fontsize=14)
    axL.set_title('High-res Multi-taper PSD (~1 Hz)', fontsize=15)
    axL.legend(fontsize=12)
    axL.grid(True, alpha=0.3)

    # --- Right: PD - SNN difference spectrum (isolates oscillatory component) ---
    if diffs:
        diffs = np.asarray(diffs)
        mean_d = diffs.mean(axis=0)
        sem_d = diffs.std(axis=0, ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else np.zeros_like(mean_d)
        axR.plot(grid[fm], mean_d[fm], color='#5b5bd6', lw=2.5)
        axR.fill_between(grid[fm], mean_d[fm] - sem_d[fm], mean_d[fm] + sem_d[fm],
                         color='#5b5bd6', alpha=0.25)
        axR.axhline(0, color='k', lw=1, alpha=0.6)
        for x in (7, 35):
            axR.axvline(x, color='gray', ls='--', alpha=0.8, lw=1.2)
        axR.axvspan(0, 7, color='gray', alpha=0.3)
        axR.axvspan(35, 50, color='gray', alpha=0.3)
        axR.set_xlabel('Frequency (Hz)', fontsize=14)
        axR.set_ylabel('PSD difference (PD - SNN)', fontsize=14)
        axR.set_title('Difference Spectrum (oscillatory component)', fontsize=15)
        axR.grid(True, alpha=0.3)
    else:
        axR.text(0.5, 0.5, 'No paired difference data', ha='center', va='center',
                 transform=axR.transAxes, fontsize=14)

    plt.tight_layout()
    png = os.path.join(output_dir, f'fig1_highres_psd_diagnostic_{timestamp}.png')
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(png.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"High-res PSD diagnostic saved to {png}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Baseline Therapeutic Efficacy')
    parser.add_argument('--model', type=str, required=True, help='Path to trained SNN model (.pth)')
    parser.add_argument('--duration', type=float, default=4.0,
                        help='Simulation duration in seconds (default: 4.0)')
    parser.add_argument('--output-dir', type=str, default='data/exp1_results',
                        help='Output directory for results')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run with minimal steps for testing')
    parser.add_argument('--seeds', type=str, default='42',
                        help='Comma-separated seed list for multi-seed evaluation (default: "42"). '
                             'Use e.g. "42,123,456,789,1024" for 5-seed rigorous mode.')
    parser.add_argument('--warmup', type=int, default=30,
                        help='Steps to run before recording, letting the network settle (default: 30)')
    # Leap value arguments
    parser.add_argument('--leap-freq', type=float, default=5.0)
    parser.add_argument('--leap-pw', type=float, default=0.1)
    parser.add_argument('--leap-amp', type=float, default=5.0)
    # Initial DBS values
    parser.add_argument('--init-freq', type=float, default=40.0)
    parser.add_argument('--init-pw', type=float, default=0.3)
    parser.add_argument('--init-amp', type=float, default=300.0)
    parser.add_argument('--multi-seed-plot', action='store_true',
                        help='Plot all seeds as light traces with bold mean ± SEM on Panels B/C/D; '
                             'add per-seed scatter dots on Panel E. '
                             'Requires --seeds with multiple seeds. Panel A1/A2 always uses seed 42.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    leap = {'freq': args.leap_freq, 'pw': args.leap_pw, 'amp': args.leap_amp}
    init_dbs = {'freq': args.init_freq, 'pw': args.init_pw, 'amp': args.init_amp}

    eval_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    num_seeds = len(eval_seeds)
    num_steps = 5 if args.dry_run else int(args.duration * 10)

    print(f"Running {num_steps} steps ({num_steps * 0.1:.1f} seconds) × {num_seeds} seed(s): {eval_seeds}")
    print(f"Warmup steps: {args.warmup}")
    print(f"Leap: {leap}  |  Init DBS: {init_dbs}")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

    all_pd_beta, all_snn_beta = [], []
    # Keep results from first seed for figure generation (Panels A1/A2)
    figure_results = None
    # Collect all-seed results for multi-seed plot (Panels B/C/D/E scatter)
    all_seed_results = [] if args.multi_seed_plot else None

    for seed_idx, seed in enumerate(eval_seeds):
        print(f"\n[Seed {seed_idx+1}/{num_seeds}] seed={seed}")

        results = {}
        results['Unstimulated'] = run_experiment(
            'Unstimulated', num_steps, device=device,
            seed=seed, warmup_steps=args.warmup)

        if os.path.exists(args.model):
            results['SNN'] = run_experiment(
                'SNN', num_steps, model_path=args.model, device=device,
                leap=leap, init_dbs=init_dbs, seed=seed, warmup_steps=args.warmup)
        else:
            print(f"Warning: Model not found at {args.model}")
            break

        # Beta power per seed = mean of the per-window multi-taper biomarker
        # (obs['lfp']), the same quantity used by the RL reward and the other
        # experiments. compute_psd_multitaper is used only for Panel B's PSD curve.
        seed_beta = {}
        for cond, data in results.items():
            seed_beta[cond] = float(np.mean(data['lfp']))

        if not np.isnan(seed_beta.get('Unstimulated', np.nan)):
            all_pd_beta.append(seed_beta['Unstimulated'])
        if not np.isnan(seed_beta.get('SNN', np.nan)):
            all_snn_beta.append(seed_beta['SNN'])

        reduction = 100 * (seed_beta['Unstimulated'] - seed_beta['SNN']) / seed_beta['Unstimulated']
        print(f"  Seed {seed}: PD={seed_beta['Unstimulated']:.4e}, SNN={seed_beta['SNN']:.4e}, "
              f"Reduction={reduction:.2f}%")

        if figure_results is None:
            figure_results = results
        if all_seed_results is not None:
            all_seed_results.append(results)

    # Save raw per-seed data as NPZ so generate_exp1_figure.py can rebuild the
    # figure without re-simulating. Schema: s{i}_{cond}_{field}; fields =
    # lfp/spikes/freq/pw/amp/gpi_mt_S/gpi_mt_f.
    seeds_to_save = all_seed_results if all_seed_results else (
        [figure_results] if figure_results is not None else [])
    if seeds_to_save:
        npz_data = {'n_seeds': len(seeds_to_save),
                    'seeds': np.array(eval_seeds[:len(seeds_to_save)])}
        for i, sr in enumerate(seeds_to_save):
            for cond, data in sr.items():
                p = f's{i}_{cond}_'
                npz_data[p + 'lfp']      = data['lfp']
                npz_data[p + 'spikes']   = np.array(data['spikes'], dtype=object)
                npz_data[p + 'freq']     = np.array(data['stim']['freq'])
                npz_data[p + 'pw']       = np.array(data['stim']['pw'])
                npz_data[p + 'amp']      = np.array(data['stim']['amp'])
                npz_data[p + 'gpi_mt_S'] = np.array(data['gpi_mt_S'], dtype=object)
                npz_data[p + 'gpi_mt_f'] = np.array(data['gpi_mt_f'], dtype=object)
        npz_path = os.path.join(args.output_dir, f'exp1_data_{timestamp}.npz')
        np.savez_compressed(npz_path, **npz_data)
        print(f"NPZ data saved to {npz_path} ({len(seeds_to_save)} seed(s))")

    # Build aggregate stats for Panel E (mean ± SEM across all seeds)
    aggregate_stats = None
    if len(all_pd_beta) > 0 and len(all_snn_beta) == len(all_pd_beta):
        n = len(all_pd_beta)
        aggregate_stats = {
            'Unstimulated': {
                'mean': float(np.mean(all_pd_beta)),
                'sem':  float(np.std(all_pd_beta, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            },
            'SNN': {
                'mean': float(np.mean(all_snn_beta)),
                'sem':  float(np.std(all_snn_beta, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            },
            'n': n,
        }

    # Generate figure — Panels A1/A2 from first seed; B/C/D/E per --multi-seed-plot flag
    if figure_results is not None:
        beta_powers, _ = generate_figure1(figure_results, args.output_dir,
                                          aggregate_stats=aggregate_stats,
                                          all_results=all_seed_results)

    # Diagnostic: high-resolution multi-taper PSD + difference spectrum
    # (does the model produce a resolvable beta peak?). Multi-seed only.
    if all_seed_results and len(all_seed_results) > 1:
        generate_highres_psd_diagnostic(all_seed_results, args.output_dir, timestamp)

    # Aggregate statistics
    print("\n" + "="*60)
    print("EXPERIMENT 1 RESULTS — MULTI-SEED AGGREGATE")
    print("="*60)

    n = len(all_pd_beta)
    if n > 0 and len(all_snn_beta) == n:
        reductions = [100 * (pd - snn) / pd for pd, snn in zip(all_pd_beta, all_snn_beta)]

        mean_pd   = np.mean(all_pd_beta)
        mean_snn  = np.mean(all_snn_beta)
        mean_red  = np.mean(reductions)
        sem_pd    = np.std(all_pd_beta, ddof=1) / np.sqrt(n) if n > 1 else 0.0
        sem_snn   = np.std(all_snn_beta, ddof=1) / np.sqrt(n) if n > 1 else 0.0
        sem_red   = np.std(reductions, ddof=1) / np.sqrt(n) if n > 1 else 0.0

        print(f"Seeds: {eval_seeds}")
        print(f"Per-seed reductions: {[f'{r:.2f}%' for r in reductions]}")
        print(f"\nBeta Power (PD Unstimulated): {mean_pd:.4e} ± {sem_pd:.4e} SEM")
        print(f"Beta Power (SNN Closed-Loop):  {mean_snn:.4e} ± {sem_snn:.4e} SEM")
        print(f"Beta Reduction:                {mean_red:.2f}% ± {sem_red:.2f}% SEM  (n={n})")

        csv_path = os.path.join(args.output_dir, f'exp1_results_multiseed_{timestamp}.csv')
        df = pd.DataFrame({
            'Seed': eval_seeds[:n],
            'Beta_Power_PD': all_pd_beta,
            'Beta_Power_SNN': all_snn_beta,
            'Beta_Reduction_%': reductions,
        })
        summary_row = pd.DataFrame([{
            'Seed': 'MEAN±SEM',
            'Beta_Power_PD': f'{mean_pd:.4e}±{sem_pd:.4e}',
            'Beta_Power_SNN': f'{mean_snn:.4e}±{sem_snn:.4e}',
            'Beta_Reduction_%': f'{mean_red:.2f}±{sem_red:.2f}',
        }])
        pd.concat([df, summary_row]).to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
    else:
        print("Warning: Could not compute beta reduction (no valid seeds collected)")

    print("="*60)


if __name__ == "__main__":
    main()
