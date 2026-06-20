import numpy as np
import pandas as pd
import os
import sys
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation.sim_wrapper import SimulationWrapper
from src.models.rockpool_dqsn import RockpoolDQSN
from src.utils.action_utils_rockpool import get_action_dict
from src.models.ann_baseline import ANNBaseline
from src.models.rnn_baseline import RNNBaseline

# Copied from scripts/train_rl_rockpool_16ch.py for consistency
class AvgPoolDownsampler(torch.nn.Module):
    def __init__(self, kernel_size=5, stride=5):
        super().__init__()
        self.pool = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        # x shape: (B, T, C_in=80)
        B, T, C = x.shape
        # Reshape to (B*T, 1, C) to pool over C
        x_reshaped = x.reshape(B * T, 1, C) 
        # Output: (B*T, 1, 16)
        out = self.pool(x_reshaped)
        # Reshape back to (B, T, 16)
        return out.reshape(B, T, -1)

# Software SNN controller: loads the trained model and runs closed-loop inference.
class SoftwareSNNInterface:
    def __init__(self, model_path):
        # Load Model
        # Architecture must match training: 16 inputs, 128 hidden, 9 actions
        # IMPORTANT: use_mempot=True to match training!
        self.rockpool_model = RockpoolDQSN(16, 128, 0.95, 9, 100, 1, dt=10e-3, use_mempot=True)
        self.downsampler = AvgPoolDownsampler(kernel_size=5, stride=5)
        
        # The trained model was a Sequential(Downsampler, RockpoolDQSN)
        self.model = torch.nn.Sequential(self.downsampler, self.rockpool_model)
        
        # Load State Dict
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            try:
                self.model.load_state_dict(state_dict)
                print(f"Successfully loaded model (Sequential) from {model_path}")
            except RuntimeError as e:
                print(f"Direct load failed: {e}")
                raise e

        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
            
        self.model.eval()
        
    def get_action(self, spike_data):
        # spike_data: (T, 80)
        with torch.no_grad():
            x = torch.from_numpy(spike_data.astype(np.float32)).unsqueeze(0) # (1, T, 80)
            spk_out, mem_out, _, _, _ = self.model(x)
            action_list = get_action_dict(self.model, spk_out, mem_out)
            return action_list[0]

class SoftwareANNInterface:
    """ANN baseline controller for closed-loop DBS (mirrors SoftwareSNNInterface)."""
    def __init__(self, model_path):
        # Architecture: 80 inputs, 128 hidden, 9 outputs (matches training)
        self.model = ANNBaseline(80, 128, 9)
        
        # Load State Dict
        state_dict = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Successfully loaded ANN model from {model_path}")
        
    def get_action(self, spike_data):
        """
        Get action from ANN policy.
        
        Args:
            spike_data: np.ndarray of shape (T, 80) — raw spike matrix
            
        Returns:
            dict with keys 'freq', 'pw', 'amp', values in {-1, 0, 1}
        """
        with torch.no_grad():
            # (T, 80) -> (1, T, 80); ANN internally mean-pools over T
            x = torch.from_numpy(spike_data.astype(np.float32)).unsqueeze(0)
            q_values = self.model(x)[0]  # (9,)
            
            # Decode actions: same 3x3 partitioned scheme as SNN
            freq_act = torch.argmax(q_values[0:3]).item() - 1
            pw_act = torch.argmax(q_values[3:6]).item() - 1
            amp_act = torch.argmax(q_values[6:9]).item() - 1
            
            return {'freq': freq_act, 'pw': pw_act, 'amp': amp_act}

class SoftwareRNNInterface:
    """RNN (GRU) baseline controller for closed-loop DBS."""
    def __init__(self, model_path):
        # Architecture: 80 inputs, 128 hidden, 9 outputs (matches training)
        self.model = RNNBaseline(80, 128, 9)
        
        # Load State Dict
        state_dict = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Successfully loaded RNN model from {model_path}")
        
    def get_action(self, spike_data):
        """
        Get action from RNN policy.
        
        Args:
            spike_data: np.ndarray of shape (T, 80) — raw spike matrix
            
        Returns:
            dict with keys 'freq', 'pw', 'amp', values in {-1, 0, 1}
        """
        with torch.no_grad():
            # (T, 80) -> (1, T, 80); RNN processes full temporal sequence
            x = torch.from_numpy(spike_data.astype(np.float32)).unsqueeze(0)
            q_values = self.model(x)[0]  # (9,)
            
            # Decode actions: same 3x3 partitioned scheme as SNN
            freq_act = torch.argmax(q_values[0:3]).item() - 1
            pw_act = torch.argmax(q_values[3:6]).item() - 1
            amp_act = torch.argmax(q_values[6:9]).item() - 1
            
            return {'freq': freq_act, 'pw': pw_act, 'amp': amp_act}

def run_on_off_experiment(condition, num_steps=1000, model_path=None):
    print(f"Running Condition: {condition}")
    sim = SimulationWrapper(n_neurons=10, t_step=100, dt=0.01)
    obs = sim.reset()
    
    lfp_history = []
    stim_freqs = []
    stim_pws = []
    stim_amps = []
    pd_states = []
    
    # Controller Setup
    snn_controller = None
    if condition == 'SNN':
        if model_path is None: raise ValueError("model_path required for SNN condition")
        snn_controller = SoftwareSNNInterface(model_path)
    
    ann_controller = None
    if condition == 'ANN':
        if model_path is None: raise ValueError("model_path required for ANN condition")
        ann_controller = SoftwareANNInterface(model_path)
    
    rnn_controller = None
    if condition == 'RNN':
        if model_path is None: raise ValueError("model_path required for RNN condition")
        rnn_controller = SoftwareRNNInterface(model_path)

    # Initial State (must match training env defaults in gym_pd.py)
    current_freq = 40.0
    current_pw = 0.3
    current_amp = 300.0  # Will be clipped to 250 on first SNN step
    
    # Leap values
    leap = {'freq': 5.0, 'pw': 0.1, 'amp': 5.0}
    
    action = {'freq': 0, 'pw': 0, 'amp': 0} # Default Off
    
    # Define Cycling Blocks (steps)
    # Each block is 200 steps (20 seconds), matching the high-fidelity training max_steps.
    # Evens (0-200, 400-600, ...): Healthy
    # Odds (200-400, 600-800, ...): PD
 
    for i in tqdm(range(num_steps), desc=f"Condition: {condition}"):
        # Determine PD State: Cycle every 200 steps (100 Healthy, 100 PD)
        cycle_pos = i % 200
        if cycle_pos < 100:
            current_pd_state = 0 # Healthy
        else:
            current_pd_state = 1 # PD
            
        # Reset controller state on block transitions to prevent carry-over
        if i > 0 and current_pd_state != pd_states[-1]:
            if condition == 'SNN':
                try:
                    # RockpoolDQSN is wrapped in a Sequential model 
                    # index 1 is the rockpool model
                    snn_controller.model[1].reset()
                    print(f"Step {i}: Switched to PD State {current_pd_state}. Reset SNN membrane state.")
                except Exception as e:
                    print(f"Warning: Failed to reset SNN state at step {i}: {e}")
                
        sim.set_pd_state(current_pd_state)
        pd_states.append(current_pd_state)
        
        # Determine Action
        if condition == 'Unstimulated':
            current_freq = 0
            current_pw = 0
            current_amp = 0
            
        elif condition == 'cDBS':
            current_freq = 130
            current_pw = 0.3
            current_amp = 300
            
        elif condition == 'aDBS_Threshold':
            # Clinical-style dual-threshold adaptive DBS (Little et al. 2013, Piña-Fuentes et al. 2017)
            # Uses LFP (β power) as biomarker with hysteresis band to prevent oscillation.
            # When β > upper: turn ON with clinical parameters
            # When β < lower: turn OFF
            # Between: maintain current state
            upper_thresh = 160.0   # Turn ON above this
            lower_thresh = 140.0   # Turn OFF below this
            
            # Use previous step's LFP as the biomarker
            current_beta = lfp_history[-1] if lfp_history else 300.0  # Assume high on first step
            
            if current_beta > upper_thresh:
                # β is high → turn ON with clinical DBS parameters
                current_freq = 130
                current_pw = 0.3
                current_amp = 300
            elif current_beta < lower_thresh:
                # β is low → turn OFF
                current_freq = 0
                current_pw = 0
                current_amp = 0
            # else: between thresholds → maintain current state (hysteresis)
            
        elif condition == 'SNN':
            # Get increments
            increments = snn_controller.get_action(obs['spike_matrix'])
            
            # Update State
            current_freq = current_freq + increments['freq'] * leap['freq']
            current_pw = current_pw + increments['pw'] * leap['pw']
            current_amp = current_amp + increments['amp'] * leap['amp']
            
            # Clip to match training env bounds (gym_pd.py MousePDEnv.step)
            current_freq = max(0.0, min(180.0, current_freq))
            current_pw = max(0.06, min(0.4, current_pw))
            current_amp = max(0.0, min(250.0, current_amp))
            
        elif condition == 'ANN':
            # Get increments (same interface as SNN)
            increments = ann_controller.get_action(obs['spike_matrix'])
            
            # Update State (same leap and clipping as SNN)
            current_freq = current_freq + increments['freq'] * leap['freq']
            current_pw = current_pw + increments['pw'] * leap['pw']
            current_amp = current_amp + increments['amp'] * leap['amp']
            
            # Clip to match training env bounds (gym_pd.py MousePDEnv.step)
            current_freq = max(0.0, min(180.0, current_freq))
            current_pw = max(0.06, min(0.4, current_pw))
            current_amp = max(0.0, min(250.0, current_amp))
            
        elif condition == 'RNN':
            # Get increments (same interface as SNN/ANN)
            increments = rnn_controller.get_action(obs['spike_matrix'])
            
            # Update State (same leap and clipping as SNN/ANN)
            current_freq = current_freq + increments['freq'] * leap['freq']
            current_pw = current_pw + increments['pw'] * leap['pw']
            current_amp = current_amp + increments['amp'] * leap['amp']
            
            # Clip to match training env bounds (gym_pd.py MousePDEnv.step)
            current_freq = max(0.0, min(180.0, current_freq))
            current_pw = max(0.06, min(0.4, current_pw))
            current_amp = max(0.0, min(250.0, current_amp))

        action = {'freq': current_freq, 'pw': current_pw, 'amp': current_amp}

        # Step Sim
        obs = sim.step(action)
        
        # Record Data
        lfp_history.append(obs['lfp'])
        stim_freqs.append(current_freq)
        stim_pws.append(current_pw)
        stim_amps.append(current_amp)
        
        # Log per-step metrics to wandb
        step_energy = current_freq * current_amp * current_pw * 0.1
        wandb.log({
            f"{condition}/LFP": obs['lfp'],
            f"{condition}/Freq": current_freq,
            f"{condition}/PW": current_pw,
            f"{condition}/Amp": current_amp,
            f"{condition}/PD_State": current_pd_state,
            f"{condition}/Step_Energy": step_energy,
            "step": i,
        })
        
    return lfp_history, stim_freqs, stim_pws, stim_amps, pd_states

def compute_energy_metrics(results, conditions, step_duration_s=0.1):
    """
    Compute stimulation energy metrics for each condition.
    
    Energy proxy: Charge delivered per step ∝ Freq × Amp × PW
    Integrated over time with step_duration_s (default 100ms per step).
    
    Also computes:
    - Cumulative energy over time
    - Per-step energy
    - Duty cycle (fraction of steps where freq > threshold)
    - Energy savings vs cDBS
    """
    metrics = {}
    FREQ_ACTIVE_THRESHOLD = 10.0  # Hz; below this, consider stimulation "off"
    
    for cond in conditions:
        freqs = np.array(results[cond]['freq'])
        amps = np.array(results[cond]['amp'])
        pws = np.array(results[cond]['pw'])
        
        # Per-step energy proxy (charge per step)
        per_step_energy = freqs * amps * pws * step_duration_s
        
        # Cumulative energy
        cumulative_energy = np.cumsum(per_step_energy)
        
        # Total energy
        total_energy = cumulative_energy[-1]
        
        # Duty cycle: fraction of steps where stimulation is "active"
        active_steps = np.sum(freqs > FREQ_ACTIVE_THRESHOLD)
        duty_cycle = active_steps / len(freqs)
        
        # Mean stimulation parameters when active
        active_mask = freqs > FREQ_ACTIVE_THRESHOLD
        mean_freq_active = np.mean(freqs[active_mask]) if active_mask.any() else 0
        mean_amp_active = np.mean(amps[active_mask]) if active_mask.any() else 0
        
        metrics[cond] = {
            'per_step_energy': per_step_energy,
            'cumulative_energy': cumulative_energy,
            'total_energy': total_energy,
            'duty_cycle': duty_cycle,
            'active_steps': int(active_steps),
            'total_steps': len(freqs),
            'mean_freq_active': mean_freq_active,
            'mean_amp_active': mean_amp_active,
        }
    
    # Compute savings relative to cDBS
    if 'cDBS' in metrics:
        cdbs_total = metrics['cDBS']['total_energy']
        for cond in conditions:
            if cdbs_total > 0:
                savings_pct = (1 - metrics[cond]['total_energy'] / cdbs_total) * 100
            else:
                savings_pct = 0.0
            metrics[cond]['savings_vs_cdbs_pct'] = savings_pct
    
    return metrics


def generate_energy_figure(results, metrics, conditions, output_dir='data/on_off_results', timestamp=None):
    """
    Generate a publication-quality figure showing:
    1. GPi Beta Power (LFP) with PD state shading
    2. Stimulation Frequency traces 
    3. Per-step stimulation energy
    4. Cumulative energy comparison
    5. Summary bar chart (total energy + duty cycle)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Color palette
    colors = {
        'SNN': '#2196F3',           # Blue
        'ANN': '#FF9800',           # Orange
        'RNN': '#9C27B0',           # Purple
        'cDBS': '#F44336',          # Red
        'Unstimulated': '#9E9E9E',  # Gray
        'aDBS_Threshold': '#4CAF50', # Green
    }
    
    # Line styles and markers to distinguish overlapping traces
    line_styles = {
        'SNN':            {'linestyle': '-',  'marker': None, 'markevery': None},
        'ANN':            {'linestyle': '--', 'marker': None, 'markevery': None},
        'RNN':            {'linestyle': '-.', 'marker': None, 'markevery': None},
        'cDBS':           {'linestyle': '-',  'marker': 's',  'markevery': 50, 'markersize': 4},
        'Unstimulated':   {'linestyle': '-',  'marker': 'x',  'markevery': 50, 'markersize': 4},
        'aDBS_Threshold': {'linestyle': '-',  'marker': '^',  'markevery': 50, 'markersize': 4},
    }
    
    fig, axs = plt.subplots(5, 1, figsize=(14, 16), 
                            gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.8]})
    
    num_steps = len(results[conditions[0]]['lfp'])
    time_axis = np.arange(num_steps)
    
    # Identify PD regions for shading
    pd_states = np.array(results[conditions[0]]['pd_state'])
    
    def shade_pd_regions(ax):
        """Add PD state shading to an axis."""
        in_pd = False
        start = 0
        for i in range(len(pd_states)):
            if pd_states[i] == 1 and not in_pd:
                start = i
                in_pd = True
            elif pd_states[i] == 0 and in_pd:
                ax.axvspan(start, i, color='#FFCDD2', alpha=0.3, zorder=0)
                in_pd = False
        if in_pd:
            ax.axvspan(start, len(pd_states), color='#FFCDD2', alpha=0.3, zorder=0)
    
    # --- Panel A: GPi Beta Power ---
    ax = axs[0]
    for cond in conditions:
        ax.plot(time_axis, results[cond]['lfp'], label=cond, color=colors[cond], 
                alpha=0.85, linewidth=1.2, **line_styles.get(cond, {}))
    shade_pd_regions(ax)
    ax.set_ylabel('GPi Beta Power', fontsize=11)
    ax.set_title('A. Neural Biomarker (GPi Beta Oscillation Power)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # --- Panel B: Stimulation Frequency ---
    ax = axs[1]
    for cond in ['SNN', 'ANN', 'RNN', 'cDBS', 'aDBS_Threshold']:
        if cond not in results: continue
        ax.plot(time_axis, results[cond]['freq'], label=cond, color=colors[cond], 
                alpha=0.85, linewidth=1.2, **line_styles.get(cond, {}))
    shade_pd_regions(ax)
    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_title('B. Stimulation Frequency', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # --- Panel C: Per-Step Stimulation Energy ---
    ax = axs[2]
    for cond in ['SNN', 'ANN', 'RNN', 'cDBS', 'aDBS_Threshold']:
        if cond not in metrics: continue
        ax.plot(time_axis, metrics[cond]['per_step_energy'], label=cond, 
                color=colors[cond], alpha=0.7, linewidth=1.0, **line_styles.get(cond, {}))
    shade_pd_regions(ax)
    ax.set_ylabel('Energy Proxy\n(Freq×Amp×PW×dt)', fontsize=11)
    ax.set_title('C. Per-Step Stimulation Energy', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # --- Panel D: Cumulative Energy ---
    ax = axs[3]
    for cond in ['SNN', 'ANN', 'RNN', 'cDBS', 'aDBS_Threshold']:
        if cond not in metrics: continue
        ax.plot(time_axis, metrics[cond]['cumulative_energy'], label=cond, 
                color=colors[cond], linewidth=2.0, **line_styles.get(cond, {}))
        # Annotate final value
        final_val = metrics[cond]['cumulative_energy'][-1]
        ax.annotate(f'{final_val:.0f}', 
                    xy=(num_steps - 1, final_val),
                    xytext=(-60, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=colors[cond],
                    arrowprops=dict(arrowstyle='->', color=colors[cond], lw=1.5))
    shade_pd_regions(ax)
    ax.set_ylabel('Cumulative Energy', fontsize=11)
    ax.set_title('D. Cumulative Stimulation Energy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Simulation Step (100 ms each)', fontsize=11)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add savings annotation
    if 'SNN' in metrics and 'savings_vs_cdbs_pct' in metrics['SNN']:
        savings = metrics['SNN']['savings_vs_cdbs_pct']
        ax.text(0.98, 0.5, f'Energy Savings: {savings:.1f}%',
                transform=ax.transAxes, fontsize=13, fontweight='bold',
                ha='right', va='center', color='#1B5E20',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#C8E6C9', edgecolor='#4CAF50', alpha=0.9))
    
    # --- Panel E: Summary Bar Chart ---
    ax = axs[4]
    bar_conds = [c for c in ['cDBS', 'aDBS_Threshold', 'SNN', 'ANN', 'RNN'] if c in metrics]
    x_pos = np.arange(len(bar_conds))
    bar_width = 0.35
    
    # Total energy bars
    energies = [metrics[c]['total_energy'] for c in bar_conds]
    bars1 = ax.bar(x_pos - bar_width/2, energies, bar_width, 
                   label='Total Energy', color=[colors[c] for c in bar_conds], alpha=0.8)
    
    # Duty cycle bars (on secondary y-axis)
    ax2 = ax.twinx()
    duty_cycles = [metrics[c]['duty_cycle'] * 100 for c in bar_conds]
    bars2 = ax2.bar(x_pos + bar_width/2, duty_cycles, bar_width,
                    label='Duty Cycle (%)', color=[colors[c] for c in bar_conds], 
                    alpha=0.4, hatch='//')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_conds, fontsize=11)
    ax.set_ylabel('Total Stimulation Energy', fontsize=11)
    ax2.set_ylabel('Duty Cycle (%)', fontsize=11)
    ax2.set_ylim(0, 110)
    ax.set_title('E. Energy & Duty Cycle Comparison', fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar, val in zip(bars1, energies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, duty_cycles):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    
    fig_path_png = os.path.join(output_dir, f'on_off_energy_analysis_{timestamp}.png')
    fig_path_pdf = os.path.join(output_dir, f'on_off_energy_analysis_{timestamp}.pdf')
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    print(f"Figure saved to {fig_path_png}")
    print(f"Figure saved to {fig_path_pdf}")
    
    return fig_path_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model in an on-off experiment.")
    parser.add_argument('--model-path', type=str,
                        default="models/final_rockpool_16ch_curriculum/teacher.pth",
                        help='Path to the trained PyTorch model (.pth)')
    parser.add_argument('--ann-model-path', type=str, 
                        default="models/ann_baseline_curriculum/ann_baseline.pth",
                        help='Path to the trained ANN model (.pth)')
    parser.add_argument('--rnn-model-path', type=str, 
                        default="models/rnn_baseline_curriculum/rnn_baseline.pth",
                        help='Path to the trained RNN model (.pth)')
    parser.add_argument('--num-steps', type=int, default=500,
                        help='Number of simulation steps for each condition')
    parser.add_argument('--project', type=str, default="cl-dbs-rl-on-off",
                        help='WandB project name')
    args = parser.parse_args()

    model_path = args.model_path
    num_steps = args.num_steps
    
    wandb.init(project=args.project, config={
        "model_path": model_path,
        "num_steps": num_steps,
        "experiment": "on_off_energy_analysis",
    })
    
    # Run Conditions
    results = {}
    conditions = ['SNN', 'ANN', 'RNN', 'Unstimulated', 'cDBS', 'aDBS_Threshold'] 
    
    for cond in conditions:
        if cond == 'ANN':
            cond_model_path = args.ann_model_path
        elif cond == 'RNN':
            cond_model_path = args.rnn_model_path
        else:
            cond_model_path = model_path
        lfp, freq, pw, amp, pd_state = run_on_off_experiment(
            cond, num_steps=num_steps, model_path=cond_model_path
        )
        results[cond] = {
            'lfp': lfp,
            'freq': freq,
            'pw': pw,
            'amp': amp,
            'pd_state': pd_state
        }
        
    # Save Results to CSV
    output_dir = 'data/on_off_results'
    os.makedirs(output_dir, exist_ok=True)
    
    df_data = {'PD_State': results['SNN']['pd_state']}
    for cond in conditions:
        df_data[f'{cond}_LFP'] = results[cond]['lfp']
        df_data[f'{cond}_Freq'] = results[cond]['freq']
        df_data[f'{cond}_PW'] = results[cond]['pw']
        df_data[f'{cond}_Amp'] = results[cond]['amp']
        
    df = pd.DataFrame(df_data)
    
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    csv_path = os.path.join(output_dir, f'on_off_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # --- Energy Efficiency Analysis ---
    metrics = compute_energy_metrics(results, conditions)
    
    print("\n" + "="*60)
    print("STIMULATION ENERGY ANALYSIS")
    print("="*60)
    for cond in conditions:
        m = metrics[cond]
        print(f"\n{cond}:")
        print(f"  Total Energy (charge proxy):  {m['total_energy']:.2f}")
        print(f"  Duty Cycle:                   {m['duty_cycle']*100:.1f}% ({m['active_steps']}/{m['total_steps']} steps)")
        if 'savings_vs_cdbs_pct' in m:
            print(f"  Energy Savings vs cDBS:       {m['savings_vs_cdbs_pct']:.1f}%")
        if m['duty_cycle'] > 0:
            print(f"  Mean Freq (when active):      {m['mean_freq_active']:.1f} Hz")
            print(f"  Mean Amp (when active):       {m['mean_amp_active']:.1f}")
    
    print("\n" + "="*60)
    if 'SNN' in metrics:
        print(f"\n>>> KEY RESULT: Adaptive DBS (SNN) reduces stimulation energy by "
              f"{metrics['SNN']['savings_vs_cdbs_pct']:.1f}% compared to constant DBS <<<")
    print("="*60)
    
    # --- Log summary metrics to wandb ---
    summary_table = wandb.Table(
        columns=["Condition", "Total Energy", "Duty Cycle (%)", "Energy Savings vs cDBS (%)",
                 "Mean Freq (active)", "Mean Amp (active)"],
    )
    for cond in conditions:
        m = metrics[cond]
        summary_table.add_data(
            cond,
            round(m['total_energy'], 2),
            round(m['duty_cycle'] * 100, 1),
            round(m.get('savings_vs_cdbs_pct', 0), 1),
            round(m['mean_freq_active'], 1),
            round(m['mean_amp_active'], 1),
        )
    wandb.log({"Energy Summary": summary_table})
    
    # --- Generate Figure ---
    fig_path = generate_energy_figure(results, metrics, conditions, output_dir=output_dir, timestamp=timestamp)
    
    # Log figure to wandb
    wandb.log({"Energy Analysis Figure": wandb.Image(fig_path)})
    
    wandb.finish()
