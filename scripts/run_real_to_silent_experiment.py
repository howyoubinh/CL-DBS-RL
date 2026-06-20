import numpy as np
import pandas as pd
import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation.sim_wrapper import SimulationWrapper
from src.models.rockpool_dqsn import RockpoolDQSN
from src.utils.action_utils_rockpool import get_action_dict

# Nature Communications Aesthetics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

class AvgPoolDownsampler(torch.nn.Module):
    def __init__(self, kernel_size=5, stride=5):
        super().__init__()
        self.pool = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        B, T, C = x.shape
        x_reshaped = x.reshape(B * T, 1, C) 
        out = self.pool(x_reshaped)
        return out.reshape(B, T, -1)

class SoftwareSNNInterface:
    def __init__(self, model_path):
        self.rockpool_model = RockpoolDQSN(16, 128, 0.95, 9, 100, 1, dt=10e-3, use_mempot=True)
        self.downsampler = AvgPoolDownsampler(kernel_size=5, stride=5)
        self.model = torch.nn.Sequential(self.downsampler, self.rockpool_model)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def get_action(self, spike_data):
        with torch.no_grad():
            x = torch.from_numpy(spike_data.astype(np.float32)).unsqueeze(0)
            spk_out, mem_out, _, _, _ = self.model(x)
            action_list = get_action_dict(self.model, spk_out, mem_out)
            return action_list[0]
    
    def reset(self):
        for layer in self.model.modules():
            if hasattr(layer, 'reset'):
                layer.reset()

def generate_hybrid_figure(df, steps_per_block):
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1, 1, 1])

    axs = []
    ax0 = fig.add_subplot(gs[0])
    axs.append(ax0)
    for i in range(1, 4):
        axs.append(fig.add_subplot(gs[i], sharex=ax0))
    
    colors = {'line': '#2c3e50', 'SNN': '#3498db', 'Healthy': '#3498db', 'PD': '#e74c3c', 'Silent': '#7f8c8d'}
    time_steps = np.arange(len(df))

    # Plot Data
    # Fill NAs to prevent matplotlib from breaking on rolling edge cases
    smoothed_lfp = df['LFP'].fillna(method='bfill').fillna(method='ffill').rolling(window=5, center=True).mean()
    axs[0].plot(time_steps, smoothed_lfp, color=colors['line'], lw=1.5)
    axs[0].axhline(150, color='black', linestyle=':', alpha=0.5, label='Clinical Threshold')
    
    max_lfp = max(160, df['LFP'].max() * 1.1)
    axs[0].set_ylim(0, max_lfp)
    axs[0].set_ylabel('GPi Power ($\mu V^2$)', fontweight='bold')
    axs[0].set_title('A. Neural Biomarker Suppression (Agent blinded during Silent block)', loc='left', fontweight='bold', fontsize=12)
    axs[0].legend(loc='upper right', frameon=True)

    axs[1].plot(time_steps, df['Freq'], color=colors['SNN'], lw=2.5)
    axs[1].set_ylabel('Frequency (Hz)', fontweight='bold')
    axs[1].set_ylim(-5, 200)
    axs[1].set_title('B. Frequency Modulation', loc='left', fontweight='bold', fontsize=12)

    axs[2].plot(time_steps, df['Amp'], color=colors['SNN'], lw=2.5)
    axs[2].set_ylabel('Amplitude ($\mu A$)', fontweight='bold')
    axs[2].set_ylim(-5, 275)
    axs[2].set_title('C. Amplitude Modulation', loc='left', fontweight='bold', fontsize=12)

    axs[3].plot(time_steps, df['PW'], color=colors['SNN'], lw=2.5)
    axs[3].set_ylabel('Pulse Width (ms)', fontweight='bold')
    axs[3].set_ylim(0, 0.45)
    axs[3].set_title('D. Pulse Width Modulation', loc='left', fontweight='bold', fontsize=12)
    axs[3].set_xlabel('Simulation Step (100 ms each)', fontweight='bold', fontsize=12)

    blocks = [
        (0 * steps_per_block, 1 * steps_per_block, 'Healthy'),
        (1 * steps_per_block, 2 * steps_per_block, 'PD'),
        (2 * steps_per_block, 3 * steps_per_block, 'Silent'),
        (3 * steps_per_block, 4 * steps_per_block, 'Healthy'),
        (4 * steps_per_block, 5 * steps_per_block, 'Silent'),
        (5 * steps_per_block, 6 * steps_per_block, 'PD'),
        (6 * steps_per_block, 7 * steps_per_block, 'Healthy')
    ]

    for start, end, label in blocks:
        color = colors[label]
        alpha = 0.05 if label != 'Silent' else 0.15
        for ax in axs:
            ax.axvspan(start, end, color=color, alpha=alpha)
            if ax == axs[0]:  # Add block labels to the top plot
                y_pos = ax.get_ylim()[1] * 0.9
                ax.text(start + steps_per_block/2, y_pos, label, ha='center', va='top', 
                        fontweight='bold', color=color, alpha=0.8, fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7))

    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    
    os.makedirs('artifacts', exist_ok=True)
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    out_path = f'artifacts/Supplementary_Real_to_Silent_{timestamp}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    pdf_path = f'artifacts/Supplementary_Real_to_Silent_{timestamp}.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved successfully to: {out_path} and {pdf_path}")
    return out_path


def run_hybrid_experiment(model_path, steps_per_block=500, project="cl-dbs-rl-real-to-silent"):
    wandb.init(project=project, config={
        "model_path": model_path,
        "steps_per_block": steps_per_block,
        "experiment": "real_to_silent_transition",
    })
    
    print(f"Loading SNN Model from {model_path}...")
    snn_controller = SoftwareSNNInterface(model_path)
    snn_controller.reset()
    
    print("Initializing simulation environment...")
    sim = SimulationWrapper(n_neurons=10, t_step=100, dt=0.01)
    obs = sim.reset()
    
    # State tracking variables
    current_freq = 40.0
    current_pw = 0.3
    current_amp = 300.0
    leap = {'freq': 5.0, 'pw': 0.1, 'amp': 5.0}
    
    history = {'LFP': [], 'Freq': [], 'PW': [], 'Amp': [], 'State': []}
    
    num_blocks = 7
    num_steps = steps_per_block * num_blocks
    
    print(f"Running Real-to-Silent Experiment ({num_steps} total steps)...")
    
    for i in range(num_steps):
        # 1. Determine Current Block
        block_idx = i // steps_per_block
        if block_idx == 0:
            state_label = 'Healthy'
            sim.set_pd_state(0)
            is_silent = False
        elif block_idx == 1:
            state_label = 'PD'
            sim.set_pd_state(1)
            is_silent = False
        elif block_idx == 2:
            state_label = 'Silent'
            # Let the brain run in PD state underneath to show LFP reacting to the 'blind' agent
            sim.set_pd_state(1) 
            is_silent = True
        elif block_idx == 3:
            state_label = 'Healthy'
            sim.set_pd_state(0)
            is_silent = False
        elif block_idx == 4:
            state_label = 'Silent'
            sim.set_pd_state(1)
            is_silent = True
        elif block_idx == 5:
            state_label = 'PD'
            sim.set_pd_state(1)
            is_silent = False
        else: # block_idx == 6
            state_label = 'Healthy'
            sim.set_pd_state(0)
            is_silent = False
            
        # 2. Process Observation
        if is_silent:
            agent_obs = np.zeros_like(obs['spike_matrix'])
        else:
            agent_obs = obs['spike_matrix']
            
        # 3. Get action from controller
        increments = snn_controller.get_action(agent_obs)
        
        # 4. Decode action
        current_freq += increments['freq'] * leap['freq']
        current_pw += increments['pw'] * leap['pw']
        current_amp += increments['amp'] * leap['amp']
        
        # 5. Clip bounds
        current_freq = max(0.0, min(180.0, current_freq))
        current_pw = max(0.06, min(0.4, current_pw))
        current_amp = max(0.0, min(250.0, current_amp))
        
        # 6. Log parameters
        history['LFP'].append(obs.get('lfp') if 'lfp' in obs else obs.get('gpi_alpha_beta_area', 0))
        history['Freq'].append(current_freq)
        history['PW'].append(current_pw)
        history['Amp'].append(current_amp)
        history['State'].append(state_label)
        
        # 7. Step environment (Feed blind action to real brain)
        action = {'freq': current_freq, 'pw': current_pw, 'amp': current_amp}
        obs = sim.step(action)
        
        if (i+1) % 20 == 0:
            print(f"Step {i+1:3d}/{num_steps} [{state_label:>7}] - F: {current_freq:5.1f}, A: {current_amp:5.1f}, LFP: {history['LFP'][-1]:5.1f}")

    df = pd.DataFrame(history)
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    csv_path = f'artifacts/Supplementary_Real_to_Silent_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    out_path = generate_hybrid_figure(df, steps_per_block)
    
    wandb.log({"Real-to-Silent Figure": wandb.Image(out_path)})
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--steps-per-block', type=int, default=500, help='Steps per simulation block')
    parser.add_argument('--project', type=str, default="cl-dbs-rl-real-to-silent", help='WandB project name')
    args = parser.parse_args()
    
    run_hybrid_experiment(args.model_path, args.steps_per_block, args.project)
