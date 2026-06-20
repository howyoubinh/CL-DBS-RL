#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export RL Model to Xylo Hardware Package
==========================================
Converts a trained RockpoolDQSN model to a Xylo-compatible deployment package.

Output:
- hw_config.json: Quantized Xylo specification
- sample_spikes.npy: Test spike data (from brain simulation or random)
- metadata.json: Model and export information
"""

import torch
import numpy as np
import json
import os
import argparse
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.rockpool_dqsn import RockpoolDQSN
from rockpool.devices.xylo import syns65302 as xa3
from rockpool.transform import quantize_methods as q


def generate_real_spike_data(num_samples=20, num_steps=100, n_observations=16):
    """
    Generate real spike data from brain simulation.
    Uses the AvgPool downsampler to match training preprocessing.
    """
    print("Generating real spike data from brain simulation...")
    
    try:
        from src.environment.gym_pd import MousePDEnv
        import torch.nn as nn
        
        # AvgPool downsampler (80 -> 16 channels)
        class AvgPoolDownsampler(nn.Module):
            def __init__(self, kernel_size=5, stride=5):
                super().__init__()
                self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
                
            def forward(self, x):
                B, T, C = x.shape
                x_reshaped = x.reshape(B * T, 1, C) 
                out = self.pool(x_reshaped)
                return out.reshape(B, T, -1)
        
        downsampler = AvgPoolDownsampler()
        
        # Initialize environment
        leap = {'pw': 0.1, 'amp': 5.0, 'freq': 5.0}
        env = MousePDEnv(
            leap=leap, 
            num_steps=num_steps, 
            tau_beta_max=150., 
            tau_reward=3000., 
            delta=0.01, 
            max_steps=25, 
            TMAX=100, 
            thresh_time_req=5
        )
        
        spike_data_list = []
        
        for i in range(num_samples):
            raw_spikes, _ = env.reset()  # [100, 80]
            # Add batch dim and convert to tensor
            raw_tensor = torch.tensor(raw_spikes, dtype=torch.float32).unsqueeze(0)  # [1, 100, 80]
            # Downsample to 16 channels
            downsampled = downsampler(raw_tensor)  # [1, 100, 16]
            spike_data_list.append(downsampled.squeeze(0).numpy())  # [100, 16]
            
            if (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        sample_spikes = np.stack(spike_data_list)  # [num_samples, 100, 16]
        print(f"Generated {num_samples} real spike samples with shape {sample_spikes.shape}")
        return sample_spikes
        
    except Exception as e:
        print(f"Warning: Could not generate real data ({e}), using random spikes")
        return generate_random_spike_data(num_samples, num_steps, n_observations)


def generate_random_spike_data(num_samples=20, num_steps=100, n_observations=16):
    """Generate random Poisson-like spike data for testing."""
    print("Generating random spike data...")
    # Sparse spike input (typical ~5% activity)
    sample_spikes = (np.random.rand(num_samples, num_steps, n_observations) < 0.05).astype(np.float32)
    print(f"Generated {num_samples} random spike samples with shape {sample_spikes.shape}")
    return sample_spikes


def export_model(model_path, output_dir, generate_real_data=False, num_samples=20, device='cpu'):
    """
    Export a trained RockpoolDQSN model to Xylo hardware package.
    
    Args:
        model_path: Path to trained .pth model file
        output_dir: Directory to save output package
        generate_real_data: If True, generate real brain sim data; else random
        num_samples: Number of test samples to generate
        device: Device for model loading
    """
    print(f"\n{'='*60}")
    print(f"Exporting model: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # --- 1. Model Architecture Parameters (must match training) ---
    n_observations = 16  # After downsampling
    num_hidden = 128
    n_actions = 9
    beta = 0.95
    num_steps = 100
    batch_size = 1
    dt = 10e-3
    
    # --- 2. Reconstruct and Load Model ---
    print("Loading model...")
    model = RockpoolDQSN(n_observations, num_hidden, beta, n_actions, num_steps, batch_size, dt=dt)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    model.eval()
    
    # --- 3. Extract Graph and Map to Xylo ---
    print("Extracting computational graph...")
    rockpool_net = model.model  # Access internal Rockpool Sequential
    
    print("Mapping to XyloAudio 3 specification...")
    spec = xa3.mapper(
        rockpool_net.as_graph(), 
        weight_dtype='float', 
        threshold_dtype='float', 
        dash_dtype='float'
    )
    spec['dt'] = dt
    
    # --- 4. Quantize ---
    print("Quantizing specification (channel-wise)...")
    quant_spec = q.channel_quantize(**spec)
    
    # Add back parameters that might be dropped
    for key in ['dt', 'aliases']:
        if key in spec:
            quant_spec[key] = spec[key]
    
    # Enforce hardware data types for neuron parameters
    for key in ["dash_mem", "dash_mem_out", "dash_syn", "dash_syn_2", "dash_syn_out"]:
        if key in quant_spec:
            quant_spec[key] = np.abs(quant_spec[key]).astype(np.uint8)
    
    # --- 5. Create Output Directory ---
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 6. Save Hardware Config ---
    config_path = os.path.join(output_dir, 'hw_config.json')
    json_spec = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in quant_spec.items()}
    
    with open(config_path, 'w') as f:
        json.dump(json_spec, f, indent=4)
    print(f"Hardware configuration saved to: {config_path}")
    
    # --- 7. Generate Test Data ---
    if generate_real_data:
        sample_spikes = generate_real_spike_data(num_samples, num_steps, n_observations)
    else:
        sample_spikes = generate_random_spike_data(num_samples, num_steps, n_observations)
    
    np.save(os.path.join(output_dir, 'sample_spikes.npy'), sample_spikes)
    print(f"Sample spikes saved to: {os.path.join(output_dir, 'sample_spikes.npy')}")
    
    # --- 8. Save Metadata ---
    metadata = {
        'model_path': model_path,
        'export_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'architecture': {
            'n_observations': n_observations,
            'num_hidden': num_hidden,
            'n_actions': n_actions,
            'num_steps': num_steps,
            'dt': dt
        },
        'data': {
            'num_samples': num_samples,
            'data_type': 'real_brain_sim' if generate_real_data else 'random_poisson'
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {os.path.join(output_dir, 'metadata.json')}")
    
    print(f"\n{'='*60}")
    print("Export complete!")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export trained Rockpool RL model to Xylo hardware package.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to trained .pth model file')
    parser.add_argument('--output_dir', type=str, default='data/xylo_rl_package', 
                        help='Directory to save output package')
    parser.add_argument('--generate-real-data', action='store_true',
                        help='Generate real spike data from brain simulation (slower)')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of test samples to generate (default: 20)')
    
    args = parser.parse_args()
    
    success = export_model(
        args.model_path, 
        args.output_dir,
        generate_real_data=args.generate_real_data,
        num_samples=args.num_samples
    )
    
    if success:
        print(f"\nTo run hardware benchmark:\n")
        print(f"  python scripts/run_xylo_rl_benchmark.py {args.output_dir}")
