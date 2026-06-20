#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Xylo Hardware Benchmark for RL Models
======================================
Runs inference benchmarks on Xylo Audio 3 HDK for RL SNN models.

Measures:
- Inference latency (per step)
- Power consumption (idle vs inference vs dynamic)
- Energy-Delay Product (EDP)

Usage:
    python scripts/run_xylo_rl_benchmark.py data/xylo_rl_package/teacher --num-runs 10
"""

import numpy as np
import json
import argparse
from tqdm import tqdm
import sys
import os
import time
import csv
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import samna
    from rockpool.devices.xylo import find_xylo_hdks
    import rockpool.devices.xylo.syns65302 as xa3
    from rockpool.devices.xylo.syns65302 import xa3_devkit_utils as xa3utils
    XYLO_AVAILABLE = True
except ImportError:
    XYLO_AVAILABLE = False
    print("Warning: Rockpool/Xylo not available. Install rockpool and samna for hardware benchmarks.")


def run_single_benchmark(model_package_dir, dt=10e-3, clock_freq=6.25):
    """
    Run benchmark on a single model package.
    
    Args:
        model_package_dir: Directory containing hw_config.json and sample_spikes.npy
        dt: Simulation timestep (default: 10ms)
        clock_freq: Xylo core clock frequency in MHz (default: 6.25)
    
    Returns:
        dict with latency, power, and EDP metrics
    """
    # --- 1. Load Package Files ---
    config_path = os.path.join(model_package_dir, 'hw_config.json')
    spikes_path = os.path.join(model_package_dir, 'sample_spikes.npy')
    metadata_path = os.path.join(model_package_dir, 'metadata.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"hw_config.json not found in {model_package_dir}")
    if not os.path.exists(spikes_path):
        raise FileNotFoundError(f"sample_spikes.npy not found in {model_package_dir}")
    
    with open(config_path, 'r') as f:
        spec = {k: np.array(v) if isinstance(v, list) else v for k, v in json.load(f).items()}
    
    sample_spikes = np.load(spikes_path)
    num_samples = sample_spikes.shape[0]
    
    # Load metadata if available
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    print(f"Loaded {num_samples} samples from {model_package_dir}")
    
    # --- 2. Connect to Hardware ---
    print("Connecting to Xylo HDK...")
    hdk_nodes, _, versions = find_xylo_hdks()
    if not hdk_nodes:
        raise RuntimeError("No Xylo HDK found. Please connect the device.")
    hdk = hdk_nodes[0]
    print(f"Connected to Xylo HDK (Version: {versions[0]})")
    
    # --- 3. Build Hardware Configuration ---
    print("Building hardware configuration...")
    config = xa3.XyloConfiguration()
    
    # RL model dimensions
    Nin = 16
    Nhidden = 128
    Nout = 9  # 9 actions for RL
    
    # Set weights (sliced to match Nhidden)
    config.input.weights = np.array(spec['weights_in'][:, :Nhidden, :], dtype=np.int8)
    config.hidden.weights = np.array(spec['weights_rec'][:Nhidden, :Nhidden, :], dtype=np.int8)
    config.readout.weights = np.array(spec['weights_out'][:Nhidden, :], dtype=np.int8)
    
    # Configure hidden neurons
    hidden_neurons = []
    for i in range(Nhidden):
        neuron = samna.xyloAudio3.configuration.HiddenNeuron()
        neuron.v_mem_decay = spec['dash_mem'][i]
        neuron.i_syn_decay = spec['dash_syn'][i]
        if 'bias_hidden' in spec:
            config.bias_enable = True
            neuron.v_mem_bias = spec['bias_hidden'][i]
        hidden_neurons.append(neuron)
    config.hidden.neurons = hidden_neurons
    
    # Configure output neurons
    config.readout.neurons = [samna.xyloAudio3.configuration.OutputNeuron() for _ in range(Nout)]
    
    # Validate configuration
    is_valid, msg = samna.xyloAudio3.validate_configuration(config)
    if not is_valid:
        raise ValueError(f"Invalid hardware configuration: {msg}")
    print("Configuration validated successfully.")
    
    # --- 4. Deploy and Benchmark ---
    print("Deploying to Xylo...")
    modSamna = None
    
    try:
        # Set clock frequency and deploy
        xa3utils.set_xylo_core_clock_freq(hdk, clock_freq)
        modSamna = xa3.XyloSamna(hdk, config, dt=dt, power_frequency=20.)
        time.sleep(1.0)  # Allow configuration to settle
        
        # Clear power buffer before starting
        modSamna._power_buf.get_events()
        
        # --- Warmup (not timed, no power recording) ---
        warmup_samples = min(5, num_samples)
        print(f"Warming up with {warmup_samples} samples...")
        for i in range(warmup_samples):
            modSamna(sample_spikes[i], record_power=False)
        
        # Clear power buffer again before timed run
        modSamna._power_buf.get_events()
        
        # --- Timed Inference (power collected in background) ---
        print("Running timed inference...")
        timed_samples = num_samples - warmup_samples
        
        start_time = time.perf_counter()
        for i in tqdm(range(warmup_samples, num_samples), desc="Hardware Inference"):
            # Use record_power=False - power is collected in background by power_frequency
            output_spike_raster, _, _ = modSamna(sample_spikes[i], record_power=False)
        end_time = time.perf_counter()
        
        # Collect inference power events (accumulated during timed run)
        inference_power_events = modSamna._power_buf.get_events()
        
        # --- Measure Idle Power ---
        print("Measuring idle power...")
        modSamna._power_buf.get_events()  # Clear buffer
        time.sleep(2.0)
        idle_power_events = modSamna._power_buf.get_events()
        
        # --- Process Results ---
        total_time = end_time - start_time
        avg_latency_per_sample = total_time / timed_samples if timed_samples > 0 else 0
        
        def process_power_events(events):
            if not events:
                return np.zeros(3)
            power_readings = ([], [], [])
            for p in events:
                power_readings[p.channel].append(p.value)
            return np.array([np.mean(r) if r else 0 for r in power_readings])
        
        mean_inference_power = process_power_events(inference_power_events)
        mean_idle_power = process_power_events(idle_power_events)
        dynamic_power = mean_inference_power - mean_idle_power
        
        # SNN Core is channel 2
        snn_core_power = mean_inference_power[2]
        snn_dynamic_power = dynamic_power[2]
        
        # Calculate EDP
        edp = snn_core_power * avg_latency_per_sample
        
        results = {
            'model_name': os.path.basename(model_package_dir),
            'num_samples': timed_samples,
            'latency_s': avg_latency_per_sample,
            'latency_ms': avg_latency_per_sample * 1000,
            'inference_power_w': mean_inference_power.tolist(),
            'idle_power_w': mean_idle_power.tolist(),
            'dynamic_power_w': dynamic_power.tolist(),
            'snn_core_power_w': snn_core_power,
            'snn_dynamic_power_w': snn_dynamic_power,
            'edp': edp
        }
        
        return results
        
    finally:
        if modSamna is not None:
            del modSamna


def print_results_summary(all_results):
    """Print formatted results summary."""
    channel_names = ["All IO", "Analog & AFE", "SNN Core Logic"]
    
    print(f"\n{'='*70}")
    print("XYLO HARDWARE BENCHMARK RESULTS")
    print(f"{'='*70}")
    
    for model_name, results_list in all_results.items():
        if not results_list:
            continue
            
        print(f"\n--- {model_name} ({len(results_list)} runs) ---")
        
        latencies = np.array([r['latency_ms'] for r in results_list])
        snn_powers = np.array([r['snn_core_power_w'] for r in results_list])
        edps = np.array([r['edp'] for r in results_list])
        
        print(f"  Latency:    {np.mean(latencies):.3f} ± {np.std(latencies):.3f} ms")
        print(f"  SNN Power:  {np.mean(snn_powers) * 1e6:.1f} ± {np.std(snn_powers) * 1e6:.1f} µW")
        print(f"  EDP:        {np.mean(edps):.6e}")
        
        # Detailed power breakdown
        inf_powers = np.stack([r['inference_power_w'] for r in results_list])
        print("\n  Power Breakdown (Inference):")
        for i, name in enumerate(channel_names):
            print(f"    {name:<16}: {np.mean(inf_powers[:, i]) * 1e6:7.1f} µW")
    
    print(f"\n{'='*70}")


def save_results_csv(all_results, output_file):
    """Save results to CSV file."""
    header = [
        'model_name', 'run', 'latency_ms', 
        'inf_power_io_W', 'inf_power_analog_W', 'inf_power_snn_W',
        'dyn_power_io_W', 'dyn_power_analog_W', 'dyn_power_snn_W',
        'edp'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for model_name, results_list in all_results.items():
            for i, result in enumerate(results_list):
                row = [
                    model_name,
                    i + 1,
                    result['latency_ms'],
                    *result['inference_power_w'],
                    *result['dynamic_power_w'],
                    result['edp']
                ]
                writer.writerow(row)
    
    print(f"Results saved to: {output_file}")


def plot_results(all_results, output_file=None):
    """Generate benchmark comparison plot."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plot.")
        return
    
    model_names = list(all_results.keys())
    
    # Calculate means and stds
    mean_latencies = [np.mean([r['latency_ms'] for r in results_list]) 
                      for results_list in all_results.values()]
    std_latencies = [np.std([r['latency_ms'] for r in results_list]) 
                     for results_list in all_results.values()]
    
    mean_powers = [np.mean([r['snn_core_power_w'] for r in results_list]) * 1e6 
                   for results_list in all_results.values()]
    std_powers = [np.std([r['snn_core_power_w'] for r in results_list]) * 1e6 
                  for results_list in all_results.values()]
    
    mean_edps = [np.mean([r['edp'] for r in results_list]) 
                 for results_list in all_results.values()]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(model_names))
    
    # Colors
    colors = ['#2ecc71' if 'teacher' in m.lower() else '#27ae60' for m in model_names]
    
    # Latency
    axes[0].bar(x, mean_latencies, yerr=std_latencies, color=colors, capsize=5, edgecolor='black')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('A) Inference Latency')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=20, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Power
    axes[1].bar(x, mean_powers, yerr=std_powers, color=colors, capsize=5, edgecolor='black')
    axes[1].set_ylabel('Power (µW)')
    axes[1].set_title('B) SNN Core Power')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=20, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # EDP
    axes[2].bar(x, mean_edps, color=colors, edgecolor='black')
    axes[2].set_ylabel('EDP (J·s)')
    axes[2].set_title('C) Energy-Delay Product')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_names, rotation=20, ha='right')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Run hardware benchmark for RL models on Xylo HDK.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('model_package_dirs', nargs='+', type=str,
                        help='One or more paths to model deployment package directories.')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of benchmark runs per model (default: 3)')
    parser.add_argument('--dt', type=float, default=10e-3,
                        help='Simulation timestep in seconds (default: 0.01)')
    parser.add_argument('--clock-freq', type=float, default=6.25,
                        help='Xylo core clock frequency in MHz (default: 6.25)')
    parser.add_argument('--output-file', type=str,
                        help='Path to save results CSV')
    parser.add_argument('--output-dir', type=str, default='data/exp3_results',
                        help='Directory for output files (default: data/exp3_results)')
    
    args = parser.parse_args()
    
    if not XYLO_AVAILABLE:
        print("Error: Xylo/Rockpool not available. Cannot run hardware benchmark.")
        sys.exit(1)
    
    all_results = {}
    
    for model_dir in args.model_package_dirs:
        if not os.path.isdir(model_dir):
            print(f"Warning: Directory not found, skipping: {model_dir}")
            continue
        
        model_name = os.path.basename(model_dir)
        all_results[model_name] = []
        
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name} ({args.num_runs} runs)")
        print(f"{'='*60}")
        
        for run in range(args.num_runs):
            print(f"\n--- Run {run + 1}/{args.num_runs} ---")
            try:
                result = run_single_benchmark(model_dir, args.dt, args.clock_freq)
                all_results[model_name].append(result)
            except Exception as e:
                print(f"Error in run {run + 1}: {e}")
    
    # Print summary
    print_results_summary(all_results)
    
    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    
    # CSV
    csv_file = args.output_file or os.path.join(args.output_dir, f'xylo_benchmark_{timestamp}.csv')
    save_results_csv(all_results, csv_file)
    
    # Plot
    plot_file = os.path.join(args.output_dir, f'xylo_benchmark_{timestamp}.png')
    plot_results(all_results, plot_file)


if __name__ == "__main__":
    main()
