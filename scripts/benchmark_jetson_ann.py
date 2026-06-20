#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Jetson Orin Nano Benchmark for ANN Baseline
=============================================
Runs inference benchmarks for the ANN baseline model on Jetson GPU.

Measures:
- Inference latency (per step)
- Power consumption (via tegrastats on Jetson, estimated on other platforms)
- Energy-Delay Product (EDP)

Usage (on Jetson):
    python scripts/benchmark_jetson_ann.py \
        --model models/ann_baseline_curriculum/ann_baseline.pth \
        --device cuda \
        --num-runs 3
"""

import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import csv
import subprocess
import threading
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ann_baseline import ANNBaseline


class PowerMonitor:
    """
    Power monitoring for Jetson using tegrastats.
    Falls back to estimated values on other platforms.
    """
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.power_readings = []
        self.is_jetson = self._check_jetson()
        self._running = False
        self._process = None
        self._thread = None
        
    def _check_jetson(self):
        """Check if running on Jetson platform."""
        try:
            result = subprocess.run(['tegrastats', '--help'], 
                                   capture_output=True, timeout=2)
            return True  # If tegrastats exists, we're on Jetson
        except:
            return False
    
    def _parse_power(self, line):
        """Parse power from tegrastats output line."""
        import re
        # Try different power rail patterns (varies by Jetson model)
        # Orin format: VDD_IN 5000mW/5000mW or VDD_CPU_GPU_CV 1234mW/5678mW
        patterns = [
            r'VDD_IN\s+(\d+)mW',           # Total input power
            r'VDD_CPU_GPU_CV\s+(\d+)mW',   # CPU+GPU power (Orin)
            r'VDD_GPU_SOC\s+(\d+)mW',      # GPU+SoC power
            r'POM_5V_IN\s+(\d+)',          # Older Jetson format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return float(match.group(1)) / 1000.0  # Convert mW to W
        return None
    
    def _monitor_loop(self):
        """Background thread reading from tegrastats subprocess."""
        try:
            self._process = subprocess.Popen(
                ['tegrastats', '--interval', str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            
            while self._running and self._process.poll() is None:
                line = self._process.stdout.readline()
                if line:
                    power = self._parse_power(line)
                    if power is not None:
                        self.power_readings.append(power)
        except Exception as e:
            print(f"Power monitor error: {e}")
        finally:
            if self._process:
                self._process.terminate()
    
    def start(self):
        """Start power monitoring."""
        self.power_readings = []
        if self.is_jetson:
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            time.sleep(0.2)  # Let tegrastats start
    
    def stop(self):
        """Stop power monitoring and return mean power."""
        self._running = False
        if self._process:
            self._process.terminate()
            self._process = None
        if self._thread:
            self._thread.join(timeout=1)
        
        if self.power_readings:
            return np.mean(self.power_readings)
        else:
            # Return estimated power for non-Jetson platforms
            return 5.0  # Estimated 5W for Jetson Orin Nano


def run_benchmark(model_path, device='cpu', num_samples=100, num_timesteps=100):
    """
    Run ANN benchmark on specified device.
    
    Args:
        model_path: Path to trained .pth model file
        device: 'cpu' or 'cuda'
        num_samples: Number of inference samples
        num_timesteps: Time steps per sample (for input shape)
    
    Returns:
        dict with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"ANN Benchmark")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Architecture parameters (must match training)
    n_observations = 80  # Full channels (no downsampling for ANN)
    num_hidden = 128
    n_actions = 9
    
    # Initialize model
    model = ANNBaseline(n_observations, num_hidden, n_actions).to(device)
    
    # Load trained weights
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        print(f"Loaded model from: {model_path}")
    else:
        print("Warning: Using untrained model (random weights)")
    
    model.eval()
    
    # Generate sample input data [Batch, Time, Channels]
    # Simulates brain spike data
    sample_input = torch.randn(1, num_timesteps, n_observations).to(device)
    
    # Initialize power monitor
    power_monitor = PowerMonitor()
    
    # Warmup
    print("Warming up...")
    for _ in range(50):
        with torch.no_grad():
            _ = model(sample_input)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    # Start power monitoring
    power_monitor.start()
    
    # Benchmark
    print(f"Running {num_samples} inferences...")
    latencies = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate fresh random input for each sample
            sample_input = torch.randn(1, num_timesteps, n_observations).to(device)
            
            start = time.perf_counter()
            output = model(sample_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # Convert to ms
    
    # Stop power monitoring
    mean_power = power_monitor.stop()
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    # Calculate EDP (Energy-Delay Product)
    edp = mean_power * (avg_latency / 1000)  # W * s = J
    
    results = {
        'model_name': os.path.basename(model_path) if model_path else 'untrained',
        'device': device,
        'num_samples': num_samples,
        'latency_ms': avg_latency,
        'latency_std_ms': std_latency,
        'power_w': mean_power,
        'edp': edp,
        'is_jetson': power_monitor.is_jetson
    }
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Device:       {device} {'(Jetson)' if power_monitor.is_jetson else '(Simulated)'}")
    print(f"Latency:      {avg_latency:.4f} ± {std_latency:.4f} ms")
    print(f"Throughput:   {1000/avg_latency:.2f} inferences/sec")
    print(f"Power:        {mean_power * 1000:.1f} mW {'(measured)' if power_monitor.is_jetson else '(estimated)'}")
    print(f"EDP:          {edp:.6e} J·s")
    print(f"{'='*60}\n")
    
    return results


def save_results_csv(results_list, output_file):
    """Save results to CSV file."""
    header = ['model_name', 'device', 'run', 'latency_ms', 'latency_std_ms', 
              'power_w', 'edp', 'is_jetson']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for i, result in enumerate(results_list):
            row = [
                result['model_name'],
                result['device'],
                i + 1,
                result['latency_ms'],
                result['latency_std_ms'],
                result['power_w'],
                result['edp'],
                result['is_jetson']
            ]
            writer.writerow(row)
    
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark ANN baseline on Jetson or CPU.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, 
                        default='models/ann_baseline_curriculum/ann_baseline.pth',
                        help='Path to trained .pth model file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda)')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of benchmark runs (default: 3)')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of inference samples per run (default: 100)')
    parser.add_argument('--output-dir', type=str, default='data/exp3_results',
                        help='Directory for output files')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Run benchmarks
    all_results = []
    
    for run in range(args.num_runs):
        print(f"\n--- Run {run + 1}/{args.num_runs} ---")
        result = run_benchmark(args.model, args.device, args.num_samples)
        all_results.append(result)
    
    # Print summary
    latencies = [r['latency_ms'] for r in all_results]
    powers = [r['power_w'] for r in all_results]
    edps = [r['edp'] for r in all_results]
    
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(all_results)} runs)")
    print(f"{'='*60}")
    print(f"Latency:  {np.mean(latencies):.4f} ± {np.std(latencies):.4f} ms")
    print(f"Power:    {np.mean(powers) * 1000:.1f} ± {np.std(powers) * 1000:.1f} mW")
    print(f"EDP:      {np.mean(edps):.6e}")
    print(f"{'='*60}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    csv_file = os.path.join(args.output_dir, f'jetson_benchmark_{timestamp}.csv')
    save_results_csv(all_results, csv_file)


if __name__ == "__main__":
    main()
