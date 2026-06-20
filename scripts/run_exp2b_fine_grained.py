#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 2B: Fine-Grained Pareto Front Analysis
==================================================
Extends Experiment 2 to find the precise location of the discontinuity
in the Pareto front of distilled sparse SNN models.

This script:
1. Trains NEW student models with finer hyperparameter granularity
2. Evaluates them with reduced simulation duration for speed
3. Combines with existing results to identify the transition boundary

For supplementary information documentation.

Usage:
    # Train new models and evaluate (full run)
    python scripts/run_exp2b_fine_grained.py --train --epochs 100

    # Dry-run (quick test)
    python scripts/run_exp2b_fine_grained.py --train --dry-run

    # Evaluate existing models only
    python scripts/run_exp2b_fine_grained.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import wandb

from src.simulation.sim_wrapper import SimulationWrapper
from src.models.rockpool_dqsn import RockpoolDQSN
from src.utils.action_utils_rockpool import get_action_dict
from src.environment.gym_pd import MousePDEnvXylo, MousePDEnvAdaptive


# --- Model Architecture ---
class AvgPoolDownsampler(nn.Module):
    def __init__(self, kernel_size=5, stride=5):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        B, T, C = x.shape
        x_reshaped = x.reshape(B * T, 1, C) 
        out = self.pool(x_reshaped)
        return out.reshape(B, T, -1)


# --- Model Constants ---
N_INPUT_CHANNELS = 16
NUM_HIDDEN = 128
N_ACTIONS = 9
BETA = 0.95
NUM_STEPS = 100
BATCH_SIZE = 1
DT = 10e-3


class SNNEvaluator:
    """Evaluate SNN model performance and efficiency."""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model_path = model_path
        
        self.rockpool_model = RockpoolDQSN(16, 128, 0.95, 9, 100, 1, dt=10e-3, use_mempot=True)
        self.downsampler = AvgPoolDownsampler(kernel_size=5, stride=5)
        self.model = nn.Sequential(self.downsampler, self.rockpool_model).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.reset()
        
    def reset(self):
        self.freq = 40.0
        self.pw = 0.3
        self.amp = 300.0
        self.leap = {'freq': 5.0, 'pw': 0.1, 'amp': 5.0}
        self.total_spikes = 0
        self.inference_count = 0
        self.rockpool_model.reset()
        
    def get_action(self, spike_data):
        with torch.no_grad():
            x = torch.from_numpy(spike_data.astype(np.float32)).unsqueeze(0).to(self.device)
            spk_out, mem_out, _, _, rec = self.model(x)
            
            hidden_spk1 = rec['1_LIFTorch']['spikes'].sum().item()
            hidden_spk2 = rec['3_LIFTorch']['spikes'].sum().item()
            self.total_spikes += hidden_spk1 + hidden_spk2
            self.inference_count += 1
            
            action_list = get_action_dict(self.model, spk_out, mem_out)
            increments = action_list[0]
            
            self.freq = np.clip(self.freq + increments['freq'] * self.leap['freq'], 0, 180)
            self.pw = np.clip(self.pw + increments['pw'] * self.leap['pw'], 0.06, 0.4)
            self.amp = np.clip(self.amp + increments['amp'] * self.leap['amp'], 0, 250)
            
            return {'freq': self.freq, 'pw': self.pw, 'amp': self.amp}
    
    def get_synops_per_ms(self):
        if self.inference_count == 0:
            return 0
        total_time_ms = self.inference_count * 100
        return self.total_spikes / total_time_ms


# --- Training Functions ---
def distillation_loss(student_outputs, teacher_outputs, temperature, student_sparsity_penalty):
    """Calculates the combined distillation and sparsity loss."""
    student_q_values = student_outputs.sum(0)
    teacher_q_values = teacher_outputs.sum(0)

    teacher_scaled = teacher_q_values / temperature
    student_scaled = student_q_values / temperature
    
    soft_targets = F.softmax(teacher_scaled, dim=1)
    soft_prob = F.log_softmax(student_scaled, dim=1)

    kl_raw = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
    distill_loss = kl_raw * (temperature ** 2)

    total_loss = distill_loss + student_sparsity_penalty
    return total_loss, distill_loss


def train_student_model(teacher_net, env, target_sparsity, sparsity_weight,
                        num_epochs=100, temperature=4.0, learning_rate=0.001, device='cpu',
                        use_wandb=True, model_dir='models/distilled_rockpool_16ch'):
    """
    Train a single student model via knowledge distillation.
    Reduced epochs for speed in fine-grained exploration.
    """
    # Initialize student
    downsampler = AvgPoolDownsampler(kernel_size=5, stride=5)
    rockpool_student = RockpoolDQSN(
        N_INPUT_CHANNELS, NUM_HIDDEN, BETA, N_ACTIONS,
        NUM_STEPS, BATCH_SIZE, use_mempot=True,
        target_sparsity=target_sparsity,
        sparsity_weight=sparsity_weight, dt=DT
    )
    student_net = nn.Sequential(downsampler, rockpool_student).to(device)
    optimizer = optim.Adam(student_net.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        student_net.train()
        optimizer.zero_grad()

        dummy_obs, info = env.reset()
        raw_spike_matrix = info['raw_spike_data']
        state = torch.from_numpy(raw_spike_matrix.astype(np.float32)).to(device).unsqueeze(0)

        with torch.no_grad():
            _, teacher_mem, _, _, _ = teacher_net(state)

        _, student_mem, sparsity_penalty, avg_rate, _ = student_net(state)

        # Gradual sparsity introduction
        sparsity_schedule = min(1.0, 2.0 * epoch / num_epochs)
        scheduled_sparsity_penalty = sparsity_penalty * sparsity_schedule

        total_loss, distill_loss = distillation_loss(
            student_mem, teacher_mem, temperature, scheduled_sparsity_penalty
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=1.0)
        optimizer.step()

        # WandB logging
        if use_wandb:
            wandb.log({
                "train/total_loss": total_loss.item(),
                "train/distill_loss": distill_loss.item(),
                "train/sparsity_penalty": sparsity_penalty.item(),
                "train/avg_firing_rate": avg_rate.item(),
                "train/sparsity_schedule": sparsity_schedule,
                "train/epoch": epoch
            })

    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    model_filename = f'student_s{target_sparsity}_t{temperature}_w{int(sparsity_weight)}_{timestamp}.pth'
    model_path = os.path.join(model_dir, model_filename)
    torch.save(student_net.state_dict(), model_path)

    return model_path, avg_rate.cpu().item()


def run_baseline_pd(num_steps, seed=None, warmup_steps=0):
    """Run unstimulated PD to get baseline beta power."""
    sim = SimulationWrapper(n_neurons=10, t_step=100, dt=0.01, pd=1, seed=seed)
    sim.set_pd_state(1)
    obs = sim.reset()

    for _ in range(warmup_steps):
        obs = sim.step({'freq': 0, 'pw': 0, 'amp': 0})

    lfp_history = []
    for _ in tqdm(range(num_steps), desc="Baseline PD", leave=False):
        obs = sim.step({'freq': 0, 'pw': 0, 'amp': 0})
        lfp_history.append(obs['lfp'])

    return np.mean(lfp_history)


def evaluate_model(model_path, num_steps, device='cpu', verbose=False, seed=None, warmup_steps=0):
    """Evaluate a single model for performance and efficiency."""
    evaluator = SNNEvaluator(model_path, device)

    sim = SimulationWrapper(n_neurons=10, t_step=100, dt=0.01, pd=1, seed=seed)
    sim.set_pd_state(1)
    obs = sim.reset()

    # Warmup: let SNN settle before recording beta power
    for _ in range(warmup_steps):
        action = evaluator.get_action(obs['spike_matrix'])
        obs = sim.step(action)

    # Reset spike counter so SynOps measures steady-state efficiency only
    evaluator.total_spikes = 0
    evaluator.inference_count = 0

    lfp_history = []
    desc = f"Evaluating {os.path.basename(model_path)}" if verbose else None
    iterator = tqdm(range(num_steps), desc=desc, leave=False) if verbose else range(num_steps)

    for _ in iterator:
        action = evaluator.get_action(obs['spike_matrix'])
        obs = sim.step(action)
        lfp_history.append(obs['lfp'])

    return {
        'beta_power': np.mean(lfp_history),
        'synops_per_ms': evaluator.get_synops_per_ms()
    }


def evaluate_model_multiseed(model_path, num_steps, device, seeds, verbose=False, warmup_steps=0):
    """Evaluate model across multiple seeds, returning mean +/- std."""
    all_beta = []
    all_synops = []

    for seed in seeds:
        metrics = evaluate_model(model_path, num_steps, device, verbose=verbose, seed=seed, warmup_steps=warmup_steps)
        all_beta.append(metrics['beta_power'])
        all_synops.append(metrics['synops_per_ms'])
    
    n = len(seeds)
    return {
        'beta_power_mean': np.mean(all_beta),
        'beta_power_std': np.std(all_beta, ddof=1) if n > 1 else 0.0,
        'beta_power_sem': np.std(all_beta, ddof=1) / np.sqrt(n) if n > 1 else 0.0,
        'synops_per_ms_mean': np.mean(all_synops),
        'synops_per_ms_std': np.std(all_synops, ddof=1) if n > 1 else 0.0,
        'synops_per_ms_sem': np.std(all_synops, ddof=1) / np.sqrt(n) if n > 1 else 0.0,
        'n_seeds': n,
        'seeds': list(seeds),
        'beta_power_all': all_beta,
        'synops_per_ms_all': all_synops
    }


def run_baseline_pd_multiseed(num_steps, seeds, warmup_steps=0):
    """Run unstimulated PD across multiple seeds."""
    all_beta = []
    for seed in seeds:
        beta = run_baseline_pd(num_steps, seed=seed, warmup_steps=warmup_steps)
        all_beta.append(beta)
    
    n = len(seeds)
    return {
        'mean': np.mean(all_beta),
        'std': np.std(all_beta, ddof=1) if n > 1 else 0.0,
        'sem': np.std(all_beta, ddof=1) / np.sqrt(n) if n > 1 else 0.0,
        'all': all_beta,
        'n_seeds': n
    }


def load_models_from_index(model_dir):
    """Load models from the models_index.csv file."""
    index_path = os.path.join(model_dir, 'models_index.csv')
    if not os.path.exists(index_path):
        return []
    
    df = pd.read_csv(index_path)
    models = []
    for _, row in df.iterrows():
        model_path = os.path.join(model_dir, row['model_file'])
        if os.path.exists(model_path):
            models.append({
                'path': model_path,
                'target_sparsity': row['target_sparsity'],
                'sparsity_weight': row['sparsity_weight'],
                'student_avg_rate': row.get('student_avg_rate', None)
            })
    return models


def classify_performance(beta_power, baseline_beta, threshold_reduction=50):
    """Classify model as 'working' or 'failing' based on beta reduction."""
    reduction = 100 * (baseline_beta - beta_power) / baseline_beta
    return 'working' if reduction >= threshold_reduction else 'failing'


def update_models_index(model_dir, new_entries):
    """Append new entries to the models_index.csv."""
    index_path = os.path.join(model_dir, 'models_index.csv')
    
    new_df = pd.DataFrame(new_entries)
    
    if os.path.exists(index_path):
        existing_df = pd.read_csv(index_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    combined_df.to_csv(index_path, index=False)
    print(f"Updated models index: {index_path}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 2B: Fine-Grained Pareto Front Analysis')
    parser.add_argument('--model-dir', type=str,
                        default='models/distilled_rockpool_16ch_new',
                        help='Directory containing distilled student models')
    parser.add_argument('--teacher-path', type=str,
                        default='models/final_rockpool_16ch_curriculum/teacher.pth',
                        help='Path to the trained teacher model')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Evaluation duration in seconds (default: 3.0)')
    parser.add_argument('--output-dir', type=str, default='data/exp2b_results',
                        help='Output directory for results')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run with minimal epochs/steps for testing')
    
    # Training options
    parser.add_argument('--train', action='store_true',
                        help='Train new models with fine-grained hyperparameters')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of distillation epochs per model (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                        help='Learning rate for training (default: 0.0005)')
    
    # Multi-seed evaluation options
    parser.add_argument('--num-seeds', type=int, default=1,
                        help='Number of random seeds per model evaluation (default: 1)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Explicit comma-separated seed list, e.g. "42,123,456,789,1024"')
    parser.add_argument('--rigorous', action='store_true',
                        help='Shorthand for --num-seeds 5 --duration 10.0 (publication mode)')
    parser.add_argument('--warmup', type=int, default=30,
                        help='Steps to run before recording beta power, letting the SNN settle (default: 30)')
    
    args = parser.parse_args()
    
    # Apply --rigorous defaults (can still be overridden by explicit args)
    if args.rigorous:
        if args.num_seeds == 1:  # not explicitly set
            args.num_seeds = 5
        if args.duration == 3.0:  # not explicitly set
            args.duration = 10.0
    
    # Determine seed list
    if args.seeds:
        eval_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        # Fixed seeds for reproducibility
        DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
        eval_seeds = DEFAULT_SEEDS[:args.num_seeds]
    
    use_multiseed = len(eval_seeds) > 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps = 3 if args.dry_run else int(args.duration * 10)
    num_epochs = 5 if args.dry_run else args.epochs
    
    print("="*60)
    print("EXPERIMENT 2B: FINE-GRAINED PARETO FRONT ANALYSIS")
    print("="*60)
    print(f"Mode: {'TRAINING + EVALUATION' if args.train else 'EVALUATION ONLY'}")
    print(f"Device: {device}")
    print(f"Evaluation steps: {num_steps} ({num_steps * 0.1:.1f} seconds)")
    print(f"Seeds: {eval_seeds} (n={len(eval_seeds)})")
    if use_multiseed:
        total_evals = len(eval_seeds) * num_steps
        print(f"Total steps per model: {total_evals} ({total_evals * 0.1:.0f}s simulated time)")
    if args.train:
        print(f"Training epochs: {num_epochs}")
    print()
    
    # --- Initialize parent WandB run for evaluation sweep ---
    sweep_run_name = f"exp2b_eval_{'rigorous' if use_multiseed else 'standard'}"
    sweep_wandb = wandb.init(
        project="cl-dbs-rl-exp2b-eval",
        name=sweep_run_name,
        config={
            "mode": "rigorous" if use_multiseed else "standard",
            "num_seeds": len(eval_seeds),
            "seeds": eval_seeds,
            "duration_seconds": args.duration,
            "num_steps": num_steps,
            "warmup_steps": args.warmup,
            "train": args.train,
            "dry_run": args.dry_run
        },
        reinit=True
    )
    models_evaluated = 0
    
    # Complete (target_sparsity x lambda) square. The coarse search (s=0.01/0.02/
    # 0.05/0.1, lambda=500/1000/2000) plus the finer extension left some cells
    # untested (gray in the heatmap); listing the full square here lets --train
    # fill ONLY the missing combos (the dedup against models_index skips existing
    # ones), so the published heatmap is a complete grid with no gaps.
    FINE_GRID = {
        'target_sparsity': [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1],
        'sparsity_weight': [500, 600, 700, 800, 900, 1000, 1100, 1200, 1500, 2000]
    }
    
    # Get baseline beta power
    print("--- Getting Baseline PD Beta Power ---")
    if use_multiseed:
        baseline_result = run_baseline_pd_multiseed(num_steps, eval_seeds, warmup_steps=args.warmup)
        baseline_beta = baseline_result['mean']
        print(f"Baseline PD Beta Power: {baseline_beta:.4f} ± {baseline_result['std']:.4f} (n={baseline_result['n_seeds']})")
        print(f"  Per-seed values: {[f'{v:.4f}' for v in baseline_result['all']]}")
    else:
        baseline_beta = run_baseline_pd(num_steps, seed=eval_seeds[0] if eval_seeds else None, warmup_steps=args.warmup)
        baseline_result = None
        print(f"Baseline PD Beta Power: {baseline_beta:.4f}")
    print()
    
    # Log baseline to WandB
    wandb.log({"baseline/beta_power": baseline_beta})
    if use_multiseed and baseline_result:
        wandb.log({
            "baseline/beta_std": baseline_result['std'],
            "baseline/beta_sem": baseline_result['sem']
        })
    
    results_data = []
    
    # TRAIN NEW MODELS if requested
    if args.train:
        print("--- Training New Models (Fine-Grained Grid) ---")
        
        # Load teacher model
        if not os.path.exists(args.teacher_path):
            print(f"ERROR: Teacher model not found at {args.teacher_path}")
            return
        
        # Initialize teacher
        downsampler_teacher = AvgPoolDownsampler(kernel_size=5, stride=5)
        rockpool_teacher = RockpoolDQSN(
            N_INPUT_CHANNELS, NUM_HIDDEN, BETA, N_ACTIONS, 
            NUM_STEPS, BATCH_SIZE, use_mempot=True,
            target_sparsity=1.0, sparsity_weight=0.0, dt=DT
        )
        teacher_net = nn.Sequential(downsampler_teacher, rockpool_teacher).to(device)
        teacher_net.load_state_dict(torch.load(args.teacher_path, map_location=device))
        teacher_net.eval()
        print("Teacher model loaded.\n")
        
        # Initialize environment
        leap = {'pw': 0.1, 'amp': 5.0, 'freq': 5.0}
        env = MousePDEnvAdaptive(leap=leap, num_steps=100, tau_beta_max=150.,
                                 tau_reward=3000., delta=0.5, max_steps=100,
                                 TMAX=100, thresh_time_req=999)
        
        # Check existing models to avoid retraining
        existing_models = load_models_from_index(args.model_dir)
        existing_combos = set()
        for m in existing_models:
            existing_combos.add((float(m['target_sparsity']), float(m['sparsity_weight'])))
        
        # Calculate total new combinations
        new_combos = []
        for ts in FINE_GRID['target_sparsity']:
            for sw in FINE_GRID['sparsity_weight']:
                if (ts, sw) not in existing_combos:
                    new_combos.append((ts, sw))
        
        print(f"Existing model combinations: {len(existing_combos)}")
        print(f"New combinations to train: {len(new_combos)}")
        
        if len(new_combos) == 0:
            print("All fine-grained combinations already trained!")
        else:
            new_index_entries = []
            
            for idx, (ts, sw) in enumerate(tqdm(new_combos, desc="Training models")):
                print(f"\n  [{idx+1}/{len(new_combos)}] Training: target_sparsity={ts}, λ={sw}")
                
                # Initialize WandB run for this model
                run_name = f"exp2b_s{ts}_w{int(sw)}"
                wandb.init(
                    project="cl-dbs-rl-exp2b-finegrained",
                    name=run_name,
                    config={
                        "target_sparsity": ts,
                        "sparsity_weight": sw,
                        "temperature": 4.0,
                        "num_epochs": num_epochs,
                        "learning_rate": args.learning_rate
                    },
                    reinit=True
                )
                
                model_path, avg_rate = train_student_model(
                    teacher_net, env, ts, sw,
                    num_epochs=num_epochs,
                    temperature=4.0,
                    learning_rate=args.learning_rate,
                    device=device,
                    use_wandb=True,
                    model_dir=args.model_dir
                )
                
                # Evaluate immediately
                if use_multiseed:
                    metrics = evaluate_model_multiseed(model_path, num_steps, device, eval_seeds, warmup_steps=args.warmup)
                    beta_val = metrics['beta_power_mean']
                    synops_val = metrics['synops_per_ms_mean']
                else:
                    single_metrics = evaluate_model(model_path, num_steps, device, seed=eval_seeds[0], warmup_steps=args.warmup)
                    beta_val = single_metrics['beta_power']
                    synops_val = single_metrics['synops_per_ms']
                    metrics = None
                
                reduction = 100 * (baseline_beta - beta_val) / baseline_beta
                classification = classify_performance(beta_val, baseline_beta)
                
                # Log evaluation results to WandB
                wandb_log = {
                    "eval/beta_power": beta_val,
                    "eval/beta_reduction": reduction,
                    "eval/synops_per_ms": synops_val,
                    "eval/classification": 1 if classification == 'working' else 0,
                    "eval/baseline_beta": baseline_beta
                }
                if use_multiseed:
                    wandb_log["eval/beta_power_std"] = metrics['beta_power_std']
                    wandb_log["eval/synops_per_ms_std"] = metrics['synops_per_ms_std']
                wandb.log(wandb_log)
                
                wandb.finish()  # End the per-model training sub-run
                
                # Resume parent sweep run
                wandb.init(project="cl-dbs-rl-exp2b-eval", id=sweep_wandb.id, resume="allow")
                
                # Log to parent sweep run
                models_evaluated += 1
                wandb.log({
                    "sweep/models_evaluated": models_evaluated,
                    "sweep/beta_power": beta_val,
                    "sweep/synops_per_ms": synops_val,
                    "sweep/beta_reduction": reduction,
                    "sweep/classification": 1 if classification == 'working' else 0,
                    "sweep/target_sparsity": ts,
                    "sweep/sparsity_weight": sw
                })
                
                if use_multiseed:
                    print(f"    Beta: {beta_val:.1f}±{metrics['beta_power_std']:.1f}, Reduction: {reduction:.1f}%, Class: {classification}")
                else:
                    print(f"    Beta Power: {beta_val:.1f}, Reduction: {reduction:.1f}%, Class: {classification}")
                
                row = {
                    'Target_Sparsity': ts,
                    'Sparsity_Weight': sw,
                    'Beta_Power': beta_val,
                    'Beta_Reduction_%': reduction,
                    'SynOps_per_ms': synops_val,
                    'Classification': classification,
                    'Model_Path': model_path
                }
                if use_multiseed:
                    row.update({
                        'Beta_Power_Mean': metrics['beta_power_mean'],
                        'Beta_Power_Std': metrics['beta_power_std'],
                        'Beta_Power_SEM': metrics['beta_power_sem'],
                        'SynOps_per_ms_Mean': metrics['synops_per_ms_mean'],
                        'SynOps_per_ms_Std': metrics['synops_per_ms_std'],
                        'SynOps_per_ms_SEM': metrics['synops_per_ms_sem'],
                        'N_Seeds': metrics['n_seeds'],
                        'Seeds': ','.join(str(s) for s in metrics['seeds']),
                        'Beta_Power_All': ','.join(f'{v:.4f}' for v in metrics['beta_power_all']),
                        'SynOps_per_ms_All': ','.join(f'{v:.4f}' for v in metrics['synops_per_ms_all'])
                    })
                results_data.append(row)
                
                new_index_entries.append({
                    'model_file': os.path.basename(model_path),
                    'target_sparsity': ts,
                    'temperature': 4.0,
                    'sparsity_weight': sw,
                    'student_avg_rate': avg_rate
                })
            
            # Update models index
            if new_index_entries:
                update_models_index(args.model_dir, new_index_entries)
    
    # EVALUATE ALL MODELS (existing + newly trained)
    print("\n--- Evaluating All Models ---")
    all_models = load_models_from_index(args.model_dir)
    print(f"Total models to evaluate: {len(all_models)}")
    if use_multiseed:
        print(f"Multi-seed evaluation: {len(eval_seeds)} seeds × {num_steps} steps per model")
    print()
    
    for i_model, model_info in enumerate(tqdm(all_models, desc="Evaluating models")):
        ts = model_info['target_sparsity']
        sw = model_info['sparsity_weight']
        
        # Skip if already evaluated during training
        already_evaluated = any(
            r['Target_Sparsity'] == ts and r['Sparsity_Weight'] == sw 
            for r in results_data
        )
        if already_evaluated:
            continue
        
        if use_multiseed:
            metrics = evaluate_model_multiseed(model_info['path'], num_steps, device, eval_seeds, warmup_steps=args.warmup)
            beta_val = metrics['beta_power_mean']
            synops_val = metrics['synops_per_ms_mean']
        else:
            single_metrics = evaluate_model(model_info['path'], num_steps, device, seed=eval_seeds[0] if eval_seeds else None, warmup_steps=args.warmup)
            beta_val = single_metrics['beta_power']
            synops_val = single_metrics['synops_per_ms']
            metrics = None
        
        reduction = 100 * (baseline_beta - beta_val) / baseline_beta
        classification = classify_performance(beta_val, baseline_beta)
        
        row = {
            'Target_Sparsity': ts,
            'Sparsity_Weight': sw,
            'Beta_Power': beta_val,
            'Beta_Reduction_%': reduction,
            'SynOps_per_ms': synops_val,
            'Classification': classification,
            'Model_Path': model_info['path']
        }
        if use_multiseed and metrics:
            row.update({
                'Beta_Power_Mean': metrics['beta_power_mean'],
                'Beta_Power_Std': metrics['beta_power_std'],
                'Beta_Power_SEM': metrics['beta_power_sem'],
                'SynOps_per_ms_Mean': metrics['synops_per_ms_mean'],
                'SynOps_per_ms_Std': metrics['synops_per_ms_std'],
                'SynOps_per_ms_SEM': metrics['synops_per_ms_sem'],
                'N_Seeds': metrics['n_seeds'],
                'Seeds': ','.join(str(s) for s in metrics['seeds']),
                'Beta_Power_All': ','.join(f'{v:.4f}' for v in metrics['beta_power_all']),
                'SynOps_per_ms_All': ','.join(f'{v:.4f}' for v in metrics['synops_per_ms_all'])
            })
        results_data.append(row)
        
        # Log to parent WandB sweep
        models_evaluated += 1
        wandb.log({
            "sweep/models_evaluated": models_evaluated,
            "sweep/total_models": len(all_models),
            "sweep/progress_pct": 100 * models_evaluated / max(len(all_models), 1),
            "sweep/beta_power": beta_val,
            "sweep/synops_per_ms": synops_val,
            "sweep/beta_reduction": reduction,
            "sweep/classification": 1 if classification == 'working' else 0,
            "sweep/target_sparsity": float(ts) if ts != 'Teacher' else -1,
            "sweep/sparsity_weight": float(sw),
            "sweep/model_name": os.path.basename(model_info['path'])
        })
    
    # Evaluate Teacher
    if os.path.exists(args.teacher_path):
        print("\n--- Evaluating Teacher Model ---")
        if use_multiseed:
            teacher_metrics = evaluate_model_multiseed(args.teacher_path, num_steps, device, eval_seeds, verbose=True, warmup_steps=args.warmup)
            teacher_beta = teacher_metrics['beta_power_mean']
            teacher_synops = teacher_metrics['synops_per_ms_mean']
            print(f"  Teacher Beta Power: {teacher_beta:.4f} ± {teacher_metrics['beta_power_std']:.4f}")
        else:
            teacher_single = evaluate_model(args.teacher_path, num_steps, device, verbose=True, seed=eval_seeds[0] if eval_seeds else None, warmup_steps=args.warmup)
            teacher_beta = teacher_single['beta_power']
            teacher_synops = teacher_single['synops_per_ms']
            teacher_metrics = None
            print(f"  Teacher Beta Power: {teacher_beta:.4f}")
        
        teacher_reduction = 100 * (baseline_beta - teacher_beta) / baseline_beta
        print(f"  Teacher Beta Reduction: {teacher_reduction:.2f}%")
        
        teacher_row = {
            'Target_Sparsity': 'Teacher',
            'Sparsity_Weight': 0,
            'Beta_Power': teacher_beta,
            'Beta_Reduction_%': teacher_reduction,
            'SynOps_per_ms': teacher_synops,
            'Classification': 'working',
            'Model_Path': args.teacher_path
        }
        if use_multiseed and teacher_metrics:
            teacher_row.update({
                'Beta_Power_Mean': teacher_metrics['beta_power_mean'],
                'Beta_Power_Std': teacher_metrics['beta_power_std'],
                'Beta_Power_SEM': teacher_metrics['beta_power_sem'],
                'SynOps_per_ms_Mean': teacher_metrics['synops_per_ms_mean'],
                'SynOps_per_ms_Std': teacher_metrics['synops_per_ms_std'],
                'SynOps_per_ms_SEM': teacher_metrics['synops_per_ms_sem'],
                'N_Seeds': teacher_metrics['n_seeds'],
                'Seeds': ','.join(str(s) for s in teacher_metrics['seeds']),
                'Beta_Power_All': ','.join(f'{v:.4f}' for v in teacher_metrics['beta_power_all']),
                'SynOps_per_ms_All': ','.join(f'{v:.4f}' for v in teacher_metrics['synops_per_ms_all'])
            })
        results_data.append(teacher_row)
    
    results_df = pd.DataFrame(results_data)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    suffix = '_rigorous' if use_multiseed else ''
    csv_path = os.path.join(args.output_dir, f'exp2b_fine_grained{suffix}_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Save baseline info alongside results
    if use_multiseed and baseline_result:
        baseline_path = os.path.join(args.output_dir, f'exp2b_baseline{suffix}_{timestamp}.json')
        import json
        baseline_info = {
            'baseline_beta_mean': baseline_result['mean'],
            'baseline_beta_std': baseline_result['std'],
            'baseline_beta_sem': baseline_result['sem'],
            'baseline_beta_all': baseline_result['all'],
            'n_seeds': baseline_result['n_seeds'],
            'seeds': list(eval_seeds),
            'num_steps': num_steps,
            'duration_seconds': args.duration
        }
        with open(baseline_path, 'w') as f:
            json.dump(baseline_info, f, indent=2)
        print(f"Baseline info saved to {baseline_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT 2B RESULTS SUMMARY")
    print("="*60)
    
    student_results = results_df[results_df['Target_Sparsity'] != 'Teacher']
    working = student_results[student_results['Classification'] == 'working']
    failing = student_results[student_results['Classification'] == 'failing']
    
    print(f"\nTotal models evaluated: {len(student_results)}")
    print(f"Working models (>50% reduction): {len(working)}")
    print(f"Failing models (<50% reduction): {len(failing)}")
    
    if len(working) > 0 and len(failing) > 0:
        print("\n--- Transition Boundary Analysis ---")
        print(f"Working models beta power range: {working['Beta_Power'].min():.1f} - {working['Beta_Power'].max():.1f}")
        print(f"Failing models beta power range: {failing['Beta_Power'].min():.1f} - {failing['Beta_Power'].max():.1f}")
        
        gap_lower = working['Beta_Power'].max()
        gap_upper = failing['Beta_Power'].min()
        print(f"\nGap/Discontinuity region: {gap_lower:.1f} - {gap_upper:.1f}")
        print(f"Gap size: {gap_upper - gap_lower:.1f} beta power units")
        
        # Log summary to WandB
        wandb.log({
            "summary/total_models": len(student_results),
            "summary/working_models": len(working),
            "summary/failing_models": len(failing),
            "summary/gap_lower": gap_lower,
            "summary/gap_upper": gap_upper,
            "summary/gap_size": gap_upper - gap_lower
        })
    
    print("\n" + "="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    # Finish parent WandB run
    wandb.finish()
    print("\nWandB sweep run finished.")


if __name__ == "__main__":
    main()
