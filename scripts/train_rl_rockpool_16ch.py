#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for Reinforcement Learning-based DBS parameter optimization
with a 16-channel Rockpool model, compatible with Xylo hardware.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import math
import random
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm
from datetime import datetime
import os
import time
import argparse
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.models.rockpool_dqsn import RockpoolDQSN
from src.environment.gym_pd import MousePDEnvXylo, MousePDEnvSoftReward, MousePDEnvAdaptive
from src.environment.replay_memory import ReplayMemory, Transition

# Import utility modules
from src.utils.action_utils_rockpool import get_action_dict, select_action, flatten_action
from src.utils.optimization_utils_rockpool import optimize_model, update_target_network
from src.utils.io_utils import save_results, save_checkpoint, save_config, load_checkpoint

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TAU = 0.005
LR = 1e-3
BETA = 0.95
REWARD_SCALE = 1.0

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(policy_net, target_net, memory, optimizer, env, num_episodes, save_freq, device,
          start_episode=0,
          initial_steps_done=0,
          sparse=False,
          initial_results=None,
          curriculum=False,
          pd_duration=100,
          healthy_duration=100):
    """
    Main training loop.
    """
    steps_done = initial_steps_done
    
    if initial_results:
        episode_rewards = initial_results.get('episode_rewards', [])
        episode_lengths = initial_results.get('episode_lengths', [])
        episode_epsilon = initial_results.get('episode_epsilon', [])
        episode_actions = initial_results.get('episode_actions', [])
        episode_final_states = initial_results.get('episode_final_states', [])
        episode_losses = initial_results.get('episode_losses', [])
        episode_alpha_beta = initial_results.get('episode_alpha_beta', [])
        episode_dbs_params = initial_results.get('episode_dbs_params', [])
        episode_brain_spikes = initial_results.get('episode_brain_spikes', [])
        episode_stim_energy = initial_results.get('episode_stim_energy', [])
        episode_snn_spikes = initial_results.get('episode_snn_spikes', [])
        loss_per_batch = initial_results.get('loss_per_batch', [])
        episode_sparsity = initial_results.get('episode_sparsity', [])
    else:
        episode_rewards, episode_lengths, episode_epsilon, episode_actions = [], [], [], []
        episode_final_states, episode_losses, episode_alpha_beta, episode_dbs_params = [], [], [], []
        episode_brain_spikes, episode_stim_energy, episode_snn_spikes = [], [], []
        loss_per_batch, episode_sparsity = [], []
    
    steps_since_switch = 0
    current_pd_state = 1 # Default assumption
    
    print(f'\nStarting training from episode {start_episode + 1} for {num_episodes - start_episode} episodes...')
    
    for i_episode in tqdm(range(start_episode, num_episodes)):
        # Curriculum Learning Logic
        if curriculum:
            cycle_len = pd_duration + healthy_duration
            cycle_pos = i_episode % cycle_len
            if cycle_pos < pd_duration:
                new_pd_state = 1 # PD
            else:
                new_pd_state = 0 # Healthy
            
            # Check for state switch to reset exploration
            if i_episode > start_episode and new_pd_state != current_pd_state:
                steps_since_switch = 0
                tqdm.write(f"Episode {i_episode}: Switched to PD State = {new_pd_state}. Resetting Epsilon Decay.")
            
            current_pd_state = new_pd_state
            env.set_pd_state(current_pd_state)
        
        dummy_obs, info = env.reset()

        # Reset network states at the beginning of each episode
        policy_net[1].reset()
        target_net[1].reset()
        
        raw_spike_matrix = info['raw_spike_data']

        # Prepare state for the sequential model (B, T, C) -> (1, 100, 80)
        state = torch.from_numpy(raw_spike_matrix.astype(np.float32)).to(device).unsqueeze(0)

        episode_reward, episode_length, episode_loss = 0, 0, 0
        episode_brain_spike, episode_stim_energy_val, episode_snn_spike = 0, 0, 0
        episode_sparsity_val = 0
        episode_action_list = []
        env.t = 0
        
        for t in tqdm(range(env.max_steps), desc=f"Episode {i_episode + 1}/{num_episodes}", leave=False): 
            env.t += 1
            
            # Determine step count for epsilon decay
            if curriculum:
                eps_steps = steps_since_switch
            else:
                eps_steps = steps_done
            
            action, spk_counts = select_action(policy_net, target_net, state, env, eps_steps, device, 
                                              EPS_START, EPS_END, EPS_DECAY)
            steps_done += 1
            if curriculum:
                steps_since_switch += 1
            
            dummy_observation, reward, terminated, truncated, info = env.step(action)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            next_raw_spike_matrix = info['raw_spike_data']

            episode_reward += reward.item()
            episode_length += 1
            episode_action_list.append(action)
            
            current_step_spikes = np.sum(next_raw_spike_matrix)
            episode_brain_spike += current_step_spikes
            
            episode_stim_energy_val += env.E
            episode_snn_spike += spk_counts[0]

            if terminated:
                next_state = None
            else:
                # Prepare next_state for the sequential model
                next_state = torch.from_numpy(next_raw_spike_matrix.astype(np.float32)).to(device).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state

            loss, current_sparsity = optimize_model(policy_net, target_net, optimizer, memory, device, 
                                                  BATCH_SIZE, GAMMA)
            episode_sparsity_val += current_sparsity
            
            if loss is not None:
                episode_loss += loss
                loss_per_batch.append(loss)

            update_target_network(policy_net, target_net, TAU)

            if done:
                break
        
        # Save episode data regardless of how the episode ended (terminated, truncated, or max steps)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_epsilon.append(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (steps_since_switch if curriculum else steps_done) / EPS_DECAY))
        episode_actions.extend(episode_action_list)
        episode_final_states.append({'raw_spikes': next_raw_spike_matrix, 'E': env.E})
        episode_losses.append(episode_loss / episode_length if episode_length > 0 else 0)
        episode_alpha_beta.append(env.gpi_alpha_beta_area)
        episode_dbs_params.append({'freq': env.freq, 'pw': env.pw, 'amp': env.amp})
        episode_brain_spikes.append(episode_brain_spike)
        episode_stim_energy.append(episode_stim_energy_val)
        episode_snn_spikes.append(episode_snn_spike)
        episode_sparsity.append(episode_sparsity_val / episode_length if episode_length > 0 else 0)

        wandb.log({
            "Episode Reward": episode_reward,
            "Episode Length": episode_length,
            "Epsilon": EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (steps_since_switch if curriculum else steps_done) / EPS_DECAY),
            "Episode Loss": episode_loss / episode_length if episode_length > 0 else 0,
            "GPi Alpha-beta Oscillation": env.gpi_alpha_beta_area,
            "Frequency": env.freq,
            "Pulse Width": env.pw,
            "Amplitude": env.amp,
            "Avg Brain Spikes per Step": episode_brain_spike / episode_length if episode_length > 0 else 0,
            "Total Stimulus Energy": episode_stim_energy_val,
            "Avg SNN Spikes per Step": episode_snn_spike / episode_length if episode_length > 0 else 0,
            "Avg Sparsity per Step": episode_sparsity_val / episode_length if episode_length > 0 else 0,
            "PD State": env.PD if hasattr(env, 'PD') else 1,
        })
        
        if (i_episode + 1) % save_freq == 0:
            checkpoint_dir = 'models/checkpoints_rockpool_16ch'
            if curriculum:
                checkpoint_dir += '_curriculum'
            save_checkpoint(policy_net, optimizer, i_episode + 1, loss, steps_done, checkpoint_dir)
            
            current_results = {
                'episode_rewards': episode_rewards, 'episode_lengths': episode_lengths,
                'episode_epsilon': episode_epsilon, 'episode_actions': episode_actions,
                'episode_final_states': episode_final_states, 'episode_losses': episode_losses,
                'loss_per_batch': loss_per_batch, 'episode_alpha_beta': episode_alpha_beta,
                'episode_dbs_params': episode_dbs_params, 'episode_brain_spikes': episode_brain_spikes,
                'episode_stim_energy': episode_stim_energy, 'episode_snn_spikes': episode_snn_spikes,
                'episode_sparsity': episode_sparsity
            }
            
            results_dir = 'data/results_rockpool_16ch'
            if curriculum:
                results_dir += '_curriculum'
            os.makedirs(results_dir, exist_ok=True)
            results_filepath = os.path.join(results_dir, 'intermediate_results.pth')
            save_results(results_filepath, current_results)
            tqdm.write(f"\nSaved intermediate results for episode {i_episode + 1} to {results_filepath}")
    
    return {
        'episode_rewards': episode_rewards, 'episode_lengths': episode_lengths,
        'episode_epsilon': episode_epsilon, 'episode_actions': episode_actions,
        'episode_final_states': episode_final_states, 'episode_losses': episode_losses,
        'loss_per_batch': loss_per_batch, 'episode_alpha_beta': episode_alpha_beta,
        'episode_dbs_params': episode_dbs_params, 'episode_brain_spikes': episode_brain_spikes,
        'episode_stim_energy': episode_stim_energy, 'episode_snn_spikes': episode_snn_spikes,
        'episode_sparsity': episode_sparsity
    }

def main():
    """Main function to run the training process.

    Resume training from a checkpoint with:
    python -m scripts.train_rl_rockpool_16ch \
     --checkpoint "path/to/your/checkpoint.pth" \
     --force-steps 8725
    """
    parser = argparse.ArgumentParser(description="Train a 16-channel Rockpool model for DBS optimization.")
    parser.add_argument('--checkpoint', type=str, help='Path to a checkpoint file to continue training from.')
    parser.add_argument('--force-steps', type=int, help='Manually set the starting steps_done value, overriding the checkpoint value.')
    parser.add_argument('--target-sparsity', type=float, default=0.0, help='Target sparsity for hidden layers.')
    parser.add_argument('--sparsity-weight', type=float, default=0.0, help='Weight for the sparsity penalty.')
    parser.add_argument('--delta', type=float, default=0.5,
                        help='Energy penalty weight. Default: 0.5 (for MousePDEnvAdaptive), use 0.01 with --legacy-reward')
    parser.add_argument('--legacy-reward', action='store_true',
                        help='Use MousePDEnvXylo (original reward) instead of MousePDEnvAdaptive')
    parser.add_argument('--num-episodes', type=int, default=500, help='Number of episodes to train.')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Max steps per episode.')
    parser.add_argument('--thresh-time-req', type=int, default=999,
                        help='Steps below threshold to terminate. Set to 999 to disable early termination.')
    parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning (alternating PD/Healthy).')
    parser.add_argument('--pd-duration', type=int, default=50, help='Number of episodes for PD block.')
    parser.add_argument('--healthy-duration', type=int, default=50, help='Number of episodes for Healthy block.')
    args = parser.parse_args()

    project_name = "cl-dbs-rl-rockpool-16ch"
    if args.curriculum:
        project_name += "-curriculum"

    # Select environment class based on --legacy-reward flag
    leap = {'pw': 0.1, 'amp': 5.0, 'freq': 5.0}
    EnvClass = MousePDEnvXylo if args.legacy_reward else MousePDEnvAdaptive
    env = EnvClass(
        leap=leap, 
        num_steps=100, 
        tau_beta_max=150., 
        tau_reward=3000.,  
        delta=args.delta, 
        max_steps=args.max_steps,
        TMAX=100,
        thresh_time_req=args.thresh_time_req
    )
    
    # --- Define Network Structure ---
    NUM_NEURON_TYPES = 8
    NEURONS_PER_TYPE = 10
    N_INPUT_CHANNELS = 16 # Target channels for Xylo
    N_RAW_CHANNELS = NUM_NEURON_TYPES * NEURONS_PER_TYPE # Should be 80

    n_observations = N_INPUT_CHANNELS
    num_steps = env.num_steps

    wandb.init(project=project_name, config={
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "eps_start": EPS_START,
        "eps_end": EPS_END,
        "eps_decay": EPS_DECAY,
        "tau": TAU,
        "beta": BETA,
        "target_sparsity": args.target_sparsity,
        "sparsity_weight": args.sparsity_weight,
        "curriculum": args.curriculum,
        "pd_duration": args.pd_duration,
        "healthy_duration": args.healthy_duration,
        "env_class": env.__class__.__name__,
        "max_steps": env.max_steps,
        "num_episodes": args.num_episodes,
        "TMAX": env.TMAX
    })

    print('Training Xylo-compatible Rockpool model with Fixed AvgPool1d Downsampler')
    print('='*40)
    print(f'Raw Input Channels: {N_RAW_CHANNELS}')
    print(f'Downsampled Input Obs: {n_observations} channels')
    print(f'Time Steps per Observation: {num_steps}')
    print(f'Target Sparsity: {args.target_sparsity}')
    print(f'Sparsity Weight: {args.sparsity_weight}')
    print('='*40)

    num_hidden = 128
    n_actions = 9
    use_mempot = True

    dt = 10e-3  # Rockpool simulation time-step

    # Downsample the 80 raw neuron channels to 16 by averaging groups of 5,
    # so the network fits Xylo's 16 input channels. AvgPool1d expects (N, C, L),
    # so the module below reshapes (B, T, 80) -> pools over the 80 -> (B, T, 16).

    class AvgPoolDownsampler(nn.Module):
        def __init__(self, kernel_size=5, stride=5):
            super().__init__()
            self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
            
        def forward(self, x):
            # x: (B, T, 80) -> (B*T, 1, 80) -> pool -> (B*T, 1, 16) -> (B, T, 16)
            B, T, C = x.shape
            x_reshaped = x.reshape(B * T, 1, C)
            out = self.pool(x_reshaped)
            return out.reshape(B, T, -1)

    downsampler_policy = AvgPoolDownsampler(kernel_size=5, stride=5)
    rockpool_policy = RockpoolDQSN(
        n_observations, num_hidden, BETA, n_actions, num_steps, BATCH_SIZE, 
        use_mempot=use_mempot, dt=dt, 
        target_sparsity=args.target_sparsity, sparsity_weight=args.sparsity_weight
    )
    policy_net = nn.Sequential(downsampler_policy, rockpool_policy).to(device)

    # Target Network
    downsampler_target = AvgPoolDownsampler(kernel_size=5, stride=5)
    rockpool_target = RockpoolDQSN(
        n_observations, num_hidden, BETA, n_actions, num_steps, BATCH_SIZE, 
        use_mempot=use_mempot, dt=dt, 
        target_sparsity=args.target_sparsity, sparsity_weight=args.sparsity_weight
    )
    target_net = nn.Sequential(downsampler_target, rockpool_target).to(device)
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    
    start_episode = 0
    steps_done = 0
    initial_results = None

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        start_episode, steps_done = load_checkpoint(policy_net, optimizer, args.checkpoint)
        print(f"Resuming from episode {start_episode}, with {steps_done} steps completed.")

        if args.curriculum:
            intermediate_results_path = 'data/results_rockpool_16ch_curriculum/intermediate_results.pth'
        else:
            intermediate_results_path = 'data/results_rockpool_16ch/intermediate_results.pth'
        if os.path.exists(intermediate_results_path):
            try:
                initial_results = torch.load(intermediate_results_path, map_location=device)
                print(f"Successfully loaded intermediate results from {intermediate_results_path}")
                loaded_episodes = len(initial_results.get('episode_rewards', []))
                if loaded_episodes != start_episode:
                    print(f"Warning: Mismatch between checkpoint episode ({start_episode}) and loaded results episodes ({loaded_episodes}).")
            except Exception as e:
                print(f"Could not load intermediate results file: {e}")
        else:
            print("No intermediate results file found. Starting with fresh results.")

    if args.force_steps is not None:
        steps_done = args.force_steps
        print(f"Manual override: Starting with {steps_done} steps.")

    target_net.load_state_dict(policy_net.state_dict())

    memory = ReplayMemory(100000)


    num_episodes = args.num_episodes
    save_freq = 25

    results = train(policy_net, target_net, memory, optimizer, env, num_episodes, save_freq, device,
                    start_episode=start_episode,
                    initial_steps_done=steps_done,
                    initial_results=initial_results,
                    curriculum=args.curriculum,
                    pd_duration=args.pd_duration,
                    healthy_duration=args.healthy_duration)

    time_now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    
    results_dir = 'data/results_rockpool_16ch'
    models_dir = 'models/final_rockpool_16ch'
    configs_dir = 'configs_rockpool_16ch'
    
    if args.curriculum:
        results_dir += '_curriculum'
        models_dir += '_curriculum'
        configs_dir += '_curriculum'
        
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)
    
    results_filename = f'training_results_{time_now}.pth'
    model_filename = f'model_{time_now}.pth'
    config_filename = f'config_{time_now}.pth'
    
    save_results(os.path.join(results_dir, results_filename), results)
    torch.save(policy_net.state_dict(), os.path.join(models_dir, model_filename))

    config = {
        'n_observations': n_observations, 'num_steps': num_steps, 'num_hidden': num_hidden,
        'n_actions': n_actions, 'use_mempot': use_mempot, 'BATCH_SIZE': BATCH_SIZE, 'GAMMA': GAMMA,
        'EPS_START': EPS_START, 'EPS_END': EPS_END, 'EPS_DECAY': EPS_DECAY, 'TAU': TAU, 'LR': LR,
        'BETA': BETA, 'num_episodes': num_episodes, 'save_freq': save_freq,
        'tau_beta_max': env.tau_beta_max, 'thresh_time_req': env.thresh_time_req,
        'tau_reward': env.tau_reward, 'delta': env.delta, 'TMAX': env.TMAX, 'max_steps': env.max_steps,
        'target_sparsity': args.target_sparsity, 'sparsity_weight': args.sparsity_weight,
        'curriculum': args.curriculum, 'pd_duration': args.pd_duration, 'healthy_duration': args.healthy_duration,
        'env_class': env.__class__.__name__
    }
    save_config(config, os.path.join(configs_dir, config_filename))

    print("\nTraining completed.")
    print(f"New 16-channel Rockpool model saved to: {os.path.join(models_dir, model_filename)}")

    wandb.finish()

if __name__ == "__main__":
    main()