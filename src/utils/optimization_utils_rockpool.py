#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for model optimization in the RL-based DBS parameter optimization.
"""

import torch
import torch.nn as nn

def optimize_model(policy_net, target_net, optimizer, memory, device, batch_size, gamma):
    """
    Perform one step of optimization on the policy network.
    
    Args:
        policy_net (nn.Sequential): Policy network (Sequential model)
        target_net (nn.Sequential): Target network (Sequential model)
        optimizer (Optimizer): Optimizer
        memory (ReplayMemory): Replay memory
        device (torch.device): Device to run computations on
        batch_size (int): Size of batch to sample from memory
        gamma (float): Discount factor for future rewards
        
    Returns:
        tuple: (loss, current sparsity)
    """
    from src.utils.action_utils_rockpool import flatten_action, get_action_dict
    from src.environment.replay_memory import Transition
    
    if len(memory) < batch_size:
        return None, 0
    
    # Sample a batch of transitions
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Compute mask of non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                 device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    # Prepare batch and ensure it's on the correct device
    state_batch = torch.cat(batch.state).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Transpose the batch to be (Time, Batch, Features) for Rockpool
    state_batch = state_batch.permute(1, 0, 2)

    # Forward pass
    spk, mem, sparsity_penalty, current_sparsity, _ = policy_net(state_batch)

    # Prepare actions and ensure they're on the correct device
    action_indices_flat = flatten_action(batch.action, device).to(device)
    
    summed_mempot = mem.sum(0).to(device)
    summed_spk = spk.sum(0).to(device)

    # Get state-action values
    # Access the RockpoolDQSN model inside the Sequential container
    # Access the RockpoolDQSN model inside the Sequential container if applicable
    if isinstance(policy_net, nn.Sequential):
        rockpool_policy = policy_net[1]
    else:
        rockpool_policy = policy_net

    if rockpool_policy.use_mempot:
        state_action_values = summed_mempot.gather(1, action_indices_flat)
    else: 
        state_action_values = summed_spk.gather(1, action_indices_flat)

    # Compute next state values
    next_state_values = torch.zeros(batch_size, 3, device=device)
    if non_final_next_states is not None:
        with torch.no_grad():
            # Transpose the batch to be (Time, Batch, Features) for Rockpool and move to device
            non_final_next_states = non_final_next_states.permute(1, 0, 2).to(device)

            next_spk_batch, next_mem_batch, next_sparsity_penalty, next_current_sparsity, _ = target_net(non_final_next_states)

            next_spk_sum = next_spk_batch.sum(0).to(device)
            next_mem_sum = next_mem_batch.sum(0).to(device)

            next_action_dict_batch = get_action_dict(target_net, next_spk_batch, next_mem_batch)
            next_action_flattened = flatten_action(next_action_dict_batch, device).to(device)
            
            # Access the RockpoolDQSN model inside the Sequential container
            # Access the RockpoolDQSN model inside the Sequential container if applicable
            if isinstance(target_net, nn.Sequential):
                rockpool_target = target_net[1]
            else:
                rockpool_target = target_net

            if rockpool_target.use_mempot:
                next_state_values[non_final_mask] = next_mem_sum.gather(1, next_action_flattened)
            else: 
                next_state_values[non_final_mask] = next_spk_sum.gather(1, next_action_flattened)
    
    # Compute expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch.unsqueeze(1)

    # Compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    
    # Add sparsity penalty
    loss += sparsity_penalty

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item(), current_sparsity.item()

def update_target_network(policy_net, target_net, tau):
    """
    Update target network with policy network weights using soft update.
    
    Args:
        policy_net (nn.Sequential): Policy network (Sequential model)
        target_net (nn.Sequential): Target network to be updated (Sequential model)
        tau (float): Interpolation parameter for soft update
    """
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
        
    target_net.load_state_dict(target_net_state_dict)
