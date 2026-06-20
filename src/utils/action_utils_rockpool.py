#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for handling actions in the RL-based DBS parameter optimization.
"""

import torch
import random
import math

def get_action_dict(target_net, spk, mem, optimizing=False):
    """
    Convert network output to action dictionary.
    
    Args:
        target_net (nn.Sequential): Target network (Sequential model)
        spk (Tensor): Spike tensor
        mem (Tensor): Membrane potential tensor
        optimizing (bool): Whether called during optimization
        
    Returns:
        list: List of action dictionaries
    """
    action_batch = [] 
    # Access the RockpoolDQSN model inside the Sequential container if applicable
    if isinstance(target_net, torch.nn.Sequential):
        rockpool_model = target_net[1]
    else:
        rockpool_model = target_net

    if rockpool_model.use_mempot:
        # For the refactored model, `mem` is the full membrane potential history.
        # It has shape [time_steps, batch_size, num_actions], so we sum over time first.
        mem_summed = mem.sum(0)  # Sum over time steps -> [batch_size, num_actions]
        for mem_sum in mem_summed:
            freq_act = torch.argmax(mem_sum[0:3], dim=0).item() - 1 
            pw_act = torch.argmax(mem_sum[3:6], dim=0).item() - 1 
            amp_act = torch.argmax(mem_sum[6:9], dim=0).item() - 1
            action_batch.append({
                'freq': freq_act,
                'pw': pw_act,
                'amp': amp_act})
    else: 
        # The spike output `spk` from the refactored model is the full history.
        spk_sum_batch = spk.sum(0) # Sum over time steps -> [batch_size, num_actions]
        for spk_sum in spk_sum_batch:
            freq_act = torch.argmax(spk_sum[0:3], dim=0).item() - 1 
            pw_act = torch.argmax(spk_sum[3:6], dim=0).item() - 1 
            amp_act = torch.argmax(spk_sum[6:9], dim=0).item() - 1 
            action_batch.append({
                    'freq': freq_act,
                    'pw': pw_act,
                    'amp': amp_act})
    return action_batch

def select_action(policy_net, target_net, state, env, steps_done, device, eps_start, eps_end, eps_decay):
    """
    Select an action using epsilon-greedy policy.
    
    Args:
        policy_net (nn.Sequential): Policy network (Sequential model)
        target_net (nn.Sequential): Target network (Sequential model)
        state (Tensor): Current state
        env (MousePDEnv): Environment
        steps_done (int): Number of steps done so far
        device (torch.device): Device to run computations on
        eps_start (float): Starting value of epsilon
        eps_end (float): Final value of epsilon
        eps_decay (float): Rate of epsilon decay
        
    Returns:
        tuple: (selected action, spike counts)
    """
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    
    if sample > eps_threshold:
        with torch.no_grad():
            # policy_net is a sequential model. The RockpoolDQSN is the second element.
            spk, mem, sparsity_penalty, current_sparsity, _ = policy_net(state)
            
            if isinstance(policy_net, torch.nn.Sequential):
                spike_count = policy_net[1].get_spike_count()
            else:
                spike_count = policy_net.get_spike_count()
                
            return get_action_dict(target_net, spk, mem)[0], spike_count
    else:
        return env.action_space.sample(), [0]

def flatten_action(actions, device):
    """
    Flatten action dictionaries to tensor.
    
    Args:
        actions (list): List of action dictionaries
        device (torch.device): Device to run computations on
        
    Returns:
        Tensor: Flattened action indices
    """
    action_indices = []
    for action_dict in actions:
        # Transform action values to indices
        action_values = [
            action_dict['freq'] + 1, 
            action_dict['pw'] + 1 + 3, 
            action_dict['amp'] + 1 + 6
        ]
        action_indices.append(action_values)
    action_batch = torch.tensor(action_indices, dtype=torch.long).to(device)
    return action_batch
