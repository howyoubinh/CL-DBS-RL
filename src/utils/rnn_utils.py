import torch
import torch.nn as nn
import random
import math
from src.environment.replay_memory import Transition

def select_action_rnn(policy_net, state, env, steps_done, device, eps_start, eps_end, eps_decay):
    """
    Select action for RNN using epsilon-greedy policy.
    """
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    
    if sample > eps_threshold:
        with torch.no_grad():
            # policy_net returns q_values [Batch, Actions]
            q_values = policy_net(state)
            
            # Get discrete actions from the 9 outputs
            # Output structure: [Freq(3), PW(3), Amp(3)]
            # We need to argmax each group
            
            # If batch size is 1 (inference)
            q_values = q_values[0] # [9]
            
            freq_act = torch.argmax(q_values[0:3]).item() - 1
            pw_act = torch.argmax(q_values[3:6]).item() - 1
            amp_act = torch.argmax(q_values[6:9]).item() - 1
            
            action = {
                'freq': freq_act,
                'pw': pw_act,
                'amp': amp_act
            }
            return action
    else:
        return env.action_space.sample()

def flatten_action_rnn(actions, device):
    """
    Flatten action dictionaries to tensor for RNN optimization.
    """
    action_indices = []
    for action_dict in actions:
        action_values = [
            action_dict['freq'] + 1, 
            action_dict['pw'] + 1 + 3, 
            action_dict['amp'] + 1 + 6
        ]
        action_indices.append(action_values)
    return torch.tensor(action_indices, dtype=torch.long).to(device)

def optimize_model_rnn(policy_net, target_net, optimizer, memory, device, batch_size, gamma):
    """
    Perform one step of optimization on the RNN policy network.
    """
    if len(memory) < batch_size:
        return None
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Compute mask of non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                 device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)
    
    # Forward pass
    # state_batch: [Batch, Time, Channels]
    state_action_values_full = policy_net(state_batch) # [Batch, 9]
    
    # Prepare actions
    action_indices_flat = flatten_action_rnn(batch.action, device) # [Batch, 3]
    
    # Gather Q-values for the taken actions
    # state_action_values_full is [Batch, 9]
    # action_indices_flat is [Batch, 3] (indices 0-8)
    # We need to gather the specific Q-values for the 3 sub-actions
    state_action_values = state_action_values_full.gather(1, action_indices_flat) # [Batch, 3]

    # Compute next state values
    next_state_values = torch.zeros(batch_size, 3, device=device)
    if len(non_final_next_states) > 0:
        with torch.no_grad():
            next_q_full = target_net(non_final_next_states) # [NonFinalBatch, 9]
            
            # Standard DQN: max over next-state actions
            # But we have 3 independent action heads (effectively)
            # Max for Freq (0-2), PW (3-5), Amp (6-8)
            
            max_freq = next_q_full[:, 0:3].max(1)[0]
            max_pw = next_q_full[:, 3:6].max(1)[0]
            max_amp = next_q_full[:, 6:9].max(1)[0]
            
            next_state_values[non_final_mask, 0] = max_freq
            next_state_values[non_final_mask, 1] = max_pw
            next_state_values[non_final_mask, 2] = max_amp

    # Compute expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch.unsqueeze(1)

    # Compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()
