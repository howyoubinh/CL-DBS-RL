#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for saving and loading model checkpoints and results.
"""

import os
import torch

def save_results(filename, results):
    """
    Save training results to file.
    
    Args:
        filename (str): Output filename
        results (dict): Results dictionary
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    torch.save(results, filename)
    print(f'Saved results to: {filename}')

def save_checkpoint(model, optimizer, episode, loss, steps_done, save_dir):
    """
    Save a checkpoint of the training state.
    
    Args:
        model (DQSN): Model to save
        optimizer (Optimizer): Optimizer state to save
        episode (int): Current episode number
        loss (float): Current loss value
        steps_done (int): Total steps completed
        save_dir (str): Directory to save checkpoint in
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'episode': episode,
        'steps_done': steps_done,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    filename = f'checkpoint_episode_{episode}.pth'
    torch.save(checkpoint, os.path.join(save_dir, filename))

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load a checkpoint to resume training.
    
    Args:
        model (DQSN): Model to load weights into
        optimizer (Optimizer): Optimizer to load state into
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        tuple: episode, steps_done
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    loss = checkpoint['loss']
    # Get steps_done, default to 0 for backward compatibility with old checkpoints
    steps_done = checkpoint.get('steps_done', 0)
    
    print(f"Loaded checkpoint from episode {episode} with loss {loss}")
    return episode, steps_done

def save_config(config, filename):
    """
    Save the training configuration dictionary to a file.
    
    Args:
        config (dict): Configuration dictionary
        filename (str): Output filename
    """
    torch.save(config, filename)
    print(f'Configuration saved to: {filename}')