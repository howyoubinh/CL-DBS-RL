import os
import sys
import argparse # Added for command-line parsing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import wandb # Added for logging


from datetime import datetime
from tqdm import tqdm
from itertools import count

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Adjusted imports for CL-DBS-RL project structure
from src.environment.gym_pd import MousePDEnvXylo, MousePDEnvAdaptive
from src.models.rockpool_dqsn import RockpoolDQSN

###### Setup hyperparmeter variables ######

num_epochs = 100 # Number of distillation epochs per student

class AvgPoolDownsampler(nn.Module):
    def __init__(self, kernel_size=5, stride=5):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        # x shape: (B, T, C_in=80)
        B, T, C = x.shape
        # Reshape to (B*T, 1, C) to pool over C
        # AvgPool1d expects (N, C, L). We want to pool over the last dimension (80).
        # So we treat 80 as L.
        x_reshaped = x.reshape(B * T, 1, C) 
        # Output: (B*T, 1, 16)
        out = self.pool(x_reshaped)
        # Reshape back to (B, T, 16)
        return out.reshape(B, T, -1)

# Initialize environment (using CL-DBS-RL's environment)
leap = {'pw': 0.1, 'amp': 5.0, 'freq': 5.0}
env = MousePDEnvAdaptive(leap=leap, num_steps=100, tau_beta_max=150., tau_reward=3000., delta=0.5, max_steps=100, TMAX=100, thresh_time_req=999)

# Get a sample state to determine the shapes
dummy_obs, info = env.reset()
sample_spikes = info['raw_spike_data']

# Constants
GROUP_SIZE = 5
N_RAW_CHANNELS = 80
N_INPUT_CHANNELS = N_RAW_CHANNELS // GROUP_SIZE # 16
num_steps = sample_spikes.shape[0]

num_hidden = 128 
n_actions = 9 
BETA = 0.95 
BATCH_SIZE = 1 
dt = 10e-3 # Match training script

# --- Define Helper Functions ---

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def get_action_dict(spk, mem, model, optimizing=False):
    """ Extracts discrete actions from network output (spikes or membrane potential). """
    action_batch = []
    # Use membrane potential sum if use_mempot is True
    use_mempot = False
    if hasattr(model, 'use_mempot'):
        use_mempot = model.use_mempot
    elif isinstance(model, nn.Sequential):
        # Assuming RockpoolDQSN is the last layer or we search for it
        for module in model:
            if hasattr(module, 'use_mempot'):
                use_mempot = module.use_mempot
                break
    
    if use_mempot:
        # Sum over time steps -> [batch_size, num_actions]
        mem_sum_batch = mem.sum(0)
        for mem_sum in mem_sum_batch: # Iterate through batch
            # Argmax over the 3 actions for each parameter
            freq_act = torch.argmax(mem_sum[0:3], dim=0).item() - 1
            pw_act = torch.argmax(mem_sum[3:6], dim=0).item() - 1
            amp_act = torch.argmax(mem_sum[6:9], dim=0).item() - 1

            action_batch.append({
                'freq': freq_act,
                'pw': pw_act,
                'amp': amp_act})
    else: # Use spike counts otherwise
        # The spike output `spk` from the refactored model is the full history.
        # Sum over time steps -> [batch_size, num_actions]
        spk_sum_batch = spk.sum(0)

        for spk_sum in spk_sum_batch: # Iterate through batch
            freq_act = torch.argmax(spk_sum[0:3], dim=0).item() - 1
            pw_act = torch.argmax(spk_sum[3:6], dim=0).item() - 1
            amp_act = torch.argmax(spk_sum[6:9], dim=0).item() - 1
            action_batch.append({
                'freq': freq_act,
                'pw': pw_act,
                'amp': amp_act})
    return action_batch

def select_action(state: torch.tensor, model):
    """ Selects an action based on the model's output for a given state. """
    model.eval() # Ensure model is in eval mode for inference
    with torch.no_grad():
        # Add batch dimension if state is single sample
        if state.dim() == 2:
            # Input state is (T, 80). We need (B, T, 80) for AvgPoolDownsampler
            state = state.unsqueeze(0)

        # Ensure state is on the correct device
        state = state.to(device)
        # Forward pass
        spk, mem, _, _, _ = model(state)

        # Get action dictionary (take the first action if batch size is 1)
        action_dict = get_action_dict(spk, mem, model)
        # Handle case where model output might be on GPU
        if isinstance(action_dict, list) and len(action_dict) > 0:
             return action_dict[0]
        elif isinstance(action_dict, dict): # If batch size was 1, it might return dict directly
             return action_dict
        else:
             # Fallback or error handling if action_dict is unexpected
             print(f"Warning: Unexpected action_dict format: {action_dict}")
             # Provide a default action or raise an error
             return {'freq': 0, 'pw': 0, 'amp': 0} # Example default

def test_model(model, env, num_episodes=10, model_params=None):
    """ Evaluates the model's performance in the environment. """
    model.eval() # Set model to evaluation mode

    # Testing Variables (oae = over an episode)
    oae_alpha_beta = []
    oae_amp = []
    oae_freq = []
    oae_pw = []
    oae_e = []
    oae_steps = []

    print(f'\n--- Testing Model ---')
    print(f'Parameters: {model_params}')
    print(f'Episodes: {num_episodes}, Max Steps/Episode: {env.max_steps}')

    for i_episode in tqdm(range(num_episodes), desc="Testing Episodes"):
        
        dummy_obs, info = env.reset()
        raw_spike_matrix = info['raw_spike_data']
        # State: (T, 80) -> (1, T, 80)
        state = torch.from_numpy(raw_spike_matrix.astype(np.float32)).to(device).unsqueeze(0)

        env.t = 0 # Reset internal step counter for truncation

        # Test tracking lists for current episode
        alpha_beta_hist = [env.gpi_alpha_beta_area]
        amp_hist = [env.amp]
        freq_hist = [env.freq]
        pw_hist = [env.pw]
        e_hist = [env.E]

        for t in count():
            env.t += 1 # Increment step counter for truncation check

            action = select_action(state, model)

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_raw_spike_matrix = info['raw_spike_data']

            # Track metrics
            alpha_beta_hist.append(env.gpi_alpha_beta_area)
            amp_hist.append(env.amp)
            freq_hist.append(env.freq)
            pw_hist.append(env.pw)
            e_hist.append(env.E)

            if terminated:
                next_state = None
            else:
                next_state = torch.from_numpy(next_raw_spike_matrix.astype(np.float32)).to(device).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:
                oae_alpha_beta.append(alpha_beta_hist)
                oae_amp.append(amp_hist)
                oae_freq.append(freq_hist)
                oae_pw.append(pw_hist)
                oae_e.append(e_hist)
                oae_steps.append(t + 1) # Record number of steps taken
                
                # --- WandB Logging for Test Episode ---
                wandb.log({
                    "test/episode_length": t + 1,
                    "test/mean_alpha_beta": np.mean(alpha_beta_hist),
                    "test/mean_amp": np.mean(amp_hist),
                    "test/mean_freq": np.mean(freq_hist),
                    "test/mean_pw": np.mean(pw_hist),
                    "test/mean_energy": np.mean(e_hist),
                    "test/episode": i_episode
                })
                
                break

    results = {
        'oae_alpha_beta': oae_alpha_beta,
        'oae_amp': oae_amp,
        'oae_freq': oae_freq,
        'oae_pw': oae_pw,
        'oae_e': oae_e,
        'oae_steps': oae_steps,
        'model_params': model_params # Store params with results
    }

    print("Test completed.")

    # --- Saving Test Results ---
    results_dir = os.path.join('data', 'results', 'distillation')
    os.makedirs(results_dir, exist_ok=True) # Ensure directory exists

    date_now = datetime.now().strftime("%m-%d-%Y")
    time_now = datetime.now().strftime("%H-%M-%S")
    if model_params:
        # Use descriptive filename based on parameters
        results_filename = f"test_s{model_params['target_sparsity']}_t{model_params['temperature']}_{date_now}_{time_now}.pth"
    else:
        # Fallback to timestamp if parameters not provided
        results_filename = f'test_results_{date_now}_{time_now}.pth'

    save_results_path = os.path.join(results_dir, results_filename)
    torch.save(results, save_results_path)
    print(f"Test data saved to {save_results_path}")

    return results

def distillation_loss(student_outputs, teacher_outputs, temperature, student_sparsity_penalty, debug=False):
    """
    Calculates the combined distillation and sparsity loss.

    Args:
        student_outputs (Tensor): Membrane potential history from student [time, batch, actions]
        teacher_outputs (Tensor): Membrane potential history from teacher [time, batch, actions]
        temperature (float): Temperature for softening probabilities.
        student_sparsity_penalty (Tensor): Sparsity penalty from student forward pass.
        debug (bool): Whether to print debugging information

    Returns:
        (Tensor, Tensor): A tuple containing the total_loss and the distill_loss component.
    """
    # Sum membrane potential over time steps to get Q-value estimates
    student_q_values = student_outputs.sum(0)
    teacher_q_values = teacher_outputs.sum(0)

    # Debug Q-value statistics before processing
    if debug:
        print(f"    DEBUG - Raw Q-values Student: mean={student_q_values.mean().item():.3f}, std={student_q_values.std().item():.3f}")
        print(f"    DEBUG - Raw Q-values Teacher: mean={teacher_q_values.mean().item():.3f}, std={teacher_q_values.std().item():.3f}")

    # Soften probabilities using temperature
    # Use log_softmax for numerical stability with KLDivLoss
    teacher_scaled = teacher_q_values / temperature
    student_scaled = student_q_values / temperature
    
    soft_targets = F.softmax(teacher_scaled, dim=1)
    soft_prob = F.log_softmax(student_scaled, dim=1)

    # Debug softmax statistics
    if debug:
        print(f"    DEBUG - Soft targets: mean={soft_targets.mean().item():.6f}, max={soft_targets.max().item():.6f}")
        print(f"    DEBUG - Soft prob: mean={soft_prob.mean().item():.6f}, min={soft_prob.min().item():.6f}")

    # Compute KL divergence loss
    # kl_div expects input in log-space, target in probability-space
    # reduction='batchmean' averages over the batch dimension
    kl_raw = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
    distill_loss = kl_raw * (temperature ** 2)
    
    if debug:
        print(f"    DEBUG - KL raw: {kl_raw.item():.6f}, Temp^2: {temperature**2}, Distill loss: {distill_loss.item():.6f}")
        print(f"    DEBUG - Sparsity penalty: {student_sparsity_penalty.item():.6f}")

    # Add sparsity regularization (already weighted in the model's forward pass)
    total_loss = distill_loss + student_sparsity_penalty
    return total_loss, distill_loss

def train_distilled_student(teacher_net, env, target_sparsity, temperature, num_epochs=100, sparsity_weight=20000, learning_rate=0.001, model_dir='models/distilled_rockpool_16ch'):
    """ Trains a single student network using knowledge distillation. """

    print(f'\n--- Training Student ---')
    print(f"Target Sparsity: {target_sparsity}, Temperature: {temperature}, Sparsity Weight: {sparsity_weight}, Epochs: {num_epochs}")

    # Initialize student network
    downsampler_student = AvgPoolDownsampler(kernel_size=5, stride=5)
    rockpool_student = RockpoolDQSN(N_INPUT_CHANNELS, num_hidden, BETA, n_actions,
                     num_steps, BATCH_SIZE, use_mempot=True,
                     target_sparsity=target_sparsity,
                     sparsity_weight=sparsity_weight, dt=dt)
    student_net = nn.Sequential(downsampler_student, rockpool_student).to(device)

    optimizer = optim.Adam(student_net.parameters(), lr=learning_rate)
    
    # Initial baseline measurements for debugging
    print(f"Initial baseline measurements:")
    dummy_obs, info = env.reset()
    raw_spike_matrix = info['raw_spike_data']
    initial_state = torch.from_numpy(raw_spike_matrix.astype(np.float32)).to(device).unsqueeze(0)
    
    with torch.no_grad():
        _, teacher_mem_init, _, teacher_rate_init, _ = teacher_net(initial_state)
        _, student_mem_init, sparsity_init, student_rate_init, _ = student_net(initial_state)
        
        print(f"  Teacher initial rate: {teacher_rate_init.item():.6f}")
        print(f"  Student initial rate: {student_rate_init.item():.6f}")
        print(f"  Initial sparsity penalty: {sparsity_init.item():.6f}")

    # --- Loss History Tracking ---
    loss_history = {
        'total': [],
        'distill': [],
        'sparsity': []
    }

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Distillation Epochs"):
        student_net.train()
        optimizer.zero_grad()

        dummy_obs, info = env.reset()
        raw_spike_matrix = info['raw_spike_data']
        state = torch.from_numpy(raw_spike_matrix.astype(np.float32)).to(device).unsqueeze(0)

        with torch.no_grad():
            teacher_spk, teacher_mem, _, _, _ = teacher_net(state)

        student_spk, student_mem, sparsity_penalty, avg_rate, _ = student_net(state)

        # Implement loss scheduling - gradually introduce sparsity over first 50% of training
        sparsity_schedule = min(1.0, 2.0 * epoch / num_epochs)  # 0 to 1 over first half of training
        scheduled_sparsity_penalty = sparsity_penalty * sparsity_schedule

        # Enable debug mode if losses are getting large
        debug_mode = (epoch + 1) % 20 == 0 or (epoch > 0 and 'prev_loss' in locals() and prev_loss > 100)
        total_loss, distill_loss = distillation_loss(student_mem, teacher_mem, temperature, scheduled_sparsity_penalty, debug=debug_mode)
        prev_loss = total_loss.item()

        total_loss.backward()
        # Add gradient clipping to prevent oscillation between loss objectives
        torch.nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=1.0)
        optimizer.step()

         # record losses
        total_loss_item = total_loss.item()
        distill_loss_item = distill_loss.item()
        sparsity_penalty_item = sparsity_penalty.item()
        
        loss_history['distill'].append(distill_loss_item)
        loss_history['sparsity'].append(sparsity_penalty_item)
        
        # Log to WandB
        wandb.log({
            "train/total_loss": total_loss_item,
            "train/distill_loss": distill_loss_item,
            "train/sparsity_penalty": sparsity_penalty_item,
            "train/avg_firing_rate": avg_rate.item(),
            "train/epoch": epoch
        })

        # Enhanced debugging output every 10 epochs
        if (epoch + 1) % 10 == 0 or total_loss_item > 1000 or distill_loss_item > 1000:
            # Get Q-value statistics for debugging
            with torch.no_grad():
                student_q_sum = student_mem.sum(0).detach()
                teacher_q_sum = teacher_mem.sum(0).detach()
                
                student_q_min, student_q_max = student_q_sum.min().item(), student_q_sum.max().item()
                teacher_q_min, teacher_q_max = teacher_q_sum.min().item(), teacher_q_sum.max().item()
                
                # Calculate individual layer firing rates for detailed analysis
                _, _, _, layer_rates, rec = student_net(state)
                spk1 = rec['1_LIFTorch']['spikes']
                spk2 = rec['3_LIFTorch']['spikes'] 
                spk3 = rec['5_LIFTorch']['spikes']
                
                rate1 = (spk1.mean() / num_steps).item()
                rate2 = (spk2.mean() / num_steps).item() 
                rate3 = (spk3.mean() / num_steps).item()
            
            print(f"  Epoch [{epoch+1}/{num_epochs}]:")
            print(f"    Losses - Total: {total_loss_item:.4f}, Distill: {distill_loss_item:.4f}, Sparsity: {sparsity_penalty_item:.4f}")
            print(f"    Firing Rates - Avg: {avg_rate.item():.6f}, L1: {rate1:.6f}, L2: {rate2:.6f}, L3: {rate3:.6f}")
            print(f"    Q-values Student: [{student_q_min:.3f}, {student_q_max:.3f}], Teacher: [{teacher_q_min:.3f}, {teacher_q_max:.3f}]")

    final_avg_rate = avg_rate.cpu().item()
    print(f"Training finished. Final Loss: {total_loss.item():.4f}, Final Avg Rate: {final_avg_rate:.4f}")

    os.makedirs(model_dir, exist_ok=True)

    date_now = datetime.now().strftime("%m-%d-%Y")
    time_now = datetime.now().strftime("%H-%M-%S")
    # Include sparsity_weight in filename for easier identification
    model_filename = f'student_s{target_sparsity}_t{temperature}_w{sparsity_weight}_{date_now}_{time_now}.pth'
    save_model_path = os.path.join(model_dir, model_filename)
    torch.save(student_net.state_dict(), save_model_path)
    print(f"Student model saved to {save_model_path}")

    print(f"Student training finished. Final Loss: {total_loss.item():.4f}, Final Avg Rate: {avg_rate.item():.4f}")

    return student_net, final_avg_rate, loss_history, save_model_path

def calibrate_sparsity_weight_notrain(teacher_net, env, num_calibration_steps=50):
    """
    Runs a short calibration process on an UNTRAINED network to determine a good starting sparsity_weight.
    """
    print("\n===== Starting Sparsity Weight Calibration ====")
    print(f"Measuring losses over {num_calibration_steps} initial states...")

    # Use a fixed set of parameters for calibration
    calibration_params = {
        'target_sparsity': 0.05, # A representative target
        'temperature': 4.0,
        'sparsity_weight': 1.0, # This is just a placeholder now
        'learning_rate': 0.0005
    }

    # Initialize a temporary student network for EACH step to get a better average
    # on randomly initialized weights.
    
    distill_losses = []
    unweighted_sparsity_losses = []

    for step in tqdm(range(num_calibration_steps), desc="Calibration Steps"):
        # Initialize a new, untrained student network for each measurement
        downsampler_student = AvgPoolDownsampler(kernel_size=5, stride=5)
        rockpool_student = RockpoolDQSN(N_INPUT_CHANNELS, num_hidden, BETA, n_actions,
                                   num_steps, BATCH_SIZE, use_mempot=True,
                                   target_sparsity=calibration_params['target_sparsity'],
                                   sparsity_weight=calibration_params['sparsity_weight'], # Weight is 1.0
                                   dt=dt)
        student_net = nn.Sequential(downsampler_student, rockpool_student).to(device)
        
        # We do NOT use an optimizer and we do NOT train the network.
        student_net.eval() # Set to evaluation mode

        dummy_obs, info = env.reset()
        raw_spike_matrix = info['raw_spike_data']
        state = torch.from_numpy(raw_spike_matrix.astype(np.float32)).to(device).unsqueeze(0)

        with torch.no_grad(): # Use no_grad for the entire process
            _, teacher_mem, _, _, _ = teacher_net(state)

            # In the student's forward pass, sparsity_penalty is already weighted by its internal sparsity_weight of 1.0.
            # This means it's effectively the unweighted sum of KL penalties.
            _, student_mem, unweighted_sparsity_loss, _, _ = student_net(state)
            
            _, distill_loss_val = distillation_loss(student_mem, teacher_mem, calibration_params['temperature'], unweighted_sparsity_loss)

        distill_losses.append(distill_loss_val.item())
        unweighted_sparsity_losses.append(unweighted_sparsity_loss.item())

    # --- Calculate Recommended Weight ---
    avg_distill_loss = np.mean(distill_losses)
    avg_unweighted_sparsity_loss = np.mean(unweighted_sparsity_losses)

    print(f"\nAverage Initial Distillation Loss: {avg_distill_loss:.6f}")
    print(f"Average Initial Unweighted Sparsity Loss: {avg_unweighted_sparsity_loss:.6f}")

    if avg_unweighted_sparsity_loss < 1e-9:
        recommended_weight = 0 
    else:
        # The goal is to find W such that: W * sparsity_loss ≈ distill_loss
        recommended_weight = avg_distill_loss / avg_unweighted_sparsity_loss
        print(f"\nRecommended Sparsity Weight: {recommended_weight:.2f}")

    print("===== Calibration Complete ====")
    return recommended_weight


# Remove default sparsity_weight from signature as it's now in param_grid
def evaluate_students(teacher_net, teacher_avg_rate, env, param_grid, num_distill_epochs=100, learning_rate=0.001, model_dir='models/distilled_rockpool_16ch'):
    """ Trains and evaluates multiple student networks across a parameter grid. """

    print("\n===== Starting Student Evaluation ====")
    all_results = {}

    total_configs = len(param_grid['target_sparsity']) * len(param_grid['temperatures']) * len(param_grid['sparsity_weights'])
    current_config_num = 0

    for sparsity in param_grid['target_sparsity']:
        for temp in param_grid['temperatures']:
            for weight in param_grid['sparsity_weights']: # Loop through sparsity weights
                current_config_num += 1
                print(f"\n--- Processing Configuration {current_config_num}/{total_configs} ---")

                # Define parameters for this run, including the current weight
                current_params = {
                    'target_sparsity': sparsity,
                    'temperature': temp,
                    'sparsity_weight': weight, # Use the current weight from the loop
                    'num_distill_epochs': num_distill_epochs,
                    'learning_rate': learning_rate
                }
                print(f"Params: {current_params}")

                # Initialize WandB run for this configuration
                run_name = f"distill_s{sparsity}_t{temp}_w{weight}"
                wandb.init(
                    project="cl-dbs-rl-rockpool-16ch-distill",
                    name=run_name,
                    config=current_params,
                    reinit=True
                )

                # Train student using the current weight
                _, student_avg_rate, loss_history, model_path = train_distilled_student(
                    teacher_net, env, sparsity, temp,
                    num_epochs=num_distill_epochs,
                    sparsity_weight=weight,
                    learning_rate=learning_rate,
                    model_dir=model_dir)

                key = f"s{sparsity}_t{temp}_w{weight}"
                all_results[key] = {
                    'teacher_avg_rate': teacher_avg_rate,
                    'student_avg_rate': student_avg_rate,
                    'loss_history': loss_history,
                    'model_path': model_path,
                    'model_params': current_params
                }

                wandb.log({"train/student_avg_rate": student_avg_rate})
                wandb.finish()

    # --- Save Combined Results ---
    results_dir = os.path.join('data', 'results', 'distillation')
    os.makedirs(results_dir, exist_ok=True)

    date_now = datetime.now().strftime("%m-%d-%Y")
    time_now = datetime.now().strftime("%H-%M-%S")
    save_combined_results_path = os.path.join(results_dir, f'evaluation_summary_rockpool_{date_now}_{time_now}.pth')
    torch.save(all_results, save_combined_results_path)
    print(f"\nSaved combined evaluation results to {save_combined_results_path}")
    
    # --- Create Models Index CSV for Experiment 2 ---
    index_data = []
    for key, result in all_results.items():
        params = result['model_params']
        index_data.append({
            'model_file': os.path.basename(result['model_path']),
            'target_sparsity': params['target_sparsity'],
            'temperature': params['temperature'],
            'sparsity_weight': params['sparsity_weight'],
            'student_avg_rate': result['student_avg_rate']
        })
    
    import pandas as pd
    new_df = pd.DataFrame(index_data)
    index_path = os.path.join(model_dir, 'models_index.csv')
    if os.path.exists(index_path):
        existing_df = pd.read_csv(index_path)
        new_df = pd.concat([existing_df, new_df], ignore_index=True)
    new_df.to_csv(index_path, index=False)
    print(f"Models index saved to {index_path}")
    
    print("===== Student Evaluation Complete =====")

    return all_results





def plot_activation_sparsity_comparison(results):
    """ Plots the activation sparsity (avg firing rate) comparison. """
    print("\n--- Plotting Activation Sparsity (Avg Firing Rate) Comparison ---")
    if not results:
        print("No results to plot.")
        return

    # Extract data for plotting
    labels = []
    student_avg_rates = []
    teacher_avg_rate = None

    # Sort results by target sparsity then temperature for consistent plotting
    # Sort results including sparsity weight
    sorted_keys = sorted(results.keys(), key=lambda k: (
        results[k]['model_params']['target_sparsity'],
        results[k]['model_params']['temperature'],
        results[k]['model_params']['sparsity_weight']
    ))

    for key in sorted_keys:
        result = results[key]
        params = result['model_params']
        # Include weight in label
        labels.append(f"S={params['target_sparsity']}, T={params['temperature']}, W={params['sparsity_weight']}")
        student_avg_rates.append(result['student_avg_rate'])
        if teacher_avg_rate is None:
            teacher_avg_rate = result.get('teacher_avg_rate', None)

    if teacher_avg_rate is None:
        print("Warning: Teacher average firing rate not found in results.")
        return
    if not student_avg_rates:
        print("Warning: No student average firing rates found in results.")
        return

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot teacher rate as a horizontal line for reference
    ax.axhline(teacher_avg_rate, color='dodgerblue', linestyle='--', linewidth=2, label=f'Teacher Avg Rate ({teacher_avg_rate:.4f})')
    rects_students = ax.bar(x, student_avg_rates, width, label='Student Avg Rate', color='lightgreen')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Average Firing Rate')
    ax.set_title('Teacher vs Student Average Firing Rate after Distillation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc='upper right')

    ax.bar_label(rects_students, padding=3, fmt='%.4f')

    fig.tight_layout()

    # Save the plot
    plot_dir = os.path.join('data', 'plots', 'distillation')
    os.makedirs(plot_dir, exist_ok=True)
    date_now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    plot_filename = f'activation_sparsity_comparison_rockpool_{date_now}.png'
    save_plot_path = os.path.join(plot_dir, plot_filename)
    plt.savefig(save_plot_path)
    print(f"Activation sparsity comparison plot saved to {save_plot_path}")


# ===== Main Execution =====


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate distilled student networks with optional sparsity weight calibration.")
    parser.add_argument('--teacher-checkpoint', type=str, required=True, help="Path to the trained teacher model checkpoint.")
    parser.add_argument('--model-dir', type=str, default='models/distilled_rockpool_16ch_new', help="Directory to save student model files and models_index.csv.")
    parser.add_argument('--calibrate-sparsity', action='store_true', help="Run a short calibration process to find a recommended sparsity_weight and exit.")
    parser.add_argument('--baseline-only', action='store_true', help="Train only a single baseline model with sparsity_weight=0 (no sparsity penalty).")
    parser.add_argument('--num-cal-steps', type=int, default=200, help="Number of steps for sparsity calibration.")
    parser.add_argument('--cal-target-sparsity', type=float, default=0.05, help="Target sparsity to use during calibration.")
    parser.add_argument('--cal-temp', type=float, default=4.0, help="Temperature to use during calibration.")

    args = parser.parse_args()

    # --- Load Teacher Model ---
    print(f"Loading teacher model from {args.teacher_checkpoint}")
    if not os.path.exists(args.teacher_checkpoint):
        raise FileNotFoundError(f"Teacher model not found at {args.teacher_checkpoint}")

    # Initialize teacher network structure
    downsampler_teacher = AvgPoolDownsampler(kernel_size=5, stride=5)
    rockpool_teacher = RockpoolDQSN(N_INPUT_CHANNELS, num_hidden, BETA, n_actions, num_steps, BATCH_SIZE, use_mempot=True, target_sparsity=1.0, sparsity_weight=0.0, dt=dt)
    teacher_net = nn.Sequential(downsampler_teacher, rockpool_teacher).to(device)
    
    # Load weights
    # Note: The checkpoint might be the full state_dict of the Sequential model
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    teacher_net.load_state_dict(checkpoint)
    teacher_net.eval()
    print("Teacher model loaded and set to eval mode.")

    # --- Main Execution Logic ---
    if args.calibrate_sparsity:
        calibrate_sparsity_weight_notrain(
            teacher_net=teacher_net,
            env=env,
            num_calibration_steps=args.num_cal_steps
            )
    elif args.baseline_only:
        # Train a single baseline student with sparsity_weight=0
        print("\n===== Training Baseline Model (sparsity_weight=0) =====")
        baseline_params = {
            'target_sparsity': 0.01,  # Doesn't matter when weight=0
            'temperature': 4.0,
            'sparsity_weight': 0,
            'num_distill_epochs': 300,
            'learning_rate': 0.0001
        }
        
        wandb.init(
            project="cl-dbs-rl-rockpool-16ch-distill",
            name="distill_baseline_w0",
            config=baseline_params,
            reinit=True
        )
        
        _, student_avg_rate, loss_history, model_path = train_distilled_student(
            teacher_net, env,
            target_sparsity=baseline_params['target_sparsity'],
            temperature=baseline_params['temperature'],
            num_epochs=baseline_params['num_distill_epochs'],
            sparsity_weight=baseline_params['sparsity_weight'],
            learning_rate=baseline_params['learning_rate'],
            model_dir=args.model_dir
        )

        wandb.finish()

        # Save baseline to index CSV
        index_path = os.path.join(args.model_dir, 'models_index.csv')
        
        import pandas as pd
        baseline_entry = {
            'model_file': os.path.basename(model_path),
            'target_sparsity': baseline_params['target_sparsity'],
            'temperature': baseline_params['temperature'],
            'sparsity_weight': baseline_params['sparsity_weight'],
            'student_avg_rate': student_avg_rate
        }
        
        # Append to existing index or create new one
        if os.path.exists(index_path):
            existing_df = pd.read_csv(index_path)
            updated_df = pd.concat([existing_df, pd.DataFrame([baseline_entry])], ignore_index=True)
        else:
            updated_df = pd.DataFrame([baseline_entry])
        updated_df.to_csv(index_path, index=False)
        print(f"Baseline model added to index: {index_path}")
        print("===== Baseline Training Complete =====")
    else:
        # Define the grid of hyperparameters for Pareto front
        # More sparsity weights = better Pareto coverage
        param_grid = {
            'target_sparsity': [0.01, 0.02, 0.05, 0.10],  # 4 sparsity targets
            'temperatures': [4.0],                         # Single temperature for simplicity
            'sparsity_weights': [500, 1000, 2000]          # 3 weights for Pareto coverage
        }
        # Total: 4 x 1 x 3 = 12 student models

        # Define other training parameters
        num_distill_epochs = 300 # Number of epochs for distilling each student
        # sparsity_weight is now part of the grid search
        learning_rate = 0.0001 # Reduced learning rate for more stable training

        # Calculate teacher average firing rate (using a sample state)
        teacher_net.eval()
        dummy_obs, info = env.reset() # Get a sample state
        raw_spike_matrix = info['raw_spike_data']
        sample_state_for_rate = torch.from_numpy(raw_spike_matrix.astype(np.float32)).to(device).unsqueeze(0)
        with torch.no_grad():
            _, _, _, teacher_avg_rate, _ = teacher_net(sample_state_for_rate)
        
        # Move teacher rate to CPU and get the Python number for plotting/logging
        teacher_avg_rate = teacher_avg_rate.cpu().item()
        print(f"Teacher Model Average Firing Rate: {teacher_avg_rate:.4f}")


        # Run the evaluation process
        results = evaluate_students(
            teacher_net,
            teacher_avg_rate,
            env,
            param_grid,
            num_distill_epochs=num_distill_epochs,
            learning_rate=learning_rate,
            model_dir=args.model_dir
        )

        # Plot the activation sparsity (avg firing rate) comparison
        plot_activation_sparsity_comparison(results)
