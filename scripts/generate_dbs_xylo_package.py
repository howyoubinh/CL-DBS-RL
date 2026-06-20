import torch
import os
import argparse
import numpy as np
import json
from tqdm import tqdm

# Import DBS components
from src.models.rockpool_dqsn import RockpoolDQSN
from src.environment.gym_pd import MousePDEnvXylo
from rockpool.devices.xylo.syns65302 import mapper
from rockpool.transform import quantize_methods as q

def generate_dbs_package(model_path, base_output_dir):
    # --- Create a unique subdirectory for this model ---
    config_name = os.path.basename(model_path).replace('.pth', '')
    output_dir = os.path.join(base_output_dir, config_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing model: {model_path}")
    print(f"Output directory: {output_dir}")

    # --- Parameters (must match the trained DBS model) ---
    # Parameters below match the trained DBS model (see train_rl_rockpool_16ch.py).
    GROUP_SIZE = 5
    N_INPUT_CHANNELS = 80 // GROUP_SIZE # 16
    num_hidden = 128
    n_actions = 9
    num_steps = 100
    BATCH_SIZE = 1
    dt = 1e-5
    
    # --- Load the trained DBS model ---
    device = torch.device("cpu")
    # Initialize with same params as training
    model = RockpoolDQSN(N_INPUT_CHANNELS, num_hidden, beta=0.95, n_actions=n_actions, 
                         num_steps=num_steps, batch_size=BATCH_SIZE, use_mempot=True, dt=dt).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try loading with strict=False if keys don't match exactly
        print("Retrying with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    print("Successfully loaded trained PyTorch DBS model.")

    # --- Perform mapping and quantization ---
    print("Mapping SNN backend to hardware specification...")
    # Access the internal Sequential model which has as_graph()
    graph = model.model.as_graph() 
    spec = mapper(graph, weight_dtype="float")
    spec['dt'] = dt

    # --- Quantize the specification ---
    print("Quantizing parameters using channel-wise method...")
    quant_spec = q.channel_quantize(**spec)

    # Add back necessary parameters
    for key in ['dt', 'aliases']:
        if key in spec:
            quant_spec[key] = spec[key]

    # Enforce hardware data types
    for key in ["dash_mem", "dash_mem_out", "dash_syn", "dash_syn_2", "dash_syn_out"]:
        if key in quant_spec:
            quant_spec[key] = np.abs(quant_spec[key]).astype(np.uint8)

    # --- Save the hardware specification ---
    config_path = os.path.join(output_dir, 'hw_config.json')
    serializable_spec = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in quant_spec.items()}
    with open(config_path, 'w') as f:
        json.dump(serializable_spec, f, indent=4)
    print(f"SUCCESS: Quantized hardware specification saved to: {config_path}")

    # --- Prepare and save sample test data from Environment ---
    print("Generating sample spike data from MousePDEnvXylo...")
    
    # Initialize Env
    leap = {'pw': 0.1, 'amp': 5.0, 'freq': 5.0}
    env = MousePDEnvXylo(leap=leap, num_steps=num_steps, tau_beta_max=150., tau_reward=3000., delta=0.01, max_steps=25, TMAX=100, thresh_time_req=5)
    
    # Collect a few samples
    num_samples = 10
    sample_spikes_list = []
    sample_labels_list = []  # RL has no class labels; store the model's own prediction as the
                             # "label" so validate_on_hardware.py has a labels array to check against.

    for _ in range(num_samples):
        dummy_obs, info = env.reset()
        raw_spike_matrix = info['raw_spike_data'] # [Time, 80]
        
        # Process to 16 channels
        # Logic from train_rl_rockpool_16ch.py
        num_timesteps, num_original_neurons = raw_spike_matrix.shape
        num_target_channels = num_original_neurons // GROUP_SIZE
        summed_spikes = raw_spike_matrix.reshape(num_timesteps, num_target_channels, GROUP_SIZE).sum(axis=2)
        # Ensure binary spikes for Xylo (it handles counts, but usually we feed binary events or counts per bin)
        # Xylo input is typically [Time, Channels]
        
        sample_spikes_list.append(summed_spikes.astype(np.float32))
        
        # Get model prediction for this state to use as "label"
        with torch.no_grad():
            input_tensor = torch.from_numpy(summed_spikes.astype(np.float32)).unsqueeze(1).to(device) # [Time, Batch, Channels]
            spk, mem, _, _, _ = model(input_tensor)
            # Get action
            # mem is [Time, Batch, Actions]
            # Sum over time
            q_values = mem.sum(0)
            # We can just store the argmax of the first parameter (freq) as a simple label
            # Or store the full action index if we flatten the action space.
            # For validate_on_hardware.py which expects a single integer class, let's just use the Freq action (0,1,2)
            freq_act = torch.argmax(q_values[0, 0:3]).item()
            sample_labels_list.append(freq_act)

    sample_spikes_np = np.array(sample_spikes_list) # [N, Time, Channels]
    sample_labels_np = np.array(sample_labels_list)

    data_path = os.path.join(output_dir, 'sample_spikes.npy')
    labels_path = os.path.join(output_dir, 'sample_labels.npy')
    
    np.save(data_path, sample_spikes_np)
    np.save(labels_path, sample_labels_np)
    print(f"SUCCESS: Sample spike data and labels saved in '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert trained DBS models to Xylo hardware deployment packages.')
    parser.add_argument('model_paths', nargs='+', type=str, help='A list of paths to the trained .pth model files.')
    args = parser.parse_args()

    base_output_dir = "hardware_deployment"

    for model_path in args.model_paths:
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Skipping.")
            continue
        generate_dbs_package(model_path, base_output_dir)
        
    print(f"\nAll packages are ready in the '{base_output_dir}' directory.")
