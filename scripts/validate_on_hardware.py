import numpy as np
import json
import argparse
from tqdm import tqdm
import sys
import os
import samna
import time

# Rockpool hardware imports
from rockpool.devices.xylo import find_xylo_hdks
import rockpool.devices.xylo.syns65302 as xa3

def run_hardware_validation(model_package_dir, dt):
    """
    Loads a hardware spec and manually builds a SLICED, TIGHT configuration
    to exactly match the logic of the working incremental_test.py script.
    """
    print(f"\n{'='*20} Validating: {os.path.basename(model_package_dir)} {'='*20}")
    
    # --- 1. Load deployment files ---
    config_path = os.path.join(model_package_dir, 'hw_config.json')
    spikes_path = os.path.join(model_package_dir, 'sample_spikes.npy')
    labels_path = os.path.join(model_package_dir, 'sample_labels.npy')
    
    for p in [config_path, spikes_path, labels_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    with open(config_path, 'r') as f:
        spec = {k: np.array(v) if isinstance(v, list) else v for k, v in json.load(f).items()}
    print("Loaded hardware specification from JSON.")

    sample_spikes = np.load(spikes_path)
    sample_labels = np.load(labels_path)
    num_samples = sample_spikes.shape[0]

    # --- 2. Connect to Hardware ---
    print("Searching for Xylo hardware...")
    hdk_nodes, _, versions = find_xylo_hdks()
    if not hdk_nodes: raise RuntimeError("No Xylo HDK found.")
    hdk = hdk_nodes[0]
    print(f"Successfully connected to Xylo HDK (Version: {versions[0]})")

    # --- 3. Manually Build Sliced Hardware Configuration ---
    print("Manually building sliced hardware configuration...")
    config = xa3.XyloConfiguration()
    Nin = 16
    Nhidden = 128 # Use the actual model size
    Nout = 10

    # --- Weights (SLICE all matrices to match Nhidden) ---
    config.input.weights = spec['weights_in'][:, :Nhidden, :]
    config.hidden.weights = spec['weights_rec'][:Nhidden, :Nhidden, :]
    config.readout.weights = spec['weights_out'][:Nhidden, :]

    # --- Neurons (Create exactly Nhidden neurons with TUNED thresholds) ---
    threshold_scaling_factor_hidden = 1 # HYPERPARAMETER: Tune this value (e.g., 1.1, 1.5, 2.0)
    print(f"--- Applying HIDDEN layer threshold scaling factor of: {threshold_scaling_factor_hidden} ---")
    
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
    
    
    config.readout.neurons = [samna.xyloAudio3.configuration.OutputNeuron() for _ in range(Nout)]

    is_valid, msg = samna.xyloAudio3.validate_configuration(config)
    if not is_valid: raise ValueError(f"Manually built config is invalid: {msg}")
    print("Manually built sliced configuration is valid.")

    # --- 4. Deploy and Run ---
    print("Deploying configuration to the Xylo SNN core...")
    modSamna = None
    try:
        modSamna = xa3.XyloSamna(hdk, config, dt=dt)
        print(modSamna)
        print("Pausing for 2 seconds after configuration...")
        time.sleep(2.0)

        correct_predictions = 0
        print("Starting inference on the hardware...")
        for i in tqdm(range(num_samples), desc="Hardware Inference"):
            single_spike_data = sample_spikes[i]
            single_label = sample_labels[i]


            output_spike_raster, _, record_dict = modSamna(single_spike_data, record_power = True)
            spike_counts = output_spike_raster.sum(axis=0)

            print(record_dict)

            if spike_counts.sum() > 0:
                prediction = np.argmax(spike_counts)
            else:
                prediction = 0

            if prediction == single_label:
                correct_predictions += 1

        accuracy = 100 * correct_predictions / num_samples
        print(f"\n--- Hardware Validation Complete ---")
        print(f"  -> FINAL ACCURACY ON HARDWARE: {accuracy:.2f} %")
        print("------------------------------------")

    finally:
        if modSamna is not None:
            del modSamna

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a packaged SNN model on physical Xylo hardware.')
    parser.add_argument('model_package_dirs', nargs='+', type=str, 
                        help='One or more paths to the deployment package directories.')
    
    args = parser.parse_args()
    
    simulation_dt = 10e-3
    
    for model_dir in args.model_package_dirs:
        if os.path.isdir(model_dir):
            run_hardware_validation(model_dir, simulation_dt)
        else:
            print(f"Warning: Directory not found, skipping: {model_dir}")