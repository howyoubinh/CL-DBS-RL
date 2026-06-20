#!/bin/bash

# This script creates a conda environment from a specified YAML file.

# --- Usage --- #
if [ -z "$1" ]; then
    echo "Usage: ./install.sh <environment_file.yml>"
    echo "Example: ./install.sh environment.yml"
    exit 1
fi

ENV_FILE=$1

# --- Pre-flight Checks --- #
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file not found at '$ENV_FILE'"
    exit 1
fi

if ! command -v conda &> /dev/null; then
    echo "Error: Conda could not be found. Please install Miniconda or Anaconda."
    echo "Visit https://docs.conda.io/en/latest/miniconda.html for installation instructions."
    exit 1
fi

# --- Environment Creation --- #

# Extract environment name from the YAML file
ENV_NAME=$(grep "name:" "$ENV_FILE" | head -n 1 | cut -d ' ' -f 2)

if [ -z "$ENV_NAME" ]; then
    echo "Error: Could not find the environment name in '$ENV_FILE'."
    echo "Make sure the file contains a line like 'name: my-env-name'"
    exit 1
fi

echo "Conda found. Creating environment named '$ENV_NAME' from '$ENV_FILE'..."

# Create the conda environment
conda env create -f "$ENV_FILE"

# --- Final Instructions --- #
if [ $? -eq 0 ]; then
    echo ""
    echo "---------------------------------------------------"
    echo "Conda environment '$ENV_NAME' created successfully."
    echo "To activate the environment, run:"
    echo ""
    echo "conda activate $ENV_NAME"
    echo "---------------------------------------------------"
    echo ""
else
    echo ""
    echo "---------------------------------------------------"
    echo "Failed to create conda environment."
    echo "Please check the error messages above."
    echo "You may want to try creating the environment manually:"
    echo "conda env create -f $ENV_FILE"
    echo "---------------------------------------------------"
    exit 1
fi

echo "Setup complete."
