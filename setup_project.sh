#!/bin/bash

# Exit immediately if a command fails
set -e

# Name of the environment file and environment
ENV_FILE="environment.yaml"

# Extract environment name from YAML
ENV_NAME=$(grep 'name:' "$ENV_FILE" | awk '{print $2}')

echo "Creating Conda environment: $ENV_NAME"
conda env create -f "$ENV_FILE"

echo "Activating Conda environment: $ENV_NAME"
# Use conda's shell integration to enable "conda activate"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Running data download script..."
bash download_logs_models_data.sh

echo "Setup complete."
