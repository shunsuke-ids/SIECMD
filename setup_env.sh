#!/bin/bash

# Environment setup script for SIECMD
echo "Setting up SIECMD environment..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install numpy scipy opencv-python tensorflow keras keras-cv

# Set Python path
export PYTHONPATH=$(pwd)

echo "Environment setup complete!"
echo ""
echo "To use SIECMD:"
echo "1. Run: source setup_env.sh"
echo "2. Prepare dataset: ./prepare_dataset.sh <data_dir> <dataset_name> <save_dir>"
echo "3. Train probing model: ./run_probing.sh <prepared_data_dir> <dataset_name>"
echo "4. Train fine-tuning model: ./run_fine_tuning.sh <prepared_data_dir> <dataset_name>"
