#!/bin/bash

# SIECMD Unit Circle Probing Script
# Usage: ./run_probing_unit_circle.sh <data_dir> <dataset_name>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <data_dir> <dataset_name>"
    echo "Example: $0 ./prepared_data my_dataset"
    echo ""
    echo "This script runs SIECMD probing with unit circle representation:"
    echo "- 2 output neurons (x, y coordinates on unit circle)"
    echo "- linear_dist_squared_loss"
    echo "- sigmoid activation"
    exit 1
fi

# Get parameters
DATA_DIR=$1
DATASET_NAME=$2
ROOT_DIR=$(pwd)

echo "Running SIECMD Unit Circle Probing with:"
echo "  Root directory: $ROOT_DIR"
echo "  Data directory: $DATA_DIR"
echo "  Dataset name: $DATASET_NAME"
echo ""

# Set environment
export PYTHONPATH=$ROOT_DIR
source venv/bin/activate

# Run probing
python src/regression/probing_unit_circle.py "$ROOT_DIR" "$DATA_DIR" "$DATASET_NAME"
