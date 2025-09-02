#!/bin/bash

# SIECMD Fine-tuning Script
# Usage: ./run_fine_tuning.sh <data_dir> <dataset_name>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_dir> <dataset_name>"
    echo "Example: $0 ./prepared_data my_dataset"
    exit 1
fi

DATA_DIR=$1
DATASET_NAME=$2
ROOT_DIR=$(pwd)

export PYTHONPATH=$ROOT_DIR

echo "Running SIECMD Fine-tuning with:"
echo "  Root directory: $ROOT_DIR"
echo "  Data directory: $DATA_DIR"
echo "  Dataset name: $DATASET_NAME"
echo ""

python3 src/regression/fine_tuning.py "$ROOT_DIR" "$DATA_DIR" "$DATASET_NAME" \
    --n 4 \
    --epochs 10 \
    --fine_tuning_epochs 50 \
    --batch_size 32 \
    --prediction_tolerance 45
