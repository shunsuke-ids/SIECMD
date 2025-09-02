#!/bin/bash

# SIECMD Dataset Preparation Script
# Usage: ./prepare_dataset.sh <data_dir> <dataset_name> <save_dir>

if [ $# -lt 3 ]; then
    echo "Usage: $0 <data_dir> <dataset_name> <save_dir>"
    echo "Example: $0 ./raw_data my_dataset ./prepared_data"
    echo ""
    echo "The dataset should be formatted as follows:"
    echo "data_dir/"
    echo "├── 0/"
    echo "│   ├── image_name0.format"
    echo "│   └── image_name1.format"
    echo "├── 1/"
    echo "│   ├── image_name2.format"
    echo "│   ..."
    echo "..."
    echo "├── 360/"
    echo "│   ├── ..."
    echo "│   ..."
    echo "with 0, 1, ... being the GT angles"
    exit 1
fi

DATA_DIR=$1
DATASET_NAME=$2
SAVE_DIR=$3
ROOT_DIR=$(pwd)

export PYTHONPATH=$ROOT_DIR

echo "Preparing SIECMD dataset:"
echo "  Source directory: $DATA_DIR"
echo "  Dataset name: $DATASET_NAME"
echo "  Save directory: $SAVE_DIR"
echo ""

mkdir -p "$SAVE_DIR"

python3 src/regression/prepare_dataset.py "$DATA_DIR" "$DATASET_NAME" "$SAVE_DIR" \
    --folds 4
