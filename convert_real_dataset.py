#!/usr/bin/env python3
"""
Convert real dataset with CSV angle annotations to SIECMD format
"""
import os
import pandas as pd
import numpy as np
import cv2 as cv
import shutil
from pathlib import Path

def extract_angle_from_csv(csv_path):
    """
    Extract the migration angle from CSV file
    """
    try:
        df = pd.read_csv(csv_path)
        # Use the final angle_net value (last row)
        if 'angle_net' in df.columns:
            angle = df['angle_net'].iloc[-1]
        elif 'angle_i' in df.columns:
            angle = df['angle_i'].iloc[-1]
        else:
            print(f"Warning: No angle column found in {csv_path}")
            return None
        
        # Convert to 0-360 range
        angle = angle % 360
        return int(round(angle))
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def convert_real_dataset(source_dir, output_dir, dataset_name, use_single_fold=True):
    """
    Convert fold-based dataset with CSV annotations to angle-based dataset
    Only use fold0 to avoid data duplication
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {dataset_name} dataset...")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    
    if use_single_fold:
        print("Using only fold0 to avoid data duplication")
    
    total_images = 0
    angle_counts = {}
    
    # Process only specified folds
    folds_to_process = ['fold0'] if use_single_fold else list(source_path.glob('fold*'))
    
    for fold_dir in folds_to_process:
        if isinstance(fold_dir, str):
            fold_dir = source_path / fold_dir
        
        if not fold_dir.exists():
            continue
            
        print(f"\nProcessing {fold_dir.name}...")
        
        # Process train and test directories
        for split_dir in ['train', 'test']:
            split_path = fold_dir / split_dir
            if not split_path.exists():
                continue
                
            print(f"  Processing {split_dir}...")
            
            # Process each quadrant directory
            for quadrant_dir in split_path.iterdir():
                if not quadrant_dir.is_dir():
                    continue
                    
                print(f"    Processing {quadrant_dir.name}...")
                
                # Find all TIF files
                tif_files = list(quadrant_dir.glob('*.tif'))
                
                for tif_file in tif_files:
                    # Look for corresponding CSV file
                    csv_file = tif_file.with_suffix('.csv')
                    
                    if not csv_file.exists():
                        print(f"      Warning: No CSV for {tif_file.name}")
                        continue
                    
                    # Extract angle from CSV
                    angle = extract_angle_from_csv(csv_file)
                    if angle is None:
                        continue
                    
                    # Create angle directory
                    angle_dir = output_path / str(angle)
                    angle_dir.mkdir(exist_ok=True)
                    
                    # Create unique filename
                    new_name = f"{split_dir}_{tif_file.name}"
                    dest_path = angle_dir / new_name
                    
                    # Copy image file
                    shutil.copy2(tif_file, dest_path)
                    total_images += 1
                    
                    # Count angles
                    angle_counts[angle] = angle_counts.get(angle, 0) + 1
    
    print(f"\nConversion complete!")
    print(f"Total images processed: {total_images}")
    print(f"Unique angles found: {len(angle_counts)}")
    print(f"Dataset saved to: {output_path}")
    
    # Show angle distribution
    print(f"\nAngle distribution (top 10):")
    sorted_angles = sorted(angle_counts.items(), key=lambda x: x[1], reverse=True)
    for angle, count in sorted_angles[:10]:
        print(f"  Angle {angle}: {count} images")

def main():
    # Install pandas if needed
    try:
        import pandas as pd
    except ImportError:
        print("Installing pandas...")
        os.system("pip install pandas")
        import pandas as pd
    
    # Remove existing output directory and create fresh
    output_base = "/home/shunsuke/lab/siecmd/real_data"
    if Path(output_base).exists():
        print(f"Removing existing {output_base}...")
        shutil.rmtree(output_base)
    
    # Convert NIH3T3 dataset using only fold0
    nih_source = "/home/shunsuke/datasets/datasets/NIH3T3_4foldcv"
    nih_output = "/home/shunsuke/lab/siecmd/real_data/NIH3T3"
    
    if os.path.exists(nih_source):
        convert_real_dataset(nih_source, nih_output, "NIH3T3", use_single_fold=True)
    else:
        print(f"NIH3T3 dataset not found at {nih_source}")
    
    # Convert U373 dataset using only fold0
    u373_source = "/home/shunsuke/datasets/datasets/U373_4foldcv"
    u373_output = "/home/shunsuke/lab/siecmd/real_data/U373"
    
    if os.path.exists(u373_source):
        convert_real_dataset(u373_source, u373_output, "U373", use_single_fold=True)
    else:
        print(f"U373 dataset not found at {u373_source}")

if __name__ == "__main__":
    main()
