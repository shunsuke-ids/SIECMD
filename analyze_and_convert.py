#!/usr/bin/env python3
"""
Analyze and convert real cell migration dataset to SIECMD format
"""
import os
import pandas as pd
import numpy as np
import cv2 as cv
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

def normalize_angle(angle):
    """
    Convert angle to 0-360 range
    """
    return (angle % 360)

def load_and_normalize_image(img_path):
    """
    Load 16-bit TIF image and normalize to 8-bit
    """
    # Load as 16-bit
    img = cv.imread(str(img_path), cv.IMREAD_UNCHANGED)
    
    if img is None:
        return None
    
    # Normalize 16-bit to 8-bit
    img_normalized = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    img_8bit = img_normalized.astype(np.uint8)
    
    return img_8bit

def extract_angle_from_csv(csv_path):
    """
    Extract the final migration angle from CSV file
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Use angle_net (net migration direction) as it's more stable
        if 'angle_net' in df.columns:
            angle = df['angle_net'].iloc[-1]  # Use final value
        elif 'angle_i' in df.columns:
            angle = df['angle_i'].iloc[-1]
        else:
            print(f"Warning: No angle column found in {csv_path}")
            return None
        
        # Convert to 0-360 range
        normalized_angle = normalize_angle(angle)
        return int(round(normalized_angle))
        
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def analyze_dataset(source_dir, dataset_name):
    """
    Analyze the dataset structure and angle distribution
    """
    source_path = Path(source_dir)
    
    print(f"Analyzing {dataset_name} dataset...")
    print(f"Source: {source_path}")
    
    angles_found = []
    total_images = 0
    sample_images = []
    
    # Process each fold
    for fold_dir in source_path.glob('fold*'):
        # Process train and test directories
        for split_dir in ['train', 'test']:
            split_path = fold_dir / split_dir
            if not split_path.exists():
                continue
            
            # Process each quadrant directory
            for quadrant_dir in split_path.iterdir():
                if not quadrant_dir.is_dir():
                    continue
                
                # Find all TIF files
                tif_files = list(quadrant_dir.glob('*.tif'))[:3]  # Sample first 3
                
                for tif_file in tif_files:
                    csv_file = tif_file.with_suffix('.csv')
                    
                    if csv_file.exists():
                        angle = extract_angle_from_csv(csv_file)
                        if angle is not None:
                            angles_found.append(angle)
                            total_images += 1
                            
                            if len(sample_images) < 3:
                                # Load and check image
                                img = load_and_normalize_image(tif_file)
                                if img is not None:
                                    sample_images.append({
                                        'path': str(tif_file),
                                        'angle': angle,
                                        'image': img,
                                        'original_path': quadrant_dir.name
                                    })
    
    # Show analysis results
    print(f"\nAnalysis Results:")
    print(f"Total images found: {total_images}")
    print(f"Angle range: {min(angles_found)} to {max(angles_found)}")
    print(f"Unique angles: {len(set(angles_found))}")
    
    # Angle distribution
    angle_counts = {}
    for angle in angles_found:
        angle_counts[angle] = angle_counts.get(angle, 0) + 1
    
    print(f"\nTop 10 most common angles:")
    sorted_angles = sorted(angle_counts.items(), key=lambda x: x[1], reverse=True)
    for angle, count in sorted_angles[:10]:
        print(f"  {angle}°: {count} images")
    
    # Save sample images for verification
    print(f"\nSaving sample images...")
    for i, sample in enumerate(sample_images):
        save_path = f"/home/shunsuke/lab/siecmd/sample_image_{i}_angle_{sample['angle']}.png"
        cv.imwrite(save_path, sample['image'])
        print(f"  Saved: {save_path} (angle: {sample['angle']}°, from: {sample['original_path']})")
    
    return angles_found, sample_images

def convert_real_dataset(source_dir, output_dir, dataset_name):
    """
    Convert with proper angle extraction and image normalization
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConverting {dataset_name} dataset...")
    
    total_images = 0
    angle_counts = {}
    conversion_log = []
    
    # Process each fold
    for fold_dir in source_path.glob('fold*'):
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
                processed_count = 0
                
                for tif_file in tif_files:
                    csv_file = tif_file.with_suffix('.csv')
                    
                    if not csv_file.exists():
                        continue
                    
                    # Extract angle from CSV
                    angle = extract_angle_from_csv(csv_file)
                    if angle is None:
                        continue
                    
                    # Load and normalize image
                    img = load_and_normalize_image(tif_file)
                    if img is None:
                        continue
                    
                    # Create angle directory
                    angle_dir = output_path / str(angle)
                    angle_dir.mkdir(exist_ok=True)
                    
                    # Create unique filename
                    new_name = f"{fold_dir.name}_{split_dir}_{tif_file.stem}.png"
                    dest_path = angle_dir / new_name
                    
                    # Save normalized image as PNG
                    cv.imwrite(str(dest_path), img)
                    total_images += 1
                    processed_count += 1
                    
                    # Count angles
                    angle_counts[angle] = angle_counts.get(angle, 0) + 1
                    
                    # Log conversion
                    conversion_log.append({
                        'original': str(tif_file),
                        'converted': str(dest_path),
                        'angle': angle
                    })
                
                print(f"      Processed {processed_count}/{len(tif_files)} images")
    
    print(f"\nConversion complete!")
    print(f"Total images processed: {total_images}")
    print(f"Unique angles found: {len(angle_counts)}")
    
    return angle_counts, conversion_log

def main():
    # Analyze NIH3T3 dataset first
    nih_source = "/home/shunsuke/datasets/datasets/NIH3T3_4foldcv"
    
    if os.path.exists(nih_source):
        print("="*60)
        print("STEP 1: ANALYZING DATASET")
        print("="*60)
        angles, samples = analyze_dataset(nih_source, "NIH3T3")
        
        print("\n" + "="*60)
        print("STEP 2: CONVERTING DATASET")
        print("="*60)
        nih_output = "/home/shunsuke/lab/siecmd/real_data/NIH3T3"
        angle_counts, log = convert_real_dataset(nih_source, nih_output, "NIH3T3")
        
        print(f"\nFinal angle distribution:")
        sorted_angles = sorted(angle_counts.items())
        for angle, count in sorted_angles:
            print(f"  {angle}°: {count} images")
            
    else:
        print(f"NIH3T3 dataset not found at {nih_source}")

if __name__ == "__main__":
    main()
