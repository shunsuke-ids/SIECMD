#!/usr/bin/env python3
"""
Create sample dataset for testing SIECMD
"""
import os
import numpy as np
import cv2 as cv

def create_sample_dataset(output_dir="sample_data", n_angles=36, images_per_angle=5):
    """
    Create a sample dataset with synthetic cell images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create images for different angles
    angles = np.linspace(0, 360, n_angles, endpoint=False).astype(int)
    
    for angle in angles:
        angle_dir = os.path.join(output_dir, str(angle))
        os.makedirs(angle_dir, exist_ok=True)
        
        for i in range(images_per_angle):
            # Create a simple synthetic image (64x64 grayscale)
            img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            
            # Add some directional pattern (simple gradient)
            x = np.linspace(0, 1, 64)
            y = np.linspace(0, 1, 64)
            X, Y = np.meshgrid(x, y)
            
            # Create directional gradient based on angle
            direction_x = np.cos(np.radians(angle))
            direction_y = np.sin(np.radians(angle))
            gradient = (X * direction_x + Y * direction_y) * 50
            
            # Add gradient to random noise
            img = np.clip(img.astype(float) + gradient, 0, 255).astype(np.uint8)
            
            # Save image
            filename = f"image_{i:03d}.png"
            cv.imwrite(os.path.join(angle_dir, filename), img)
    
    print(f"Sample dataset created in {output_dir}")
    print(f"Created {n_angles} angle directories with {images_per_angle} images each")

if __name__ == "__main__":
    create_sample_dataset()
