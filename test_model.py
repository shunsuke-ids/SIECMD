#!/usr/bin/env python3
"""
Test the trained SIECMD model manually
"""
import os
import sys
import numpy as np
import pickle
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from DL.models import SIECMD_simple
from DL.losses import circular_mean_squared_error as LOSS
from DL.metrics import prediction_mean_deviation
from regression.fine_tuning import TTA, circular_mean

def test_model(data_dir, dataset_name):
    """Test the trained model"""
    print(f"Testing {dataset_name} model...")
    
    # Load test data
    data_filepath = os.path.join(data_dir, f'{dataset_name}_0.pkl')
    with open(data_filepath, 'rb') as f:
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Load model
    model = SIECMD_simple(224)
    model.compile(optimizer='adam', loss=LOSS)
    
    # Load weights
    weight_files = [f for f in os.listdir('weights') if f.startswith('weights_') and f.endswith('.weights.h5')]
    weight_files.sort()
    
    results = []
    
    for weight_file in weight_files:
        weight_path = os.path.join('weights', weight_file)
        print(f"\nTesting with {weight_file}...")
        
        try:
            model.load_weights(weight_path)
            
            # Test-time augmentation (TTA)
            n_rotations = 4
            X, y, rotations = TTA(X_test, y_test, n=n_rotations)
            pred = model.predict(X, verbose=0)
            pred = pred.reshape((len(pred)))
            pred = (pred - rotations) % 360
            
            predictions = np.zeros(y_test.shape)
            for j in range(int(pred.shape[0] / n_rotations)):
                values = pred[n_rotations * j: n_rotations * j + n_rotations]
                predictions[j] = np.round(circular_mean(values))
            
            deviation = prediction_mean_deviation(y_test, predictions)
            results.append((weight_file, deviation))
            print(f"  Deviation: {deviation:.2f}")
            
        except Exception as e:
            print(f"  Error loading {weight_file}: {e}")
    
    return results

if __name__ == "__main__":
    import sys
    
    data_dir = "./prepared_data"
    dataset_name = "NIH3T3_real"
    
    results = test_model(data_dir, dataset_name)
    
    print(f"\n=== Final Results for {dataset_name} ===")
    for weight_file, deviation in results:
        print(f"{weight_file}: {deviation:.2f}")
    
    if results:
        best_result = min(results, key=lambda x: x[1])
        print(f"\nBest result: {best_result[0]} with deviation {best_result[1]:.2f}")
