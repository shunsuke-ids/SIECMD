import argparse
import os
import pickle as pkl
import numpy as np

import keras
from keras.callbacks import ModelCheckpoint

from src.preprocessing.augment import *
from src.regression.circular_operations import *
from src.DL.models import *
from src.DL.losses import *
from src.DL.activation_functions import *
from src.DL.metrics import *

parser = argparse.ArgumentParser(description='Train circular regression model with unit circle coordinates')
parser.add_argument('root_dir', help='Path to SIECMD Project')
parser.add_argument('data_dir', help='Path to dataset')
parser.add_argument('dataset', help='Name of dataset')

parser.add_argument('--n', '-n', type=int, default=4)
parser.add_argument('--epochs', '-e', type=int, default=50)
parser.add_argument('--batch_size', '-b', type=int, default=32)
parser.add_argument('--prediction_tolerance', '-pt', type=int, default=45)

args = parser.parse_args()

# Use 2 neurons (unit circle coordinates) with linear distance loss and custom sigmoid activation
LOSS = linear_dist_squared_loss
ACTIVATION = sigmoid_activation  # Custom sigmoid_activation outputs -1 to 1, suitable for unit circle coordinates

def angles_to_unit_circle(angles):
    """Convert angles in degrees to unit circle coordinates (x, y)"""
    angles_rad = np.deg2rad(angles)
    x = np.cos(angles_rad)
    y = np.sin(angles_rad)
    return np.column_stack([x, y])

def unit_circle_to_angles(coordinates):
    """Convert unit circle coordinates (x, y) to angles in degrees"""
    angles_rad = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    angles_deg = np.rad2deg(angles_rad)
    # Convert to 0-360 range
    angles_deg = (angles_deg + 360) % 360
    return angles_deg

print(f'Results for Training (n={args.n}), {args.epochs} Epochs')
print(f'Using 2-neuron unit circle representation with linear_dist_squared_loss and sigmoid_activation\n\n')

mean_deviations = np.zeros(args.n, dtype=np.float32)

for i in range(args.n):
    print(f'{i + 1}.\n')
    with open(f'{args.data_dir}/{args.dataset}_{i}.pkl', 'rb') as f:
        data = pkl.load(f)

    ((X_train, X_val, X_test), (y_train, y_val, y_test)) = data

    # Convert angle labels to unit circle coordinates
    y_train_circle = angles_to_unit_circle(y_train)
    y_val_circle = angles_to_unit_circle(y_val)
    y_test_circle = angles_to_unit_circle(y_test)

    # Create model with 2 output neurons for unit circle coordinates
    model = get_cnn_regression_model(X_train.shape[1:], output_size=2, activation=ACTIVATION, summary=False)

    weights_path = f'{args.root_dir}/weights'
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)

    checkpoint_filepath = f'{weights_path}/weights_unit_circle_{i}.weights.h5'

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1)

    model.compile(optimizer='adam',
                  loss=LOSS)

    if os.path.isfile(checkpoint_filepath):
        print(f'Loads checkpoint file...\n{checkpoint_filepath}')
        model.load_weights(checkpoint_filepath)
    else:
        print(f'Starts training ...\n{checkpoint_filepath}')

        training_history = model.fit(X_train, y_train_circle,
                                     epochs=args.epochs, batch_size=args.batch_size,
                                     validation_data=(X_val, y_val_circle), callbacks=[model_checkpoint_callback])

    # Test-time augmentation (TTA) for unit circle coordinates
    n_rotations = 4
    X_tta, y_tta, rotations = TTA(X_test, y_test, n=n_rotations)
    
    # Convert rotated angles to unit circle coordinates for TTA
    y_tta_circle = angles_to_unit_circle(y_tta)
    
    # Predict unit circle coordinates
    pred_circle = model.predict(X_tta, verbose=0)
    
    # Normalize predictions to unit circle
    pred_norms = np.linalg.norm(pred_circle, axis=1, keepdims=True)
    pred_circle = pred_circle / (pred_norms + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Convert predictions back to angles
    pred_angles = unit_circle_to_angles(pred_circle)
    
    # Apply rotation correction for TTA
    pred_angles_corrected = (pred_angles - rotations) % 360

    # Average predictions from TTA
    predictions = np.zeros(y_test.shape)
    for j in range(int(pred_angles_corrected.shape[0] / n_rotations)):
        values = pred_angles_corrected[n_rotations * j: n_rotations * j + n_rotations]
        predictions[j] = np.round(circular_mean(values))

    mean_deviations[i] = prediction_mean_deviation(y_test, predictions)
    print(f'\tdeviation: {mean_deviations[i]}')

print('\nFinished training')
mean_deviation = np.mean(mean_deviations)
std_diviation = np.sqrt(np.mean((mean_deviations - np.mean(mean_deviations)) ** 2))
print(f'\tMean diviation: {mean_deviation} +- {std_diviation}')
