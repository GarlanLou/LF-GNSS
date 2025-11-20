# LF-GNSS_Test.py
# LF-GNSS test script for evaluating trained models

import torch
import numpy as np
import json
import os
import pickle  # Add pickle import
from torch.utils.data import DataLoader
from datetime import datetime
from gnss_dataset import GNSSDataset, collate_fn
from neural_network import BasicModel
from gnss_util import test_model

# ==================== Configuration ====================
MODEL_PATH = "trained_model/example_model.pth"  # Path to trained model. An example model is provided; you can use this code to test your own models.
VAL_CONFIG_PATH = "config/LF-GNSS-A.json"   # Validation configuration file
OUTPUT_DIR = "lf-gnss_results"              # Output directory for results
# ======================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

def main():
    """Main function to test the trained model"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Model configuration - you can customize these parameters
    input_size = 4  # Number of input features - using SNR, Azimuth, Elevation, Residual as an example - you can design your own input dimensions
    hidden_sizes = [64, 128, 64]  # Hidden layer architecture - feel free to modify
    num_layers = len(hidden_sizes)  # Number of hidden layers
    output_size = 2  # Number of output features - please modify if you want to change the model structure
    batchsize_val = 1  # Validation batch size - keep as 1 for sequential processing
    
    # Initialize model - you can freely explore your own network architecture design
    net = BasicModel(input_size, hidden_sizes, num_layers, output_size).to(DEVICE).double()

    # Load trained model weights
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Model loaded from {MODEL_PATH}")

    # Load normalization parameters from training
    model_dir = os.path.dirname(MODEL_PATH)
    norm_params_path = os.path.join(model_dir, "norm_params.pkl")

    if os.path.exists(norm_params_path):
        with open(norm_params_path, "rb") as f:
            norm_params = pickle.load(f)
        print(f"Normalization parameters loaded from {norm_params_path}")
    else:
        norm_params = None
        print("Warning: Normalization parameters not found, using default values")

    # Load validation dataset with training normalization parameters
    val_dataset = GNSSDataset(VAL_CONFIG_PATH, is_train=False, norm_params=norm_params)
    val_loader = DataLoader(val_dataset, batchsize_val, shuffle=False, collate_fn=collate_fn)
    print(f"Test dataset loaded from {VAL_CONFIG_PATH}")

    # Test model
    print("Starting test...")
    val_result = test_model(net, val_dataset, val_loader, DEVICE)

    # Display results
    dataset_name = os.path.basename(VAL_CONFIG_PATH).split('.')[0]
    print(f"\nTest Results for {dataset_name}:")
    print(f"  East RMSE:  {val_result['east_rmse']:.2f} m")
    print(f"  North RMSE: {val_result['north_rmse']:.2f} m")
    print(f"  Up RMSE:    {val_result['up_rmse']:.2f} m")
    print(f"  2D RMSE:    {val_result['2d_rmse']:.2f} m")
    print(f"  3D RMSE:    {val_result['3d_rmse']:.2f} m")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_output_dir = f"{OUTPUT_DIR}/{dataset_name}"
    os.makedirs(dataset_output_dir, exist_ok=True)

    # Save ENU errors
    enu_output_file = f"{dataset_output_dir}/lf-gnss_enu_errors_{timestamp}.csv"
    np.savetxt(enu_output_file, val_result["enu_errors_with_time"], delimiter=',', 
               header="time,E,N,U", comments='', fmt="%.6f")
    print(f"ENU errors saved to {enu_output_file}")

    # Save position data
    pos_output_file = f"{dataset_output_dir}/lf-gnss_pos_{timestamp}.csv"
    np.savetxt(pos_output_file, val_result["pos_with_time"], delimiter=',', 
               header="time,lat,lon,height", comments='', fmt="%.6f")
    print(f"Position data saved to {pos_output_file}")

    # Save statistics
    stats_file = f"{dataset_output_dir}/stats_{timestamp}.json"
    stats = {
        "east_rmse": float(val_result['east_rmse']),
        "north_rmse": float(val_result['north_rmse']),
        "up_rmse": float(val_result['up_rmse']),
        "2d_rmse": float(val_result['2d_rmse']),
        "3d_rmse": float(val_result['3d_rmse'])
    }
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {stats_file}")

if __name__ == "__main__":
    main()