# LF-GNSS_Train.py
# LF-GNSS training script

# Import necessary libraries
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import torch.optim as optim
from gnss_foundation.gnss_type import gtime_t
import os
from datetime import datetime
import argparse
# Import custom modules
from neural_network import BasicModel, FocalLoss, hem_calculate_loss
from gnss_dataset import GNSSDataset, collate_fn
from gnss_util import validate_model
import pickle

# Device configuration: Use GPU if available, otherwise use CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Neural network model configuration parameters - feel free to explore and customize
input_size = 4  # # Number of input features - using SNR, Azimuth, Elevation, Residual as an example - you can design your own input dimensions
                # you can design your own input dimensions based on your new input features
hidden_sizes = [64,128,64]  # Hidden layer architecture - feel free to explore different structures
num_layers = len(hidden_sizes)  # Number of layers automatically defined by hidden_sizes
output_size = 2  # Number of output features - please modify if you want to change the model structure
batchsize_train = 16  # Training batch size - adjust based on your GPU memory
batchsize_val = 1     # Validation batch size - keep as 1 for sequential processing

# Initialize neural network model and move to specified device
net = BasicModel(input_size, hidden_sizes, num_layers, output_size).to(DEVICE).double()
# Initialize loss function and move to specified device
criterion = FocalLoss().to(DEVICE).double()
# Initialize optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
# Define learning rate scheduler: Cosine annealing scheduler
# T_max represents the complete cosine cycle of the scheduler (in epochs), eta_min is the minimum learning rate
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-4)

# Dataset configuration dictionary: Define configuration file paths for different datasets
dataset_configs = {
    "LF-GNSS": {
        "train": "config/LF-GNSS-train.json",
        "val1": "config/LF-GNSS-A.json",
        "val2": "config/LF-GNSS-B.json",
        "val3": "config/LF-GNSS-C.json",
    }
}

# Parse command line arguments: Allow users to select the dataset to train
parser = argparse.ArgumentParser(description="Select dataset for GNSS training.")
parser.add_argument("dataset", type=str, help="Dataset name (LF-GNSS)")
args = parser.parse_args()

# Get dataset configuration: Select corresponding dataset configuration based on user input
if args.dataset in dataset_configs:
    config = dataset_configs[args.dataset]
    train_config = config["train"]
    val1_config = config["val1"]
    val2_config = config["val2"]
    val3_config = config["val3"]
else:
    raise ValueError(f"Unknown dataset name: {args.dataset}")

# Extract validation set names for result output
val1_name = os.path.basename(val1_config).split(".")[0]
val2_name = os.path.basename(val2_config).split(".")[0]
val3_name = os.path.basename(val3_config).split(".")[0]

# Data loading section: Load training and validation datasets
try:
    # Load training dataset
    train_dataset = GNSSDataset(train_config, is_train=True)
    # Create training data loader
    train_loader = DataLoader(train_dataset, batchsize_train, shuffle=False, collate_fn=collate_fn)
    
    # Load validation datasets - use training normalization parameters
    val1_dataset = GNSSDataset(val1_config, is_train=False, norm_params=train_dataset.norm_params) 
    val1_loader = DataLoader(val1_dataset, batchsize_val, shuffle=False, collate_fn=collate_fn)
    
    val2_dataset = GNSSDataset(val2_config, is_train=False, norm_params=train_dataset.norm_params)
    val2_loader = DataLoader(val2_dataset, batchsize_val, shuffle=False, collate_fn=collate_fn)
    
    val3_dataset = GNSSDataset(val3_config, is_train=False, norm_params=train_dataset.norm_params)
    val3_loader = DataLoader(val3_dataset, batchsize_val, shuffle=False, collate_fn=collate_fn)

    print("------------------Dataset loaded successfully------------------")
except Exception as e:
    print(f"------------------Failed to load dataset: {e}------------------")
    raise

# Get current timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
best_avg_3d_rmse = float('inf')
best_model_state = None

# Extract training set name for model saving
train_name = os.path.basename(train_config).split("-")[0]
save_dir = "trained_model"
os.makedirs(save_dir, exist_ok=True)
# Define model save path
save_path = f"{save_dir}/{train_name}_{timestamp}.pth"

# Save normalization parameters
norm_params_path = f"{save_dir}/norm_params.pkl"
with open(norm_params_path, "wb") as f:
    pickle.dump(train_dataset.norm_params, f)
print(f"Normalization parameters saved to {norm_params_path}")

num_epochs = 200
best_val_results = {}
best_enu_errors = {
    val1_name: [],
    val2_name: [],
    val3_name: []
}
pos = {
    val1_name: [],
    val2_name: [],
    val3_name: []
}

print("------------------Training starts------------------")

# Main training loop
for epoch in range(num_epochs):
    net.train()
    epoch_loss = 0.0
    
    # Reset training dataset state
    train_dataset.std.nav.t = gtime_t(0, 0.0)
    train_dataset.std.nav.x = np.zeros(train_dataset.nav.nx)
    train_dataset.std.initialize_covariance()
    
    # Iterate through training data batches
    for inputs, labels, masks, obss in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
        # Move data to specified device
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        R_diag, bias = net(inputs, masks)

        # Collect prediction results
        pred_ecef_list = []
        gt_llh_list = []

        # Process each sample
        for i, input_data in enumerate(inputs):
            o = obss[i]
            gt_row = labels[i]

            valid_mask = masks[i].bool()
            valid_R = R_diag[i][valid_mask]
            valid_bias = bias[i][valid_mask]
            # Process observation data
            result = train_dataset.std.process(obs=o, Net_R=valid_R, bias=valid_bias)

            if result['status']:
                pred_ecef = result['pos'][:3]
                pred_ecef_list.append(pred_ecef)
                gt_llh_list.append(gt_row.to(torch.double).to(DEVICE))

        # If there are successfully processed samples
        if pred_ecef_list and gt_llh_list:
            pred_ecef_batch = torch.stack(pred_ecef_list)
            gt_llh_batch = torch.stack(gt_llh_list)

            # Calculate loss
            loss = hem_calculate_loss(pred_ecef_batch, gt_llh_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    scheduler.step()
    # Print epoch training information
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.6f}, LR: {scheduler.get_last_lr()[0]:.6e}")

    # Validate every 5 epochs
    if (epoch + 1) % 5 == 0:
        # Validate on three validation sets - add DEVICE parameter
        val1 = validate_model(net, val1_dataset, val1_loader)
        val2 = validate_model(net, val2_dataset, val2_loader)
        val3 = validate_model(net, val3_dataset, val3_loader)

        # Calculate average 3D RMSE
        avg_3d_rmse = (val1["3d_rmse"] + val2["3d_rmse"] + val3["3d_rmse"]) / 3
        # Print validation results
        print(f"\nValidation Results at Epoch {epoch + 1}:")
        print(f"  {val1_name}  ->  E: {val1['east_rmse']:.2f}, N: {val1['north_rmse']:.2f}, U: {val1['up_rmse']:.2f}, 2D: {val1['2d_rmse']:.2f}, 3D: {val1['3d_rmse']:.2f}")
        print(f"  {val2_name}  ->  E: {val2['east_rmse']:.2f}, N: {val2['north_rmse']:.2f}, U: {val2['up_rmse']:.2f}, 2D: {val2['2d_rmse']:.2f}, 3D: {val2['3d_rmse']:.2f}")
        print(f"  {val3_name}  ->  E: {val3['east_rmse']:.2f}, N: {val3['north_rmse']:.2f}, U: {val3['up_rmse']:.2f}, 2D: {val3['2d_rmse']:.2f}, 3D: {val3['3d_rmse']:.2f}")
        print(f"Average 3D RMSE: {avg_3d_rmse:.2f}")

        # Save if current model performs better
        if avg_3d_rmse < best_avg_3d_rmse:
            best_avg_3d_rmse = avg_3d_rmse
            best_model_state = net.state_dict()
            
            # Save best model
            torch.save(best_model_state, save_path)
            print(f"\nNew best model saved at epoch {epoch + 1}, Avg 3D RMSE: {avg_3d_rmse:.2f}")

# Training completed
print("------------------Training completed------------------")
print(f"Best model Avg 3D RMSE: {best_avg_3d_rmse:.2f}")

