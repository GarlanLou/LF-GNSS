# losses.py
# neural network loss functions

# You can freely design and implement your own loss functions here

import torch
from torch.nn import Module
import gnss_util as util

# Import coordinate transformation functions from gnss_util
ecef_to_enu_torch = util.ecef_to_enu_torch

class FocalLoss(Module):
    def __init__(self, alpha=1.0, gamma=2.0, dynamic_gamma=True, scale_factor=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight coefficient
        self.gamma = gamma  # Initial hard sample weight coefficient
        self.dynamic_gamma = dynamic_gamma  # Whether to use dynamic gamma
        self.scale_factor = scale_factor  # Scaling factor

    def forward(self, inputs, targets):
        """
        Calculate improved Focal Loss with geodetic coordinates (lat, lon, h) input.
        inputs: Predicted ECEF coordinates, shape (batch_size, 3) -> [x, y, z]
        targets: Ground truth geodetic coordinates (batch_size, 3) -> [lat, lon, h]
        """
        # If single sample, expand shape from (3) to (1, 3)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)

        # Calculate ENU errors
        enu_errors = torch.stack([
            ecef_to_enu_torch(pred, gt) for pred, gt in zip(inputs, targets)
        ])

        # Calculate squared Euclidean distance (square of L2 distance)
        squared_dist = torch.norm(enu_errors, p=2, dim=1)

        # Base loss
        base_loss = squared_dist

        # Calculate dynamic gamma (optional)
        if self.dynamic_gamma:
            dynamic_gamma = self.gamma * torch.exp(-self.scale_factor * base_loss)
        else:
            dynamic_gamma = self.gamma

        # Calculate Focal Loss weights
        focal_weight = torch.pow(1 - torch.sqrt(squared_dist / (squared_dist.max() + 1e-8)), dynamic_gamma)

        focal_loss = self.alpha * focal_weight * base_loss

        # Return RMSE form loss
        focal_loss_rmse = torch.sqrt((focal_loss ** 2).mean())

        return focal_loss_rmse

# Using improved loss function
def hem_calculate_loss(pred_ecef, gt_llh):
    """
    Calculate loss between model output and ground truth labels
    pred_ecef: Predictions, shape (batch_size, 3) -> [x, y, z]
    gt_llh: Ground truth, shape (batch_size, 3) -> [lat, lon, h]
    """
    loss_fn = FocalLoss(alpha=0.75, gamma=1.5)  # Use focal loss
    return loss_fn(pred_ecef, gt_llh)