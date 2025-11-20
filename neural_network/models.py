# models.py
# Neural network models for GNSS positioning

# You can freely design and implement your own models here
# This is a basic model that can be adapted and extended with various architectures.

import torch
import torch.nn as nn

class BasicModel(nn.Module):
    """Basic neural network model for GNSS positioning"""
    """You can freely explore and add various architectures."""
    def __init__(self, input_size, hidden_sizes, num_layers, output_size):
        super(BasicModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_sizes[i-1], hidden_sizes[i]) 
            for i in range(num_layers)
        ])

        # Output layer
        self.fc_output = nn.Linear(hidden_sizes[-1], output_size)

        # Activation functions
        self.softplus = nn.Softplus()  # For estimating R

    def forward(self, x, mask=None):

        for fc in self.fc_layers:
            x = fc(x)
            x = torch.relu(x)  # Use ReLU activation function

        # Map to output
        output = self.fc_output(x)

        # Apply activation functions
        R_diag = self.softplus(output[:, :, 0])  # Learn diagonal elements of R
        v_c = output[:, :, 1]  # bias remains as real values

        return R_diag, v_c

