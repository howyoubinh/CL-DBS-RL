import torch
import torch.nn as nn
import torch.nn.functional as F

class ANNBaseline(nn.Module):
    def __init__(self, n_observations, num_hidden, n_actions):
        """
        Standard Artificial Neural Network (MLP) baseline.
        Architecture matches the RockpoolDQSN hidden layer structure but replaces LIF with ReLU.
        
        Note: Uses full 80 input channels (no Xylo-like downsampling constraint).
        
        Args:
            n_observations (int): Number of input features (80 for full spike channels).
            num_hidden (int): Number of hidden units (e.g., 128).
            n_actions (int): Number of output actions (e.g., 9).
        """
        super(ANNBaseline, self).__init__()
        
        # MLP architecture: 2 hidden layers with ReLU (same depth as SNN)
        self.model = nn.Sequential(
            nn.Linear(n_observations, num_hidden),  # 80 -> 128
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),       # 128 -> 128
            nn.ReLU(),
            nn.Linear(num_hidden, n_actions)         # 128 -> 9
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (batch, time, channels=80).
                        
        Returns:
            q_values (Tensor): Output Q-values of shape (batch, n_actions).
        """
        # Aggregate over time (rate-based equivalent of SNN temporal integration)
        # Input: (B, T, 80) -> Output: (B, 80)
        if x.dim() == 3:
            x = x.mean(dim=1)  # Mean over time dimension
        
        # Pass through MLP: (B, 80) -> (B, 9)
        return self.model(x)
