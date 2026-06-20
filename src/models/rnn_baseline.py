import torch
import torch.nn as nn

class RNNBaseline(nn.Module):
    def __init__(self, n_observations=80, num_hidden=128, n_actions=9):
        """
        GRU-based Deep Q-Network baseline.
        Processes the full temporal sequence via GRU recurrence,
        then outputs Q-values from the final hidden state.
        
        Architecture: GRU(80, 128) → GRU(128, 128) → Linear(128, 9)
        
        Unlike the ANN which mean-pools the temporal dimension,
        this model processes each timestep sequentially through the GRU,
        preserving temporal dynamics.
        
        Args:
            n_observations (int): Number of input features (80 for full spike channels).
            num_hidden (int): Number of hidden units (128).
            n_actions (int): Number of output actions (9).
        """
        super(RNNBaseline, self).__init__()
        
        # 2-layer GRU with 128 hidden units per layer
        self.gru = nn.GRU(
            input_size=n_observations,
            hidden_size=num_hidden,
            num_layers=2,
            batch_first=True
        )
        
        # Output layer mapping the final hidden state to the 9 Q-values
        self.fc = nn.Linear(num_hidden, n_actions)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (batch, time, channels=80) or (time, channels=80).
                        
        Returns:
            q_values (Tensor): Output Q-values of shape (batch, n_actions).
        """
        # Ensure 3D tensor: (batch, time, channels)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (T, 80) -> (1, T, 80)
            
        # GRU forward pass
        # out shape: (batch, time, hidden_size)
        # h_n shape: (num_layers, batch, hidden_size)
        out, _ = self.gru(x)
        
        # Extract the final hidden state (last timestep's output)
        # final_hidden shape: (batch, hidden_size)
        final_hidden = out[:, -1, :]
        
        # Output layer
        # q_values shape: (batch, n_actions)
        return self.fc(final_hidden)
