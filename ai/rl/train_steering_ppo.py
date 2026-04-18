import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class SDAStateEncoder(nn.Module):
    def __init__(self, n_objects=128, feature_dim=8):
        super().__init__()
        self.n_objects = n_objects
        self.feature_dim = feature_dim
        
        # Input: [Batch, N, Features]
        # Features: [CovarianceTrace, TimeSinceLastObs, IsProtected, PriorityClass, Eccentricity, Altitude, SNR_Estimate, Reserved]
        self.input_proj = nn.Linear(feature_dim, 64)
        
        # Small Transformer for SDA Correlation
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Latent space for the whole catalog
        self.latent_proj = nn.Linear(64, 32)
        
        # Policy head: Priority Scores for each object
        self.actor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Value head: State value
        self.critic = nn.Sequential(
            nn.Linear(32 * n_objects, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: [Batch, N, Features]
        x = F.relu(self.input_proj(x))
        x = self.transformer(x)
        latents = F.relu(self.latent_proj(x)) # [Batch, N, 32]
        
        # Priority scores
        scores = self.actor(latents).squeeze(-1) # [Batch, N]
        
        # Value estimation (flatten for simplicity)
        v = self.critic(latents.view(latents.size(0), -1))
        
        return scores, v

def train_and_export():
    n_objects = 128
    features = 8
    model = SDAStateEncoder(n_objects=n_objects, feature_dim=features)
    
    # Simulate a dummy training step (representing the PPO optimization)
    # In a real environment, we would collect trajectories here.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Mock data
    dummy_input = torch.randn(1, n_objects, features)
    
    # Export to ONNX
    os.makedirs("models/rl", exist_ok=True)
    export_path = "models/rl/sensor_steering.onnx"
    
    # We want to export the policy part mainly
    # For inference, we only need scores
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['sda_state'],
        output_names=['priority_scores', 'state_value'],
        dynamic_axes={'sda_state': {0: 'batch_size'},
                      'priority_scores': {0: 'batch_size'},
                      'state_value': {0: 'batch_size'}}
    )
    print(f"✅ PPO Steering Policy exported to {export_path}")

if __name__ == "__main__":
    train_and_export()
