"""
DEPRECATED: Conservative-only GHNN (special case of Port-Hamiltonian System)

This module implements the old conservative-only GHNN.
GHNN is the special case of Port-Hamiltonian System with R=0, g=0.

USE INSTEAD: port_hamiltonian_pytorch.PortHamiltonianSystem
- Handles non-conservative dynamics (drag, thrust)
- PSD dissipation matrix via Cholesky parameterization
- Control coupling for active maneuvers
- Latent conditioning for regime-aware dynamics

Migration Guide:
    OLD: from ghnn import GHNNVectorField, HamiltonianNet
    NEW: from port_hamiltonian_pytorch import PortHamiltonianSystem
    
    OLD: ghnn = GHNNVectorField(HamiltonianNet())
    NEW: phs = PortHamiltonianSystem(use_latent_conditioning=False)  # R=0, g=0 if not needed

Kept for backwards compatibility only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

# Canonical Physics Constants (WGS-84)
# Used for DU/TU normalization to ensure µ = 1.0 and RE = 1.0
RE_KM = 6378.137             # Earth Equator Radius (km)
MU_KM = 398600.4418          # Earth Gravitational Parameter (km^3/s^2)
TU_SEC = np.sqrt(RE_KM**3 / MU_KM) # Time Unit (s) approx 806.81s

class HamiltonianNet(nn.Module):
    """
    Learns the scalar Hamiltonian potential H(q, p).
    Hardened Architecture: 4-layer MLP with 128 units and Tanh activation.
    Optimized for symplectic conservation and ultra-low latency inference.
    """
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        layers = []
        input_dim = 6  # [q, p] in Canonical Units (DU, DU/TU)
        
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.Tanh(), # Tanh ensures smooth double-differentiability
            ])
        
        layers.append(nn.Linear(hidden_dim, 1)) # Scalar Potential H
        self.net = nn.Sequential(*layers)
        
        # Initialize with smaller weights to keep initial manifold near flat
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for scalar potential H."""
        return self.net(x)

class GHNNVectorField(nn.Module):
    """
    Wraps HamiltonianNet to provide the symplectic vector field [dH/dp, -dH/dq].
    Bakes gradients into the graph for traceable ONNX export.
    """
    def __init__(self, h_net: HamiltonianNet):
        super().__init__()
        self.h_net = h_net
        self.iteration_count = 0  # For periodic GPU cache cleanup
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hamiltonian dynamics via automatic differentiation.
        Enforces symplectic geometry: z_dot = J * grad(H)
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            H = self.h_net(x)
            
            # Use PyTorch's built-in training flag for autograd graph retention
            if self.training:
                # Training: retain graph for higher-order derivatives
                grad_H = torch.autograd.grad(
                    H, x, 
                    grad_outputs=torch.ones_like(H),
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                # Periodic memory cleanup every 100 iterations
                self.iteration_count += 1
                if self.iteration_count % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # Inference: don't retain graph to prevent memory leak
                grad_H = torch.autograd.grad(
                    H, x, 
                    grad_outputs=torch.ones_like(H),
                    create_graph=False,
                    retain_graph=False
                )[0]
        
        # NaN guards: numerical gradient fallback for Hamiltonian autograd failures
        if torch.isnan(grad_H).any():
            print("[WARNING] NaN detected in Hamiltonian gradient, using fallback")
            # Fallback: numerical gradient via finite differences
            eps = 1e-6
            grad_H_numerical = torch.zeros_like(x)
            for i in range(x.shape[1]):
                x_plus = x.clone()
                x_plus[:, i] += eps
                H_plus = self.h_net(x_plus)
                grad_H_numerical[:, i] = (H_plus - H) / eps
            grad_H = grad_H_numerical
        
        if torch.isinf(grad_H).any():
            print("[WARNING] Inf detected in Hamiltonian gradient, clipping")
            grad_H = torch.clamp(grad_H, -1e6, 1e6)
        
        # Gradient clipping to prevent exploding gradients
        grad_H = torch.nn.utils.clip_grad_norm_(grad_H, max_norm=1.0)
            
        # Hamiltonian Equations:
        # q_dot =  dH/dp (velocity components)
        # p_dot = -dH/dq (acceleration components)
        dH_dq = grad_H[:, :3]
        dH_dp = grad_H[:, 3:]
        
        return torch.cat([dH_dp, -dH_dq], dim=1)

def export_ghnn_to_onnx(model: GHNNVectorField, output_path: str):
    """
    Exports the hardened GHNN to ONNX using opset 15 for gradient support.
    """
    model.eval()
    # Dummy input in Canonical Units [1, 6]
    dummy_input = torch.randn(1, 6)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['state'],
        output_names=['dynamics'],
        dynamic_axes={'state': {0: 'batch'}, 'dynamics': {0: 'batch'}},
        opset_version=15,
        do_constant_folding=True
    )
    print(f"Sovereign GHNN (4x128 Tanh) exported to {output_path}")

if __name__ == "__main__":
    h_net = HamiltonianNet()
    ghnn = GHNNVectorField(h_net)
    export_ghnn_to_onnx(ghnn, "axiom_hnode_v1.onnx")
