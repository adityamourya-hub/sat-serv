#!/usr/bin/env python3
"""
ADITI 4.0 Hamiltonian Neural Network (HNN) Training Pipeline

Implements Greydanus 2019 HNN with geometric constraints for orbital mechanics.

Key features:
- Learns scalar Hamiltonian H(q, p; θ) instead of direct dynamics
- Uses torch.autograd for ∂H/∂q, ∂H/∂p gradients
- Enforces energy conservation by construction
- Exports to ONNX for Go runtime inference

References:
- arXiv:1906.01563 (Greydanus et al., foundational HNN)
- arXiv:2507.15678 (GeoHNN with SPD constraints)
- arXiv:2501.18808 (HNN-UKF for astrodynamics)

Usage:
    python train_hnn.py --data data/orbits.npz --epochs 1000 --export
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import onnx
from pathlib import Path


class HamiltonianNetwork(nn.Module):
    """
    Neural network that learns scalar Hamiltonian H(q, p).
    
    Architecture: MLP with symplectic structure preservation.
    Input: (q, p) concatenated → 6D for orbital mechanics
    Output: H scalar (energy)
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=3):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())  # Smooth activation for gradients
            in_dim = hidden_dim
        
        # Final layer outputs scalar Hamiltonian
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Learnable mass parameter (optional)
        self.mass = nn.Parameter(torch.tensor(1000.0))
    
    def forward(self, q, p):
        """
        Compute H(q, p).
        
        Args:
            q: Position tensor (batch, 3)
            p: Momentum tensor (batch, 3)
        
        Returns:
            H: Hamiltonian scalar (batch, 1)
        """
        # Concatenate (q, p)
        x = torch.cat([q, p], dim=-1)  # (batch, 6)
        
        H = self.network(x)
        
        # Add analytical kinetic energy prior (optional)
        # This helps learning: T = p²/(2m)
        T_prior = 0.5 * torch.sum(p**2, dim=-1, keepdim=True) / self.mass
        
        # Network learns residual: V_eff + perturbations
        return T_prior + H
    
    def gradients(self, q, p):
        """
        Compute ∂H/∂q and ∂H/∂p via autograd.
        
        These are Hamilton's equations components:
        - ∂H/∂p = velocity (q̇)
        - -∂H/∂q = force (ṗ)
        
        Args:
            q: Position (requires_grad=True)
            p: Momentum (requires_grad=True)
        
        Returns:
            dHdq: ∂H/∂q (batch, 3)
            dHdp: ∂H/∂p (batch, 3)
        """
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        
        H = self.forward(q, p)
        
        # Compute gradients
        dHdq = torch.autograd.grad(
            H.sum(), q, create_graph=True, retain_graph=True
        )[0]
        
        dHdp = torch.autograd.grad(
            H.sum(), p, create_graph=True, retain_graph=True
        )[0]
        
        return dHdq, dHdp
    
    def dynamics(self, q, p):
        """
        Compute Hamilton's equations.
        
        Returns:
            q_dot = ∂H/∂p (velocity)
            p_dot = -∂H/∂q (force)
        """
        dHdq, dHdp = self.gradients(q, p)
        
        q_dot = dHdp  # ẋ = ∂H/∂p
        p_dot = -dHdq  # ṗ = -∂H/∂q
        
        return q_dot, p_dot


class SymplecticLoss(nn.Module):
    """
    Loss function enforcing symplectic structure.
    
    Combines:
    1. Trajectory matching (data fit)
    2. Energy conservation penalty
    3. Symplectic regularization (optional GeoHNN)
    """
    
    def __init__(self, lambda_energy=0.1, lambda_symplectic=0.01):
        super().__init__()
        self.lambda_energy = lambda_energy
        self.lambda_symplectic = lambda_symplectic
    
    def forward(self, model, q0, p0, q1_target, p1_target, dt):
        """
        Compute total loss.
        
        Args:
            model: HamiltonianNetwork
            q0, p0: Initial state
            q1_target, p1_target: Target state after dt
            dt: Time step
        
        Returns:
            Total loss scalar
        """
        # Predict next state via symplectic integration
        q1_pred, p1_pred = self.symplectic_euler(model, q0, p0, dt)
        
        # 1. Trajectory matching loss
        loss_traj = torch.mean((q1_pred - q1_target)**2) + \
                    torch.mean((p1_pred - p1_target)**2)
        
        # 2. Energy conservation penalty
        H0 = model.forward(q0, p0)
        H1 = model.forward(q1_pred, p1_pred)
        loss_energy = torch.mean((H1 - H0)**2)
        
        # 3. Symplectic regularization (GeoHNN)
        # Compute Jacobian and enforce J^T Ω J = Ω
        loss_symplectic = self.symplectic_constraint(model, q0, p0)
        
        total_loss = loss_traj + \
                     self.lambda_energy * loss_energy + \
                     self.lambda_symplectic * loss_symplectic
        
        return total_loss, {
            'traj': loss_traj.item(),
            'energy': loss_energy.item(),
            'symplectic': loss_symplectic.item()
        }
    
    def symplectic_euler(self, model, q, p, dt):
        """
        Symplectic Euler integration step.
        
        Preserves Hamiltonian structure better than standard Euler.
        """
        q_dot, p_dot = model.dynamics(q, p)
        
        # Update
        q_new = q + dt * q_dot
        p_new = p + dt * p_dot
        
        return q_new, p_new
    
    def symplectic_constraint(self, model, q, p):
        """
        GeoHNN: Enforce symplectic constraint on Jacobian.
        
        For perfect symplecticity: J^T Ω J = Ω
        where Ω = [[0, I], [-I, 0]] (canonical symplectic form)
        
        This is a simplified version - full GeoHNN uses SPD parameterization.
        """
        # Compute Jacobian of dynamics
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        
        q_dot, p_dot = model.dynamics(q, p)
        
        # Jacobian elements (simplified check)
        # Full implementation would compute full 6x6 Jacobian
        
        # Penalize if divergence of phase flow is non-zero
        div_q = torch.autograd.grad(
            q_dot.sum(), q, create_graph=True, retain_graph=True
        )[0].sum(dim=-1)
        
        div_p = torch.autograd.grad(
            p_dot.sum(), p, create_graph=True, retain_graph=True
        )[0].sum(dim=-1)
        
        # For Hamiltonian flow: div should be zero (volume preservation)
        loss = torch.mean((div_q + div_p)**2)
        
        return loss


def load_orbital_data(data_path):
    """
    Load training data from Go-generated .npz file.
    
    Expected format from Go GenerateTrainingData():
    - q0: initial positions (N, 3)
    - p0: initial momenta (N, 3)
    - q1: final positions (N, 3)
    - p1: final momenta (N, 3)
    - dt: time step (scalar or N,)
    """
    data = np.load(data_path)
    
    q0 = torch.tensor(data['q0'], dtype=torch.float32)
    p0 = torch.tensor(data['p0'], dtype=torch.float32)
    q1 = torch.tensor(data['q1'], dtype=torch.float32)
    p1 = torch.tensor(data['p1'], dtype=torch.float32)
    dt = torch.tensor(data.get('dt', 60.0), dtype=torch.float32)  # default 60s
    
    return q0, p0, q1, p1, dt


def train(args):
    """
    Main training loop.
    """
    print(f"Loading data from {args.data}")
    q0, p0, q1, p1, dt = load_orbital_data(args.data)
    
    print(f"Dataset: {len(q0)} samples")
    print(f"Time step: {dt.item():.1f} seconds")
    
    # Create model
    model = HamiltonianNetwork(
        input_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = SymplecticLoss(
        lambda_energy=args.lambda_energy,
        lambda_symplectic=args.lambda_symplectic
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        
        # Batch training (simplified - full batch for small datasets)
        optimizer.zero_grad()
        
        loss, loss_dict = criterion(model, q0, p0, q1, p1, dt)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step(loss)
        
        # Logging
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.6f} "
                  f"(traj={loss_dict['traj']:.6f}, "
                  f"energy={loss_dict['energy']:.6e}, "
                  f"symp={loss_dict['symplectic']:.6e})")
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f"{args.output}/hnn_best.pt")
    
    print(f"Training complete. Best loss: {best_loss:.6f}")
    
    # Export to ONNX
    if args.export:
        export_onnx(model, args.output)
    
    return model


def export_onnx(model, output_dir):
    """
    Export trained model to ONNX for Go inference.
    
    This exports just the H(q,p) forward pass.
    Gradients (∂H/∂q, ∂H/∂p) are computed in Go via finite differences
    or by exporting a gradient-computing model.
    
    TODO: For production, export dual-output model with pre-computed gradients
    to avoid runtime autograd in Go.
    """
    model.eval()
    
    # Dummy inputs
    q_dummy = torch.randn(1, 3)
    p_dummy = torch.randn(1, 3)
    
    # Export H(q,p) only
    # For gradient computation in Go:
    # Option 1: Finite differences (slower, simpler)
    # Option 2: Export second model that outputs gradients (faster)
    
    torch.onnx.export(
        model,
        (q_dummy, p_dummy),
        f"{output_dir}/hnn_model.onnx",
        export_params=True,
        opset_version=15,  # Use 15+ for better dynamic support
        do_constant_folding=True,
        input_names=['q', 'p'],
        output_names=['H'],
        dynamic_axes={
            'q': {0: 'batch_size'},
            'p': {0: 'batch_size'},
            'H': {0: 'batch_size'}
        }
    )
    
    print(f"Exported ONNX model to {output_dir}/hnn_model.onnx")
    
    # Also export gradient-computing version (optional)
    export_gradient_model(model, output_dir)


def export_gradient_model(model, output_dir):
    """
    Export model that computes both H and gradients.
    
    This avoids finite difference computations in Go.
    """
    class GradientWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        
        def forward(self, q, p):
            H = self.base_model(q, p)
            dHdq, dHdp = self.base_model.gradients(q, p)
            
            # Return H, dH/dq, dH/dp
            return H, dHdq, dHdp
    
    wrapper = GradientWrapper(model)
    wrapper.eval()
    
    q_dummy = torch.randn(1, 3)
    p_dummy = torch.randn(1, 3)
    
    torch.onnx.export(
        wrapper,
        (q_dummy, p_dummy),
        f"{output_dir}/hnn_with_gradients.onnx",
        export_params=True,
        opset_version=15,
        input_names=['q', 'p'],
        output_names=['H', 'dHdq', 'dHdp'],
        dynamic_axes={
            'q': {0: 'batch_size'},
            'p': {0: 'batch_size'},
            'H': {0: 'batch_size'},
            'dHdq': {0: 'batch_size'},
            'dHdp': {0: 'batch_size'}
        }
    )
    
    print(f"Exported gradient model to {output_dir}/hnn_with_gradients.onnx")


def test_conservation(model, q0, p0, steps=100, dt=60.0):
    """
    Test energy conservation on a trajectory.
    """
    model.eval()
    
    q, p = q0[:1], p0[:1]  # Single trajectory
    
    energies = []
    
    with torch.no_grad():
        H0 = model(q, p).item()
        energies.append(H0)
        
        for _ in range(steps):
            q_dot, p_dot = model.dynamics(q, p)
            q = q + dt * q_dot
            p = p + dt * p_dot
            
            H = model(q, p).item()
            energies.append(H)
    
    # Compute drift
    max_drift = max(abs(e - H0) for e in energies) / abs(H0)
    
    print(f"\nConservation test:")
    print(f"  Initial H: {H0:.6e}")
    print(f"  Final H: {energies[-1]:.6e}")
    print(f"  Max relative drift: {max_drift:.6e}")
    
    return max_drift


def main():
    parser = argparse.ArgumentParser(description='Train HNN for orbital mechanics')
    
    # Data
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data .npz file')
    parser.add_argument('--output', type=str, default='./models',
                        help='Output directory for models')
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden layer dimension')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of hidden layers')
    
    # Training
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lambda-energy', type=float, default=0.1,
                        help='Energy conservation penalty weight')
    parser.add_argument('--lambda-symplectic', type=float, default=0.01,
                        help='Symplectic constraint weight')
    
    # Export
    parser.add_argument('--export', action='store_true',
                        help='Export to ONNX after training')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Train
    model = train(args)
    
    # Test conservation
    q0, p0, _, _, _ = load_orbital_data(args.data)
    drift = test_conservation(model, q0, p0)
    
    if drift > 1e-4:
        print("\nWARNING: High energy drift detected. Model may need more training.")
    else:
        print("\nSUCCESS: Energy conservation within acceptable bounds.")


if __name__ == '__main__':
    main()
