"""
Sovereign Training Pipeline for Axiom-Zero.

⚠️  DEPRECATION NOTICE: This PyTorch training pipeline is deprecated.
    JAX Port-Hamiltonian System (ai/v3/train_phs_jax.py) is now the canonical
    training backend for structural correctness (Forest-Ruth 4th-order symplectic).
    
    This PyTorch version is maintained for:
    - Backwards compatibility
    - Deployment/ONNX export path
    - Users without JAX/CUDA support

Enforces physical invariants via:
1. Canonical DU/TU Normalization (µ=1.0)
2. Port-Hamiltonian dynamics with learned dissipation R(x)
3. Augmented Lagrangian hard constraints (energy, momentum)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import logging
import argparse
from pathlib import Path
import json
import warnings

# UNIFIED ARCHITECTURE: Port-Hamiltonian System is the default
# Both PyTorch (deployment) and JAX (training) versions exist
# JAX version in ai/v3/port_hamiltonian.py is canonical for training
# PyTorch version below is for deployment compatibility
from port_hamiltonian_pytorch import (
    PortHamiltonianSystem,
    create_port_hamiltonian_system,
    GHNNVectorField,  # DEPRECATED: Use PortHamiltonianSystem instead
)

# Emit deprecation warning for training usage
warnings.warn(
    "PyTorch training pipeline is deprecated. "
    "Use ai/v3/train_phs_jax.py for canonical JAX training with correct Yoshida-4 integrator. "
    "This PyTorch version is maintained for deployment/ONNX export only.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export constants from ghnn for backwards compatibility
RE_KM = 6378.137
TU_SEC = 806.8116
MU_KM = 398600.4418

def export_ghnn_to_onnx(model, output_path):
    """Export model to ONNX (wrapper for compatibility)."""
    # For PHS, we would need a more sophisticated export
    # This is a placeholder for now
    import torch
    model.eval()
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
    print(f"Model exported to {output_path}")

# Import physics constraints with learned Lagrange multipliers
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'v3'))
from physics_constraints import AugmentedLagrangianLoss as PhysicsAugmentedLagrangianLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CanonicalTrajectoryDataset(Dataset):
    """
    Dataset that normalizes high-fidelity trajectories to Canonical Units (DU, TU).
    Distance: DU = RE_KM (6378.137 km)
    Time: TU = 806.81 sec
    Velocity: DU/TU
    Acceleration: DU/TU^2
    """
    def __init__(self, high_fidelity_file: str):
        self.samples = []
        logger.info(f"Loading high-fidelity data from {high_fidelity_file}")
        
        with open(high_fidelity_file, 'r') as f:
            data = json.load(f)
            
        for sat in data:
            states = np.array(sat['states'], dtype=np.float32)
            times = np.array(sat['time_points'], dtype=np.float32)
            
            # Normalize to Canonical Units
            # states: [T, 6] -> [x, y, z, vx, vy, vz]
            q = states[:, :3] / RE_KM
            p = states[:, 3:] * (TU_SEC / RE_KM)
            normalized_states = np.concatenate([q, p], axis=-1)
            normalized_times = times / TU_SEC
            
            self.samples.append({
                'sat_id': sat['sat_id'],
                'initial_state': normalized_states[0],
                'times': normalized_times,
                'states': normalized_states
            })
        
        logger.info(f"Normalized {len(self.samples)} trajectories to DU/TU units.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch: List[Dict]) -> Dict:
    max_len = max(len(s['times']) for s in batch)
    batch_size = len(batch)
    
    x0 = torch.stack([torch.tensor(s['initial_state']) for s in batch])
    times = torch.zeros(batch_size, max_len)
    states = torch.zeros(batch_size, max_len, 6)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, s in enumerate(batch):
        n = len(s['times'])
        lengths[i] = n
        times[i, :n] = torch.tensor(s['times'])
        states[i, :n] = torch.tensor(s['states'])
        
    return {
        'initial_state': x0,
        'times': times,
        'states': states,
        'lengths': lengths
    }

# NOTE: AugmentedLagrangianLoss imported from physics_constraints.py (ai/v3/)
# Uses dual variables (μ) updated via dual ascent, NOT learned parameters.
# Formula: L(θ, μ, ρ) = L_obs(θ) + μᵀ c(θ) + (ρ/2) ‖c(θ)‖²
# Dual ascent: μ_{k+1} = μ_k + ρ · c(θ_k)
# Call loss.dual_ascent_update() after each training epoch.
AugmentedLagrangianLoss = PhysicsAugmentedLagrangianLoss

def implicit_midpoint_step(model, z, dt, max_iter=10, tol=1e-10):
    """
    Implicit midpoint rule - symplectic for general Hamiltonians.
    
    The implicit midpoint rule z_{n+1} = z_n + dt * J * grad(H((z_n+z_{n+1})/2))
    is symplectic for ALL Hamiltonian systems when the fixed-point iteration converges.
    This implementation serves as a baseline comparison to Yoshida-4.
    
    Yoshida-4 is preferred for production because:
        (1) Explicit evaluation: no fixed-point iteration, ~4x faster per step
        (2) 4th-order accuracy: O(dt^5) error vs O(dt^3) for midpoint
        (3) Modified Hamiltonian is closer to original H for neural networks
    
    Algorithm: z_{n+1} = z_n + dt * J * grad(H(z_mid))
    where z_mid = (z_n + z_{n+1}) / 2
    
    Solved via fixed-point iteration for z_mid.
    
    Args:
        model: GHNNVectorField providing dz/dt = [dH/dp, -dH/dq]
        z: State tensor [batch, 6] = [q, p]
        dt: Time step
        max_iter: Maximum fixed-point iterations
        tol: Convergence tolerance
        
    Returns:
        z_next: Next state [batch, 6]
    """
    z_n = z.clone()
    z_mid = z_n.clone()  # Initial guess: z_mid = z_n
    
    for _ in range(max_iter):
        z_mid_prev = z_mid.clone()
        
        # Compute vector field at midpoint
        dz_dt = model(z_mid)  # [dH/dp, -dH/dq]
        
        # Fixed-point update: z_mid = z_n + (dt/2) * dz_dt
        z_mid = z_n + (dt / 2.0) * dz_dt
        
        # Check convergence
        if torch.max(torch.abs(z_mid - z_mid_prev)) < tol:
            break
    
    # Final step: z_{n+1} = z_n + dt * dz_dt(z_mid)
    dz_dt_mid = model(z_mid)
    z_next = z_n + dt * dz_dt_mid
    
    return z_next


def yoshida4_step(model, q, p, z_latent, dt):
    """
    Yoshida-4 4th-order symplectic integration step with gradient checkpointing.
    
    CORRECT IMPLEMENTATION with proper q/p alternation:
        Stage 1: p = p - c1*dt*dH/dq,  q = q + d1*dt*dH/dp
        Stage 2: p = p - c2*dt*dH/dq,  q = q + d2*dt*dH/dp
        Stage 3: p = p - c3*dt*dH/dq,  q = q + d3*dt*dH/dp
        Stage 4: p = p - c4*dt*dH/dq
    
    This preserves phase-space volume under the modified Hamiltonian H̃_θ
    (Yoshida 1990, §3) with O(dt^5) error in the original H_θ.
    Uses torch.utils.checkpoint to trade compute for memory (M2 Air compatibility).
    
    Args:
        model: PortHamiltonianSystem with compute_dH_dq, compute_dH_dp methods
        q: [batch, 3] position vectors
        p: [batch, 3] momentum vectors
        z_latent: [batch, latent_dim] physics code (optional, can be None)
        dt: Time step
        
    Returns:
        q_next: [batch, 3] next position
        p_next: [batch, 3] next momentum
    """
    import torch.utils.checkpoint as checkpoint
    
    # Yoshida-4 coefficients (Forest-Ruth 4th order)
    # Yoshida, H. (1990). Construction of higher order symplectic integrators.
    # Physics Letters A, 150(5-7), 262-268. doi:10.1016/0375-9601(90)90092-3
    c1 = 0.6756035959798288
    c2 = -0.1756035959798288
    c3 = -0.1756035959798288
    c4 = 0.6756035959798288
    
    d1 = 1.3512071919596576
    d2 = -1.7024143839193153
    d3 = 1.3512071919596576
    
    # Stage 1: Momentum kick then position drift
    def get_dH_dq_stage1(q1, p1):
        return model.compute_dH_dq(q1, p1, z_latent)
    
    dH_dq1 = checkpoint.checkpoint(get_dH_dq_stage1, q, p, use_reentrant=False)
    p = p - c1 * dt * dH_dq1
    
    def get_dH_dp_stage1(q1, p1):
        return model.compute_dH_dp(q1, p1, z_latent)
    
    dH_dp1 = checkpoint.checkpoint(get_dH_dp_stage1, q, p, use_reentrant=False)
    q = q + d1 * dt * dH_dp1
    
    # Stage 2
    def get_dH_dq_stage2(q2, p2):
        return model.compute_dH_dq(q2, p2, z_latent)
    
    dH_dq2 = checkpoint.checkpoint(get_dH_dq_stage2, q, p, use_reentrant=False)
    p = p - c2 * dt * dH_dq2
    
    def get_dH_dp_stage2(q2, p2):
        return model.compute_dH_dp(q2, p2, z_latent)
    
    dH_dp2 = checkpoint.checkpoint(get_dH_dp_stage2, q, p, use_reentrant=False)
    q = q + d2 * dt * dH_dp2
    
    # Stage 3
    def get_dH_dq_stage3(q3, p3):
        return model.compute_dH_dq(q3, p3, z_latent)
    
    dH_dq3 = checkpoint.checkpoint(get_dH_dq_stage3, q, p, use_reentrant=False)
    p = p - c3 * dt * dH_dq3
    
    def get_dH_dp_stage3(q3, p3):
        return model.compute_dH_dp(q3, p3, z_latent)
    
    dH_dp3 = checkpoint.checkpoint(get_dH_dp_stage3, q, p, use_reentrant=False)
    q = q + d3 * dt * dH_dp3
    
    # Stage 4: Final momentum kick (no position drift)
    def get_dH_dq_stage4(q4, p4):
        return model.compute_dH_dq(q4, p4, z_latent)
    
    dH_dq4 = checkpoint.checkpoint(get_dH_dq_stage4, q, p, use_reentrant=False)
    p = p - c4 * dt * dH_dq4
    
    return q, p


def compute_sparse_loss(z_sequence, noise_floor=1e-5, lambda_sparse=0.01):
    """
    Enforces L1 Sparsity on the latent thrust/control vector.
    
    Prevents Micro-Thrust Hallucination: The network cannot use tiny, continuous
    latent adjustments to fit sensor noise (atmospheric scintillation). Instead,
    it must recognize true ballistic coasting (z=0) with occasional real maneuvers.
    
    Mathematical formulation:
        L_sparse = λ * mean(max(0, ||z_t||_1 - ε_noise))
    
    Where:
        - ||z_t||_1: L1 norm of latent vector at time t (sum of absolute values)
        - ε_noise: Physical deadband (sensor noise floor threshold)
        - λ: Sparsity penalty weight
    
    This is the mathematical translation of "satellites coast 99.9% of the time".
    
    Args:
        z_sequence: [batch, seq_len, latent_dim] latent vectors from SINDy inference
        noise_floor: ε_noise threshold - forces z to exactly 0 below this level
        lambda_sparse: λ weight for sparsity penalty
        
    Returns:
        loss: Scalar L1 sparsity loss with deadband
    """
    if z_sequence is None or z_sequence.numel() == 0:
        return torch.tensor(0.0, device=z_sequence.device if z_sequence is not None else 'cpu')
    
    # 1. Compute L1 norm along latent dimension: ||z_t||_1
    # Shape: [batch, seq_len]
    l1_norms = torch.norm(z_sequence, p=1, dim=-1)
    
    # 2. Apply Physical Deadband: penalize only when exceeding noise floor
    # If ||z||_1 < noise_floor, force to exactly 0 (heavy penalty for non-zero)
    # If ||z||_1 >= noise_floor, allow non-zero (penalty for excess)
    deadband_penalty = torch.relu(l1_norms - noise_floor)
    
    # 3. Average over batch and sequence
    # This encourages the network to keep z=0 whenever possible
    sparsity_loss = torch.mean(deadband_penalty)
    
    return lambda_sparse * sparsity_loss


class SymplecticAugmentedLoss(nn.Module):
    """
    Augmented Lagrangian Loss with Symplectic Jacobian Penalty.
    
    Combines:
    1. Data fidelity (MSE)
    2. Energy/momentum constraints (Augmented Lagrangian)
    3. Symplectic structure preservation (Jacobian penalty)
    
    This ensures the learned Hamiltonian is not just accurate,
    but topologically correct on the phase space manifold.
    """
    
    def __init__(self, rho_init=1.0, rho_scale=2.0, rho_min=0.1, rho_max=1000.0, mu_init=0.1,
                 lambda_symp_init=1e-5, lambda_symp_max=1e-2, symp_ramp_epochs=50,
                 lambda_sparse=0.01, noise_floor=1e-5):
        super().__init__()
        self.rho = rho_init  # Penalty parameter (adaptive per Boyd et al. 2011)
        self.rho_scale = rho_scale
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.mu = mu_init    # Dual variable step size

        # Symplectic penalty with scheduled ramp-up
        self.lambda_symp = lambda_symp_init
        self.lambda_symp_max = lambda_symp_max
        self.symp_ramp_epochs = symp_ramp_epochs
        self.current_epoch = 0

        # Dual variables for constraints — NOT nn.Parameter (updated via dual ascent, not AdamW)
        self.register_buffer('lambda_energy', torch.tensor(0.0))
        self.register_buffer('lambda_momentum', torch.tensor(0.0))
        
        # SINDy sparse regularization (prevents micro-thrust hallucination)
        self.lambda_sparse = lambda_sparse  # L1 sparsity weight
        self.noise_floor = noise_floor      # Physical deadband threshold
        
    def ramp_symplectic_weight(self):
        """Gradually increase symplectic penalty after MSE stabilizes."""
        if self.current_epoch < self.symp_ramp_epochs:
            # Linear ramp: 0 -> max over ramp_epochs
            progress = self.current_epoch / self.symp_ramp_epochs
            self.lambda_symp = self.lambda_symp_max * (progress ** 2)  # Quadratic ease-in
        else:
            self.lambda_symp = self.lambda_symp_max
        self.current_epoch += 1
        
    def forward(self, model, pred_traj, true_traj, z_initial=None, dt=None, z_sequence=None):
        """
        Compute combined loss with symplectic penalty and SINDy sparsity.
        
        Args:
            model: GHNNVectorField or PortHamiltonianSystem
            pred_traj: [batch, seq, 6] Predicted trajectory
            true_traj: [batch, seq, 6] Ground truth trajectory
            z_initial: [batch, 6] Initial states (for Jacobian computation)
            dt: Time step (for Jacobian computation)
            z_sequence: [batch, seq, latent_dim] Latent vectors from SINDy (for sparsity)
            
        Returns:
            loss_total: Combined loss
            metrics: Dictionary of loss components
        """
        # 1. Observation Loss (data fidelity)
        loss_obs = nn.functional.mse_loss(pred_traj, true_traj)
        
        # 2. Physics Constraints (Augmented Lagrangian)
        batch_size, seq_len, _ = pred_traj.shape
        flat_traj = pred_traj.view(-1, 6)
        
        q = flat_traj[:, :3]
        p = flat_traj[:, 3:6]
        
        # Energy constraint
        H = model.h_net(flat_traj).view(batch_size, seq_len)
        H_mean = H.mean(dim=1, keepdim=True)
        energy_residual = H - H_mean
        h_energy = energy_residual.pow(2).mean()
        energy_augmented = (self.lambda_energy * energy_residual.mean() + 
                           (self.rho / 2) * h_energy)
        
        # Momentum constraint
        L = torch.cross(q, p, dim=1)
        L_mean = L.mean(dim=0, keepdim=True)
        momentum_residual = L - L_mean
        h_momentum = momentum_residual.pow(2).mean()
        momentum_augmented = (self.lambda_momentum * momentum_residual.mean() +
                              (self.rho / 2) * h_momentum)
        
        # 3. Symplectic Jacobian Penalty (compute on initial states) - DEPRECATED
        # Yoshida-4 integration now provides hard symplectic constraint
        loss_symp = 0.0
        symp_residual_norm = 0.0
        
        # 4. SINDy Sparse Regularization (prevents micro-thrust hallucination)
        loss_sparse = 0.0
        z_active_fraction = 0.0
        if z_sequence is not None and self.lambda_sparse > 0:
            loss_sparse = compute_sparse_loss(
                z_sequence, 
                noise_floor=self.noise_floor,
                lambda_sparse=self.lambda_sparse
            )
            # Track what fraction of z values are "active" (above noise floor)
            z_l1 = torch.norm(z_sequence, p=1, dim=-1)
            z_active_fraction = (z_l1 > self.noise_floor).float().mean().item()
        
        # Total loss
        loss_total = loss_obs + energy_augmented + momentum_augmented + loss_sparse

        # Primal residual = combined constraint violation (for dual ascent, done per-epoch)
        primal_residual = math.sqrt(h_energy + h_momentum)

        return loss_total, {
            "obs": loss_obs,
            "energy_violation": h_energy,
            "momentum_violation": h_momentum,
            "primal_residual": primal_residual,
            "sparse_loss": loss_sparse,
            "z_active_fraction": z_active_fraction,
            "lambda_sparse": self.lambda_sparse,
            "noise_floor": self.noise_floor,
            "rho": self.rho,
            "lambda_energy": self.lambda_energy.item(),
            "lambda_momentum": self.lambda_momentum.item(),
        }

    def dual_ascent_update(self, primal_residual: float):
        """Dual ascent and adaptive rho per Boyd et al. (2011) §3.4.1.

        Primal residual = ||r||_2 (constraint violation).
        Dual residual = ||s||_2 (change in dual variables scaled by rho).
        Adapt rho if primal/dual ratio exceeds tolerance (default 10x).
        """
        lambda_e_old = self.lambda_energy.item()
        lambda_m_old = self.lambda_momentum.item()

        # Gradient ascent on Lagrangian dual
        self.lambda_energy.add_(self.mu * primal_residual)
        self.lambda_momentum.add_(self.mu * primal_residual)

        # Dual residual = change in dual variables (normalized by rho)
        dual_residual = abs(self.lambda_energy.item() - lambda_e_old) + \
                        abs(self.lambda_momentum.item() - lambda_m_old)
        if self.rho > 0:
            dual_residual /= self.rho

        # Boyd 2011 adaptive rho heuristic
        if primal_residual > 10.0 * dual_residual:
            self.rho = min(self.rho * self.rho_scale, self.rho_max)
        elif dual_residual > 10.0 * primal_residual:
            self.rho = max(self.rho / self.rho_scale, self.rho_min)


class GHNNTrainer:
    def __init__(self, model, device, lr=1e-3, use_symplectic_loss=True,
                 lambda_sparse=0.01, noise_floor=1e-5):
        """
        Sovereign GHNN Trainer with Yoshida-4 symplectic integration and SINDy sparsity.
        
        Args:
            model: PortHamiltonianSystem or GHNNVectorField
            device: torch device
            lr: Learning rate
            use_symplectic_loss: Whether to use Augmented Lagrangian loss
            lambda_sparse: L1 sparsity weight for SINDy latent regularization
            noise_floor: Physical deadband threshold (forces z to 0 below this level)
        """
        self.model = model.to(device)
        self.device = device
        self.use_symplectic_loss = use_symplectic_loss
        
        # Initialize loss criterion with SINDy sparse regularization
        self.criterion = SymplecticAugmentedLoss(
            lambda_sparse=lambda_sparse,
            noise_floor=noise_floor
        )
        logger.info(f"Using SymplecticAugmentedLoss with SINDy sparsity "
                   f"(λ={lambda_sparse}, noise_floor={noise_floor})")
        
        # Optimizer: model parameters only (dual variables are buffers, updated via dual ascent)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        epoch_primal_residual = 0.0
        primal_count = 0
        for batch in loader:
            x0 = batch['initial_state'].to(self.device).requires_grad_(True)
            true_states = batch['states'].to(self.device)
            times = batch['times'][0].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Yoshida-4 Symplectic Integration
            # Hard-coded 4th-order symplectic integrator with proper q/p alternation.
            # Preserves phase-space volume under modified Hamiltonian H̃_θ (Yoshida 1990, §3);
            # for neural H_θ the modified H̃_θ differs from H_θ by O(h^5).
            # Uses gradient checkpointing for M2 Air memory efficiency.
            
            dt = times[1] - times[0]
            current_x = x0
            preds = [current_x]
            
            # Split initial state into q (position) and p (momentum)
            q = current_x[:, :3]
            p = current_x[:, 3:6]
            
            # SINDy Latent Sequence Tracking (for sparse regularization)
            # In production: z_sequence comes from SlidingSINDy residual inference
            # z_sequence shape: [batch, seq_len, latent_dim]
            # For now: placeholder - will be populated when SINDy is integrated
            z_sequence_list = []  # Track latent vectors at each timestep
            batch_size = x0.shape[0]
            latent_dim = 64  # Default latent dimension
            
            for step in range(1, true_states.shape[1]):
                # Get z_latent from SINDy (placeholder for now)
                # In production: z_t = sindy_inference(residual_t)
                # z_t should be zero during ballistic coasting, non-zero during maneuvers
                z_t = torch.zeros(batch_size, latent_dim, device=self.device)
                z_sequence_list.append(z_t)
                
                # Yoshida-4 symplectic step with gradient checkpointing
                # Note: z_latent integration into dynamics is future work
                q, p = yoshida4_step(self.model, q, p, z_t, dt)
                
                # Recombine for storage
                current_x = torch.cat([q, p], dim=1)
                preds.append(current_x)
            
            pred_states = torch.stack(preds, dim=1)
            
            # Stack latent sequence for sparse regularization
            # Shape: [batch, seq_len, latent_dim]
            if len(z_sequence_list) > 0:
                z_sequence = torch.stack(z_sequence_list, dim=1)
            else:
                z_sequence = None
            
            # Compute loss with sparse regularization (z_sequence passed for SINDy sparsity)
            loss, metrics = self.criterion(
                self.model, pred_states, true_states, 
                z_initial=x0, dt=dt, z_sequence=z_sequence
            )
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Accumulate primal residual for epoch-end dual ascent
            if 'primal_residual' in metrics:
                epoch_primal_residual += metrics['primal_residual']
                primal_count += 1

        # Dual ascent and rho adaptation (once per epoch, per Boyd et al. 2011)
        if primal_count > 0:
            self.criterion.dual_ascent_update(epoch_primal_residual / primal_count)

        return total_loss / len(loader)

def validate_yoshida4_integrator(model, state, z_latent, steps=10000, dt=0.01):
    """
    Benchmarking script for Yoshida-4 symplectic integrator validation.
    
    Target: <1e-8 energy drift over 10000 steps (matches Sovereign Handshake σ_H/H threshold)
    Target: <1e-10 constraint violation (manifold integrity)
    
    Yoshida-4 preserves phase-space volume under the modified Hamiltonian H̃_θ
    (Yoshida 1990, §3); for neural H_θ this eliminates O(h^3) energy drift from
    penalty-based "symplectic" training.
    
    Args:
        model: PortHamiltonianSystem with compute_dH_dq/dp methods
        state: Initial state [batch, 6]
        z_latent: [batch, latent_dim] physics code (optional)
        steps: Number of integration steps
        dt: Time step
        
    Returns:
        avg_drift: Average relative energy drift
        max_drift: Maximum energy drift (worst case)
    """
    device = state.device
    model = model.to(device)
    model.eval()
    
    # Compute initial Hamiltonian
    H_initial = model.get_hamiltonian(state, z_latent).sum()
    
    drifts = []
    
    # Split state into q and p
    q = state[:, :3]
    p = state[:, 3:6]
    
    for i in range(steps):
        # Yoshida-4 symplectic step (no gradient checkpointing in eval mode)
        q, p = yoshida4_step(model, q, p, z_latent, dt)
        
        # Recombine for Hamiltonian computation
        current_state = torch.cat([q, p], dim=1)
        
        # Energy Drift: |(H_t - H_0) / H_0|
        H_t = model.get_hamiltonian(current_state, z_latent).sum()
        drift = torch.abs((H_t - H_initial) / (H_initial + 1e-10)).item()
        drifts.append(drift)
    
    avg_drift = sum(drifts) / len(drifts)
    max_drift = max(drifts)
    
    logger.info(f"Yoshida-4 Integrator Validation:")
    logger.info(f"  Average Energy Drift: {avg_drift:.2e} (target < 1e-08)")
    logger.info(f"  Max Energy Drift: {max_drift:.2e} (target < 1e-07)")
    logger.info(f"  Steps: {steps}, dt: {dt}")
    
    if avg_drift < 1e-8:
        logger.info(f"  ✅ Energy drift target PASSED (Yoshida-4 working)")
    else:
        logger.warning(f"  ❌ Energy drift target FAILED (check implementation)")
    
    if max_drift < 1e-7:
        logger.info(f"  ✅ Max drift target PASSED")
    else:
        logger.warning(f"  ❌ Max drift target FAILED")
    
    return avg_drift, max_drift


def main():
    parser = argparse.ArgumentParser(
        description='Sovereign Training Pipeline for Axiom-Zero with Yoshida-4 and SINDy Sparsity'
    )
    parser.add_argument('--data', type=str, required=True, help='Path to trajectory dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--output-dir', type=str, default='models/ghnn', help='Model output directory')
    parser.add_argument('--validate-integrator', action='store_true', help='Run Yoshida-4 integrator benchmark')
    
    # SINDy Sparse Regularization Arguments
    parser.add_argument('--lambda-sparse', type=float, default=0.01, 
                        help='L1 sparsity weight for SINDy latent regularization (default: 0.01)')
    parser.add_argument('--noise-floor', type=float, default=1e-5,
                        help='Physical deadband threshold for z vector (default: 1e-5)')
    parser.add_argument('--use-sparsity', action='store_true', default=True,
                        help='Enable SINDy sparse regularization (default: True)')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    logger.info(f"SINDy Sparse Regularization: λ={args.lambda_sparse}, noise_floor={args.noise_floor}")
    
    dataset = CanonicalTrajectoryDataset(args.data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # Use Port-Hamiltonian System by default (unified architecture)
    # This handles non-conservative dynamics (drag, thrust) via R and g
    # For conservative-only behavior, use latent_conditioning=False
    model = create_port_hamiltonian_system(
        state_dim=6,
        control_dim=3,
        latent_dim=64,
        use_latent_conditioning=False,  # Set False for conservative-only (equivalent to GHNN)
    )
    
    # Initialize trainer with SINDy sparse regularization
    trainer = GHNNTrainer(
        model, device, 
        lambda_sparse=args.lambda_sparse if args.use_sparsity else 0.0,
        noise_floor=args.noise_floor
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run integrator validation if requested
    if args.validate_integrator:
        logger.info("Running Yoshida-4 Symplectic Integrator Validation...")
        # Sample a state from dataset for validation
        sample_batch = next(iter(loader))
        sample_state = sample_batch['initial_state'][0:1].to(device)
        
        avg_drift, max_drift = validate_yoshida4_integrator(
            model, sample_state, z_latent=None, steps=10000, dt=0.01
        )
        
        # Save validation results
        with open(output_dir / 'yoshida4_validation.txt', 'w') as f:
            f.write(f"Yoshida-4 Symplectic Integrator Validation Results\n")
            f.write(f"===================================================\n")
            f.write(f"Average Energy Drift: {avg_drift:.2e}\n")
            f.write(f"Max Energy Drift: {max_drift:.2e}\n")
            f.write(f"\nTargets:\n")
            f.write(f"Energy Drift: < 1e-08 {'PASSED' if avg_drift < 1e-8 else 'FAILED'}\n")
            f.write(f"Max Drift: < 1e-07 {'PASSED' if max_drift < 1e-7 else 'FAILED'}\n")
        
        logger.info(f"Validation results saved to {output_dir / 'yoshida4_validation.txt'}")
        return  # Exit after validation
    
    logger.info("Starting Sovereign GHNN Training...")
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(loader)
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss = {loss:.6f}")
            
    # Final Export
    onnx_path = output_dir / "axiom_hnode_v1.onnx"
    export_ghnn_to_onnx(model, str(onnx_path))
    logger.info(f"Training complete. Model exported to {onnx_path}")

if __name__ == "__main__":
    main()
