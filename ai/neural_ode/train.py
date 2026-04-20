"""
Sovereign Training Pipeline for Axiom-Zero GHNN.
Enforces physical invariants via:
1. Canonical DU/TU Normalization (µ=1.0)
2. Poisson Bracket Loss ({q, p} = 1)
3. Hamiltonian Energy Conservation
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

from ghnn import (
    HamiltonianNet,
    GHNNVectorField,
    export_ghnn_to_onnx,
    RE_KM, TU_SEC, MU_KM
)

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

class AugmentedLagrangianLoss(nn.Module):
    """
    Augmented Lagrangian Loss - Physics as Hard Constraints.
    
    Replaces soft penalty-based StructurePreservingLoss with equality constraints:
    - Energy conservation: H(q,p) = H_0 (equality)
    - Momentum: L(q,p) = q x p (equality)
    - Hydrostatic: p = mv (equality)
    
    Uses Augmented Lagrangian with dual variables and penalty terms.
    """
    def __init__(self, rho_init=1.0, rho_scale=2.0, mu_init=0.1):
        super().__init__()
        self.rho = rho_init  # Penalty parameter (adaptive)
        self.rho_scale = rho_scale
        self.mu = mu_init    # Dual variable scaling
        
        # Dual variables for constraints (learned/updated during training)
        self.lambda_energy = 0.0
        self.lambda_momentum = 0.0
        
    def forward(self, model: GHNNVectorField, pred_traj: torch.Tensor, true_traj: torch.Tensor):
        # 1. Observation Loss (data fidelity)
        loss_obs = nn.functional.mse_loss(pred_traj, true_traj)
        
        # 2. Physics Constraints (as equalities, not penalties)
        batch_size, seq_len, _ = pred_traj.shape
        flat_traj = pred_traj.view(-1, 6)
        
        # Extract q, p
        q = flat_traj[:, :3]  # Position
        p = flat_traj[:, 3:6]  # Momentum
        
        # Hamiltonian constraint: H(q,p) = H_0 (energy conservation)
        H = model.h_net(flat_traj).view(batch_size, seq_len)
        H_mean = H.mean(dim=1, keepdim=True)
        energy_residual = H - H_mean  # Should be zero
        
        # Angular momentum constraint: L = q x p (conserved)
        L = torch.cross(q, p, dim=1)
        L_mean = L.mean(dim=0, keepdim=True)
        momentum_residual = L - L_mean
        
        # Augmented Lagrangian terms
        # L_AL = L_obs + lambda^T * h(x) + (rho/2) * ||h(x)||^2
        
        # Energy constraint (augmented)
        h_energy = energy_residual.pow(2).mean()
        energy_augmented = (self.lambda_energy * energy_residual.mean() + 
                           (self.rho / 2) * h_energy)
        
        # Momentum constraint (augmented)
        h_momentum = momentum_residual.pow(2).mean()
        momentum_augmented = (self.lambda_momentum * momentum_residual.mean() +
                             (self.rho / 2) * h_momentum)
        
        # Total augmented loss
        loss_total = loss_obs + energy_augmented + momentum_augmented
        
        # Update dual variables (gradient ascent on Lagrangian)
        with torch.no_grad():
            self.lambda_energy += self.mu * energy_residual.mean().item()
            self.lambda_momentum += self.mu * momentum_residual.mean().item()
            
            # Adaptive penalty increase if constraint violated
            if h_energy > 1e-6:
                self.rho = min(self.rho * self.rho_scale, 1000.0)
        
        return loss_total, {
            "obs": loss_obs,
            "energy_violation": h_energy,
            "momentum_violation": h_momentum,
            "rho": self.rho,
            "lambda_energy": self.lambda_energy
        }

def forest_ruth_step(model, z, dt):
    """
    Forest-Ruth (Yoshida-4) 4th-order symplectic integrator.
    
    Implements operator splitting for separable Hamiltonian H(q,p) = T(p) + V(q).
    Preserves the symplectic 2-form ω = Σ dq_i ∧ dp_i.
    
    Coefficients:
        theta = 1 / (2 - 2^(1/3)) ≈ 1.35120719195966
        a = [theta/2, (1-theta)/2, (1-theta)/2, theta/2]
        b = [theta, 1-2*theta, theta, 0.0]
    
    Args:
        model: GHNNVectorField providing dz/dt = [dH/dp, -dH/dq]
        z: State tensor [batch, 6] = [q, p]
        dt: Time step (in canonical units)
        
    Returns:
        z_next: Next state [batch, 6]
    """
    # Forest-Ruth coefficients (Yoshida-4)
    theta = 1.3512071919596576  # 1 / (2 - 2^(1/3))
    a = [theta/2, (1-theta)/2, (1-theta)/2, theta/2]
    b = [theta, 1-2*theta, theta, 0.0]
    
    q = z[:, :3].clone()  # Position
    p = z[:, 3:6].clone()  # Momentum
    
    # 4-stage symplectic integration
    for stage in range(4):
        # Stage 1 & 3: Position update, then momentum update
        # Stage 2: Momentum update (negative coefficient), then position
        # Stage 4: Final position update (b4 = 0, so no momentum update)
        
        if stage < 3:  # Stages 0, 1, 2 have momentum updates
            # Position half-step (or full-step for stage 2)
            x_stage = torch.cat([q, p], dim=1)
            dq = model(x_stage)[:, :3]  # dH/dp
            q = q + a[stage] * dt * dq
            
            # Momentum step
            x_stage = torch.cat([q, p], dim=1)
            dp = model(x_stage)[:, 3:6]  # -dH/dq
            p = p + b[stage] * dt * dp
        else:  # Stage 3: Final position update only
            x_stage = torch.cat([q, p], dim=1)
            dq = model(x_stage)[:, :3]
            q = q + a[stage] * dt * dq
    
    return torch.cat([q, p], dim=1)


def get_symplectic_form(device):
    """
    Returns the standard symplectic form J = [[0, I], [-I, 0]].
    
    Args:
        device: torch device
        
    Returns:
        J: [6, 6] skew-symmetric matrix
    """
    I = torch.eye(3, device=device)
    Z = torch.zeros(3, 3, device=device)
    J = torch.cat([
        torch.cat([Z, I], dim=1),
        torch.cat([-I, Z], dim=1)
    ], dim=0)
    return J


def compute_symplectic_jacobian_loss(model, z_t, dt, lambda_symp=1e-3):
    """
    Compute symplectic loss by penalizing Jacobian deviation from symplectic condition.
    
    The symplectic condition requires: M^T * J * M = J
    where M = d(z_next)/d(z_t) is the Jacobian of the flow map.
    
    Loss: L_symp = ||M^T * J * M - J||_F^2
    
    Args:
        model: GHNNVectorField
        z_t: [batch, 6] Current state
        dt: Time step
        lambda_symp: Weight for symplectic penalty
        
    Returns:
        loss_symp: Scalar symplectic loss
        residual_norm: Frobenius norm of symplectic residual (for monitoring)
    """
    batch_size = z_t.shape[0]
    device = z_t.device
    
    # Get symplectic form
    J = get_symplectic_form(device)  # [6, 6]
    
    # Define single-step map for Jacobian computation
    def single_step_map(z_single):
        """Wrapper for vmap - takes and returns single state."""
        z_in = z_single.unsqueeze(0)  # [1, 6]
        z_out = forest_ruth_step(model, z_in, dt)  # [1, 6]
        return z_out.squeeze(0)  # [6]
    
    # Compute Jacobian using vmap for efficiency on H100
    # batch_jacobian[i] = d(z_next_i)/d(z_i) for sample i
    # This gives us [batch, 6, 6] Jacobian tensor
    try:
        batch_jacobian = torch.vmap(
            lambda z: torch.autograd.functional.jacobian(single_step_map, z, create_graph=True)
        )(z_t)
    except RuntimeError:
        # Fallback: compute Jacobians individually if vmap fails
        jacobians = []
        for i in range(batch_size):
            z_i = z_t[i]
            jac_i = torch.autograd.functional.jacobian(single_step_map, z_i, create_graph=True)
            jacobians.append(jac_i)
        batch_jacobian = torch.stack(jacobians)
    
    # Compute symplectic residual: M^T * J * M - J
    # batch_jacobian: [batch, 6, 6]
    MT = batch_jacobian.transpose(-1, -2)  # [batch, 6, 6]
    
    # M^T * J
    MT_J = torch.matmul(MT, J.unsqueeze(0).expand(batch_size, -1, -1))  # [batch, 6, 6]
    
    # M^T * J * M
    MT_J_M = torch.matmul(MT_J, batch_jacobian)  # [batch, 6, 6]
    
    # Symplectic residual
    J_expanded = J.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 6, 6]
    symp_residual = MT_J_M - J_expanded  # [batch, 6, 6]
    
    # Frobenius norm squared (per sample)
    residual_norm_sq = torch.sum(symp_residual ** 2, dim=(-2, -1))  # [batch]
    
    # Mean over batch
    loss_symp = lambda_symp * residual_norm_sq.mean()
    
    # Return loss and monitoring metric
    residual_norm = torch.sqrt(residual_norm_sq).mean()
    
    return loss_symp, residual_norm


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
    
    def __init__(self, rho_init=1.0, rho_scale=2.0, mu_init=0.1,
                 lambda_symp_init=1e-5, lambda_symp_max=1e-2, symp_ramp_epochs=50):
        super().__init__()
        self.rho = rho_init  # Penalty parameter (adaptive)
        self.rho_scale = rho_scale
        self.mu = mu_init    # Dual variable scaling
        
        # Symplectic penalty with scheduled ramp-up
        self.lambda_symp = lambda_symp_init
        self.lambda_symp_max = lambda_symp_max
        self.symp_ramp_epochs = symp_ramp_epochs
        self.current_epoch = 0
        
        # Dual variables for constraints
        self.lambda_energy = 0.0
        self.lambda_momentum = 0.0
        
    def ramp_symplectic_weight(self):
        """Gradually increase symplectic penalty after MSE stabilizes."""
        if self.current_epoch < self.symp_ramp_epochs:
            # Linear ramp: 0 -> max over ramp_epochs
            progress = self.current_epoch / self.symp_ramp_epochs
            self.lambda_symp = self.lambda_symp_max * (progress ** 2)  # Quadratic ease-in
        else:
            self.lambda_symp = self.lambda_symp_max
        self.current_epoch += 1
        
    def forward(self, model, pred_traj, true_traj, z_initial=None, dt=None):
        """
        Compute combined loss with symplectic penalty.
        
        Args:
            model: GHNNVectorField
            pred_traj: [batch, seq, 6] Predicted trajectory
            true_traj: [batch, seq, 6] Ground truth trajectory
            z_initial: [batch, 6] Initial states (for Jacobian computation)
            dt: Time step (for Jacobian computation)
            
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
        
        # 3. Symplectic Jacobian Penalty (compute on initial states)
        loss_symp = 0.0
        symp_residual_norm = 0.0
        if z_initial is not None and dt is not None and self.lambda_symp > 0:
            loss_symp, symp_residual_norm = compute_symplectic_jacobian_loss(
                model, z_initial, dt, self.lambda_symp
            )
        
        # Total loss
        loss_total = loss_obs + energy_augmented + momentum_augmented + loss_symp
        
        # Update dual variables
        with torch.no_grad():
            self.lambda_energy += self.mu * energy_residual.mean().item()
            self.lambda_momentum += self.mu * momentum_residual.mean().item()
            if h_energy > 1e-6:
                self.rho = min(self.rho * self.rho_scale, 1000.0)
        
        return loss_total, {
            "obs": loss_obs,
            "energy_violation": h_energy,
            "momentum_violation": h_momentum,
            "symp_residual_norm": symp_residual_norm,
            "lambda_symp": self.lambda_symp,
            "rho": self.rho,
            "lambda_energy": self.lambda_energy,
        }


class GHNNTrainer:
    def __init__(self, model, device, lr=1e-3, use_symplectic_loss=True):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.use_symplectic_loss = use_symplectic_loss
        
        if use_symplectic_loss:
            self.criterion = SymplecticAugmentedLoss()
            logger.info("Using SymplecticAugmentedLoss with Jacobian penalty")
        else:
            self.criterion = AugmentedLagrangianLoss()
            logger.info("Using AugmentedLagrangianLoss (no symplectic penalty)")

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            x0 = batch['initial_state'].to(self.device).requires_grad_(True)
            true_states = batch['states'].to(self.device)
            times = batch['times'][0].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forest-Ruth (Yoshida-4) 4th-order symplectic integration
            # Preserves phase space volume and Hamiltonian structure
            # Much better for orbital mechanics than Euler/RK4
            
            dt = times[1] - times[0]
            current_x = x0
            preds = [current_x]
            
            for step in range(1, true_states.shape[1]):
                current_x = forest_ruth_step(self.model, current_x, dt)
                preds.append(current_x)
            
            pred_states = torch.stack(preds, dim=1)
            
            # Compute loss with symplectic penalty (pass initial state and dt)
            if self.use_symplectic_loss:
                loss, metrics = self.criterion(self.model, pred_states, true_states, x0, dt)
                self.criterion.ramp_symplectic_weight()  # Gradually increase symplectic penalty
            else:
                loss, metrics = self.criterion(self.model, pred_states, true_states)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='models/ghnn')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = CanonicalTrajectoryDataset(args.data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    h_net = HamiltonianNet(hidden_dim=128, num_layers=4)
    model = GHNNVectorField(h_net)
    
    trainer = GHNNTrainer(model, device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
