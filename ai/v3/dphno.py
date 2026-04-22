"""
Dissipative Port-Hamiltonian Neural Operator (DPHNO)

Architecture:
- H(x; θ_H): Hamiltonian network (4×128 Tanh) - conserves energy
- R(x; θ_R): Dissipation matrix (Cholesky L·Lᵀ) - captures drag/thermal
- g(x; θ_g): Input coupling MLP - maps state to control effectiveness
- J: Canonical symplectic matrix (hardcoded, geometric constraint)

Dynamics:
 ẋ = (J - R)∇H + g·u

Reference: ai/model_arch.md - HF-PGFM Dominant Architecture
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


class DPHNO(nn.Module):
 """
 Dissipative Port-Hamiltonian Neural Operator
 
 Implements the dynamics:
 ẋ = [J(x) - R(x; θ_R)] · ∇H(x; θ_H) + g(x; θ_g) · u(t)
 
 Where:
 - J is the canonical symplectic matrix (hardcoded)
 - R is a positive semi-definite dissipation matrix (learned via Cholesky factor)
 - H is the Hamiltonian (learned)
 - g is the control coupling (learned)
 - u is the control input (e.g., bank angle)
 """
 
 def __init__(self, state_dim: int = 6, hidden_dim: int = 128):
 super().__init__()
 
 self.state_dim = state_dim
 self.hidden_dim = hidden_dim
 
 # Hamiltonian network H(q,p)
 # 4×128 Tanh architecture as specified in model_arch.md
 self.hamiltonian_net = nn.Sequential(
 nn.Linear(state_dim, hidden_dim),
 nn.Tanh(),
 nn.Linear(hidden_dim, hidden_dim),
 nn.Tanh(),
 nn.Linear(hidden_dim, hidden_dim),
 nn.Tanh(),
 nn.Linear(hidden_dim, hidden_dim),
 nn.Tanh(),
 nn.Linear(hidden_dim, 1) # Scalar Hamiltonian
)
 
 # Dissipation matrix R (outputs Cholesky factor L)
 # For 6D state: R is 6×6, L is 6×6 lower triangular
 # Output dimension: 6*7/2 = 21 elements for lower triangular L
 self.dissipation_net = nn.Sequential(
 nn.Linear(state_dim, hidden_dim),
 nn.Tanh(),
 nn.Linear(hidden_dim, hidden_dim),
 nn.Tanh(),
 nn.Linear(hidden_dim, 21) # 21 elements for lower triangular L
)
 
 # Input coupling g(x)
 # Maps state to control effectiveness
 self.control_net = nn.Sequential(
 nn.Linear(state_dim, hidden_dim // 2),
 nn.Tanh(),
 nn.Linear(hidden_dim // 2, hidden_dim // 2),
 nn.Tanh(),
 nn.Linear(hidden_dim // 2, state_dim) # Maps state to control effectiveness
)
 
 # Canonical symplectic matrix J (hardcoded, not learned)
 # J = [[0, I], [-I, 0]] for 6D state [x,y,z,vx,vy,vz]
 self.register_buffer('J', torch.tensor([
 [0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 0, 1],
 [-1, 0, 0, 0, 0, 0],
 [0, -1, 0, 0, 0, 0],
 [0, 0, -1, 0, 0, 0]
 ], dtype=torch.float32))
 
 def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
 """
 Compute Hamiltonian H(x)
 
 Args:
 x: (batch, 6) state [x,y,z,vx,vy,vz] in km, km/s
 
 Returns:
 H: (batch,) scalar Hamiltonian
 """
 return self.hamiltonian_net(x).squeeze(-1)
 
 def dissipation_matrix(self, x: torch.Tensor) -> torch.Tensor:
 """
 Compute dissipation matrix R(x) = L·Lᵀ (positive semi-definite)
 
 Args:
 x: (batch, 6) state
 
 Returns:
 R: (batch, 6, 6) dissipation matrix
 """
 batch_size = x.shape[0]
 L_flat = self.dissipation_net(x) # (batch, 21)
 
 # Convert flat to lower triangular L
 L = torch.zeros(batch_size, 6, 6, device=x.device, dtype=x.dtype)
 idx = torch.tril_indices(6, 6)
 L[:, idx[0], idx[1]] = L_flat
 
 # R = L·Lᵀ (guarantees positive semi-definite)
 R = torch.bmm(L, L.transpose(1, 2))
 
 return R
 
 def control_coupling(self, x: torch.Tensor) -> torch.Tensor:
 """
 Compute control coupling g(x)
 
 Args:
 x: (batch, 6) state
 
 Returns:
 g: (batch, 6) control effectiveness vector
 """
 return self.control_net(x)
 
 def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
 """
 Compute dynamics: ẋ = (J - R)∇H + g·u
 
 Args:
 x: (batch, 6) state [x,y,z,vx,vy,vz] in km, km/s
 u: (batch, 1) control input (bank angle in radians)
 
 Returns:
 dxdt: (batch, 6) time derivative [vx,vy,vz,ax,ay,az]
 """
 batch_size = x.shape[0]
 x = x.requires_grad_(True)
 
 # Compute Hamiltonian gradient ∇H
 H = self.hamiltonian(x)
 grad_H = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
 
 # Compute dissipation matrix R
 R = self.dissipation_matrix(x)
 
 # Compute control coupling g
 g = self.control_coupling(x)
 
 # Symplectic term: J·∇H
 J_expanded = self.J.unsqueeze(0).expand(batch_size, -1, -1)
 symplectic_term = torch.bmm(J_expanded, grad_H.unsqueeze(-1)).squeeze(-1)
 
 # Dissipation term: -R·∇H
 dissipation_term = -torch.bmm(R, grad_H.unsqueeze(-1)).squeeze(-1)
 
 # Control term: g·u
 control_term = g * u
 
 # Total dynamics: (J - R)∇H + g·u
 dxdt = symplectic_term + dissipation_term + control_term
 
 return dxdt
 
 def energy_conservation_loss(self, x: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
 """
 Compute energy conservation loss (for training)
 
 In vacuum (no dissipation), Hamiltonian should be conserved:
 H(t + dt) = H(t)
 
 Args:
 x: (batch, 6) initial state
 dt: time step in seconds
 
 Returns:
 loss: scalar energy conservation loss
 """
 H_initial = self.hamiltonian(x)
 
 # Propagate one step
 u = torch.zeros(x.shape[0], 1, device=x.device) # No control
 dxdt = self.forward(x, u)
 x_next = x + dxdt * dt
 
 H_next = self.hamiltonian(x_next)
 
 # Energy drift should be zero
 energy_drift = (H_next - H_initial).abs()
 return energy_drift.mean()
 
 def to_onnx(self, path: str):
 """
 Export model to ONNX format for Go integration
 
 Args:
 path: Output ONNX file path
 """
 dummy_state = torch.randn(1, 6)
 dummy_control = torch.randn(1, 1)
 
 torch.onnx.export(
 self,
 (dummy_state, dummy_control),
 path,
 input_names=['state', 'control'],
 output_names=['dxdt'],
 dynamic_axes={
 'state': {0: 'batch'},
 'control': {0: 'batch'},
 'dxdt': {0: 'batch'}
 }
)


class DPHNOTrainer:
 """
 Training loop for DPHNO with Augmented Lagrangian constraints.
 
 : Replaces soft penalty (0.1 * energy_loss) with proper
 Augmented Lagrangian for hard physics constraint enforcement.
 """
 
 def __init__(self, model: DPHNO, lr: float = 1e-3, rho: float = 10.0):
 self.model = model
 
 # Augmented Lagrangian parameters
 self.rho = rho # Penalty parameter
 self.lambda_energy = nn.Parameter(torch.tensor(0.0)) # Lagrange multiplier (learned dual variable)
 self.mu = 0.1 # Dual variable update rate
 
 # Optimizer includes model parameters AND Lagrange multiplier for joint optimization
 self.optimizer = torch.optim.Adam(
 list(model.parameters()) + [self.lambda_energy], lr=lr
)
 
 def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
 """
 Single training step with Augmented Lagrangian constraints.
 
 Args:
 batch: Dictionary with keys:
 - 'state': (batch, 6) initial states
 - 'target_dxdt': (batch, 6) target derivatives
 - 'control': (batch, 1) control inputs
 
 Returns:
 losses: Dictionary of loss values
 """
 self.optimizer.zero_grad()
 
 x = batch['state']
 target_dxdt = batch['target_dxdt']
 u = batch['control']
 
 # Forward pass
 pred_dxdt = self.model(x, u)
 
 # Dynamics loss (MSE)
 dynamics_loss = nn.functional.mse_loss(pred_dxdt, target_dxdt)
 
 # Energy conservation constraint: c_E = H(next) - H(current) = 0
 H_initial = self.model.hamiltonian(x)
 dxdt = self.model(x, torch.zeros_like(u)) # No control for conservation check
 x_next = x + dxdt * 0.1 # Single step
 H_next = self.model.hamiltonian(x_next)
 
 c_E = H_next - H_initial # Constraint violation (should be 0)
 h_energy = c_E.pow(2).mean() # Constraint violation squared
 
 # Augmented Lagrangian: L = L_obs + lambda * c + (rho/2) * c^2
 energy_augmented = (self.lambda_energy * c_E.mean() + 
 (self.rho / 2) * h_energy)
 
 # Total loss
 total_loss = dynamics_loss + energy_augmented
 
 # Backward pass
 total_loss.backward()
 self.optimizer.step()
 
 # Update dual variable (gradient ascent on Lagrangian)
 with torch.no_grad():
 self.lambda_energy.add_(self.mu * c_E.mean().item())
 # Adaptive penalty increase if constraint violated
 if h_energy.item() > 1e-6:
 self.rho = min(self.rho * 2.0, 1000.0)
 
 return {
 'total': total_loss.item(),
 'dynamics': dynamics_loss.item(),
 'energy_violation': h_energy.item(),
 'lambda_energy': self.lambda_energy.item(),
 'rho': self.rho
 }
