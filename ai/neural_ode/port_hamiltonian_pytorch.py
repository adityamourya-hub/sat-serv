"""
Port-Hamiltonian System (PyTorch Port)

PyTorch implementation of the Port-Hamiltonian System from ai/v3/port_hamiltonian.py
This unifies the PyTorch training pipeline on PHS instead of the conservative-only GHNN.

Mathematical Framework:
    dz/dt = [J(z) - R(z)] ∇H(z) + g(z)u
    
    where:
        - J(z): Interconnection matrix (symplectic, conservative)
        - R(z): Dissipation matrix (PSD via Cholesky: R = LL^T)
        - H(z): Hamiltonian (energy function)
        - g(z): Control coupling (port)
        - u: Control input (thrust)

Key Innovation:
    Dissipation matrix R is enforced PSD via Cholesky factorization.
    This guarantees the model cannot "create" energy via friction -
    it can only dissipate (drag) or inject (thrust) through controlled ports.

Architecture:
    - H_net: 4×128 Tanh MLP (Hamiltonian/energy function)
    - R_net: 2×128 MLP → L (lower triangular) → R = LL^T (PSD dissipation)
    - g_net: 2×128 MLP (control coupling for thrust)
    - Uses implicit_midpoint_step for symplectic integration (non-separable)

Special Case:
    GHNN (conservative-only) is the special case R=0, g=0
    PHS generalizes to include dissipation and control ports
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PortHamiltonianSystem(nn.Module):
    """
    Port-Hamiltonian System for orbital dynamics with non-conservative forces.
    
    PyTorch port of ai/v3/port_hamiltonian.py PortHamiltonianSystem.
    
    The dissipation matrix R is guaranteed PSD via Cholesky parameterization.
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        control_dim: int = 3,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        use_latent_conditioning: bool = True,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        self.use_latent_conditioning = use_latent_conditioning
        
        # iteration_count for periodic GPU cache cleanup during training
        self.iteration_count = 0
        
        # Hamiltonian network: H(x, z) → energy
        # Input: [state, latent] if conditioning, else [state]
        h_input_dim = state_dim + latent_dim if use_latent_conditioning else state_dim
        self.H_net = nn.Sequential(
            nn.Linear(h_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # Scalar Hamiltonian
        )
        
        # Dissipation network: x → L (lower triangular) → R = LL^T (PSD)
        # Softplus ensures R>0 (strictly positive dissipation) with smooth gradients
        r_input_dim = state_dim + latent_dim if use_latent_conditioning else state_dim
        self.R_net = nn.Sequential(
            nn.Linear(r_input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, state_dim * state_dim),
        )
        
        # Control coupling network: x → G(z) (port matrix)
        g_input_dim = state_dim + latent_dim if use_latent_conditioning else state_dim
        self.g_net = nn.Sequential(
            nn.Linear(g_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim * control_dim),
        )
        
        # Latent embedding (if using conditioning)
        if use_latent_conditioning:
            self.latent_embed = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.Tanh(),
            )
        
        # Initialize with small weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with small weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def get_hamiltonian(self, x: torch.Tensor, z_latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Hamiltonian (energy) H(x, z), conditioned on latent code.
        
        Args:
            x: [batch, state_dim] state vector
            z_latent: [batch, latent_dim] physics code (zeros if None)
            
        Returns:
            H: [batch, 1] energy value
        """
        if self.use_latent_conditioning:
            if z_latent is None:
                z_latent = torch.zeros(x.shape[0], self.latent_dim, device=x.device)
            z_embedded = self.latent_embed(z_latent)
            x_input = torch.cat([x, z_embedded], dim=1)
        else:
            x_input = x
        
        return self.H_net(x_input)
    
    def get_dissipation_matrix(self, x: torch.Tensor, z_latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Positive Semi-Definite dissipation matrix R(x, z).
        
        Uses Cholesky parameterization: R = LL^T (guaranteed PSD).
        
        Args:
            x: [batch, state_dim] state vector
            z_latent: [batch, latent_dim] physics code (optional)
            
        Returns:
            R: [batch, state_dim, state_dim] PSD dissipation matrix
        """
        if self.use_latent_conditioning:
            if z_latent is None:
                z_latent = torch.zeros(x.shape[0], self.latent_dim, device=x.device)
            z_embedded = self.latent_embed(z_latent)
            x_input = torch.cat([x, z_embedded], dim=1)
        else:
            x_input = x
        
        # Get raw network output
        L_flat = self.R_net(x_input)
        
        # Reshape to square matrix
        batch_size = x.shape[0]
        L = L_flat.view(batch_size, self.state_dim, self.state_dim)
        
        # Force lower triangular structure (Cholesky factor)
        L = torch.tril(L)
        
        # Apply softplus to diagonal to ensure strict positivity of dissipation matrix R
        diag_indices = torch.arange(self.state_dim, device=x.device)
        diag_softplus = F.softplus(L[:, diag_indices, diag_indices]) + 1e-6
        L[:, diag_indices, diag_indices] = diag_softplus
        
        # R = LL^T (guaranteed PSD)
        R = torch.bmm(L, L.transpose(1, 2))
        
        return R
    
    def get_interconnection_matrix(self, device: torch.device) -> torch.Tensor:
        """
        Get symplectic interconnection matrix J.
        
        J = [[0, I], [-I, 0]] represents the conservative
        Hamiltonian structure (Poisson bracket).
        
        Args:
            device: torch device
            
        Returns:
            J: [state_dim, state_dim] constant symplectic matrix
        """
        half = self.state_dim // 2  # 3 for orbital state
        
        # Top-right: identity (ensure on correct device)
        I = torch.eye(half, device=device)
        Z = torch.zeros(half, half, device=device)
        
        # Full J matrix
        J_top = torch.cat([Z, I], dim=1)
        J_bottom = torch.cat([-I, Z], dim=1)
        J = torch.cat([J_top, J_bottom], dim=0)
        
        # Ensure J is on the correct device
        J = J.to(device)
        
        return J
    
    def get_control_coupling(self, x: torch.Tensor, z_latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get control coupling matrix G(x, z) (the "Port").
        
        Maps control input u to state dynamics:
            z_dot += G(x, z) @ u
        
        Args:
            x: [batch, state_dim] state vector
            z_latent: [batch, latent_dim] physics code (optional)
            
        Returns:
            G: [batch, state_dim, control_dim] control coupling matrix
        """
        if self.use_latent_conditioning:
            if z_latent is None:
                z_latent = torch.zeros(x.shape[0], self.latent_dim, device=x.device)
            z_embedded = self.latent_embed(z_latent)
            x_input = torch.cat([x, z_embedded], dim=1)
        else:
            x_input = x
        
        G_flat = self.g_net(x_input)
        batch_size = x.shape[0]
        G = G_flat.view(batch_size, self.state_dim, self.control_dim)
        return G
    
    def dynamics(
        self,
        x: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        z_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Port-Hamiltonian dynamics: dz/dt = [J - R]∇H + Gu
        
        Args:
            x: [batch, state_dim] current state
            u: [batch, control_dim] control input (thrust), optional
            z_latent: [batch, latent_dim] physics code (optional)
            
        Returns:
            dz_dt: [batch, state_dim] time derivative of state
        """
        # 1. Compute gradient of Hamiltonian
        x_grad = x.detach().requires_grad_(True)
        H = self.get_hamiltonian(x_grad, z_latent)
        
        # Use PyTorch's built-in training flag for autograd graph retention
        if self.training:
            # Training: retain graph for higher-order derivatives
            grad_H = torch.autograd.grad(
                H.sum(), x_grad,
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
                H.sum(), x_grad,
                create_graph=False,
                retain_graph=False
            )[0]
        
        # NaN guards: numerical gradient fallback for Hamiltonian autograd failures
        if torch.isnan(grad_H).any():
            print("[WARNING] NaN detected in Hamiltonian gradient, using fallback")
            # Fallback: numerical gradient via finite differences
            eps = 1e-6
            grad_H_numerical = torch.zeros_like(x_grad)
            for i in range(x_grad.shape[1]):
                x_plus = x_grad.clone()
                x_plus[:, i] += eps
                H_plus = self.get_hamiltonian(x_plus, z_latent)
                grad_H_numerical[:, i] = (H_plus - H) / eps
            grad_H = grad_H_numerical
        
        if torch.isinf(grad_H).any():
            print("[WARNING] Inf detected in Hamiltonian gradient, clipping")
            grad_H = torch.clamp(grad_H, -1e6, 1e6)
        
        # Gradient clipping to prevent exploding gradients
        grad_H = torch.nn.utils.clip_grad_norm_(grad_H, max_norm=1.0)
        
        # 2. Get structure matrices
        J = self.get_interconnection_matrix(x.device)
        R = self.get_dissipation_matrix(x, z_latent)
        
        # 3. Conservative + Dissipative dynamics
        # [J - R] @ ∇H
        # Expand J to batch for broadcasting
        J_batch = J.unsqueeze(0).expand(x.shape[0], -1, -1)
        dz_dt = torch.bmm(J_batch - R, grad_H.unsqueeze(-1)).squeeze(-1)
        
        # 4. Add control input if provided
        if u is not None:
            G = self.get_control_coupling(x, z_latent)
            dz_dt = dz_dt + torch.bmm(G, u.unsqueeze(-1)).squeeze(-1)
        
        return dz_dt
    
    def forward(self, x: torch.Tensor, z_latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for compatibility with existing training code.
        
        Returns the vector field dz/dt (same as dynamics with u=None).
        This makes PHS a drop-in replacement for GHNNVectorField.
        """
        return self.dynamics(x, u=None, z_latent=z_latent)
    
    def compute_dH_dq(self, q: torch.Tensor, p: torch.Tensor, 
                      z_latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute position gradient ∂H/∂q for symplectic integration.
        
        Args:
            q: [batch, 3] position vectors
            p: [batch, 3] momentum vectors  
            z_latent: [batch, latent_dim] physics code (optional)
            
        Returns:
            dH_dq: [batch, 3] gradient w.r.t. position
        """
        x = torch.cat([q, p], dim=1)
        x_grad = x.detach().requires_grad_(True)
        H = self.get_hamiltonian(x_grad, z_latent)
        
        grad_H = torch.autograd.grad(
            H.sum(), x_grad,
            create_graph=self.training,
            retain_graph=self.training
        )[0]
        
        return grad_H[:, :3]
    
    def compute_dH_dp(self, q: torch.Tensor, p: torch.Tensor,
                        z_latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute momentum gradient ∂H/∂p for symplectic integration.
        
        Args:
            q: [batch, 3] position vectors
            p: [batch, 3] momentum vectors
            z_latent: [batch, latent_dim] physics code (optional)
            
        Returns:
            dH_dp: [batch, 3] gradient w.r.t. momentum
        """
        x = torch.cat([q, p], dim=1)
        x_grad = x.detach().requires_grad_(True)
        H = self.get_hamiltonian(x_grad, z_latent)
        
        grad_H = torch.autograd.grad(
            H.sum(), x_grad,
            create_graph=self.training,
            retain_graph=self.training
        )[0]
        
        return grad_H[:, 3:]


class GHNNVectorField(nn.Module):
    """
    DEPRECATED: Conservative-only GHNN (special case of PHS with R=0, g=0)
    
    This is the old implementation that only handles conservative dynamics.
    Use PortHamiltonianSystem instead for full dissipative dynamics.
    
    Kept for backwards compatibility. GHNN is equivalent to PHS with:
        - R_net disabled (R = 0)
        - g_net disabled (g = 0)
        - No latent conditioning
    """
    
    def __init__(self, h_net: nn.Module):
        super().__init__()
        self.h_net = h_net
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hamiltonian dynamics via automatic differentiation.
        Enforces symplectic geometry: z_dot = J * grad(H)
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            H = self.h_net(x)
            
            # Compute dH/dx (Jacobian of scalar H is the gradient)
            grad_H = torch.autograd.grad(
                H, x, 
                grad_outputs=torch.ones_like(H),
                create_graph=True,
                retain_graph=True
            )[0]
            
        # Hamiltonian Equations:
        # q_dot =  dH/dp (velocity components)
        # p_dot = -dH/dq (acceleration components)
        dH_dq = grad_H[:, :3]
        dH_dp = grad_H[:, 3:]
        
        return torch.cat([dH_dp, -dH_dq], dim=1)


def create_port_hamiltonian_system(
    state_dim: int = 6,
    control_dim: int = 3,
    latent_dim: int = 64,
    use_latent_conditioning: bool = True,
) -> PortHamiltonianSystem:
    """
    Factory function to create a Port-Hamiltonian System.
    
    This is the recommended default for training.
    
    Args:
        state_dim: Dimension of state space (default 6 for orbital)
        control_dim: Dimension of control input (default 3 for thrust)
        latent_dim: Dimension of latent code for regime conditioning
        use_latent_conditioning: Whether to use latent code conditioning
        
    Returns:
        PortHamiltonianSystem instance
    """
    return PortHamiltonianSystem(
        state_dim=state_dim,
        control_dim=control_dim,
        latent_dim=latent_dim,
        use_latent_conditioning=use_latent_conditioning,
    )
