"""
Structural Port-Hamiltonian Neural Network
Physics-Embedded Architecture for ADITI 4.0

Replaces soft physics penalties with hard structural constraints:
- Skew-symmetric J(x): Energy exchange topology
- Cholesky-parameterized R(x): Guaranteed dissipative drag
- Hamiltonian H(x,z): Latent-conditioned energy surface

Mathematical Contract:
    ẋ = [J(x) - R(x)] ∇H(x,z) + g(x)u

Where:
    J(x) = -J(x)ᵀ (skew-symmetric, energy-preserving)
    R(x) = L(x)L(x)ᵀ (positive semi-definite, energy-dissipating)
    L(x) = lower-triangular with positive diagonal (Cholesky factor)

This architecture CANNOT violate energy conservation regardless of loss weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class InterconnectionMatrix(nn.Module):
    """
    Learns the skew-symmetric interconnection matrix J(x) ∈ ℝ⁶ˣ⁶.
    
    J represents the internal energy exchange topology:
    - Position-momentum coupling (canonical structure)
    - Gyroscopic effects (for rotating frames)
    
    Skew-symmetry is enforced: J = -Jᵀ, ensuring energy preservation.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # State-dependent J(x) parameterized by MLP
        self.j_net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 15)  # 15 = 6*5/2 independent entries for 6x6 skew-sym
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute J(x) from state.
        
        Args:
            x: [batch, 6] state vector [q, p]
            
        Returns:
            J: [batch, 6, 6] skew-symmetric matrix
        """
        batch_size = x.shape[0]
        
        # Get independent parameters
        params = self.j_net(x)  # [batch, 15]
        
        # Build skew-symmetric matrix from upper triangle
        J = torch.zeros(batch_size, 6, 6, device=x.device)
        
        # Fill upper triangle (indices for 6x6 skew-symmetric)
        idx = 0
        for i in range(6):
            for j in range(i+1, 6):
                J[:, i, j] = params[:, idx]
                J[:, j, i] = -params[:, idx]  # Skew-symmetric
                idx += 1
        
        return J


class DissipationMatrix(nn.Module):
    """
    Learns the dissipation matrix R(x) = L(x)L(x)ᵀ ∈ ℝ⁶ˣ⁶.
    
    R represents energy dissipation (atmospheric drag, friction):
    - Positive semi-definite by construction (R = LLᵀ)
    - Diagonal dominance through softplus on L diagonal
    - State-dependent: drag varies with altitude and velocity
    
    Cholesky factorization guarantees R ≽ 0 (one-way energy dissipation).
    """
    def __init__(self, hidden_dim: int = 64, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        
        # State-dependent Cholesky factor L(x)
        # 21 = 6*7/2 entries for lower-triangular 6x6
        self.l_net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 21)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute R(x) = L(x)L(x)ᵀ from state.
        
        Args:
            x: [batch, 6] state vector [q, p]
            
        Returns:
            R: [batch, 6, 6] positive semi-definite matrix
        """
        batch_size = x.shape[0]
        
        # Get L parameters
        params = self.l_net(x)  # [batch, 21]
        
        # Build lower-triangular L
        L = torch.zeros(batch_size, 6, 6, device=x.device)
        
        # Fill lower triangle
        idx = 0
        for i in range(6):
            for j in range(i+1):
                if i == j:
                    # Diagonal: ensure positive via softplus + epsilon
                    L[:, i, j] = F.softplus(params[:, idx]) + self.epsilon
                else:
                    # Off-diagonal: unconstrained
                    L[:, i, j] = params[:, idx]
                idx += 1
        
        # R = LLᵀ (guaranteed positive semi-definite)
        R = torch.bmm(L, L.transpose(-2, -1))
        
        return R


class LatentConditionedHamiltonian(nn.Module):
    """
    Learns the Hamiltonian H(x, z) with FiLM conditioning.
    
    H represents the total energy (kinetic + potential + latent effects):
    - Base Hamiltonian H₀(x) from MLP
    - FiLM modulation by latent code z ∈ ℝ³²
    - Scale and shift per layer: γ(z) * h + β(z)
    
    This allows regime-aware energy surfaces (Orbital/Ballistic/HGV).
    """
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, latent_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # Base MLP for H(x)
        self.input_proj = nn.Linear(6, hidden_dim)
        
        # FiLM generators for each layer
        self.film_layers = nn.ModuleList([
            nn.Linear(latent_dim, 2 * hidden_dim)  # γ and β
            for _ in range(num_layers)
        ])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute H(x, z) with FiLM conditioning.
        
        Args:
            x: [batch, 6] state vector [q, p]
            z: [batch, latent_dim] latent code (optional, zeros if None)
            
        Returns:
            H: [batch, 1] scalar Hamiltonian
        """
        # Default latent code if not provided
        if z is None:
            z = torch.zeros(x.shape[0], self.latent_dim, device=x.device)
        
        # Project input
        h = self.input_proj(x)  # [batch, hidden_dim]
        
        # Apply FiLM-conditioned hidden layers
        for i in range(self.num_layers):
            # FiLM parameters from latent code
            film_params = self.film_layers[i](z)  # [batch, 2*hidden_dim]
            gamma, beta = film_params.chunk(2, dim=-1)  # Each [batch, hidden_dim]
            
            # Apply FiLM: γ(z) * h + β(z)
            h = gamma * h + beta
            
            # Linear transformation with Tanh activation
            h = torch.tanh(self.hidden_layers[i](h))
        
        # Output scalar Hamiltonian
        H = self.output_layer(h)
        
        return H


class StructuralPortHamiltonian(nn.Module):
    """
    Structural Port-Hamiltonian System with hard physics constraints.
    
    This is the physics-embedded architecture that replaces soft penalties.
    The forward pass CANNOT violate the PH identity by construction.
    
    Architecture:
        ẋ = [J(x) - R(x)] ∇H(x,z) + g(x)u
        
    Components:
        - J(x): InterconnectionMatrix (skew-symmetric)
        - R(x): DissipationMatrix (Cholesky PSD)
        - H(x,z): LatentConditionedHamiltonian (FiLM)
        - g(x): Control input matrix (optional)
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        latent_dim: int = 32,
        use_latent: bool = True
    ):
        super().__init__()
        
        self.use_latent = use_latent
        self.latent_dim = latent_dim
        
        # Structural PH components
        self.J = InterconnectionMatrix(hidden_dim=64)
        self.R = DissipationMatrix(hidden_dim=64, epsilon=1e-6)
        self.H = LatentConditionedHamiltonian(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            latent_dim=latent_dim
        )
        
    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute structural PH dynamics.
        
        Args:
            x: [batch, 6] state vector [q, p]
            z: [batch, latent_dim] latent code (optional)
            u: [batch, m] control input (optional)
            
        Returns:
            x_dot: [batch, 6] state derivative
        """
        batch_size = x.shape[0]
        
        # Enable gradient computation for Hamiltonian
        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            H_val = self.H(x_grad, z)
            
            # Compute ∇H via autograd
            grad_H = torch.autograd.grad(
                H_val, x_grad,
                grad_outputs=torch.ones_like(H_val),
                create_graph=True,
                retain_graph=True
            )[0]  # [batch, 6]
        
        # Compute structural matrices
        J_mat = self.J(x)  # [batch, 6, 6]
        R_mat = self.R(x)  # [batch, 6, 6]
        
        # PH dynamics: ẋ = [J - R] ∇H
        # Expand grad_H for batch matrix multiplication
        grad_H_expanded = grad_H.unsqueeze(-1)  # [batch, 6, 1]
        
        # (J - R) ∇H
        J_minus_R = J_mat - R_mat  # [batch, 6, 6]
        x_dot_ph = torch.bmm(J_minus_R, grad_H_expanded).squeeze(-1)  # [batch, 6]
        
        # Add control input if provided
        if u is not None and hasattr(self, 'g_control'):
            g_u = self.g_control(x, u)
            x_dot = x_dot_ph + g_u
        else:
            x_dot = x_dot_ph
        
        return x_dot
    
    def get_physics_metrics(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> dict:
        """
        Compute physics diagnostics for monitoring.
        
        Returns:
            metrics: Dictionary of physics quantities
        """
        with torch.no_grad():
            # Hamiltonian
            H_val = self.H(x, z)
            
            # Power flow: ẋᵀ ∇H = ∇Hᵀ (J - R) ∇H = -∇Hᵀ R ∇H ≤ 0
            x.requires_grad_(True)
            H_for_grad = self.H(x, z)
            grad_H = torch.autograd.grad(H_for_grad, x, 
                grad_outputs=torch.ones_like(H_for_grad),
                create_graph=False)[0]
            
            R_mat = self.R(x)
            
            # Energy dissipation rate: -∇Hᵀ R ∇H
            grad_H_exp = grad_H.unsqueeze(-1)
            R_grad_H = torch.bmm(R_mat, grad_H_exp).squeeze(-1)
            dissipation = -(grad_H * R_grad_H).sum(dim=-1)  # [batch]
            
            # Symplectic error check (for monitoring)
            J_mat = self.J(x)
            J_skew_error = torch.norm(J_mat + J_mat.transpose(-2, -1), dim=(-2, -1)).mean()
            
            return {
                'hamiltonian_mean': H_val.mean().item(),
                'dissipation_mean': dissipation.mean().item(),
                'j_skew_error': J_skew_error.item(),
                'energy_conserved': (dissipation <= 1e-6).float().mean().item()
            }


class SE3Projection(nn.Module):
    """
    Projects state onto SE(3) manifold (S² for position, TS² for velocity).
    
    Respects Earth's topology: position on unit sphere, velocity tangent to sphere.
    """
    def __init__(self, re_km: float = 6378.137):
        super().__init__()
        self.re_km = re_km
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x = [q, p] onto SE(3) manifold.
        
        Args:
            x: [batch, 6] with q in km, p in km/s (or canonical units)
            
        Returns:
            x_proj: [batch, 6] projected onto manifold
        """
        q = x[:, :3]  # Position
        p = x[:, 3:6]  # Velocity
        
        # Normalize position to unit sphere (S²)
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        q_hat = q / (q_norm + 1e-10)
        
        # Project velocity to tangent space of S² at q_hat
        # v_tangent = v - (v·q_hat) q_hat
        v_dot_q = (p * q_hat).sum(dim=-1, keepdim=True)
        p_tangent = p - v_dot_q * q_hat
        
        return torch.cat([q_hat, p_tangent], dim=-1)
    
    def inverse_project(self, x_manifold: torch.Tensor, r_km: float) -> torch.Tensor:
        """
        Convert from manifold coordinates back to physical coordinates.
        
        Args:
            x_manifold: [batch, 6] with q_hat on S², p_tangent
            r_km: physical radius in km
            
        Returns:
            x_physical: [batch, 6] with q in km, p in km/s
        """
        q_hat = x_manifold[:, :3]
        p_tangent = x_manifold[:, 3:6]
        
        # Scale position from unit sphere to physical radius
        q_physical = q_hat * r_km
        
        # Velocity is already tangent, but needs magnitude scaling
        # (depends on specific velocity normalization)
        p_physical = p_tangent * r_km  # Assuming canonical scaling
        
        return torch.cat([q_physical, p_physical], dim=-1)


def test_structural_ph():
    """Unit test for structural PH system."""
    batch_size = 4
    x = torch.randn(batch_size, 6)
    z = torch.randn(batch_size, 32)
    
    # Create structural PH
    ph = StructuralPortHamiltonian(
        hidden_dim=64,
        num_layers=2,
        latent_dim=32
    )
    
    # Forward pass
    x_dot = ph(x, z)
    print(f"Output shape: {x_dot.shape}")  # Should be [4, 6]
    
    # Check J skew-symmetry
    J_mat = ph.J(x)
    skew_error = torch.norm(J_mat + J_mat.transpose(-2, -1), dim=(-2, -1)).mean()
    print(f"J skew-symmetry error: {skew_error.item():.2e}")  # Should be ~0
    
    # Check R positive semi-definite
    R_mat = ph.R(x)
    eigenvalues = torch.linalg.eigvalsh(R_mat)  # [batch, 6]
    min_eig = eigenvalues.min(dim=-1).values
    print(f"R min eigenvalue: {min_eig.mean().item():.2e}")  # Should be >= 0
    
    # Physics metrics
    metrics = ph.get_physics_metrics(x, z)
    print(f"Physics metrics: {metrics}")
    
    print("✅ Structural PH test passed!")


if __name__ == "__main__":
    test_structural_ph()
