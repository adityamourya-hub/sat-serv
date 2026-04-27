"""
Stochastic Port-Hamiltonian System (SPHS) Diffusion Network

: Atmospheric Jitter Correction

This module implements the state-dependent noise intensity network σ(x) for
Stochastic Port-Hamiltonian Systems using Stratonovich interpretation.

The Stochastic Physics Core:
 dx = (J-R)∇H dt + g u dt + σ(x) ∘ dW

Stratonovich interpretation preserves:
- Energy conservation properties of H
- Symplectic structure of J
- Standard chain rule (unlike Ito)

Key Innovation:
 The DiffusionNet learns a "Turbulence Map" - state-dependent noise intensity.
 - Near Karman line (100km): σ ≈ 0 (vacuum)
 - Dense atmosphere (40km): σ spikes (micro-turbulence, shockwave jitter)

This enables "Free Covariance Propagation" - no hardcoded R matrix needed.
The IMM filter's innovation covariance S_k is dynamically derived from learned diffusion.

References:
 - Stratonovich Calculus: Stratonovich (1966)
 - Stochastic Port-Hamiltonian Systems: van der Schaft
 - Milstein Scheme: Milstein (1974)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DiffusionNet(nn.Module):
    """
    Learns state-dependent noise coefficients sigma(x).

    Enforces positive semi-definiteness via Softplus activation.
    Outputs diagonal sigma for Milstein scheme (one sigma per state component).

    Architecture:
        Input: [batch, state_dim] - state vector [pos(3), vel(3)]
        Hidden: MLP with Softplus activation
        Output: [batch, state_dim] - diagonal diffusion coefficients σ_i(x)

    Physical Interpretation:
        - σ_pos(x): Positional noise (turbulence-induced position jitter)
        - σ_vel(x): Velocity noise (drag fluctuations, shockwave buffeting)
    """

    def __init__(self, state_dim: int = 6, hidden_dim: int = 64, depth: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Build MLP layers
        layers = []
        input_dim = state_dim

        for i in range(depth):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Softplus())  # Ensures positive outputs
            input_dim = hidden_dim

        # Output layer (no activation, handled by softplus in forward)
        layers.append(nn.Linear(hidden_dim, state_dim))

        self.net = nn.Sequential(*layers)

        # Initialize weights for small initial noise
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for small initial diffusion (near-zero noise)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute state-dependent diffusion coefficients σ(x).

        Args:
            x: State tensor [batch, state_dim]
               State: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]

        Returns:
            sigma: Diagonal diffusion coefficients [batch, state_dim]
                   All values are positive (enforced by softplus)
        """
        # Forward through MLP
        sigma = self.net(x)

        # Apply softplus to ensure positive semi-definite
        # σ(x) = log(1 + exp(x)) > 0
        sigma = F.softplus(sigma)

        return sigma

    def compute_diffusion_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute full diffusion matrix Σ(x) = σ(x)σ(x)^T.

        For diagonal sigma, this is:
            Σ = diag(σ_0², σ_1², ..., σ_{n-1}²)

        Args:
            x: State tensor [batch, state_dim]

        Returns:
            diffusion_matrix: [batch, state_dim, state_dim]
        """
        sigma = self.forward(x)  # [batch, state_dim]

        # Compute Σ = σ ⊙ σ (element-wise square for diagonal)
        # Then expand to full matrix
        batch_size = sigma.shape[0]
        diffusion_matrix = torch.zeros(batch_size, self.state_dim, self.state_dim,
                                        device=sigma.device, dtype=sigma.dtype)

        for i in range(self.state_dim):
            diffusion_matrix[:, i, i] = sigma[:, i] ** 2

        return diffusion_matrix


class MilsteinIntegrator:
    """
    Milstein scheme for Stochastic Port-Hamiltonian Systems.

    Provides O(dt) convergence for SDEs by accounting for the derivative
    of the noise term (Stratonovich correction).

    Milstein Update:
        x_{n+1} = x_n + f(x_n)Δt + σ(x_n)ΔW_n
                 + 0.5 * σ(x_n) * (∂σ/∂x) * (ΔW_n² - Δt)

    where:
        - f(x) = (J-R)∇H + g u (deterministic drift)
        - σ(x) = DiffusionNet(x) (state-dependent diffusion)
        - ΔW_n ~ N(0, Δt) (Wiener increment)
        - The correction term is the Stratonovich-to-Ito conversion
    """

    def __init__(self, diffusion_net: DiffusionNet):
        self.diffusion_net = diffusion_net

    def milstein_step(
        self,
        x: torch.Tensor,
        drift: torch.Tensor,
        dt: float,
        dw: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one Milstein integration step.

        Args:
            x: Current state [batch, state_dim]
            drift: Deterministic drift f(x) [batch, state_dim]
            dt: Time step
            dw: Wiener increment (optional, will sample if None) [batch, state_dim]

        Returns:
            x_next: Next state [batch, state_dim]
            sigma: Diffusion coefficients at x [batch, state_dim]
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample Wiener increment if not provided
        if dw is None:
            dw = torch.randn(batch_size, x.shape[1], device=device) * math.sqrt(dt)

        # Compute state-dependent diffusion
        sigma = self.diffusion_net(x)  # [batch, state_dim]

        # Compute derivative of sigma w.r.t x (Jacobian)
        # For diagonal sigma, this is ∂σ_i/∂x_j
        sigma_jacobian = self._compute_sigma_jacobian(x, sigma)

        # Stratonovich/Milstein correction term
        # 0.5 * σ(x) * (∂σ/∂x) * (ΔW² - Δt)
        dw_squared_minus_dt = dw * dw - dt  # [batch, state_dim]

        # For diagonal sigma: correction_i = 0.5 * σ_i * Σ_j (∂σ_i/∂x_j) * (ΔW_j² - Δt)
        correction = torch.zeros_like(x)
        for i in range(x.shape[1]):
            # Sum over j of (∂σ_i/∂x_j) * (ΔW_j² - Δt)
            jacobian_row = sigma_jacobian[:, i, :]  # [batch, state_dim]
            correction[:, i] = 0.5 * sigma[:, i] * torch.sum(jacobian_row * dw_squared_minus_dt, dim=1)

        # Milstein update
        x_next = x + drift * dt + sigma * dw + correction

        return x_next, sigma

    def _compute_sigma_jacobian(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of sigma w.r.t state using automatic differentiation.

        Args:
            x: State [batch, state_dim]
            sigma: Diffusion coefficients [batch, state_dim]

        Returns:
            jacobian: [batch, state_dim, state_dim]
                      jacobian[i, j] = ∂σ_i/∂x_j
        """
        # Enable gradient computation
        x.requires_grad = True

        # Recompute sigma with gradient tracking
        sigma_with_grad = self.diffusion_net(x)

        # Compute Jacobian
        jacobian = torch.zeros(x.shape[0], x.shape[1], x.shape[1],
                                device=x.device, dtype=x.dtype)

        for i in range(x.shape[1]):
            # Gradient of σ_i w.r.t x
            grad = torch.autograd.grad(
                outputs=sigma_with_grad[:, i],
                inputs=x,
                grad_outputs=torch.ones_like(sigma_with_grad[:, i]),
                create_graph=False,
                retain_graph=(i < x.shape[1] - 1)
            )[0]

            jacobian[:, i, :] = grad

        return jacobian


def negative_log_likelihood(
    observations: torch.Tensor,
    predicted_states: torch.Tensor,
    sigma: torch.Tensor
) -> torch.Tensor:
    """
    Compute Negative Log-Likelihood (NLL) of observations under stochastic flow.

    For Gaussian observations with state-dependent noise:
        p(y|x) = N(y; x, Σ(x))
        NLL = 0.5 * (y-x)^T Σ(x)^{-1} (y-x) + 0.5 * log|Σ(x)| + const

    This replaces MSE loss in training loop for SPHS.

    Args:
        observations: Observed states [batch, state_dim]
        predicted_states: Predicted states from model [batch, state_dim]
        sigma: Diffusion coefficients at predicted states [batch, state_dim]

    Returns:
        nll: Negative log-likelihood [batch]
    """
    # Residual
    residual = observations - predicted_states  # [batch, state_dim]

    # Diffusion matrix Σ = diag(σ²)
    sigma_squared = sigma ** 2  # [batch, state_dim]

    # Mahalanobis distance: (y-x)^T Σ^{-1} (y-x)
    # For diagonal Σ: Σ_i^{-1} = 1/σ_i²
    mahalanobis = torch.sum(residual ** 2 / (sigma_squared + 1e-8), dim=1)  # [batch]

    # Log determinant: log|Σ| = Σ log(σ_i²)
    log_det = torch.sum(torch.log(sigma_squared + 1e-8), dim=1)  # [batch]

    # NLL = 0.5 * (Mahalanobis + log_det) + const
    nll = 0.5 * (mahalanobis + log_det)

    return nll


def trace_regularization(sigma: torch.Tensor, lambda_trace: float = 0.01) -> torch.Tensor:
    """
    : Minimal Noise Regularization via Trace Penalty

    Prevents SPHS from "cheating" - using Brownian noise σ(x)dW to mask
    its inability to explain a maneuver. Forces the model to explain
    as much as possible through Hamiltonian H and SINDy Intent u before
    resorting to "Noise" to close the gap.

    Regularization term:
        L_trace = λ * Tr(σ σ^T) = λ * Σ σ_i²

    This keeps the Information Bottleneck tight and ensures SINDy
    coefficients remain the primary source of truth for Challenge (f).

    Args:
        sigma: Diffusion coefficients [batch, state_dim]
        lambda_trace: Regularization strength (default 0.01)

    Returns:
        trace_loss: Trace penalty [batch]
    """
    # Compute trace of diffusion matrix: Tr(σ σ^T) = Σ σ_i²
    trace = torch.sum(sigma ** 2, dim=1)  # [batch]

    # Regularization: L_trace = λ * trace
    trace_loss = lambda_trace * trace

    return trace_loss


def total_sphs_loss(
    observations: torch.Tensor,
    predicted_states: torch.Tensor,
    sigma: torch.Tensor,
    lambda_trace: float = 0.01
) -> torch.Tensor:
    """
    Total SPHS loss combining NLL and trace regularization.

    L_total = L_NLL + L_trace

    This ensures the model:
        1. Fits observations (via NLL)
        2. Uses minimal noise (via trace regularization)

    Args:
        observations: Observed states [batch, state_dim]
        predicted_states: Predicted states [batch, state_dim]
        sigma: Diffusion coefficients [batch, state_dim]
        lambda_trace: Trace regularization strength

    Returns:
        total_loss: Combined loss [batch]
    """
    nll = negative_log_likelihood(observations, predicted_states, sigma)
    trace_loss = trace_regularization(sigma, lambda_trace)

    return nll + trace_loss


if __name__ == "__main__":
    # Test DiffusionNet
    print("=" * 70)
    print("Stochastic Port-Hamiltonian System (SPHS) Diffusion Network")
    print("=" * 70)

    # Create network
    diffusion_net = DiffusionNet(state_dim=6, hidden_dim=64, depth=2)

    # Test forward pass
    batch_size = 10
    x = torch.randn(batch_size, 6)

    sigma = diffusion_net(x)
    print(f"\nInput state shape: {x.shape}")
    print(f"Diffusion coefficients shape: {sigma.shape}")
    print(f"Sigma range: [{sigma.min():.6f}, {sigma.max():.6f}]")
    print(f"Sigma mean: {sigma.mean():.6f}")

    # Test diffusion matrix
    diffusion_matrix = diffusion_net.compute_diffusion_matrix(x)
    print(f"\nDiffusion matrix shape: {diffusion_matrix.shape}")

    # Test Milstein integrator
    integrator = MilsteinIntegrator(diffusion_net)
    drift = torch.randn_like(x) * 0.1
    dt = 0.01

    x_next, sigma_step = integrator.milstein_step(x, drift, dt)
    print(f"\nMilstein step:")
    print(f" Input: {x.shape}")
    print(f" Drift: {drift.shape}")
    print(f" Output: {x_next.shape}")
    print(f" Sigma at step: {sigma_step.shape}")

    # Test NLL
    observations = x_next + torch.randn_like(x_next) * 0.01
    nll = negative_log_likelihood(observations, x_next, sigma_step)
    print(f"\nNegative Log-Likelihood:")
    print(f" NLL shape: {nll.shape}")
    print(f" NLL mean: {nll.mean():.6f}")

    print("\n" + "=" * 70)
    print("SPHS Diffusion Network initialized successfully")
    print("=" * 70)
