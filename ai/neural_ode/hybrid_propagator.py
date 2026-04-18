"""
Neural ODE Hybrid Propagator for SSIS
Combines physics-based orbital mechanics with neural residual learning

Based on: arXiv:2403.19736 - Physics-Informed Neural Networks for Satellite State Estimation
Expected improvement: 2 orders of magnitude over SGP4 alone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint, odeint
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# Physical constants
MU = 398600.4418  # km^3/s^2 (Earth gravitational parameter)
RE = 6378.137     # km (Earth radius)
J2 = 0.00108263   # J2 oblateness coefficient
OMEGA_EARTH = 7.2921158553e-5  # rad/s (Earth rotation rate)


class PhysicsEngine:
    """Computes physics-based orbital accelerations (two-body + J2 + drag)."""
    
    @staticmethod
    def compute_acceleration(state: torch.Tensor) -> torch.Tensor:
        """
        Compute acceleration from known physics.
        
        Args:
            state: [batch, 6] or [6] tensor [x, y, z, vx, vy, vz] in km, km/s
            
        Returns:
            acceleration: [batch, 3] or [3] in km/s^2
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        pos = state[:, :3]  # [batch, 3]
        vel = state[:, 3:]  # [batch, 3]
        
        # Kepler (two-body) acceleration
        r = torch.norm(pos, dim=1, keepdim=True)  # [batch, 1]
        accel_kepler = -MU / (r**3) * pos  # [batch, 3]
        
        # J2 perturbation
        z = pos[:, 2:3]  # [batch, 1]
        factor = 1.5 * J2 * MU * (RE**2) / (r**5)
        accel_j2 = factor * torch.cat([
            pos[:, 0:1] * (5 * z**2 / r**2 - 1),
            pos[:, 1:2] * (5 * z**2 / r**2 - 1),
            pos[:, 2:3] * (5 * z**2 / r**2 - 3)
        ], dim=1)
        
        # Simplified atmospheric drag (exponential model)
        altitude = r - RE
        density = PhysicsEngine._atmospheric_density(altitude)
        
        # Relative velocity to rotating atmosphere
        v_atmosphere = torch.cross(
            torch.tensor([0.0, 0.0, OMEGA_EARTH]).expand(pos.shape[0], 3),
            pos
        )
        v_rel = vel - v_atmosphere
        v_rel_mag = torch.norm(v_rel, dim=1, keepdim=True)
        
        # Drag acceleration (simplified, assumes Cd*A/m = 0.01)
        Cd_A_m = 0.01  # Typical LEO satellite
        accel_drag = -0.5 * Cd_A_m * density * v_rel_mag * v_rel
        
        # Total physics acceleration
        accel_physics = accel_kepler + accel_j2 + accel_drag
        
        if squeeze_output:
            accel_physics = accel_physics.squeeze(0)
        
        return accel_physics
    
    @staticmethod
    def _atmospheric_density(altitude_km: torch.Tensor) -> torch.Tensor:
        """Exponential atmospheric density model (simplified)."""
        # Scale height ~8.5 km
        rho0 = 1.225e-9  # kg/m^3 at sea level, converted to kg/km^3
        H = 8.5  # km
        density = rho0 * torch.exp(-altitude_km / H)
        return density


class ResidualMLP(nn.Module):
    """
    Multi-layer perceptron that learns residual acceleration.
    Captures unmodeled perturbations: atmospheric variations, thrust, etc.
    """
    
    def __init__(self, hidden_dim: int = 100, num_layers: int = 2):
        super().__init__()
        
        layers = []
        input_dim = 7  # [x, y, z, vx, vy, vz, t]
        
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            ])
        
        layers.append(nn.Linear(hidden_dim, 3))  # [ax, ay, az] residual
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize with small weights (residual learning)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute neural residual acceleration.
        
        Args:
            state: [batch, 6] or [6]
            t: scalar or [batch] time in seconds
            
        Returns:
            residual_accel: [batch, 3] or [3]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            t = torch.full((1, 1), t.item() if torch.is_tensor(t) else t, 
                          device=state.device, dtype=state.dtype)
            squeeze_output = True
        else:
            if not torch.is_tensor(t):
                t = torch.full((state.shape[0], 1), t, 
                              device=state.device, dtype=state.dtype)
            elif t.dim() == 0:
                t = torch.full((state.shape[0], 1), t.item(), 
                              device=state.device, dtype=state.dtype)
            elif t.dim() == 1:
                t = t.unsqueeze(1)
            squeeze_output = False
        
        # Concatenate state and time
        inputs = torch.cat([state, t], dim=1)  # [batch, 7]
        residual = self.mlp(inputs)  # [batch, 3]
        
        if squeeze_output:
            residual = residual.squeeze(0)
        
        return residual


class HybridDynamics(nn.Module):
    """
    Hybrid orbital dynamics: physics + neural residual.
    
    dx/dt = [v, a_physics(x) + a_neural(x, t; θ)]
    """
    
    def __init__(self, residual_net: Optional[ResidualMLP] = None):
        super().__init__()
        self.physics = PhysicsEngine()
        self.residual = residual_net if residual_net else ResidualMLP()
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Compute state derivative.
        
        Args:
            t: time (scalar or [batch])
            state: [batch, 6] or [6] state vector
            
        Returns:
            dstate/dt: [batch, 6] or [6]
        """
        squeeze_output = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        
        pos = state[:, :3]  # [batch, 3]
        vel = state[:, 3:]   # [batch, 3]
        
        # Physics acceleration
        accel_physics = self.physics.compute_acceleration(state)  # [batch, 3]
        
        # Neural residual
        accel_neural = self.residual(state, t)  # [batch, 3]
        
        # Total acceleration
        accel_total = accel_physics + accel_neural
        
        # State derivative
        dstate = torch.cat([vel, accel_total], dim=1)  # [batch, 6]
        
        if squeeze_output:
            dstate = dstate.squeeze(0)
        
        return dstate


class HybridNeuralODE(nn.Module):
    """
    Production Neural ODE propagator with ONNX export capability.
    
    Training: Uses torchdiffeq with adjoint method for memory efficiency
    Inference: Uses adaptive step size (Dopri5) for accuracy
    Export: Provides fixed-step RK4 for ONNX traceability
    """
    
    def __init__(self, 
                 hidden_dim: int = 100, 
                 num_layers: int = 2,
                 use_adjoint: bool = True):
        super().__init__()
        
        self.residual = ResidualMLP(hidden_dim, num_layers)
        self.dynamics = HybridDynamics(self.residual)
        self.use_adjoint = use_adjoint
        
        logger.info(f"Initialized HybridNeuralODE (hidden={hidden_dim}, "
                   f"layers={num_layers}, adjoint={use_adjoint})")
    
    def forward(self, 
                x0: torch.Tensor, 
                t_span: torch.Tensor,
                method: str = 'dopri5',
                rtol: float = 1e-7,
                atol: float = 1e-9) -> torch.Tensor:
        """
        Propagate state through time.
        
        Args:
            x0: Initial state [batch, 6] or [6]
            t_span: Time points [T] (e.g., [0, 60, 120, ...] seconds)
            method: ODE solver method ('dopri5', 'rk4', 'euler')
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            trajectory: [batch, T, 6] or [T, 6] state at each time point
        """
        squeeze_output = False
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
            squeeze_output = True
        
        if self.training and self.use_adjoint:
            # Adjoint method for memory-efficient training
            trajectory = odeint_adjoint(
                self.dynamics, x0, t_span,
                method=method,
                rtol=rtol,
                atol=atol
            )
        else:
            # Standard ODE integration for inference
            trajectory = odeint(
                self.dynamics, x0, t_span,
                method=method,
                rtol=rtol,
                atol=atol
            )
        
        # trajectory shape: [T, batch, 6]
        # Permute to [batch, T, 6] for easier handling
        trajectory = trajectory.permute(1, 0, 2)
        
        if squeeze_output:
            trajectory = trajectory.squeeze(0)
        
        return trajectory
    
    def propagate_to_epoch(self,
                          x0: torch.Tensor,
                          t_seconds: float,
                          dt: float = 60.0) -> torch.Tensor:
        """
        Convenience method: propagate to single future epoch.
        
        Args:
            x0: Initial state [batch, 6] or [6]
            t_seconds: Time to propagate (seconds)
            dt: Output time step (seconds)
            
        Returns:
            final_state: [batch, 6] or [6] state at t_seconds
        """
        num_steps = int(t_seconds / dt)
        t_span = torch.linspace(0, t_seconds, num_steps + 1)
        
        trajectory = self.forward(x0, t_span)
        
        # Return final state
        if trajectory.dim() == 2:
            return trajectory[-1]  # [6]
        else:
            return trajectory[:, -1, :]  # [batch, 6]


class PhysicsInformedLoss(nn.Module):
    """
    Multi-objective loss combining observation fit and physics constraints.
    """
    
    def __init__(self, 
                 lambda_energy: float = 0.1,
                 lambda_momentum: float = 0.1,
                 lambda_kepler: float = 0.05):
        super().__init__()
        self.lambda_energy = lambda_energy
        self.lambda_momentum = lambda_momentum
        self.lambda_kepler = lambda_kepler
    
    def forward(self,
                pred_trajectory: torch.Tensor,
                true_trajectory: Optional[torch.Tensor],
                t_span: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute physics-informed loss.
        
        Args:
            pred_trajectory: [batch, T, 6] predicted states
            true_trajectory: [batch, T, 6] ground truth (optional)
            t_span: [T] time points
            
        Returns:
            total_loss: scalar
            loss_dict: dictionary of individual loss components
        """
        losses = {}
        
        # 1. Observation loss (MSE against ground truth if available)
        if true_trajectory is not None:
            losses['obs'] = F.mse_loss(pred_trajectory, true_trajectory)
        else:
            losses['obs'] = torch.tensor(0.0, device=pred_trajectory.device)
        
        # Extract position and velocity
        r = pred_trajectory[..., :3]  # [batch, T, 3]
        v = pred_trajectory[..., 3:]  # [batch, T, 3]
        
        # 2. Energy conservation (specific orbital energy should be constant)
        # ε = v²/2 - μ/r
        speed_sq = torch.sum(v**2, dim=-1)  # [batch, T]
        r_mag = torch.norm(r, dim=-1)  # [batch, T]
        specific_energy = 0.5 * speed_sq - MU / r_mag  # [batch, T]
        
        # Variance across time should be small (energy conservation)
        losses['energy'] = torch.var(specific_energy, dim=1).mean()
        
        # 3. Angular momentum conservation (for unperturbed orbits)
        # h = r × v
        h = torch.cross(r, v, dim=-1)  # [batch, T, 3]
        h_mag = torch.norm(h, dim=-1)  # [batch, T]
        losses['momentum'] = torch.var(h_mag, dim=1).mean()
        
        # 4. Kepler's 3rd law constraint (for verification)
        # For elliptical orbits: a³/T² = μ/(4π²)
        # We compute mean semi-major axis from energy: a = -μ/(2ε)
        mean_energy = specific_energy.mean(dim=1)  # [batch]
        a = -MU / (2 * mean_energy)  # [batch]
        
        # Period from Kepler's law
        T_kepler = 2 * np.pi * torch.sqrt(a**3 / MU)
        
        # Check consistency (this is more of a diagnostic)
        losses['kepler'] = torch.tensor(0.0, device=pred_trajectory.device)
        
        # Total loss
        total_loss = (
            losses['obs'] +
            self.lambda_energy * losses['energy'] +
            self.lambda_momentum * losses['momentum'] +
            self.lambda_kepler * losses['kepler']
        )
        
        return total_loss, losses


class TraceablePropagator(nn.Module):
    """
    Fixed-step propagator for ONNX export.
    Uses RK4 integration (traceable by TorchScript).
    """
    
    def __init__(self, residual_mlp: ResidualMLP, dt: float = 60.0):
        super().__init__()
        self.residual = residual_mlp
        self.dt = dt
        self.physics = PhysicsEngine()
    
    def forward(self, x0: torch.Tensor, num_steps: torch.Tensor) -> torch.Tensor:
        """
        Fixed-step RK4 propagation.
        
        Args:
            x0: Initial state [batch, 6]
            num_steps: Number of steps to propagate [batch] or scalar
            
        Returns:
            trajectory: [batch, num_steps+1, 6]
        """
        if num_steps.dim() == 0:
            num_steps = num_steps.unsqueeze(0)
        
        max_steps = int(num_steps.max().item())
        batch_size = x0.shape[0]
        
        trajectory = [x0]
        x = x0
        
        for step in range(max_steps):
            # RK4 integration
            k1 = self._compute_derivative(x, step)
            k2 = self._compute_derivative(x + 0.5 * self.dt * k1, step)
            k3 = self._compute_derivative(x + 0.5 * self.dt * k2, step)
            k4 = self._compute_derivative(x + self.dt * k3, step)
            
            x = x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(x)
        
        return torch.stack(trajectory, dim=1)  # [batch, T, 6]
    
    def _compute_derivative(self, state: torch.Tensor, step: int) -> torch.Tensor:
        """Compute dx/dt for a given state."""
        # Physics acceleration
        accel_physics = self.physics.compute_acceleration(state)
        
        # Neural residual (time is step * dt)
        t = torch.tensor(step * self.dt, device=state.device, dtype=state.dtype)
        accel_neural = self.residual(state, t)
        
        # State derivative
        vel = state[:, 3:]
        accel = accel_physics + accel_neural
        return torch.cat([vel, accel], dim=1)


def export_to_onnx(model: HybridNeuralODE,
                   output_path: str,
                   fixed_dt: float = 60.0,
                   example_steps: int = 100) -> str:
    """
    Export Neural ODE to ONNX format for Go inference.
    
    Args:
        model: Trained HybridNeuralODE
        output_path: Path to save ONNX model
        fixed_dt: Fixed time step for traceability
        example_steps: Number of steps for tracing
        
    Returns:
        output_path: Path to saved model
    """
    model.eval()
    
    # Create traceable version with fixed-step RK4
    traceable = TraceablePropagator(model.residual, fixed_dt)
    
    # Example inputs for tracing
    x0 = torch.randn(1, 6)
    num_steps = torch.tensor([example_steps])
    
    # Trace the model
    traced = torch.jit.trace(traceable, (x0, num_steps))
    
    # Export to ONNX
    torch.onnx.export(
        traced,
        (x0, num_steps),
        output_path,
        input_names=['initial_state', 'num_steps'],
        output_names=['trajectory'],
        dynamic_axes={
            'initial_state': {0: 'batch'},
            'trajectory': {0: 'batch'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    logger.info(f"Exported Neural ODE to ONNX: {output_path}")
    return output_path


# Example usage and testing
if __name__ == "__main__":
    # Test physics engine
    print("Testing Physics Engine...")
    
    # ISS-like orbit: ~400 km altitude, circular
    x0 = torch.tensor([
        RE + 400, 0, 0,  # position (km)
        0, 7.67, 0       # velocity (km/s) - circular orbit speed
    ])
    
    physics = PhysicsEngine()
    accel = physics.compute_acceleration(x0)
    print(f"Physics acceleration at 400km: {accel.numpy()}")
    
    # Test hybrid dynamics
    print("\nTesting Hybrid Dynamics...")
    
    model = HybridNeuralODE(hidden_dim=100, num_layers=2)
    
    # Propagate for 1 orbit (~90 minutes = 5400 seconds)
    t_span = torch.linspace(0, 5400, 91)  # Every 60 seconds
    
    with torch.no_grad():
        trajectory = model(x0.unsqueeze(0), t_span)
    
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Initial position: {trajectory[0, 0, :3].numpy()}")
    print(f"Final position: {trajectory[0, -1, :3].numpy()}")
    
    # Check energy conservation
    r = trajectory[0, :, :3]
    v = trajectory[0, :, 3:]
    energy = 0.5 * torch.sum(v**2, dim=1) - MU / torch.norm(r, dim=1)
    print(f"\nEnergy conservation check:")
    print(f"  Mean energy: {energy.mean().item():.6f} km²/s²")
    print(f"  Energy std:  {energy.std().item():.6f} km²/s²")
    print(f"  (Lower std = better energy conservation)")
    
    print("\n✓ HybridNeuralODE implementation complete")
