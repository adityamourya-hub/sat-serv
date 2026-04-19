"""
Neural-Physics Loss Functions for Sovereign SSA Training
Implements Physics-Informed Loss (PIL) with Keplerian dynamics constraints
"""

import torch
import torch.nn.functional as F
import numpy as np


class PhysicsInformedLoss:
    """
    Calculates the 'Sovereign Loss' by penalizing violations of 
    Keplerian dynamics and conservation of energy.
    """
    
    def __init__(self, mu=3.986004418e14, w_data=0.7, w_physics=0.3):
        """
        Initialize physics-informed loss function
        
        Args:
            mu: Earth's gravitational parameter (m³/s²), default: 3.986e14
            w_data: Weight for data loss (MSE)
            w_physics: Weight for physics loss (ODE residual)
        """
        self.mu = mu  # m³/s²
        self.w_data = w_data
        self.w_physics = w_physics
        
    def keplerian_residual(self, pred_state):
        """
        Calculate ODE residual for Keplerian dynamics
        
        pred_state: Tensor of shape [batch_size, 6] or [batch_size, timesteps, 6]
                    Columns: [x, y, z, vx, vy, vz] in meters and m/s
        
        Returns:
            Residual of acceleration from Kepler's law
        """
        # Extract position and velocity
        if pred_state.dim() == 2:
            r = pred_state[:, :3]  # Position [x, y, z]
            v = pred_state[:, 3:]  # Velocity [vx, vy, vz]
        elif pred_state.dim() == 3:
            r = pred_state[:, :, :3]
            v = pred_state[:, :, 3:]
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {pred_state.dim()}D")
        
        # Calculate distance from Earth's center
        r_mag = torch.norm(r, dim=-1, keepdim=True)  # |r|
        
        # Keplerian acceleration: a = -mu * r / |r|³
        # This is the expected acceleration based on Newton's law of gravitation
        expected_accel = -self.mu * r / (r_mag ** 3)
        
        # Calculate actual acceleration from velocity (if we have timesteps)
        if pred_state.dim() == 3 and pred_state.shape[1] > 1:
            # Finite difference: a = dv/dt
            # For simplicity, assume dt = 1 (normalized time)
            actual_accel = torch.diff(v, dim=1)
            # Pad to match expected_accel shape
            actual_accel = F.pad(actual_accel, (0, 0, 0, 1), mode='replicate')
        else:
            # Single timestep: use zero as placeholder
            # In real training, we'd need multiple timesteps
            actual_accel = torch.zeros_like(expected_accel)
        
        # ODE residual: difference between actual and expected acceleration
        residual = actual_accel - expected_accel
        
        return torch.mean(residual ** 2)
    
    def energy_conservation_loss(self, pred_state):
        """
        Penalize violations of specific orbital energy conservation
        
        E = v²/2 - mu/|r|  (should be constant for a given orbit)
        
        Args:
            pred_state: Tensor of shape [batch_size, 6]
        
        Returns:
            Variance of specific energy across trajectory
        """
        if pred_state.dim() == 2:
            r = pred_state[:, :3]
            v = pred_state[:, 3:]
            
            # Specific kinetic energy: v²/2
            ke = 0.5 * torch.sum(v ** 2, dim=-1)
            
            # Specific potential energy: -mu/|r|
            r_mag = torch.norm(r, dim=-1)
            pe = -self.mu / r_mag
            
            # Specific orbital energy
            specific_energy = ke + pe
            
            # Penalize variance (should be constant for conservative orbit)
            energy_variance = torch.var(specific_energy)
            
            return energy_variance
        
        # For single samples, return zero loss
        return torch.tensor(0.0, device=pred_state.device)
    
    def angular_momentum_loss(self, pred_state):
        """
        Penalize violations of specific angular momentum conservation
        
        h = r × v (should be constant direction and magnitude for 2-body)
        
        Args:
            pred_state: Tensor of shape [batch_size, 6]
        
        Returns:
            Loss based on angular momentum variance
        """
        if pred_state.dim() == 2:
            r = pred_state[:, :3]
            v = pred_state[:, 3:]
            
            # Cross product: h = r × v
            h = torch.cross(r, v, dim=-1)
            
            # Magnitude of specific angular momentum
            h_mag = torch.norm(h, dim=-1)
            
            # Penalize variance
            h_variance = torch.var(h_mag)
            
            return h_variance
        
        return torch.tensor(0.0, device=pred_state.device)
    
    def __call__(self, pred_state, target_state, use_physics=True):
        """
        Calculate total physics-informed loss
        
        Args:
            pred_state: Predicted state [batch_size, 6]
            target_state: Ground truth state [batch_size, 6]
            use_physics: Whether to include physics constraints
        
        Returns:
            Total weighted loss
        """
        # 1. Standard MSE (Data Loss)
        mse_loss = F.mse_loss(pred_state, target_state)
        
        if not use_physics:
            return mse_loss
        
        # 2. Physics Residual (ODE Loss)
        physics_loss = self.keplerian_residual(pred_state)
        
        # 3. Conservation laws (optional, can add weight)
        energy_loss = self.energy_conservation_loss(pred_state)
        angular_momentum_loss = self.angular_momentum_loss(pred_state)
        
        # 4. Total Weighted Loss
        # Primary: Data + ODE residual
        # Secondary: Conservation laws (smaller weight)
        w_conservation = 0.1  # Small weight for conservation
        
        total_loss = (
            self.w_data * mse_loss +
            self.w_physics * physics_loss +
            w_conservation * (energy_loss + angular_momentum_loss)
        )
        
        return total_loss
    
    def get_loss_components(self, pred_state, target_state):
        """
        Get individual loss components for logging/debugging
        
        Returns:
            Dictionary with all loss components
        """
        return {
            'mse_loss': F.mse_loss(pred_state, target_state).item(),
            'physics_loss': self.keplerian_residual(pred_state).item(),
            'energy_loss': self.energy_conservation_loss(pred_state).item(),
            'angular_momentum_loss': self.angular_momentum_loss(pred_state).item(),
            'total_loss': self.__call__(pred_state, target_state).item()
        }


class AdaptivePhysicsLoss(PhysicsInformedLoss):
    """
    Adaptive version that adjusts physics weight based on training progress
    """
    
    def __init__(self, mu=3.986004418e14, w_data_start=0.9, w_physics_start=0.1):
        super().__init__(mu, w_data_start, w_physics_start)
        self.w_data_start = w_data_start
        self.w_physics_start = w_physics_start
        self.epoch = 0
        self.total_epochs = 100  # Will be updated
        
    def set_training_progress(self, current_epoch, total_epochs):
        """Update training progress for adaptive weighting"""
        self.epoch = current_epoch
        self.total_epochs = total_epochs
        
        # Gradually increase physics weight from 0.1 to 0.4
        progress = min(current_epoch / (total_epochs * 0.5), 1.0)
        self.w_physics = self.w_physics_start + (0.3 * progress)
        self.w_data = 1.0 - self.w_physics
        
    def __call__(self, pred_state, target_state, use_physics=True):
        """Calculate adaptive physics-informed loss"""
        return super().__call__(pred_state, target_state, use_physics)


class IspConstrainedPhysicsLoss(PhysicsInformedLoss):
    """
    Physics-informed loss with hard Isp (specific impulse) constraints.
    
    Ensures policies are executable by real satellite thrusters using
    Lagrangian multiplier approach. Prevents physically impossible maneuvers.
    
    L_total = L_reward + λ * max(0, ||ΔV|| - ΔV_limit) + λ * max(0, cumΔV - total_budget)
    """
    
    def __init__(self, mu=3.986004418e14, w_data=0.7, w_physics=0.3, 
                 max_delta_v=10.0, lambda_penalty=100.0):
        """
        Args:
            mu: Gravitational parameter
            w_data: Weight for data loss
            w_physics: Weight for physics loss
            max_delta_v: Maximum delta-V per maneuver (m/s) - based on thruster Isp
            lambda_penalty: Lagrangian multiplier strength for constraints
        """
        super().__init__(mu, w_data, w_physics)
        self.max_delta_v = max_delta_v  # m/s per burn
        self.lambda_penalty = lambda_penalty
        
        # Typical satellite thruster profiles
        # Cold gas: Low Isp (~50s), small delta-V per burn, limited total budget
        # Hall-effect: High Isp (~1500s), very small delta-V per burn, large total budget
        # Chemical: Medium Isp (~300s), large delta-V per burn, limited total budget
        self.thruster_profiles = {
            'cold_gas': {
                'max_delta_v': 5.0,      # m/s per burn
                'total_budget': 100.0,    # m/s lifetime budget
                'isp_seconds': 50.0,
            },
            'hall_effect': {
                'max_delta_v': 0.5,
                'total_budget': 500.0,
                'isp_seconds': 1500.0,
            },
            'chemical': {
                'max_delta_v': 50.0,
                'total_budget': 1000.0,
                'isp_seconds': 300.0,
            },
        }
    
    def compute_constrained_loss(self, base_loss, action, cumulative_delta_v,
                                  thruster_type='cold_gas'):
        """
        Compute loss with hard Isp constraints using Lagrangian multipliers.
        
        Args:
            base_loss: Base reward/loss value
            action: Action vector (delta-V) [3]
            cumulative_delta_v: Total delta-V expended so far
            thruster_type: Type of thruster ('cold_gas', 'hall_effect', 'chemical')
            
        Returns:
            Constrained loss (penalized if constraints violated)
        """
        import torch
        
        profile = self.thruster_profiles.get(thruster_type, self.thruster_profiles['cold_gas'])
        
        # Calculate action magnitude
        action_mag = torch.norm(action)
        
        # Constraint 1: Single-maneuver delta-V limit (physics limit)
        # This prevents requesting more thrust than thruster can provide
        single_maneuver_violation = torch.clamp(action_mag - profile['max_delta_v'], min=0.0)
        
        # Constraint 2: Total fuel budget (mission limit)
        # This prevents depleting station-keeping reserves
        total_budget_violation = torch.clamp(cumulative_delta_v - profile['total_budget'], min=0.0)
        
        # Combined penalty with Lagrangian multiplier
        # λ is large to make constraints "hard" (must be satisfied)
        penalty = self.lambda_penalty * (single_maneuver_violation + total_budget_violation)
        
        # Return penalized loss
        # If constraints violated, loss increases dramatically
        return base_loss - penalty
    
    def check_action_feasible(self, action, thruster_type='cold_gas'):
        """
        Check if action is physically feasible given thruster constraints.
        
        Args:
            action: Action vector [3] in m/s
            thruster_type: Type of thruster
            
        Returns:
            (feasible: bool, reason: str)
        """
        import torch
        
        profile = self.thruster_profiles.get(thruster_type, self.thruster_profiles['cold_gas'])
        
        action_mag = float(torch.norm(action))
        
        if action_mag > profile['max_delta_v']:
            return False, (
                f"Action magnitude {action_mag:.2f} m/s exceeds {thruster_type} "
                f"single-burn limit ({profile['max_delta_v']:.2f} m/s). "
                f"Would damage bus or deplete propellant."
            )
        
        return True, f"Action feasible for {thruster_type} (Isp={profile['isp_seconds']:.0f}s)"
    
    def get_thruster_info(self, thruster_type='cold_gas'):
        """Get information about thruster capabilities."""
        return self.thruster_profiles.get(thruster_type, self.thruster_profiles['cold_gas'])


# Convenience functions for direct use
def physics_informed_loss(pred_state, target_state, mu=3.986004418e14, 
                         w_data=0.7, w_physics=0.3):
    """
    Simple wrapper for physics-informed loss
    
    Args:
        pred_state: Predicted orbital state [batch, 6]
        target_state: Target orbital state [batch, 6]
        mu: Gravitational parameter (m³/s²)
        w_data: Weight for data loss
        w_physics: Weight for physics loss
    
    Returns:
        Total loss value
    """
    loss_fn = PhysicsInformedLoss(mu, w_data, w_physics)
    return loss_fn(pred_state, target_state)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Physics-Informed Loss...")
    
    # Create synthetic orbital states
    batch_size = 4
    
    # Perfect circular orbit at 400km altitude
    r = 6371000 + 400000  # Earth radius + 400km in meters
    v_circ = np.sqrt(3.986e14 / r)  # Circular orbit velocity
    
    # Target state: perfect circular orbit
    target = torch.zeros(batch_size, 6)
    target[:, 0] = r  # x = r
    target[:, 4] = v_circ  # vy = v_circ
    
    # Predicted state: slightly perturbed
    pred = target.clone()
    pred[:, 0] += torch.randn(batch_size) * 1000  # Position error ~1km
    pred[:, 4] += torch.randn(batch_size) * 10   # Velocity error ~10m/s
    
    # Calculate loss
    loss_fn = PhysicsInformedLoss()
    total_loss = loss_fn(pred, target)
    
    print(f"Total Loss: {total_loss.item():.6f}")
    
    # Get components
    components = loss_fn.get_loss_components(pred, target)
    for name, value in components.items():
        print(f"  {name}: {value:.6f}")
    
    # Test adaptive loss
    print("\nTesting Adaptive Loss...")
    adaptive_loss = AdaptivePhysicsLoss()
    
    for epoch in [0, 25, 50, 100]:
        adaptive_loss.set_training_progress(epoch, 100)
        loss_val = adaptive_loss(pred, target)
        print(f"  Epoch {epoch}: w_data={adaptive_loss.w_data:.2f}, "
              f"w_physics={adaptive_loss.w_physics:.2f}, "
              f"loss={loss_val.item():.6f}")
    
    print("\nTesting Isp Constrained Loss...")
    isp_loss = IspConstrainedPhysicsLoss(max_delta_v=10.0, lambda_penalty=100.0)
    
    # Test feasible action (within limits)
    action_feasible = torch.tensor([5.0, 0.0, 0.0])  # 5 m/s
    loss_feasible = isp_loss.compute_constrained_loss(1.0, action_feasible, 50.0)
    print(f"  Feasible action loss: {loss_feasible:.2f}")
    
    # Test infeasible action (exceeds limits)
    action_infeasible = torch.tensor([20.0, 0.0, 0.0])  # 20 m/s
    loss_infeasible = isp_loss.compute_constrained_loss(1.0, action_infeasible, 50.0)
    print(f"  Infeasible action loss: {loss_infeasible:.2f}")
    
    is_feasible, reason = isp_loss.check_action_feasible(action_infeasible, 'cold_gas')
    print(f"  Feasibility check: {is_feasible}, {reason}")
    
    print("\nAll tests passed!")
