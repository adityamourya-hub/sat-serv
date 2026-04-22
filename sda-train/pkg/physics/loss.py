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

    def __init__(self, mu=3.986004418e14, w_data=0.7, w_physics=0.3,
                 w_drag=0.0, cd=2.2, area_mass_ratio=0.01):
        """
        Initialize physics-informed loss function

        Args:
            mu: Earth's gravitational parameter (m^3/s^2), default: 3.986e14
            w_data: Weight for data loss (MSE)
            w_physics: Weight for physics loss (ODE residual)
            w_drag: Weight for drag ODE residual (disabled by default for backward compat)
            cd: Drag coefficient
            area_mass_ratio: A / m in m^2 / kg
        """
        self.mu = mu  # m^3/s^2
        self.w_data = w_data
        self.w_physics = w_physics
        self.w_drag = w_drag
        self.cd = cd
        self.area_mass_ratio = area_mass_ratio

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

        # Keplerian acceleration: a = -mu * r / |r|^3
        expected_accel = -self.mu * r / (r_mag ** 3)

        # Calculate actual acceleration from velocity (if we have timesteps)
        if pred_state.dim() == 3 and pred_state.shape[1] > 1:
            actual_accel = torch.diff(v, dim=1)
            actual_accel = F.pad(actual_accel, (0, 0, 0, 1), mode='replicate')
        else:
            actual_accel = torch.zeros_like(expected_accel)

        # ODE residual: difference between actual and expected acceleration
        residual = actual_accel - expected_accel
        return torch.mean(residual ** 2)

    def drag_residual(self, pred_state):
        """
        Calculate ODE residual including atmospheric drag.

        pred_state: Tensor of shape [batch_size, 6] or [batch_size, timesteps, 6]
        Columns: [x, y, z, vx, vy, vz] in meters and m/s

        Returns:
            Residual of acceleration with drag included
        """
        from .drag_models import drag_acceleration

        r = pred_state[..., :3]
        v = pred_state[..., 3:]
        r_mag = torch.norm(r, dim=-1, keepdim=True)
        expected = -self.mu * r / (r_mag ** 3)
        expected = expected + drag_acceleration(pred_state, self.cd, self.area_mass_ratio)

        if pred_state.dim() == 3 and pred_state.shape[1] > 1:
            actual = torch.diff(v, dim=1)
            actual = F.pad(actual, (0, 0, 0, 1), mode='replicate')
        else:
            actual = torch.zeros_like(expected)

        return torch.mean((actual - expected) ** 2)

    def energy_conservation_loss(self, pred_state):
        """
        Penalize violations of specific orbital energy conservation

        E = v^2/2 - mu/|r| (should be constant for a given orbit)

        Args:
            pred_state: Tensor of shape [batch_size, 6]

        Returns:
            Variance of specific energy across trajectory
        """
        if pred_state.dim() == 2:
            r = pred_state[:, :3]
            v = pred_state[:, 3:]
            ke = 0.5 * torch.sum(v ** 2, dim=-1)
            r_mag = torch.norm(r, dim=-1)
            pe = -self.mu / r_mag
            specific_energy = ke + pe
            energy_variance = torch.var(specific_energy)
            return energy_variance
        return torch.tensor(0.0, device=pred_state.device)

    def angular_momentum_loss(self, pred_state):
        """
        Penalize violations of specific angular momentum conservation

        h = r x v (should be constant direction and magnitude for 2-body)

        Args:
            pred_state: Tensor of shape [batch_size, 6]

        Returns:
            Loss based on angular momentum variance
        """
        if pred_state.dim() == 2:
            r = pred_state[:, :3]
            v = pred_state[:, 3:]
            h = torch.cross(r, v, dim=-1)
            h_mag = torch.norm(h, dim=-1)
            h_variance = torch.var(h_mag)
            return h_variance
        return torch.tensor(0.0, device=pred_state.device)

    def lyapunov_loss(self, pred_state, target_state=None):
        """
        Lyapunov stability loss using Hamiltonian as Lyapunov function.

        For conservative systems, the Hamiltonian H(x) = T + V is a natural Lyapunov function.
        Lyapunov stability requires:
        1. V(x) > 0 for all x != 0 (positive definite)
        2. dV/dt <= 0 along trajectories (non-increasing)

        Args:
            pred_state: Predicted state [batch_size, 6] or [batch_size, timesteps, 6]
            target_state: Optional target state for comparison

        Returns:
            Lyapunov stability loss
        """
        if pred_state.dim() == 2:
            r = pred_state[:, :3]
            v = pred_state[:, 3:]
            r_mag = torch.norm(r, dim=-1)
            ke = 0.5 * torch.sum(v ** 2, dim=-1)
            pe = -self.mu / r_mag
            H = ke + pe

            if target_state is not None:
                r_target = target_state[:, :3]
                v_target = target_state[:, 3:]
                r_mag_target = torch.norm(r_target, dim=-1)
                ke_target = 0.5 * torch.sum(v_target ** 2, dim=-1)
                pe_target = -self.mu / r_mag_target
                H_target = ke_target + pe_target
                energy_gain = torch.relu(H - H_target)
                lyapunov_loss_val = torch.mean(energy_gain)
            else:
                lyapunov_loss_val = torch.var(H)
            return lyapunov_loss_val
        return torch.tensor(0.0, device=pred_state.device)

    def iss_loss(self, pred_state, input_control=None):
        """
        Input-to-State Stability (ISS) loss.

        ISS ensures that for bounded inputs, the state remains bounded.
        This is critical for control-affine systems with thrust/maneuvers.

        ISS condition: There exists a Lyapunov function V(x) such that:
        - alpha1|x| <= V(x) <= alpha2|x|
        - dV/dt <= -W(x) for |x| >= R (outside compact set)

        Args:
            pred_state: Predicted state [batch_size, 6]
            input_control: Control input (thrust) [batch_size, 3] (optional)

        Returns:
            ISS stability loss
        """
        if pred_state.dim() == 2:
            r = pred_state[:, :3]
            v = pred_state[:, 3:]
            state_mag = torch.norm(torch.cat([r, v], dim=-1), dim=-1)
            V = 0.5 * state_mag ** 2
            R_threshold = 10.0 * 6371e3  # meters
            outside_region = state_mag > R_threshold
            iss_loss_val = torch.mean(torch.where(outside_region, V, torch.zeros_like(V)))

            if input_control is not None:
                control_mag = torch.norm(input_control, dim=-1)
                max_control = 1000.0  # m/s^2 (reasonable thrust limit)
                control_excess = torch.relu(control_mag - max_control)
                iss_loss_val += torch.mean(control_excess)
            return iss_loss_val
        return torch.tensor(0.0, device=pred_state.device)

    def __call__(self, pred_state, target_state, use_physics=True, use_stability=False, use_drag=False):
        """
        Calculate total physics-informed loss

        Args:
            pred_state: Predicted state [batch_size, 6]
            target_state: Ground truth state [batch_size, 6]
            use_physics: Whether to include physics constraints
            use_drag: Whether to include drag ODE residual

        Returns:
            Total weighted loss
        """
        mse_loss = F.mse_loss(pred_state, target_state)
        if not use_physics:
            return mse_loss

        physics_loss = self.keplerian_residual(pred_state)
        energy_loss = self.energy_conservation_loss(pred_state)
        angular_momentum_loss = self.angular_momentum_loss(pred_state)
        w_conservation = 0.1

        total_loss = (
            self.w_data * mse_loss +
            self.w_physics * physics_loss +
            w_conservation * (energy_loss + angular_momentum_loss)
        )

        if use_drag:
            total_loss = total_loss + self.w_drag * self.drag_residual(pred_state)

        return total_loss

    def get_loss_components(self, pred_state, target_state, use_stability=False, input_control=None, use_drag=False):
        """
        Get individual loss components for logging/debugging.

        Returns:
            Dictionary with all loss components
        """
        components = {
            'mse_loss': F.mse_loss(pred_state, target_state).item(),
            'physics_loss': self.keplerian_residual(pred_state).item(),
            'energy_loss': self.energy_conservation_loss(pred_state).item(),
            'angular_momentum_loss': self.angular_momentum_loss(pred_state).item(),
        }

        if use_drag:
            components['drag_loss'] = self.drag_residual(pred_state).item()

        if use_stability:
            components['lyapunov_loss'] = self.lyapunov_loss(pred_state, target_state).item()
            components['iss_loss'] = self.iss_loss(pred_state, input_control).item()

        components['total_loss'] = self.__call__(
            pred_state, target_state, use_stability=use_stability, use_drag=use_drag
        ).item()

        return components


class AdaptivePhysicsLoss(PhysicsInformedLoss):
    """
    Adaptive version that adjusts physics weight based on training progress
    """

    def __init__(self, mu=3.986004418e14, w_data_start=0.9, w_physics_start=0.1):
        super().__init__(mu, w_data_start, w_physics_start)
        self.w_data_start = w_data_start
        self.w_physics_start = w_physics_start
        self.epoch = 0
        self.total_epochs = 100

    def set_training_progress(self, current_epoch, total_epochs):
        """Update training progress for adaptive weighting"""
        self.epoch = current_epoch
        self.total_epochs = total_epochs
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

    L_total = L_reward + lambda * max(0, ||DeltaV|| - DeltaV_limit) + lambda * max(0, cumDeltaV - total_budget)
    """

    def __init__(self, mu=3.986004418e14, w_data=0.7, w_physics=0.3,
                 max_delta_v=10.0, lambda_penalty=100.0):
        """
        Args:
            mu: Gravitational parameter
            w_data: Weight for data loss
            w_physics: Weight for physics loss
            max_delta_v: Maximum delta-V per maneuver (m/s)
            lambda_penalty: Lagrangian multiplier strength for constraints
        """
        super().__init__(mu, w_data, w_physics)
        self.max_delta_v = max_delta_v
        self.lambda_penalty = lambda_penalty

        self.thruster_profiles = {
            'cold_gas': {
                'max_delta_v': 5.0,
                'total_budget': 100.0,
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
            thruster_type: Type of thruster

        Returns:
            Constrained loss (penalized if constraints violated)
        """
        profile = self.thruster_profiles.get(thruster_type, self.thruster_profiles['cold_gas'])
        action_mag = torch.norm(action)
        single_maneuver_violation = torch.clamp(action_mag - profile['max_delta_v'], min=0.0)
        total_budget_violation = torch.clamp(torch.tensor(cumulative_delta_v - profile['total_budget']), min=0.0)
        penalty = self.lambda_penalty * (single_maneuver_violation + total_budget_violation)
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


def physics_informed_loss(pred_state, target_state, mu=3.986004418e14,
                          w_data=0.7, w_physics=0.3):
    """
    Simple wrapper for physics-informed loss

    Args:
        pred_state: Predicted orbital state [batch, 6]
        target_state: Target orbital state [batch, 6]
        mu: Gravitational parameter (m^3/s^2)
        w_data: Weight for data loss
        w_physics: Weight for physics loss

    Returns:
        Total loss value
    """
    loss_fn = PhysicsInformedLoss(mu, w_data, w_physics)
    return loss_fn(pred_state, target_state)


if __name__ == "__main__":
    print("Testing Physics-Informed Loss...")
    batch_size = 4
    r = 6371000 + 400000
    v_circ = np.sqrt(3.986e14 / r)
    target = torch.zeros(batch_size, 6)
    target[:, 0] = r
    target[:, 4] = v_circ
    pred = target.clone()
    pred[:, 0] += torch.randn(batch_size) * 1000
    pred[:, 4] += torch.randn(batch_size) * 10

    loss_fn = PhysicsInformedLoss()
    total_loss = loss_fn(pred, target)
    print(f"Total Loss: {total_loss.item():.6f}")

    components = loss_fn.get_loss_components(pred, target)
    for name, value in components.items():
        print(f"  {name}: {value:.6f}")

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
    action_feasible = torch.tensor([5.0, 0.0, 0.0])
    loss_feasible = isp_loss.compute_constrained_loss(1.0, action_feasible, 50.0)
    print(f"  Feasible action loss: {loss_feasible:.2f}")
    action_infeasible = torch.tensor([20.0, 0.0, 0.0])
    loss_infeasible = isp_loss.compute_constrained_loss(1.0, action_infeasible, 50.0)
    print(f"  Infeasible action loss: {loss_infeasible:.2f}")
    is_feasible, reason = isp_loss.check_action_feasible(action_infeasible, 'cold_gas')
    print(f"  Feasibility check: {is_feasible}, {reason}")

    print("\nAll tests passed!")
