#!/usr/bin/env python3
"""
Koopman Operator Theory for Uncertainty Propagation

Linearizes nonlinear orbital dynamics for instantaneous Probability of Collision (Pc) computation.
In the lifted space, covariance propagation is matrix multiplication: Σ(t+1) = K @ Σ(t) @ K.T

Reference: Brunton et al. "Koopman Invariant Subspaces and Finite Linear Representations"
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from functools import partial
import numpy as np
from typing import Tuple, Dict

jax.config.update("jax_enable_x64", True)


class KoopmanPropagator:
    """
    Explicit Koopman operator for orbital uncertainty propagation.
    
    Lifts 6D state (position + velocity) to higher-dimensional observable space
    where dynamics are linear, enabling fast covariance propagation.
    """
    
    def __init__(self, lifted_dim: int = 20):
        """
        Args:
            lifted_dim: Dimension of lifted space (Koopman observables)
                       Higher = more accurate but more computation
        """
        self.lifted_dim = lifted_dim
        self.K = None  # Koopman operator matrix
        self.observable_fn = None  # Lifting function ψ(x)
        
    @partial(jit, static_argnums=(0,))
    def lift_state(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Lift 6D state to Koopman observable space.
        
        Observables include: position, velocity, and nonlinear combinations
        that capture orbital dynamics (r, v, r², v², r·v, etc.)
        
        Args:
            state: [6] array [x, y, z, vx, vy, vz] in km, km/s
            
        Returns:
            psi: [lifted_dim] lifted observables
        """
        x, y, z, vx, vy, vz = state
        
        # Position and velocity magnitudes
        r = jnp.sqrt(x**2 + y**2 + z**2)
        v = jnp.sqrt(vx**2 + vy**2 + vz**2)
        
        # Orbital elements (approximate from state)
        h = jnp.linalg.norm(jnp.cross(jnp.array([x, y, z]), jnp.array([vx, vy, vz])))
        
        # Energy
        mu = 398600.4418  # km³/s²
        energy = 0.5 * v**2 - mu / r
        
        # Standard observables (first 13)
        observables = jnp.array([
            x, y, z,           # Position
            vx, vy, vz,        # Velocity
            r,                 # Distance
            v,                 # Speed
            r**2,              # Squared distance
            v**2,              # Squared speed
            x*vx + y*vy + z*vz,  # Radial velocity (r · v)
            energy,            # Specific energy
            h,                 # Specific angular momentum
        ])
        
        # Extended observables (fill to lifted_dim with higher-order terms)
        if self.lifted_dim > 13:
            extended = jnp.array([
                r**3, v**3,                    # Cubic terms
                1.0/r, 1.0/(r**2),              # Inverse terms (J2-like)
                jnp.sin(x/r), jnp.cos(y/r),   # Angular periodicities
                x*y, y*z, z*x,                  # Cross position terms
                vx*vy, vy*vz, vz*vx,            # Cross velocity terms
                x/r, y/r, z/r,                  # Unit vector components
                x*v/r, y*v/r, z*v/r,            # Scaled velocity
            ])
            # Combine and truncate/pad to lifted_dim
            full = jnp.concatenate([observables, extended])
            if len(full) >= self.lifted_dim:
                return full[:self.lifted_dim]
            else:
                # Pad with zeros if not enough observables
                return jnp.pad(full, (0, self.lifted_dim - len(full)))
        
        return observables[:self.lifted_dim]
    
    def compute_koopman_operator(self, 
                                  states: jnp.ndarray, 
                                  next_states: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Koopman operator K from state transitions.
        
        K satisfies: ψ(x_{t+1}) = K @ ψ(x_t)
        
        Args:
            states: [N, 6] trajectory states
            next_states: [N, 6] states at next timestep
            
        Returns:
            K: [lifted_dim, lifted_dim] Koopman operator
        """
        # Lift all states
        Psi = vmap(self.lift_state)(states)      # [N, lifted_dim]
        Psi_next = vmap(self.lift_state)(next_states)  # [N, lifted_dim]
        
        # Solve: Psi_next = K @ Psi.T
        # Using least squares: K = Psi_next @ pinv(Psi)
        
        # SVD-based pseudo-inverse (more stable than direct solve)
        U, S, Vt = jnp.linalg.svd(Psi, full_matrices=False)
        
        # Threshold small singular values
        S_inv = jnp.where(S > 1e-10, 1.0/S, 0.0)
        
        # K = Psi_next @ V @ diag(S_inv) @ U.T
        K = Psi_next.T @ (Vt.T * S_inv) @ U.T
        K = K.T  # Transpose to get correct shape
        
        self.K = K
        return K
    
    @partial(jit, static_argnums=(0,))
    def propagate_state(self, state: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
        """
        Propagate state using Koopman operator.
        
        Linear evolution in lifted space, then project back.
        
        Args:
            state: [6] current state
            K: [lifted_dim, lifted_dim] Koopman operator
            
        Returns:
            next_state: [6] propagated state
        """
        # Lift to observable space
        psi = self.lift_state(state)
        
        # Linear evolution: ψ(t+1) = K @ ψ(t)
        psi_next = K @ psi
        
        # Project back to state space (first 6 observables are just state)
        # This is an approximation - proper Koopman requires eigenfunctions
        next_state = psi_next[:6]
        
        return next_state
    
    @partial(jit, static_argnums=(0,))
    def propagate_covariance(self, 
                             Sigma: jnp.ndarray, 
                             K: jnp.ndarray,
                             J: jnp.ndarray) -> jnp.ndarray:
        """
        Propagate covariance using Koopman linearity.
        
        In lifted space: Σ_ψ(t+1) = K @ Σ_ψ(t) @ K.T
        Then project back to state space via Jacobian J = dψ/dx.
        
        Args:
            Sigma: [6, 6] current covariance in state space
            K: [lifted_dim, lifted_dim] Koopman operator
            J: [lifted_dim, 6] Jacobian of lifting function
            
        Returns:
            Sigma_next: [6, 6] propagated covariance
        """
        # Lift covariance to observable space
        # Σ_ψ = J @ Σ @ J.T
        Sigma_lifted = J @ Sigma @ J.T
        
        # Linear propagation in lifted space
        Sigma_lifted_next = K @ Sigma_lifted @ K.T
        
        # Project back to state space via pseudo-inverse of J
        # Σ(t+1) = pinv(J) @ Σ_ψ(t+1) @ pinv(J).T
        J_pinv = jnp.linalg.pinv(J)
        Sigma_next = J_pinv @ Sigma_lifted_next @ J_pinv.T
        
        return Sigma_next
    
    def compute_jacobian(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Jacobian of lifting function: J = dψ/dx
        
        Args:
            state: [6] state vector
            
        Returns:
            J: [lifted_dim, 6] Jacobian matrix
        """
        # Use JAX automatic differentiation
        J = jax.jacfwd(self.lift_state)(state)
        return J
    
    def fast_pc_computation(self,
                           primary_state: jnp.ndarray,
                           primary_cov: jnp.ndarray,
                           secondary_state: jnp.ndarray,
                           secondary_cov: jnp.ndarray,
                           K: jnp.ndarray,
                           dt: float,
                           n_steps: int) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Fast Probability of Collision via Koopman propagation.
        
        Instead of Monte Carlo (expensive), uses linear covariance propagation.
        
        Args:
            primary_state: [6] primary object state
            primary_cov: [6, 6] primary covariance
            secondary_state: [6] secondary object state
            secondary_cov: [6, 6] secondary covariance
            K: Koopman operator
            dt: timestep (seconds)
            n_steps: number of steps to TCA (Time of Closest Approach)
            
        Returns:
            primary_state_tca: [6] at TCA
            secondary_state_tca: [6] at TCA
            pc: Probability of collision estimate
        """
        # Propagate states using Koopman
        p_state = primary_state
        s_state = secondary_state
        
        # Jacobians for covariance lifting
        J_p = self.compute_jacobian(p_state)
        J_s = self.compute_jacobian(s_state)
        
        # Propagate covariances
        p_cov = primary_cov
        s_cov = secondary_cov
        
        for _ in range(n_steps):
            # Recompute Jacobians at each step (linearization point changes)
            J_p = self.compute_jacobian(p_state)
            J_s = self.compute_jacobian(s_state)
            
            # Propagate
            p_state = self.propagate_state(p_state, K)
            s_state = self.propagate_state(s_state, K)
            
            p_cov = self.propagate_covariance(p_cov, K, J_p)
            s_cov = self.propagate_covariance(s_cov, K, J_s)
        
        # Compute Pc at TCA (simplified 2D projection)
        pc = self._compute_pc_2d(p_state[:3], p_cov[:3, :3], 
                                  s_state[:3], s_cov[:3, :3])
        
        return p_state, s_state, pc
    
    def _compute_pc_2d(self, 
                       r1: jnp.ndarray, C1: jnp.ndarray,
                       r2: jnp.ndarray, C2: jnp.ndarray) -> float:
        """
        Simplified 2D Pc computation (Foster method approximation).
        """
        # Relative position and combined covariance
        r_rel = r2 - r1
        C_comb = C1 + C2
        
        # Project to encounter plane (simplified: plane perpendicular to relative velocity)
        # For full implementation, use proper B-plane transformation
        
        # Mahalanobis distance
        try:
            C_inv = jnp.linalg.inv(C_comb[:2, :2])  # 2D projection
            dist_sq = r_rel[:2].T @ C_inv @ r_rel[:2]
            
            # Approximate Pc using Gaussian integral (simplified)
            # Full Foster method requires more sophisticated integration
            pc_approx = jnp.exp(-0.5 * dist_sq)
            
            return float(pc_approx)
        except:
            return 0.0


class KoopmanMBPOBridge:
    """
    Bridge between Koopman propagator and Model-Based RL (MBPO).
    
    Uses Koopman-discovered dynamics as the "world model" for MBPO agent.
    """
    
    def __init__(self, koopman: KoopmanPropagator):
        self.koopman = koopman
        
    def imagination_rollout(self, 
                           initial_state: jnp.ndarray,
                           action_sequence: jnp.ndarray,
                           K: jnp.ndarray) -> jnp.ndarray:
        """
        Generate imagined trajectories for MBPO planning.
        
        Much faster than ground truth simulation (physics simulator)
        because Koopman dynamics are linear in lifted space.
        
        Args:
            initial_state: [6] initial orbital state
            action_sequence: [N, 3] sequence of thrust vectors (delta-V)
            K: Koopman operator (includes perturbation model)
            
        Returns:
            trajectory: [N+1, 6] imagined states
        """
        trajectory = [initial_state]
        state = initial_state
        
        for action in action_sequence:
            # Apply action (thrust) as velocity perturbation
            perturbed_state = state.at[3:].add(action)  # Add delta-V
            
            # Propagate with Koopman
            state = self.koopman.propagate_state(perturbed_state, K)
            trajectory.append(state)
        
        return jnp.stack(trajectory)
    
    def validate_against_physics(self,
                               imagined_trajectory: jnp.ndarray,
                               true_trajectory: jnp.ndarray) -> float:
        """
        Compute validation error between Koopman imagination and ground truth.
        
        Used to detect when Koopman model needs re-training.
        """
        # Mean squared error over trajectory
        mse = jnp.mean((imagined_trajectory - true_trajectory)**2)
        
        # If error exceeds threshold, trigger model update
        return float(mse)


def train_koopman_from_orbits(orbital_states: np.ndarray, 
                               lifted_dim: int = 20) -> Tuple[KoopmanPropagator, jnp.ndarray]:
    """
    Train Koopman operator from orbital trajectory data.
    
    Args:
        orbital_states: [N_timesteps, 6] trajectory
        lifted_dim: Dimension of lifted space
        
    Returns:
        koopman: Trained propagator
        K: Koopman operator matrix
    """
    koopman = KoopmanPropagator(lifted_dim=lifted_dim)
    
    # Create state pairs: (x_t, x_{t+1})
    states = jnp.array(orbital_states[:-1])
    next_states = jnp.array(orbital_states[1:])
    
    # Compute Koopman operator
    K = koopman.compute_koopman_operator(states, next_states)
    
    return koopman, K


if __name__ == "__main__":
    # Demonstration: Fast covariance propagation
    print("🔥 Koopman Propagator Demo")
    
    # Generate simple circular orbit
    t = np.linspace(0, 100, 1000)  # 100 orbits
    omega = 0.0011  # rad/s for LEO
    r = 7000  # km
    
    orbit_states = np.array([
        [r * np.cos(omega * ti), r * np.sin(omega * ti), 0,
         -r * omega * np.sin(omega * ti), r * omega * np.cos(omega * ti), 0]
        for ti in t
    ])
    
    # Train Koopman
    koopman, K = train_koopman_from_orbits(orbit_states)
    
    print(f"Koopman operator shape: {K.shape}")
    print(f"Eigenvalue magnitudes: {np.abs(np.linalg.eigvals(K))[:5]}")
    
    # Test fast propagation
    initial_cov = np.eye(6) * 0.1  # 100m uncertainty
    J = koopman.compute_jacobian(orbit_states[0])
    
    propagated_cov = koopman.propagate_covariance(
        jnp.array(initial_cov), K, J
    )
    
    print(f"Covariance propagated from 100m to {np.sqrt(np.trace(propagated_cov)/6)*1000:.1f}m")
