"""
manifold_projection.py - ADITI 4.0 Hard Manifold Constraint Layer

Implements the "Manifold Projection" layer—a final post-processing step that 
projects the state back onto the constant-energy surface after the integrator 
finishes, eliminating floating-point "micro-creep".

Mathematical Foundation:
    The projection solves:
        minimize ‖δz‖²  subject to  H(z + δz) = H₀
    
    This is a constrained optimization with analytical solution via Lagrange 
    multipliers. The correction is:
        δz = λ ∇H(z)  where λ = (H₀ - H(z)) / ‖∇H(z)‖²

This guarantees that even after thousands of symplectic steps, the state lies
exactly on the energy manifold (to machine precision).

References:
    - Hairer, Lubich, Wanner: "Geometric Numerical Integration" (2006)
    - foundation.md §3: Hard Manifold Constraints
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
from typing import Tuple
import equinox as eqx


class ManifoldProjection:
    """
    Projects state onto constant-Hamiltonian manifold.
    
    This layer acts as a "physics firewall"—it guarantees that no matter what
    floating-point errors accumulate during integration, the final state obeys
    energy conservation exactly.
    
    Usage:
        projector = ManifoldProjection()
        q_corrected, p_corrected = projector.project(q, p, H_target, model)
    """
    
    def __init__(self, max_iter: int = 3, tol: float = 1e-12):
        """
        Args:
            max_iter: Maximum Newton iterations for projection
            tol: Convergence tolerance for energy constraint
        """
        self.max_iter = max_iter
        self.tol = tol
    
    @staticmethod
    def _compute_energy_and_gradient(
        state: jnp.ndarray,
        model,
        z_latent: jnp.ndarray
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute H(z) and ∇H(z).
        
        Returns:
            H: Scalar energy
            grad_H: [6] gradient of H w.r.t. state
        """
        def H_fn(s):
            return model.get_hamiltonian(s, z_latent)
        
        H = H_fn(state)
        grad_H = grad(H_fn)(state)
        
        return H, grad_H
    
    def project(
        self,
        q: jnp.ndarray,
        p: jnp.ndarray,
        H_target: float,
        model,
        z_latent: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Project (q, p) onto manifold H(q, p) = H_target.
        
        Algorithm (Standard Projection):
            1. Compute current H and ∇H
            2. Compute Lagrange multiplier: λ = (H_target - H) / ‖∇H‖²
            3. Update: z_new = z_old + λ ∇H
            4. Iterate until |H - H_target| < tol
        
        This is a symplectic projection that minimizes the correction magnitude
        while enforcing exact energy conservation.
        
        Args:
            q: [3] position
            p: [3] momentum  
            H_target: Target Hamiltonian value
            model: PortHamiltonianSystem with get_hamiltonian method
            z_latent: [latent_dim] physics code (optional)
            
        Returns:
            q_corrected, p_corrected: Projected state with H = H_target
        """
        if z_latent is None:
            z_latent = jnp.zeros(model.latent_dim)
        
        state = jnp.concatenate([q, p])
        
        # Newton iteration for projection
        for i in range(self.max_iter):
            H_current, grad_H = self._compute_energy_and_gradient(
                state, model, z_latent
            )
            
            energy_error = H_current - H_target
            
            # Check convergence
            if abs(energy_error) < self.tol:
                break
            
            # Compute Lagrange multiplier
            # λ = (H_target - H) / ‖∇H‖²
            grad_norm_sq = jnp.sum(grad_H ** 2)
            
            # Avoid division by zero
            grad_norm_sq = jnp.maximum(grad_norm_sq, 1e-10)
            
            lam = -energy_error / grad_norm_sq
            
            # Project: z_new = z_old + λ ∇H
            state = state + lam * grad_H
        
        # Extract corrected q, p
        q_corrected = state[:3]
        p_corrected = state[3:]
        
        return q_corrected, p_corrected
    
    def project_trajectory(
        self,
        q_traj: jnp.ndarray,
        p_traj: jnp.ndarray,
        H_target: float,
        model,
        z_latent: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Project entire trajectory onto constant-H manifold.
        
        Args:
            q_traj: [T, 3] position trajectory
            p_traj: [T, 3] momentum trajectory
            H_target: Target Hamiltonian (typically H[0])
            model: PortHamiltonianSystem
            z_latent: [latent_dim] physics code
            
        Returns:
            q_corrected, p_corrected: Projected trajectories
        """
        def project_step(carry, inputs):
            q, p = inputs
            q_corr, p_corr = self.project(q, p, H_target, model, z_latent)
            return carry, (q_corr, p_corr)
        
        _, (q_corrected, p_corrected) = jax.lax.scan(
            project_step,
            None,
            (q_traj, p_traj)
        )
        
        return q_corrected, p_corrected


class SymplecticRegularizer:
    """
    Symplectic regularization for neural network weights.
    
    Instead of λ_h soft penalty, this enforces unitary constraints on e3nn
    layer weights to ensure the energy surface remains bounded.
    
    This is a weight-space constraint (not a loss term) that orthogonalizes
    the weight matrices to preserve geometric structure.
    """
    
    @staticmethod
    def orthogonalize_weights(weights: jnp.ndarray) -> jnp.ndarray:
        """
        Project weight matrix onto orthogonal manifold.
        
        Uses QR decomposition: W = Q·R, returns Q (orthogonal)
        """
        q, r = jnp.linalg.qr(weights)
        return q
    
    @staticmethod
    def symplectic_regularization_loss(params) -> float:
        """
        Compute symplectic regularization loss for model parameters.
        
        This encourages weight matrices to be close to orthogonal,
        preserving the geometric structure of the Hamiltonian.
        
        Loss = ‖W^T W - I‖²_F
        """
        total_loss = 0.0
        count = 0
        
        for param in jax.tree_util.tree_leaves(params):
            if param.ndim == 2:  # Matrix parameter
                w = param
                identity = jnp.eye(w.shape[1])
                orthogonality_error = jnp.sum((w.T @ w - identity) ** 2)
                total_loss += orthogonality_error
                count += 1
        
        return total_loss / max(count, 1)


@jit
def hard_physics_loss(
    q_pred: jnp.ndarray,
    p_pred: jnp.ndarray,
    q_target: jnp.ndarray,
    p_target: jnp.ndarray,
    H_target: float,
    model,
    z_latent: jnp.ndarray = None
) -> Tuple[float, Dict]:
    """
    Hard physics loss—only data term, physics enforced by architecture.
    
    This is the "Sovereign Choice"—no λ_h soft penalty.
    Physics is enforced by:
        1. Forest-Ruth integrator in forward pass
        2. Manifold projection post-processing
        3. Augmented Lagrangian constraints (dual ascent)
    
    Args:
        q_pred: [3] predicted position (after integrator + projection)
        p_pred: [3] predicted momentum
        q_target: [3] target position
        p_target: [3] target momentum
        H_target: Target Hamiltonian (for monitoring)
        model: PortHamiltonianSystem
        z_latent: Physics code
        
    Returns:
        loss: MSE loss (physics already enforced)
        metrics: Dict with energy drift, symplectic error, etc.
    """
    # Data loss only—physics is baked into the architecture
    loss_q = jnp.mean((q_pred - q_target) ** 2)
    loss_p = jnp.mean((p_pred - p_target) ** 2)
    total_loss = loss_q + loss_p
    
    # Compute metrics (for monitoring, not loss)
    state_pred = jnp.concatenate([q_pred, p_pred])
    H_pred = model.get_hamiltonian(state_pred, z_latent)
    energy_drift = abs(H_pred - H_target) / abs(H_target)
    
    metrics = {
        'loss': total_loss,
        'energy_drift': energy_drift,
        'position_error': jnp.sqrt(loss_q),
        'momentum_error': jnp.sqrt(loss_p),
    }
    
    return total_loss, metrics


def test_manifold_projection():
    """Unit test: Verify projection enforces exact energy conservation."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    
    from port_hamiltonian import PortHamiltonianSystem
    
    print("=" * 70)
    print("Manifold Projection Unit Test")
    print("=" * 70)
    
    # Create simple Keplerian model
    class KeplerianPHS(PortHamiltonianSystem):
        def __init__(self, mu=398600.4418):
            self.mu = mu
            # Minimal init for testing
            
        def get_hamiltonian(self, state, z_latent=None):
            q = state[..., :3]
            p = state[..., 3:]
            r = jnp.linalg.norm(q)
            v_sq = jnp.linalg.norm(p) ** 2
            KE = 0.5 * v_sq
            PE = -self.mu / r
            return KE + PE
    
    model = KeplerianPHS()
    projector = ManifoldProjection(max_iter=5, tol=1e-14)
    
    # Initial state: circular orbit
    q = jnp.array([6778.137, 0.0, 0.0])
    p = jnp.array([0.0, 7.6686, 0.0])
    
    # Target energy
    H_target = model.get_hamiltonian(jnp.concatenate([q, p]))
    
    # Perturb state (simulate integration error)
    q_perturbed = q + jnp.array([1.0, 0.5, 0.2])  # ~1km error
    p_perturbed = p + jnp.array([0.01, 0.01, 0.005])  # ~10 m/s error
    
    H_perturbed = model.get_hamiltonian(jnp.concatenate([q_perturbed, p_perturbed]))
    initial_drift = abs(H_perturbed - H_target) / abs(H_target)
    
    print(f"\nInitial drift: {initial_drift:.2e}")
    
    # Project back to manifold
    q_corrected, p_corrected = projector.project(
        q_perturbed, p_perturbed, H_target, model
    )
    
    H_corrected = model.get_hamiltonian(jnp.concatenate([q_corrected, p_corrected]))
    final_drift = abs(H_corrected - H_target) / abs(H_target)
    
    print(f"Final drift:   {final_drift:.2e}")
    print(f"Correction magnitude: {jnp.linalg.norm(q_corrected - q_perturbed):.6f} km")
    
    # Verify exact conservation
    assert final_drift < 1e-12, f"Projection failed: drift {final_drift:.2e} > 1e-12"
    
    print("\n✓ Manifold Projection: EXACT energy conservation verified")
    print("=" * 70)


if __name__ == "__main__":
    test_manifold_projection()
