"""
JAX Training Pipeline for Port-Hamiltonian System (PHS)
: CANONICAL TRAINING BACKEND ✅

🎯 UNIFICATION COMPLETE: This is the canonical training backend for .
 PyTorch version (ai/neural_ode/port_hamiltonian_pytorch.py) exists for deployment only.

Replaces broken PyTorch GHNN path with correct JAX implementation.
Uses Forest-Ruth 4th-order symplectic integrator with proper q/p split.

Key Features:
1. Correct Yoshida-4 integration (q/p alternation with c/d coefficients)
2. Cholesky PSD R construction (guaranteed dissipation)
3. Augmented Lagrangian training (hard physics constraints)
4. JAX autodiff-on-gradient cleanliness

Acceptance Criteria:
- Energy drift < 1e-8 over 10^4 steps on Keplerian orbits
- Position error < 100m at T+6hr for orbital trajectories

References:
 - Forest-Ruth 4th order: Ruth (1983), Forest & Ruth (1990)
 - Port-Hamiltonian Systems: van der Schaft & Maschke
 - foundation.md §6: Critical files for implementation
"""

from typing import Tuple, Dict, Optional, Callable
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, value_and_grad
import equinox as eqx
import optax
from functools import partial
import numpy as np
from pathlib import Path
import json
import time

# Import PHS model
from port_hamiltonian import PortHamiltonianSystem

# Forest-Ruth 4th order coefficients (from q/core/jax_backend.py)
# These are the correct coefficients for symplectic integration
# λ = 1/(2-2^(1/3)) ≈ 1.3512071919596578
LAMBDA_FR = 1.3512071919596578
C1_R4 = LAMBDA_FR / 2.0
D1_R4 = LAMBDA_FR
C2_R4 = (1.0 - LAMBDA_FR) / 2.0
D2_R4 = 1.0 - 2.0 * LAMBDA_FR
C3_R4 = C2_R4
D3_R4 = LAMBDA_FR
C4_R4 = C1_R4


def forest_ruth_step(
 model: PortHamiltonianSystem,
 q: jnp.ndarray,
 p: jnp.ndarray,
 z_latent: jnp.ndarray,
 dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
 """
 Forest-Ruth 4th-order symplectic integration step.
 
 CORRECT IMPLEMENTATION with proper q/p alternation:
 Stage 1: q = q + C1*dt*dH/dp, p = p - D1*dt*dH/dq
 Stage 2: q = q + C2*dt*dH/dp, p = p - D2*dt*dH/dq
 Stage 3: q = q + C3*dt*dH/dp, p = p - D3*dt*dH/dq
 Stage 4: q = q + C4*dt*dH/dp
 
 This preserves phase-space volume under the modified Hamiltonian H̃_θ (Yoshida 1990, §3);
 energy drift from original H_θ is O(dt^5).
 
 Args:
 model: PortHamiltonianSystem with H_net, R_net, g_net
 q: [3] position vector
 p: [3] momentum vector
 z_latent: [latent_dim] physics code
 dt: Time step
 
 Returns:
 q_next: [3] next position
 p_next: [3] next momentum
 """
 state = jnp.concatenate([q, p])
 
 # Stage 1
 dH_dp1 = model.compute_dH_dp(state, z_latent)
 q = q + C1_R4 * dt * dH_dp1
 state = jnp.concatenate([q, p])
 dH_dq1 = model.compute_dH_dq(state, z_latent)
 p = p - D1_R4 * dt * dH_dq1
 
 # Stage 2
 state = jnp.concatenate([q, p])
 dH_dp2 = model.compute_dH_dp(state, z_latent)
 q = q + C2_R4 * dt * dH_dp2
 state = jnp.concatenate([q, p])
 dH_dq2 = model.compute_dH_dq(state, z_latent)
 p = p - D2_R4 * dt * dH_dq2
 
 # Stage 3
 state = jnp.concatenate([q, p])
 dH_dp3 = model.compute_dH_dp(state, z_latent)
 q = q + C3_R4 * dt * dH_dp3
 state = jnp.concatenate([q, p])
 dH_dq3 = model.compute_dH_dq(state, z_latent)
 p = p - D3_R4 * dt * dH_dq3
 
 # Stage 4
 state = jnp.concatenate([q, p])
 dH_dp4 = model.compute_dH_dp(state, z_latent)
 q = q + C4_R4 * dt * dH_dp4
 
 return q, p


def dissipative_step(
 model: PortHamiltonianSystem,
 q: jnp.ndarray,
 p: jnp.ndarray,
 z_latent: jnp.ndarray,
 control: jnp.ndarray,
 dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
 """
 Dissipative Port-Hamiltonian step with drag and control.
 
 Uses operator splitting:
 1. Conservative step: Yoshida-4 (J*∇H)
 2. Dissipative step: -R*∇H (drag)
 3. Control step: g*u (thrust)
 
 Args:
 model: PortHamiltonianSystem
 q: [3] position
 p: [3] momentum
 z_latent: [latent_dim] physics code
 control: [3] control input (thrust)
 dt: Time step
 
 Returns:
 q_next, p_next
 """
 # Step 1: Conservative dynamics (Yoshida-4)
 q, p = forest_ruth_step(model, q, p, z_latent, dt)
 
 # Step 2: Dissipative dynamics (drag)
 state = jnp.concatenate([q, p])
 R_matrix = model.compute_R_matrix(state, z_latent)
 dH_dz = model.compute_gradient_H(state, z_latent)
 dissipation = -R_matrix @ dH_dz * dt
 q = q + dissipation[:3]
 p = p + dissipation[3:]
 
 # Step 3: Control input (thrust)
 if control is not None:
 G_matrix = model.compute_G_matrix(state, z_latent)
 control_effect = G_matrix @ control * dt
 q = q + control_effect[:3]
 p = p + control_effect[3:]
 
 return q, p


@partial(jit, static_argnums=(0, 4, 5))
def integrate_trajectory(
 model: PortHamiltonianSystem,
 q0: jnp.ndarray,
 p0: jnp.ndarray,
 z_latent: jnp.ndarray,
 n_steps: int,
 dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
 """
 Integrate trajectory with JIT compilation.
 
 Args:
 model: PortHamiltonianSystem
 q0: [3] initial position
 p0: [3] initial momentum
 z_latent: [latent_dim] physics code
 n_steps: Number of integration steps
 dt: Time step
 
 Returns:
 q_traj: [n_steps+1, 3] position trajectory
 p_traj: [n_steps+1, 3] momentum trajectory
 """
 def step_fn(carry, _):
 q, p = carry
 q_next, p_next = dissipative_step(model, q, p, z_latent, None, dt)
 return (q_next, p_next), (q_next, p_next)
 
 _, (q_traj, p_traj) = jax.lax.scan(
 step_fn,
 (q0, p0),
 None,
 length=n_steps
)
 
 # Prepend initial state
 q_traj = jnp.concatenate([q0[None, :], q_traj], axis=0)
 p_traj = jnp.concatenate([p0[None, :], p_traj], axis=0)
 
 return q_traj, p_traj


def compute_energy_drift(
 model: PortHamiltonianSystem,
 q_traj: jnp.ndarray,
 p_traj: jnp.ndarray,
 z_latent: jnp.ndarray,
) -> float:
 """
 Compute energy conservation metric.
 
 For conservative orbits (R=0), energy should be constant.
 Returns relative energy drift.
 
 Args:
 model: PortHamiltonianSystem
 q_traj: [T, 3] position trajectory
 p_traj: [T, 3] momentum trajectory
 z_latent: [latent_dim] physics code
 
 Returns:
 max_relative_drift: max |ΔH/H_0|
 """
 def compute_H(q, p):
 state = jnp.concatenate([q, p])
 return model.get_hamiltonian(state, z_latent)
 
 H_values = vmap(compute_H)(q_traj, p_traj)
 H_0 = H_values[0]
 relative_drift = jnp.abs((H_values - H_0) / H_0)
 return jnp.max(relative_drift)


def create_optimizer(
 learning_rate: float = 1e-4,
 warmup_steps: int = 1000,
) -> optax.GradientTransformation:
 """
 Create optimizer with warmup and cosine decay.
 
 Args:
 learning_rate: Peak learning rate
 warmup_steps: Number of warmup steps
 
 Returns:
 optax optimizer
 """
 schedule = optax.warmup_cosine_decay_schedule(
 init_value=0.0,
 peak_value=learning_rate,
 warmup_steps=warmup_steps,
 decay_steps=100000,
 end_value=learning_rate * 0.1,
)
 return optax.adam(schedule)


def train_step(
 model: PortHamiltonianSystem,
 opt_state: optax.OptState,
 optimizer: optax.GradientTransformation,
 batch: Dict,
 mu: jnp.ndarray, # Dual variables for Augmented Lagrangian
 rho: float, # Penalty parameter
) -> Tuple[PortHamiltonianSystem, optax.OptState, Dict]:
 """
 Single training step with Augmented Lagrangian.
 
 Loss: L(θ, μ, ρ) = L_obs(θ) + μᵀc(θ) + (ρ/2)‖c(θ)‖²
 
 Args:
 model: Current model
 opt_state: Optimizer state
 optimizer: optax optimizer
 batch: Training batch with 'q0', 'p0', 'q_target', 'p_target', 'z_latent'
 mu: Dual variables for constraints
 rho: Penalty parameter
 
 Returns:
 model: Updated model
 opt_state: Updated optimizer state
 metrics: Dict with loss components
 """
 def loss_fn(model):
 # Unpack batch
 q0 = batch['q0']
 p0 = batch['p0']
 q_target = batch['q_target']
 p_target = batch['p_target']
 z_latent = batch['z_latent']
 n_steps = batch['n_steps']
 dt = batch['dt']
 
 # Integrate trajectory
 q_pred, p_pred = integrate_trajectory(
 model, q0, p0, z_latent, n_steps, dt
)
 
 # Observation loss (MSE)
 mse_q = jnp.mean((q_pred - q_target) ** 2)
 mse_p = jnp.mean((p_pred - p_target) ** 2)
 loss_obs = mse_q + mse_p
 
 # Physics constraints (c(θ))
 # c₁: Energy conservation variance
 H_0 = model.get_hamiltonian(jnp.concatenate([q0, p0]), z_latent)
 H_pred = vmap(lambda q, p: model.get_hamiltonian(
 jnp.concatenate([q, p]), z_latent
))(q_pred, p_pred)
 c_energy = jnp.var(H_pred) / (H_0 ** 2 + 1e-10)
 
 # c₂: Angular momentum conservation
 L_0 = jnp.cross(q0, p0)
 L_pred = vmap(lambda q, p: jnp.cross(q, p))(q_pred, p_pred)
 c_momentum = jnp.var(L_pred, axis=0).mean() / (jnp.sum(L_0 ** 2) + 1e-10)
 
 constraints = jnp.array([c_energy, c_momentum])
 
 # Augmented Lagrangian
 loss_physics = mu @ constraints + (rho / 2.0) * jnp.sum(constraints ** 2)
 
 loss_total = loss_obs + loss_physics
 
 return loss_total, {
 'loss_total': loss_total,
 'loss_obs': loss_obs,
 'loss_physics': loss_physics,
 'c_energy': c_energy,
 'c_momentum': c_momentum,
 'mse_q': mse_q,
 'mse_p': mse_p,
 }
 
 # Compute gradients
 (loss, metrics), grads = value_and_grad(loss_fn, has_aux=True)(model)
 
 # Update model
 updates, opt_state = optimizer.update(grads, opt_state, model)
 model = eqx.apply_updates(model, updates)
 
 return model, opt_state, metrics


def dual_ascent_update(
 mu: jnp.ndarray,
 constraints: jnp.ndarray,
 rho: float,
 rho_max: float = 1000.0,
) -> Tuple[jnp.ndarray, float]:
 """
 Update dual variables and penalty parameter.
 
 μ_{k+1} = μ_k + ρ · c(θ_k)
 
 If constraint violated (c > 0.01): ρ ← min(10·ρ, ρ_max)
 Else: ρ stays or decreases slightly
 
 Args:
 mu: Current dual variables
 constraints: Constraint values c(θ)
 rho: Current penalty parameter
 rho_max: Maximum penalty parameter
 
 Returns:
 mu_next: Updated dual variables
 rho_next: Updated penalty parameter
 """
 # Dual ascent
 mu_next = mu + rho * constraints
 
 # Adaptive penalty adjustment
 if jnp.any(constraints > 0.01):
 rho_next = min(rho * 10.0, rho_max)
 else:
 rho_next = rho
 
 return mu_next, rho_next


def train_epoch(
 model: PortHamiltonianSystem,
 opt_state: optax.OptState,
 optimizer: optax.GradientTransformation,
 dataloader,
 mu: jnp.ndarray,
 rho: float,
) -> Tuple[PortHamiltonianSystem, optax.OptState, Dict, jnp.ndarray, float]:
 """
 Train for one epoch.
 
 Args:
 model: PHS model
 opt_state: Optimizer state
 optimizer: optax optimizer
 dataloader: Training data iterator
 mu: Dual variables
 rho: Penalty parameter
 
 Returns:
 model: Updated model
 opt_state: Updated optimizer state
 epoch_metrics: Aggregated metrics
 mu: Updated dual variables
 rho: Updated penalty parameter
 """
 total_metrics = {
 'loss_total': 0.0,
 'loss_obs': 0.0,
 'loss_physics': 0.0,
 'c_energy': 0.0,
 'c_momentum': 0.0,
 'mse_q': 0.0,
 'mse_p': 0.0,
 }
 n_batches = 0
 
 for batch in dataloader:
 model, opt_state, metrics = train_step(
 model, opt_state, optimizer, batch, mu, rho
)
 
 for key in total_metrics:
 total_metrics[key] += float(metrics[key])
 n_batches += 1
 
 # Average metrics
 epoch_metrics = {k: v / n_batches for k, v in total_metrics.items()}
 
 # Dual ascent update after epoch
 constraints = jnp.array([epoch_metrics['c_energy'], epoch_metrics['c_momentum']])
 mu, rho = dual_ascent_update(mu, constraints, rho)
 
 return model, opt_state, epoch_metrics, mu, rho


def save_checkpoint(
 model: PortHamiltonianSystem,
 opt_state: optax.OptState,
 epoch: int,
 metrics: Dict,
 checkpoint_dir: Path,
):
 """Save training checkpoint."""
 checkpoint_dir.mkdir(parents=True, exist_ok=True)
 checkpoint_path = checkpoint_dir / f"epoch_{epoch}.eqx"
 
 # Save model
 eqx.tree_serialise_leaves(checkpoint_path, model)
 
 # Save metadata
 metadata = {
 'epoch': epoch,
 'metrics': metrics,
 'timestamp': time.time(),
 }
 metadata_path = checkpoint_dir / f"epoch_{epoch}.json"
 with open(metadata_path, 'w') as f:
 json.dump(metadata, f, indent=2)
 
 print(f"Checkpoint saved: {checkpoint_path}")


def main():
 """Main training entry point."""
 print("=" * 70)
 print(": JAX Port-Hamiltonian System Training")
 print("Forest-Ruth 4th-Order Symplectic Integrator")
 print("=" * 70)
 
 # Hyperparameters
 state_dim = 6
 latent_dim = 64
 learning_rate = 1e-4
 batch_size = 32
 n_epochs = 200
 
 # Initialize model
 key = jax.random.PRNGKey(42)
 model = PortHamiltonianSystem(
 state_dim=state_dim,
 control_dim=3,
 latent_dim=latent_dim,
 use_film=True,
 R_conditioned=True,
 g_conditioned=True,
 key=key,
)
 
 # Initialize optimizer
 optimizer = create_optimizer(learning_rate)
 opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
 
 # Initialize dual variables for Augmented Lagrangian
 mu = jnp.zeros(2) # Two constraints: energy, momentum
 rho = 1.0
 
 print(f"Model initialized: state_dim={state_dim}, latent_dim={latent_dim}")
 print(f"Optimizer: Adam with lr={learning_rate}")
 print(f"Augmented Lagrangian: mu={mu}, rho={rho}")
 print("=" * 70)
 
 # Training loop would go here with actual data
 print("\nNote: This is the training pipeline skeleton.")
 print("To train: integrate with data loader from Zarr store.")
 print("Expected data: orbital trajectories from HPOP/Orekit simulation.")


if __name__ == "__main__":
 main()
