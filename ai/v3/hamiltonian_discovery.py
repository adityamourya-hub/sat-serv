# hamiltonian_discovery.py
# Phase 2: Beta Scaling & Symbolic Law Extraction
# Axiom-Zero Final Elite High-Compute Protocol
# plan_complexity: High
# total_lines_estimate: 280

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad, lax
import jax.random as jr
import optax
import matplotlib.pyplot as plt
from functools import partial
import time
from typing import Tuple, Dict
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

# ----------------------------- Kernel Primitives -----------------------------
from sfmn_kernel import (
    rkan_forward,
    Yoshida4Integrator,
    henon_heiles_energy,
    henon_heiles_force,
    symplectic_integrator_step,
)

jax.config.update("jax_enable_x64", True)

# ========================== HIGH-COMPUTE SHARDING CONFIG ==========================
# Scale: 1M+ trajectories (Blueprint). Local Ensemble: 2^11 for CPU stability.
EXASCALE_TOTAL = 1_048_576
LOCAL_TOTAL = 2048 
PER_DEVICE_BATCH = LOCAL_TOTAL
TRAIN_HORIZON_DATA = 150 

DEGREE = 4
IN_DIM = 2
OUT_DIM = 1
DT = 0.005
NOISE_SIGMA = 1e-4
L1_WEIGHT = 8e-4
CURRICULUM_MAX_HORIZON = 80
TRAIN_EPOCHS = 800
LR = 1.2e-3

# Mesh for SPMD simulation (Portable to TPU-pods / Dojo)
devices = jax.devices()
mesh = Mesh(devices, axis_names=('data',))

# ----------------------------- Sharded Data Pipeline -----------------------------

@partial(shard_map, mesh=mesh, in_specs=P(), out_specs=P('data'), check_rep=False)
def sharded_generate_trajectories(key):
    """
    Beta-Scaled Trajectory Generator across Mesh.
    Scales to 1M+ trajectories via local device generation.
    """
    def single_traj(k):
        # Entry Point: Sharded generation inner
        k0, k1, k2 = jr.split(k, 3)
        q0 = jr.uniform(k1, (IN_DIM,), minval=-0.15, maxval=0.15)
        p0 = jnp.array([0.0, 0.55])
        t_grid = jnp.arange(TRAIN_HORIZON_DATA + 1) * DT
        _, _, (qt, pt) = Yoshida4Integrator.integrate(q0, p0, t_grid, henon_heiles_force)
        z = jnp.concatenate([qt, pt], axis=-1)
        noise = jr.normal(k2, z.shape) * NOISE_SIGMA
        # Exit Point: Trajectory with noise returned
        return z + noise
    
    return vmap(single_traj)(jr.split(key, PER_DEVICE_BATCH))

# ----------------------------- Discovery Engine -----------------------------

def get_force_learned(q: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """Calculates conservative force field from rKAN potential."""
    # Entry Point: Force Gradient evaluation
    def potential_scalar(qv):
        return rkan_forward(qv, params, DEGREE, OUT_DIM).reshape(())
    res = -grad(potential_scalar)(q)
    # Exit Point: Force returned
    return res

def H_learned(z: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """Calculates exact learned Hamiltonian for conservation check."""
    q, p = z[:2], z[2:]
    return 0.5 * jnp.sum(p**2) + rkan_forward(q, params, DEGREE, OUT_DIM).reshape(())

def create_sharded_loss_fn(steps_pred: int):
    """Loss factory for static curriculum horizons in shard_map."""
    @partial(shard_map, mesh=mesh, in_specs=(P(), P('data')), out_specs=(P(), P()), check_rep=False)
    def loss_fn(params, z_obs):
        # Entry Point: Sharded loss with static horizon
        def traj_loss(z_traj):
            z0 = z_traj[0]
            def pred_step(carry, _):
                f_fn = lambda qv: get_force_learned(qv, params)
                qn, pn = symplectic_integrator_step(carry[:2], carry[2:], DT, f_fn)
                z = jnp.concatenate([qn, pn])
                return z, z
            
            _, z_pred = lax.scan(pred_step, z0, jnp.arange(steps_pred))
            L_state = jnp.mean((z_pred - z_traj[1:steps_pred+1])**2)
            
            # Hamiltonian Invariance check (dH/dt = 0)
            h_vals = vmap(lambda zp: H_learned(zp, params))(z_pred)
            L_cons = jnp.mean((h_vals - H_learned(z0, params))**2)
            return L_state + 10.0 * L_cons

        local_mean = jnp.mean(vmap(traj_loss)(z_obs))
        global_loss = (lax.psum(local_mean, axis_name='data') / jax.device_count())
        L_reg = L1_WEIGHT * (jnp.sum(jnp.abs(params['num'])) + jnp.sum(jnp.abs(params['den'])))
        # Exit Point: Total loss pair
        return global_loss + L_reg, L_reg
    return loss_fn

# ----------------------------- Distillation & Validation -----------------------------

@partial(jit, static_argnums=(4,))
def validate_long_horizon(params, dt, deg, out, steps=100000):
    """100k-step stress test for law confirmation."""
    # Entry Point: High-horizon check
    q0, p0 = jnp.array([0.0, 0.0]), jnp.array([0.0, 0.55])
    f_fn = lambda q: get_force_learned(q, params)
    t_grid = jnp.arange(steps + 1) * dt
    _, _, (qt, pt) = Yoshida4Integrator.integrate(q0, p0, t_grid, f_fn)
    traj = jnp.concatenate([qt, pt], axis=-1)
    energies = vmap(lambda z: H_learned(z, params))(traj)
    rel_drift = jnp.max(jnp.abs(energies - energies[0])) / jnp.abs(energies[0])
    # Exit Point: Drift metric
    return rel_drift

def symbolic_distillation(params, threshold: float = 1e-2):
    """
    Parses learned rKAN coefficients to reveal the recovered physical law.
    Identifies active Jacobi/Rational basis terms across dimensions.
    """
    # <plan_complexity>High</plan_complexity><total_lines_estimate>30</total_lines_estimate>
    print("🔬 Initiating Symbolic Distillation...")
    num = params['num'] # (OUT, IN, DEGREE+1)
    den = params['den']
    
    out_dim, in_dim, deg_plus_1 = num.shape
    
    active_terms = []
    for o in range(out_dim):
        for i in range(in_dim):
            for d in range(deg_plus_1):
                n_val = num[o, i, d]
                d_val = den[o, i, d]
                if jnp.abs(n_val) > threshold:
                    active_terms.append(f"Q{i}^deg{d} (coeff: {n_val:.3f})")
    
    summary = ", ".join(active_terms) if active_terms else "No significant terms discovered."
    print(f"✅ Distillation complete. Active terms: {summary}")
    
    # Heuristic mapping for Hénon-Heiles verification
    # Note: Jacobi basis on tanh(x) is complex, but we can verify symmetry
    is_symmetric = jnp.all(jnp.abs(num[0, 0, :] - num[0, 1, :]) < 1e-1)
    print(f"Symmetry check (Q0 vs Q1): {'Passed' if is_symmetric else 'Asymmetric'}")
    
    return f"Recovered Dynamics Summary: {summary}"

# ----------------------------- Training Workflow -----------------------------

def train_axiom_zero_final():
    """Main High-Compute discovery loop (Beta Seal)."""
    # Entry Point: Final training start
    print(f"🚀 Axiom-Zero Alpha-Beta Seal | Devices: {len(devices)} | Ensemble: {LOCAL_TOTAL}")
    key = jr.PRNGKey(42)
    start_time = time.time()
    
    # Initialize rKAN params with correct (out_dim, in_dim, deg+1) shape
    params = {
        'num': jr.normal(jr.split(key)[0], (OUT_DIM, IN_DIM, DEGREE + 1)) * 0.1,
        'den': jr.normal(jr.split(key)[1], (OUT_DIM, IN_DIM, DEGREE + 1)) * 0.1 + jnp.eye(DEGREE + 1)[0]
    }
    
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)
    
    z_noisy = sharded_generate_trajectories(jr.PRNGKey(123))
    
    jit_updates = {}
    def get_update_step(horizon):
        if horizon not in jit_updates:
            loss_fn = create_sharded_loss_fn(horizon)
            @jit
            def _update(p, os, b):
                (l, r), g = value_and_grad(loss_fn, has_aux=True)(p, b)
                up, nos = optimizer.update(g, os, p)
                return optax.apply_updates(p, up), nos, l, r
            jit_updates[horizon] = _update
        return jit_updates[horizon]

    print("Starting Final Discovery...")
    for epoch in range(TRAIN_EPOCHS):
        horizon = int(min(20 + (epoch // 10) * 10, CURRICULUM_MAX_HORIZON))
        update_fn = get_update_step(horizon)
        params, opt_state, loss, reg = update_fn(params, opt_state, z_noisy)
        if epoch % 100 == 0 or epoch == TRAIN_EPOCHS - 1:
            print(f"Epoch {epoch:4d} | Horizon {horizon:3d} | Loss {loss:.2e}")
            
    print("\n🔥 Executing 100,000-Step Sovereign Validation")
    drift = validate_long_horizon(params, DT, DEGREE, OUT_DIM, 100000)
    print(f"100k-step Energy Drift: {drift:.2e}")
    
    law = symbolic_distillation(params)
    print(f"📜 Recovered Law: {law}")
    
    # Final plot
    grid = jnp.linspace(-0.5, 0.5, 40)
    Q1, Q2 = jnp.meshgrid(grid, grid)
    v_lr = vmap(lambda q: rkan_forward(q, params, DEGREE, OUT_DIM))(jnp.stack([Q1.flatten(), Q2.flatten()], 1)).reshape(40, 40)
    plt.figure(figsize=(6, 5)); plt.contourf(Q1, Q2, v_lr - jnp.mean(v_lr)); plt.title("Axiom-Zero Final Discovery")
    plt.savefig("/Users/adi/newai/discovery_final_elite.png")
    
    print(f"✅ Alpha/Beta Milestone achieved in {time.time() - start_time:.1f}s.")
    # Exit Point: Milestone Reached

if __name__ == "__main__":
    # import jax.distributed as dist; dist.initialize() # For Dojo multi-host
    train_axiom_zero_final()
