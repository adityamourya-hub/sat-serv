# sfmn_kernel.py
# Axiom-Zero: High-Fidelity Symplectic Neural Physics Kernel
# Phase 2.0: Unified Legacy-Legendre Architecture
# plan_complexity: High
# total_lines_estimate: 400

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
from functools import partial

# --- Constants matching Go Kernel ---
MU_EARTH = 398600.4418
RE_EARTH = 6378.137
J2 = 1.0826e-3
RAD_DEG = jnp.pi / 180.0

# --- Basis & Rational Logic ---

def legendre_basis(x, degree):
    """
    Computes Legendre basis polynomials up to 'degree' using recurrence.
    """
    # <plan_complexity>Medium</planning_complexity><total_lines_estimate>15</total_lines_estimate>
    def recurrence(carry, _):
        p0, p1, n = carry
        p2 = ((2.0 * n + 1.0) * x * p1 - n * p0) / (n + 1.0)
        return (p1, p2, n + 1.0), p2

    p0 = 1.0
    p1 = x
    if degree == 0:
        return jnp.array([p0])
    if degree == 1:
        return jnp.array([p0, p1])
    
    # Recurrence for n >= 1 to get p2 (n=2) up to degree
    _, pk = lax.scan(recurrence, (p0, p1, 1.0), jnp.arange(degree - 1))
    return jnp.concatenate([jnp.array([p0, p1]), pk])

@partial(jit, static_argnums=(2, 3))
def rkan_forward(x_in, params, degree, out_dim=1):
    """
    Universal Padé-rKAN Forward Pass.
    Matches Go 'RKANKernel.Forward' logic.
    params['num'] shape: (out_dim, in_dim, degree+1)
    """
    # <plan_complexity>High</plan_complexity><total_lines_estimate>20</total_lines_estimate>
    def map_to_interval(val):
        return jnp.tanh(val / 1000.0)

    # Vectorized basis computation
    x_mapped = vmap(map_to_interval)(x_in)
    # basis_matrix: (in_dim, degree+1)
    basis_matrix = vmap(lambda xi: legendre_basis(xi, degree))(x_mapped)
    
    # num, den: (out_dim, in_dim, degree+1)
    num_weights = params['num']
    den_weights = params['den']
    
    # Compute rational terms
    # Using einsum for clean (out, in, deg+1) * (in, deg+1) -> (out, in)
    num_vals = jnp.einsum('oid,id->oi', num_weights, basis_matrix)
    den_vals = jnp.einsum('oid,id->oi', den_weights, basis_matrix)
    
    # Final Potential/Scalar V = sum over features of (num / (1 + |den|))
    # Shape of num_vals / (1 + abs(den_vals)) is (out_dim, in_dim)
    # Sum over in_dim -> (out_dim,)
    v_out = jnp.sum(num_vals / (1.0 + jnp.abs(den_vals)), axis=1)
    return v_out

# --- Physical Feature Manifold (13D) ---

@jit
def get_meeus_ephemeris(jd):
    """Sun and Moon unit vectors from Julian Date."""
    T = (jd - 2451545.0) / 36525.0
    L = (280.46646 + 36000.76983 * T) * RAD_DEG
    g = (357.52911 + 35999.05029 * T) * RAD_DEG
    lambda_sun = L + (1.914602 * jnp.sin(g)) + (0.019993 * jnp.sin(2.0 * g))
    epsilon = (23.439 - 0.013 * T) * RAD_DEG
    
    sun_v = jnp.array([
        jnp.cos(lambda_sun),
        jnp.sin(lambda_sun) * jnp.cos(epsilon),
        jnp.sin(lambda_sun) * jnp.sin(epsilon)
    ])
    
    l_moon = (218.316 + 481267.881 * T) * RAD_DEG
    moon_v = jnp.array([
        jnp.cos(l_moon),
        jnp.sin(l_moon) * jnp.cos(epsilon),
        jnp.sin(l_moon) * jnp.sin(epsilon)
    ])
    return sun_v, moon_v

@jit
def get_physical_features(q, p, jd):
    """Maps 3D Cartesian state to 13D physical manifold."""
    # <plan_complexity>High</plan_complexity><total_lines_estimate>40</total_lines_estimate>
    r = jnp.linalg.norm(q) + 1e-12
    u = q[2] / r # sin(lat)
    
    p2 = 0.5 * (3 * u**2 - 1)
    p3 = 0.5 * (5 * u**3 - 3 * u)
    p4 = 0.125 * (35 * u**4 - 30 * u**2 + 3)
    
    # Resonance Law mock logic (matching Go)
    rho = jnp.exp(-(r - RE_EARTH) / 50.0)
    sun_v, moon_v = get_meeus_ephemeris(jd)
    
    qu = q / r
    sun_dot = jnp.dot(qu, sun_v)
    moon_dot = jnp.dot(qu, moon_v)
    p_norm = jnp.linalg.norm(p)
    
    # Unified Resonance Feature (matching go/orbital/rkan_kernel.go:128)
    # inc calculation
    h = jnp.cross(q, p)
    h_norm = jnp.linalg.norm(h) + 1e-12
    inc = jnp.arccos(jnp.clip(h[2] / h_norm, -1.0, 1.0))
    node = jnp.arctan2(h[0], -h[1])
    coupling = jnp.sin(inc) * jnp.cos(node)
    
    v2 = jnp.sum(p**2)
    a_inv = 2.0 / r - v2 / MU_EARTH
    a = 1.0 / jnp.maximum(a_inv, 1e-10)
    n_val = jnp.sqrt(MU_EARTH / (a**3 + 1e-12))
    
    j2_pot = (MU_EARTH / r) * J2 * (RE_EARTH / r)**2
    res_feature = j2_pot * coupling * rho * v2 * n_val
    
    return jnp.array([
        q[0], q[1], q[2],
        r, 1.0/r,
        p2, p3, res_feature,
        rho, sun_dot, moon_dot,
        p4, p_norm
    ])

# --- Integrator Architecture ---

# Yoshida-4 coefficients
C_YOSHIDA = jnp.array([1.3512071919596578, -1.7024143839193156, 1.3512071919596578, 0.0])
W_YOSHIDA = jnp.array([0.6756035959798289, -0.1756035959798289, -0.1756035959798289, 0.6756035959798289])

@partial(jit, static_argnums=(3,))
def symplectic_integrator_step(q, p, dt, force_fn, jd=2451545.0):
    """Single 4th-order symplectic step (Yoshida)."""
    # <plan_complexity>Medium</plan_complexity><total_lines_estimate>15</total_lines_estimate>
    
    def stage(carry, i):
        q_v, p_v = carry
        q_new = q_v + W_YOSHIDA[i] * dt * p_v
        p_new = p_v + C_YOSHIDA[i] * dt * force_fn(q_new)
        return (q_new, p_new), None

    (q_f, p_f), _ = lax.scan(stage, (q, p), jnp.arange(4))
    return q_f, p_f

class Yoshida4Integrator:
    """Production wrapper for Axiom-Zero symplectic rollouts."""
    
    @staticmethod
    @partial(jit, static_argnums=(3,))
    def integrate(q0, p0, t_grid, force_fn):
        """Standard JAX integration loop."""
        def step(carry, t):
            q, p, t_prev = carry
            dt = t - t_prev
            # Fixed JD for simple benchmark, in production this tracks time
            qn, pn = symplectic_integrator_step(q, p, dt, force_fn)
            return (qn, pn, t), (qn, pn)
        
        _, (qt, pt) = lax.scan(step, (q0, p0, t_grid[0]), t_grid[1:])
        return None, None, (jnp.concatenate([q0[None], qt]), jnp.concatenate([p0[None], pt]))

# --- Benchmark Physics (Hénon-Heiles) ---

@jit
def henon_heiles_energy(z):
    """E = 0.5(p^2 + q^2) + q0^2*q1 - 1/3*q1^3"""
    q, p = z[:2], z[2:]
    v = 0.5 * jnp.sum(q**2) + q[0]**2 * q[1] - (1.0/3.0) * q[1]**3
    return 0.5 * jnp.sum(p**2) + v

@jit
def henon_heiles_force(q):
    """F = -grad(V)"""
    return -grad(lambda qv: 0.5 * jnp.sum(qv**2) + qv[0]**2 * qv[1] - (1.0/3.0) * qv[1]**3)(q)

# --- Legacy Hooks (Optional Alignment) ---

def hamiltonian_potential(q, params, degree=4):
    """Matches discovery.py usage if needed."""
    return rkan_forward(q, params, degree).reshape(())

def get_force(q, params, degree=4):
    return -grad(lambda qv: hamiltonian_potential(qv, params, degree))(q)

def yoshida4_step(q, p, dt, force_fn):
    return symplectic_integrator_step(q, p, dt, force_fn)

if __name__ == "__main__":
    print("🚀 Axiom-Zero: sfmn_kernel v2.0 (Legendre-Hybrid) initialized.")
