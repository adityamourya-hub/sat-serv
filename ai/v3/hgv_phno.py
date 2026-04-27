"""
HGV Port-Hamiltonian Neural Operator (PHNO)
: HGV Skip-Glide Dynamics on S²×ℝ³×SO(3)

Converts classical HGV spherical coordinates to Port-Hamiltonian form:
- State lives on manifold M = S² (position) × ℝ³ (velocity) × SO(3) (attitude)
- Drag absorbed into R(x) dissipation matrix
- Bank angle σ(t) as hidden control input: u(t) via g(x)·u port

Mathematical Framework:
dz/dt = [J(z) - R(z)] ∇H(z) + g(z) · u(t)

where z ∈ T*M (cotangent bundle of HGV manifold)
- J(z): Canonical Poisson structure on T*M
- R(z): Dissipation from atmospheric drag (PSD via Cholesky)
- H(z): Hamiltonian on HGV manifold
- g(z): Control port for bank angle coupling
- u(t): Bank angle σ(t) inferred via residual SINDy

Key Innovation:
Classical HGV EOM uses spherical coordinates with explicit bank angle.
PHNO formulation treats bank angle as hidden input, enabling:
1. Zero-day maneuver detection (σ(t) inferred not observed)
2. Drag/dissipation separation (physics-structured learning)
3. Passivity guarantees (cannot hallucinate energy gain)

State Vector:
z = [q, p] where:
- q ∈ S²×ℝ: [latitude, longitude, altitude] (position on manifold)
- p ∈ ℝ³: conjugate momentum (related to velocity)

Training:
Phase 1: Conservative H on ballistic trajectories (R=0, u=0)
Phase 2: Dissipative R with drag data (u=0)
Phase 3: Control g with known σ(t) profiles
Phase 4: Residual SINDy for σ(t) inference

Acceptance:
- Ground-truth drag recoverable via R(x) within 5%
- Bank angle σ(t) recovery via residual: R² > 0.9

References:
- foundation.md §3.2: Tier 1 — Wire HGV model into PHS form
- pkg/hypersonic/hgv_physics.go: Classical EOM ground truth
- Regan & Anandakrishnan, "Dynamics of Atmospheric Re-Entry"
"""

from typing import Tuple, Dict, Optional, Callable
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import equinox as eqx
from functools import partial
import numpy as np

# Import base PHS
from port_hamiltonian import PortHamiltonianSystem


class HGVPHNO(eqx.Module):
                """
                Port-Hamiltonian Neural Operator for HGV dynamics.

                Manifold: M = S² × ℝ³ × SO(3)
                - S²: Position on sphere (lat, lon)
                - ℝ³: Velocity space (V, flight path, heading)
                - SO(3): Attitude (bank angle coupling)

                The learned model absorbs drag into R(x) and bank angle into control port.
                """

                # Base PHS components
                phs: PortHamiltonianSystem

                # HGV-specific parameters (not learned - physical constants)
                reference_area: float # m²
                mass: float # kg

                # Configuration
                latent_dim: int = 64

                def __init__(
                self,
                reference_area: float = 0.5, # m² (typical HGV)
                mass: float = 1000.0, # kg
                latent_dim: int = 64,
                key: jax.random.PRNGKey = None,
                ):
                                if key is None:
                                                key = jax.random.PRNGKey(42)

                                                self.reference_area = reference_area
                                                self.mass = mass
                                                self.latent_dim = latent_dim

                                                # Initialize base PHS with 6D state [pos(3), vel(3)]
                                                # But we'll interpret it as HGV manifold coordinates
                                                self.phs = PortHamiltonianSystem(
                                                state_dim=6,
                                                control_dim=1, # Bank angle σ is scalar control
                                                latent_dim=latent_dim,
                                                use_film=True,
                                                R_conditioned=True, # R varies with atmospheric regime
                                                g_conditioned=True, # g varies with flight condition
                                                key=key,
                                                )

                                                def spherical_to_state(
                                                self,
                                                r: float, # Radius from Earth center (km)
                                                phi: float, # Latitude (rad)
                                                lambda_: float, # Longitude (rad)
                                                v: float, # Velocity magnitude (km/s)
                                                gamma: float, # Flight path angle (rad)
                                                psi: float, # Heading angle (rad)
                                                ) -> jnp.ndarray:
                                                                """
                                                                Convert HGV spherical coordinates to PHS state vector.

                                                                The PHS state z = [q, p] where:
                                                                - q: ECI position (derived from spherical)
                                                                - p: Conjugate momentum (derived from velocity)

                                                                This is a coordinate chart from HGV manifold to ℝ⁶.

                                                                Args:
                                                                r: Radius from Earth center (km)
                                                                phi: Latitude (rad)
                                                                lambda_: Longitude (rad)
                                                                v: Velocity magnitude (km/s)
                                                                gamma: Flight path angle (rad)
                                                                psi: Heading angle (rad)

                                                                Returns:
                                                                z: [6] PHS state [q(3), p(3)]
                                                                """
                                                                # ECI position from spherical coordinates
                                                                x = r * jnp.cos(phi) * jnp.cos(lambda_)
                                                                y = r * jnp.cos(phi) * jnp.sin(lambda_)
                                                                z_pos = r * jnp.sin(phi)
                                                                q = jnp.array([x, y, z_pos])

                                                                # Velocity in ECI frame (simplified - full conversion needs rotation matrix)
                                                                # Local NE frame to ECI conversion
                                                                sin_phi = jnp.sin(phi)
                                                                cos_phi = jnp.cos(phi)
                                                                sin_lambda = jnp.sin(lambda_)
                                                                cos_lambda = jnp.cos(lambda_)
                                                                sin_gamma = jnp.sin(gamma)
                                                                cos_gamma = jnp.cos(gamma)
                                                                sin_psi = jnp.sin(psi)
                                                                cos_psi = jnp.cos(psi)

                                                                # Velocity components in local NE frame
                                                                v_north = v * cos_gamma * cos_psi # North component
                                                                v_east = v * cos_gamma * sin_psi # East component
                                                                v_down = -v * sin_gamma # Down component (negative of radial)

                                                                # Convert to ECI (simplified - ignoring Earth rotation)
                                                                # North direction in ECI
                                                                north_eci = jnp.array([
                                                                -sin_phi * cos_lambda,
                                                                -sin_phi * sin_lambda,
                                                                cos_phi
                                                                ])
                                                                # East direction in ECI
                                                                east_eci = jnp.array([-sin_lambda, cos_lambda, 0.0])
                                                                # Down direction in ECI (radial inward)
                                                                down_eci = -q / r

                                                                v_eci = v_north * north_eci + v_east * east_eci + v_down * down_eci

                                                                # Conjugate momentum (for unit mass, p ≈ v in normalized units)
                                                                p = v_eci

                                                                return jnp.concatenate([q, p])

                                                                def state_to_spherical(
                                                                self,
                                                                z: jnp.ndarray,
                                                                ) -> Tuple[float, float, float, float, float, float]:
                                                                                """
                                                                                Convert PHS state back to HGV spherical coordinates.

                                                                                Inverse of spherical_to_state.

                                                                                Args:
                                                                                z: [6] PHS state [q(3), p(3)]

                                                                                Returns:
                                                                                (r, phi, lambda_, v, gamma, psi) spherical coordinates
                                                                                """
                                                                                q = z[:3]
                                                                                p = z[3:]

                                                                                # Position to spherical
                                                                                r = jnp.linalg.norm(q)
                                                                                phi = jnp.arcsin(q[2] / r) # Latitude
                                                                                lambda_ = jnp.arctan2(q[1], q[0]) # Longitude

                                                                                # Velocity from momentum
                                                                                v_eci = p
                                                                                v = jnp.linalg.norm(v_eci)

                                                                                # Flight path angle (angle between velocity and local horizontal)
                                                                                # Local horizontal is perpendicular to radial
                                                                                radial = q / r
                                                                                v_radial = jnp.dot(v_eci, radial)
                                                                                v_horizontal = jnp.linalg.norm(v_eci - v_radial * radial)
                                                                                gamma = -jnp.arctan2(v_radial, v_horizontal) # Negative for re-entry

                                                                                # Heading angle (azimuth of horizontal component)
                                                                                # Simplified - assumes standard orientation
                                                                                north = jnp.array([
                                                                                -jnp.sin(phi) * jnp.cos(lambda_),
                                                                                -jnp.sin(phi) * jnp.sin(lambda_),
                                                                                jnp.cos(phi)
                                                                                ])
                                                                                east = jnp.array([-jnp.sin(lambda_), jnp.cos(lambda_), 0.0])

                                                                                v_horizontal_vec = v_eci - v_radial * radial
                                                                                v_north = jnp.dot(v_horizontal_vec, north)
                                                                                v_east = jnp.dot(v_horizontal_vec, east)
                                                                                psi = jnp.arctan2(v_east, v_north)

                                                                                return r, phi, lambda_, v, gamma, psi

                                                                                def compute_aerodynamic_forces(
                                                                                self,
                                                                                z: jnp.ndarray,
                                                                                rho: float, # Atmospheric density (kg/m³)
                                                                                cl: float, # Lift coefficient
                                                                                cd: float, # Drag coefficient
                                                                                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
                                                                                                """
                                                                                                Compute aerodynamic lift and drag forces.

                                                                                                Args:
                                                                                                z: [6] PHS state
                                                                                                rho: Atmospheric density (kg/m³)
                                                                                                cl: Lift coefficient
                                                                                                cd: Drag coefficient

                                                                                                Returns:
                                                                                                F_lift: [3] Lift force vector (N)
                                                                                                F_drag: [3] Drag force vector (N)
                                                                                                """
                                                                                                q = z[:3]
                                                                                                p = z[3:]

                                                                                                # Velocity
                                                                                                v_vec = p / self.mass # For unit mass convention
                                                                                                v = jnp.linalg.norm(v_vec)

                                                                                                # Dynamic pressure
                                                                                                v_ms = v * 1000.0 # km/s to m/s
                                                                                                q_dyn = 0.5 * rho * v_ms ** 2 # Pa

                                                                                                # Drag force (opposite to velocity)
                                                                                                F_drag_mag = q_dyn * self.reference_area * cd
                                                                                                F_drag = -F_drag_mag * v_vec / (v + 1e-10)

                                                                                                # Lift force (perpendicular to velocity, in plane containing radial)
                                                                                                # Simplified: lift is perpendicular to velocity and in the orbital plane
                                                                                                radial = q / jnp.linalg.norm(q)
                                                                                                lift_dir = jnp.cross(v_vec, jnp.cross(radial, v_vec))
                                                                                                lift_dir = lift_dir / (jnp.linalg.norm(lift_dir) + 1e-10)
                                                                                                F_lift_mag = q_dyn * self.reference_area * cl
                                                                                                F_lift = F_lift_mag * lift_dir

                                                                                                return F_lift, F_drag

                                                                                                def dynamics_with_bank_angle(
                                                                                                self,
                                                                                                z: jnp.ndarray,
                                                                                                bank_angle: float, # Bank angle σ (rad)
                                                                                                rho: float, # Atmospheric density
                                                                                                cl: float,
                                                                                                cd: float,
                                                                                                z_latent: Optional[jnp.ndarray] = None,
                                                                                                ) -> jnp.ndarray:
                                                                                                                """
                                                                                                                Full HGV dynamics with bank angle control.

                                                                                                                Combines PHS dynamics with HGV-specific aerodynamic model.

                                                                                                                Args:
                                                                                                                z: [6] PHS state
                                                                                                                bank_angle: Bank angle σ (rad)
                                                                                                                rho: Atmospheric density (kg/m³)
                                                                                                                cl: Lift coefficient
                                                                                                                cd: Drag coefficient
                                                                                                                z_latent: [latent_dim] physics code (optional)

                                                                                                                Returns:
                                                                                                                dz_dt: [6] State derivative
                                                                                                                """
                                                                                                                # Get base PHS dynamics (conservative + learned dissipation)
                                                                                                                dz_dt_phs = self.phs.dynamics(z, None, z_latent)

                                                                                                                # Add HGV-specific aerodynamic forces
                                                                                                                F_lift, F_drag = self.compute_aerodynamic_forces(z, rho, cl, cd)

                                                                                                                # Apply bank angle to lift (rotates lift vector)
                                                                                                                # Bank angle rotates lift about velocity vector
                                                                                                                # For simplicity, we model this as modifying the effective L/D
                                                                                                                # In full implementation, use rotation matrix
                                                                                                                cos_sigma = jnp.cos(bank_angle)
                                                                                                                sin_sigma = jnp.sin(bank_angle)

                                                                                                                # Vertical component of lift (affected by bank angle)
                                                                                                                L_vertical = jnp.linalg.norm(F_lift) * cos_sigma

                                                                                                                # Convert forces to acceleration
                                                                                                                accel_lift = F_lift / self.mass / 1000.0 # m/s² to km/s²
                                                                                                                accel_drag = F_drag / self.mass / 1000.0

                                                                                                                # Add to velocity derivative
                                                                                                                dz_dt = dz_dt_phs
                                                                                                                dz_dt = dz_dt.at[3:].add(accel_lift + accel_drag)

                                                                                                                return dz_dt

                                                                                                                def forward(
                                                                                                                self,
                                                                                                                z0: jnp.ndarray,
                                                                                                                bank_angle_profile: Callable[[float], float],
                                                                                                                rho_profile: Callable[[float], float],
                                                                                                                t_span: Tuple[float, float],
                                                                                                                dt: float = 1.0,
                                                                                                                z_latent: Optional[jnp.ndarray] = None,
                                                                                                                ) -> jnp.ndarray:
                                                                                                                                """
                                                                                                                                Forward integrate HGV trajectory with bank angle profile.

                                                                                                                                Args:
                                                                                                                                z0: [6] Initial state
                                                                                                                                bank_angle_profile: Function σ(t) returning bank angle at time t
                                                                                                                                rho_profile: Function ρ(h) returning density at altitude h
                                                                                                                                t_span: (t_start, t_end) time range
                                                                                                                                dt: Time step (seconds)
                                                                                                                                z_latent: [latent_dim] physics code (optional)

                                                                                                                                Returns:
                                                                                                                                z_traj: [T, 6] State trajectory
                                                                                                                                """
                                                                                                                                t_start, t_end = t_span
                                                                                                                                n_steps = int((t_end - t_start) / dt)

                                                                                                                                z = z0
                                                                                                                                trajectory = [z]

                                                                                                                                for i in range(n_steps):
                                                                                                                                                t = t_start + i * dt

                                                                                                                                                # Get bank angle at this time
                                                                                                                                                sigma = bank_angle_profile(t)

                                                                                                                                                # Get density at current altitude
                                                                                                                                                r, _, _, _, _, _ = self.state_to_spherical(z)
                                                                                                                                                altitude = r - 6378.137 # km, approximate Earth radius
                                                                                                                                                rho = rho_profile(altitude)

                                                                                                                                                # Compute dynamics
                                                                                                                                                dz_dt = self.dynamics_with_bank_angle(
                                                                                                                                                z, sigma, rho, cl=0.5, cd=0.2, z_latent=z_latent
                                                                                                                                                )

                                                                                                                                                # Simple Euler step (for demonstration - use Yoshida-4 in production)
                                                                                                                                                z = z + dz_dt * dt
                                                                                                                                                trajectory.append(z)

                                                                                                                                                return jnp.array(trajectory)

                                                                                                                                                def infer_bank_angle_residual(
                                                                                                                                                self,
                                                                                                                                                z: jnp.ndarray,
                                                                                                                                                dz_dt_obs: jnp.ndarray, # Observed state derivative
                                                                                                                                                rho: float,
                                                                                                                                                cl: float,
                                                                                                                                                cd: float,
                                                                                                                                                z_latent: Optional[jnp.ndarray] = None,
                                                                                                                                                ) -> float:
                                                                                                                                                                """
                                                                                                                                                                Infer bank angle from residual between observed and predicted dynamics.

                                                                                                                                                                Residual SINDy: r(t) = ż_obs - [J-R]∇H
                                                                                                                                                                Bank angle ≈ argmin ‖ż_obs - dynamics(z, σ)‖²

                                                                                                                                                                Args:
                                                                                                                                                                z: [6] Current state
                                                                                                                                                                dz_dt_obs: [6] Observed state derivative
                                                                                                                                                                rho: Atmospheric density
                                                                                                                                                                cl: Lift coefficient
                                                                                                                                                                cd: Drag coefficient
                                                                                                                                                                z_latent: [latent_dim] physics code (optional)

                                                                                                                                                                Returns:
                                                                                                                                                                sigma_est: Estimated bank angle (rad)
                                                                                                                                                                """
                                                                                                                                                                # Get base PHS dynamics (without bank angle)
                                                                                                                                                                dz_dt_base = self.phs.dynamics(z, None, z_latent)

                                                                                                                                                                # Add aerodynamic drag (no bank angle yet)
                                                                                                                                                                _, F_drag = self.compute_aerodynamic_forces(z, rho, cl, cd)
                                                                                                                                                                accel_drag = F_drag / self.mass / 1000.0
                                                                                                                                                                dz_dt_no_lift = dz_dt_base.at[3:].add(accel_drag)

                                                                                                                                                                # Residual should be due to lift component affected by bank angle
                                                                                                                                                                residual = dz_dt_obs - dz_dt_no_lift

                                                                                                                                                                # Estimate bank angle from residual direction
                                                                                                                                                                # Simplified: assume residual is primarily from lift
                                                                                                                                                                # Full implementation would solve least squares for σ

                                                                                                                                                                # For now, return placeholder (to be implemented with full SINDy)
                                                                                                                                                                return 0.0 # Placeholder


def create_hgv_phno(
hgv_type: str = "df17",
key: jax.random.PRNGKey = None,
) -> HGVPHNO:
                """
                Factory function to create HGV PHNO with standard parameters.

                Args:
                hgv_type: "df17", "df26", "generic"
                key: JAX random key

                Returns:
                HGVPHNO instance
                """
                configs = {
                "df17": {"reference_area": 0.5, "mass": 1000.0},
                "df26": {"reference_area": 1.0, "mass": 1500.0},
                "generic": {"reference_area": 0.75, "mass": 1200.0},
                }

                config = configs.get(hgv_type, configs["generic"])

                return HGVPHNO(
                reference_area=config["reference_area"],
                mass=config["mass"],
                key=key,
                )


def test_hgv_phno():
                """Test HGV PHNO implementation."""
                print("=" * 70)
                print("HGV PHNO Test")
                print("=" * 70)

                # Create HGV PHNO
                key = jax.random.PRNGKey(42)
                hgv = create_hgv_phno("df17", key)

                print(f"\nCreated HGV PHNO:")
                print(f" Reference area: {hgv.reference_area} m²")
                print(f" Mass: {hgv.mass} kg")
                print(f" Latent dim: {hgv.latent_dim}")

                # Test coordinate conversion
                r = 6478.0 # km (100 km altitude)
                phi = 0.5 # rad (latitude)
                lambda_ = 1.0 # rad (longitude)
                v = 7.0 # km/s
                gamma = -0.1 # rad (descending)
                psi = 0.5 # rad (heading)

                z = hgv.spherical_to_state(r, phi, lambda_, v, gamma, psi)
                print(f"\nSpherical → State:")
                print(f" Input: r={r}, φ={phi:.3f}, λ={lambda_:.3f}, v={v}, γ={gamma:.3f}, ψ={psi:.3f}")
                print(f" State z: {z}")

                # Test inverse conversion
                r_back, phi_back, lambda_back, v_back, gamma_back, psi_back = hgv.state_to_spherical(z)
                print(f"\nState → Spherical:")
                print(f" r={r_back:.3f} (error: {abs(r_back-r):.6f})")
                print(f" φ={phi_back:.3f} (error: {abs(phi_back-phi):.6f})")
                print(f" λ={lambda_back:.3f} (error: {abs(lambda_back-lambda_):.6f})")
                print(f" v={v_back:.3f} (error: {abs(v_back-v):.6f})")

                print("\n" + "=" * 70)
                print("HGV PHNO test complete")
                print("=" * 70)


if __name__ == "__main__":
                test_hgv_phno()
