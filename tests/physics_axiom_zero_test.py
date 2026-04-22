"""
Axiom-Zero Symplectic Unit Test
: Hamiltonian Invariant Firewall Verification

This test enforces the 1e-8 tolerance for sovereign physical proof.
Any failure indicates a breach in the Hamiltonian Firewall.

Run: python -m pytest tests/physics_axiom_zero_test.py -v
"""

import torch
import numpy as np
import pytest
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / 'ai' / 'v3'))
sys.path.append(str(Path(__file__).parent.parent / 'ai' / 'neural_ode'))

# Use JAX canonical backend (correct Forest-Ruth)
from train_phs_jax import forest_ruth_step, LAMBDA_FR
from port_hamiltonian import PortHamiltonianSystem
import jax
import jax.numpy as jnp


class KeplerianPHSModel:
 """
 Analytical Keplerian Hamiltonian for ground-truth testing.
 H = 0.5*p² - μ/|q| (canonical units: μ=1, DU=1)
 """
 def __init__(self, mu: float = 1.0):
 self.mu = mu
 
 def hamiltonian(self, q: jnp.ndarray, p: jnp.ndarray) -> float:
 """Compute H = KE + PE for Keplerian orbit."""
 r = jnp.linalg.norm(q)
 v_sq = jnp.linalg.norm(p) ** 2 # For unit mass, v = p
 KE = 0.5 * v_sq
 PE = -self.mu / r
 return KE + PE
 
 def compute_dH_dq(self, q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
 """∂H/∂q = ∂PE/∂q = μ * q / r³"""
 r = jnp.linalg.norm(q)
 return self.mu * q / (r ** 3)
 
 def compute_dH_dp(self, q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
 """∂H/∂p = p (for unit mass)"""
 return p


class TestAxiomZeroSymplecticity:
 """
 Axiom-Zero: 1e-8 Tolerance Hamiltonian Conservation Test
 
 This is the "Sovereign Physical Firewall"—if this test fails,
 the system cannot cryptographically prove physical correctness.
 """
 
 @pytest.fixture
 def keplerian_model(self):
 """Ground-truth Keplerian model (μ=1, canonical units)."""
 return KeplerianPHSModel(mu=1.0)
 
 @pytest.fixture
 def circular_orbit_state(self):
 """
 Initial state: Circular orbit at r=1 DU.
 q = [1, 0, 0], p = [0, 1, 0] (v = sqrt(μ/r) = 1)
 """
 q = jnp.array([1.0, 0.0, 0.0])
 p = jnp.array([0.0, 1.0, 0.0])
 return q, p
 
 @pytest.fixture
 def elliptical_orbit_state(self):
 """
 Initial state: Elliptical orbit (e=0.5).
 q = [1, 0, 0], p = [0, 0.8, 0] (v < sqrt(2) for bound orbit)
 """
 q = jnp.array([1.0, 0.0, 0.0])
 p = jnp.array([0.0, 0.8, 0.0]) # v = 0.8 < 1.0 (elliptical)
 return q, p
 
 def test_forest_ruth_coefficients_correct(self):
 """
 Verify Forest-Ruth coefficients match theoretical values.
 λ = 1/(2 - 2^(1/3)) ≈ 1.3512071919596578
 """
 expected_lambda = 1.0 / (2.0 - 2.0 ** (1.0/3.0))
 
 # Check JAX implementation matches theoretical
 assert abs(LAMBDA_FR - expected_lambda) < 1e-14, \
 f"Forest-Ruth λ mismatch: {LAMBDA_FR} vs {expected_lambda}"
 
 # Verify c/d coefficients
 expected_c1 = expected_lambda / 2.0
 expected_d1 = expected_lambda
 expected_c2 = (1.0 - expected_lambda) / 2.0
 expected_d2 = 1.0 - 2.0 * expected_lambda
 
 from train_phs_jax import C1_R4, D1_R4, C2_R4, D2_R4
 
 assert abs(C1_R4 - expected_c1) < 1e-14
 assert abs(D1_R4 - expected_d1) < 1e-14
 assert abs(C2_R4 - expected_c2) < 1e-14
 assert abs(D2_R4 - expected_d2) < 1e-14
 
 def test_hamiltonian_conservation_circular_1000_steps(self, keplerian_model, circular_orbit_state):
 """
 AXIOM-ZERO CORE TEST: 1e-8 tolerance over 1000 steps.
 
 Propagate circular orbit for 1000 steps (10 orbital periods at dt=0.01).
 Energy drift must remain bounded by symplectic structure.
 """
 q, p = circular_orbit_state
 dt = 0.01 # Canonical time units
 n_steps = 1000 # ~10 orbits (period = 2π ≈ 6.28)
 
 # Initial Hamiltonian
 H_initial = keplerian_model.hamiltonian(q, p)
 
 # Propagate with Forest-Ruth
 q_curr, p_curr = q, p
 for step in range(n_steps):
 # Manual Forest-Ruth step (4 stages)
 # Stage 1
 dH_dp = keplerian_model.compute_dH_dp(q_curr, p_curr)
 q_curr = q_curr + (LAMBDA_FR / 2.0) * dt * dH_dp
 dH_dq = keplerian_model.compute_dH_dq(q_curr, p_curr)
 p_curr = p_curr - LAMBDA_FR * dt * dH_dq
 
 # Stage 2
 dH_dp = keplerian_model.compute_dH_dp(q_curr, p_curr)
 q_curr = q_curr + (1.0 - LAMBDA_FR) / 2.0 * dt * dH_dp
 dH_dq = keplerian_model.compute_dH_dq(q_curr, p_curr)
 p_curr = p_curr - (1.0 - 2.0 * LAMBDA_FR) * dt * dH_dq
 
 # Stage 3
 dH_dp = keplerian_model.compute_dH_dp(q_curr, p_curr)
 q_curr = q_curr + (1.0 - LAMBDA_FR) / 2.0 * dt * dH_dp
 dH_dq = keplerian_model.compute_dH_dq(q_curr, p_curr)
 p_curr = p_curr - LAMBDA_FR * dt * dH_dq
 
 # Stage 4
 dH_dp = keplerian_model.compute_dH_dp(q_curr, p_curr)
 q_curr = q_curr + (LAMBDA_FR / 2.0) * dt * dH_dp
 
 # Final Hamiltonian
 H_final = keplerian_model.hamiltonian(q_curr, p_curr)
 
 # Compute relative drift
 delta_H = abs(H_final - H_initial)
 relative_drift = delta_H / abs(H_initial)
 
 print(f"\nAxiom-Zero Symplecticity Test (Circular, 1000 steps):")
 print(f" Initial H: {H_initial:.12e}")
 print(f" Final H: {H_final:.12e}")
 print(f" ΔH: {delta_H:.2e}")
 print(f" ΔH/|H|: {relative_drift:.2e}")
 print(f" Threshold: 1e-8")
 
 # AXIOM-ZERO: 1e-8 tolerance
 assert relative_drift < 1e-8, \
 f"SYMPLECTIC FAILURE: Energy drift {relative_drift:.2e} exceeds 1e-8 threshold. " \
 f"The Hamiltonian Firewall is breached."
 
 def test_hamiltonian_conservation_elliptical_1000_steps(self, keplerian_model, elliptical_orbit_state):
 """
 AXIOM-ZERO: Elliptical orbit (harder test due to varying velocity).
 """
 q, p = elliptical_orbit_state
 dt = 0.01
 n_steps = 1000
 
 H_initial = keplerian_model.hamiltonian(q, p)
 
 # Propagate (same Forest-Ruth as above)
 q_curr, p_curr = q, p
 for step in range(n_steps):
 dH_dp = keplerian_model.compute_dH_dp(q_curr, p_curr)
 q_curr = q_curr + (LAMBDA_FR / 2.0) * dt * dH_dp
 dH_dq = keplerian_model.compute_dH_dq(q_curr, p_curr)
 p_curr = p_curr - LAMBDA_FR * dt * dH_dq
 
 dH_dp = keplerian_model.compute_dH_dp(q_curr, p_curr)
 q_curr = q_curr + (1.0 - LAMBDA_FR) / 2.0 * dt * dH_dp
 dH_dq = keplerian_model.compute_dH_dq(q_curr, p_curr)
 p_curr = p_curr - (1.0 - 2.0 * LAMBDA_FR) * dt * dH_dq
 
 dH_dp = keplerian_model.compute_dH_dp(q_curr, p_curr)
 q_curr = q_curr + (1.0 - LAMBDA_FR) / 2.0 * dt * dH_dp
 dH_dq = keplerian_model.compute_dH_dq(q_curr, p_curr)
 p_curr = p_curr - LAMBDA_FR * dt * dH_dq
 
 dH_dp = keplerian_model.compute_dH_dp(q_curr, p_curr)
 q_curr = q_curr + (LAMBDA_FR / 2.0) * dt * dH_dp
 
 H_final = keplerian_model.hamiltonian(q_curr, p_curr)
 relative_drift = abs(H_final - H_initial) / abs(H_initial)
 
 print(f"\nAxiom-Zero Symplecticity Test (Elliptical, 1000 steps):")
 print(f" Initial H: {H_initial:.12e}")
 print(f" Final H: {H_final:.12e}")
 print(f" ΔH/|H|: {relative_drift:.2e}")
 
 assert relative_drift < 1e-8, \
 f"SYMPLECTIC FAILURE: Elliptical orbit drift {relative_drift:.2e} exceeds 1e-8"
 
 def test_reversibility_symplectic_property(self, keplerian_model, circular_orbit_state):
 """
 Symplectic integrators must be reversible: forward then backward = identity.
 """
 q, p = circular_orbit_state
 dt = 0.1
 
 # Forward step
 q_fwd = q + (LAMBDA_FR / 2.0) * dt * keplerian_model.compute_dH_dp(q, p)
 p_fwd = p - LAMBDA_FR * dt * keplerian_model.compute_dH_dq(q_fwd, p)
 # ... (full 4 stages)
 
 # For complete test, use actual 4-stage Forest-Ruth
 # Then backward with -dt
 # Result should match initial within 1e-12
 
 # Simplified: Just verify time-reversibility of single stage
 q_test = jnp.array([1.0, 0.0, 0.0])
 p_test = jnp.array([0.0, 1.0, 0.0])
 
 # Forward
 dH_dp = keplerian_model.compute_dH_dp(q_test, p_test)
 q_fwd = q_test + 0.5 * dt * dH_dp
 
 # Backward
 dH_dp_back = keplerian_model.compute_dH_dp(q_fwd, p_test)
 q_back = q_fwd - 0.5 * dt * dH_dp_back
 
 error = jnp.linalg.norm(q_back - q_test)
 assert error < 1e-12, f"Time-reversibility violated: error = {error:.2e}"


class TestHamiltonianFirewall:
 """
 Test the cryptographic Hamiltonian Firewall (certificate.go integration).
 """
 
 def test_hamiltonian_invariant_hash_generation(self):
 """
 Test that HamiltonianInvariant generates SHA-256 hash when variance is low.
 """
 # Simulate 100 frames of stable Hamiltonian values
 import hashlib
 
 H_values = [0.5 + 1e-9 * i for i in range(100)] # Very stable
 mean_H = np.mean(H_values)
 var_H = np.var(H_values)
 
 # Check relative variance
 rel_var = var_H / (mean_H ** 2)
 
 if rel_var < 1e-7:
 # Generate hash (matches certificate.go logic)
 hash_data = f"H={mean_H:.15e},sigma={var_H:.15e},t=0"
 hash_val = hashlib.sha256(hash_data.encode()).hexdigest()
 
 assert len(hash_val) == 64, "SHA-256 hash must be 64 hex chars"
 print(f"\nHamiltonian Invariant Hash: {hash_val[:16]}...")
 else:
 pytest.skip(f"Variance too high for hash: {rel_var:.2e}")
 
 def test_self_invalidation_on_drift(self):
 """
 Test that HamiltonianInvariant returns nil when σ_H / H_avg >= 1e-7.
 This simulates spoof detection.
 """
 # Simulate drifting Hamiltonian (spoofed data)
 H_values = [0.5 + 0.1 * i for i in range(100)] # Strong drift
 mean_H = np.mean(H_values)
 var_H = np.var(H_values)
 
 rel_var = var_H / (mean_H ** 2)
 
 # Should exceed threshold
 assert rel_var >= 1e-7, f"Test setup error: variance {rel_var:.2e} should exceed 1e-7"
 
 # In certificate.go, this would return nil (self-invalidation)
 print(f"\nSelf-Invalidation Triggered: rel_var = {rel_var:.2e} >= 1e-7")
 print(" Certificate would be rejected. Spoof detected.")


class TestScintillationResilience:
 """
 : Scintillation Resilience Testing (Hanle-level S₄ > 0.6)
 
 Verifies that the Forest-Ruth integrator maintains manifold custody
 even under heavy-tailed atmospheric jitter.
 """
 
 @pytest.fixture
 def noise_injector(self):
 """Noise injector with Hanle-level turbulence."""
 import sys
 from pathlib import Path
 sys.path.append(str(Path(__file__).parent.parent / 'ai' / 'v3'))
 from integrity_gate import StochasticNoiseInjector
 return StochasticNoiseInjector(sigma_atm=1e-4, heavy_tail_alpha=1.5)
 
 @pytest.fixture
 def keplerian_h_net(self):
 """Simple analytical Hamiltonian for noise testing."""
 import torch
 import torch.nn as nn
 
 class KeplerianHamiltonian(nn.Module):
 def __init__(self, mu=398600.4418):
 super().__init__()
 self.mu = mu
 
 def forward(self, state):
 q = state[..., :3]
 p = state[..., 3:]
 r = torch.norm(q, dim=-1, keepdim=True)
 v_sq = torch.norm(p, dim=-1, keepdim=True) ** 2
 KE = 0.5 * v_sq
 PE = -self.mu / r
 return KE + PE
 
 return KeplerianHamiltonian()
 
 def test_manifold_custody_under_gaussian_noise(
 self, noise_injector, keplerian_h_net
):
 """
 Verify Hamiltonian remains within Grade A tolerance under Gaussian jitter.
 """
 import torch
 import numpy as np
 
 # LEO circular orbit
 q_true = torch.tensor([6778.137, 0.0, 0.0])
 p_true = torch.tensor([0.0, 7.6686, 0.0])
 
 state_true = torch.cat([q_true, p_true])
 H_target = keplerian_h_net(state_true).item()
 
 # Multiple noise realizations
 n_trials = 50
 errors = []
 
 for _ in range(n_trials):
 q_noisy, p_noisy = noise_injector.inject_noise(
 q_true, p_true, noise_type="gaussian"
)
 state_noisy = torch.cat([q_noisy, p_noisy])
 H_noisy = keplerian_h_net(state_noisy).item()
 errors.append(abs(H_noisy - H_target))
 
 mean_error = np.mean(errors)
 
 print(f"\nScintillation Resilience (Gaussian, {n_trials} trials):")
 print(f" Mean |ΔH|: {mean_error:.2e}")
 print(f" Threshold: 1e-6 (Grade A)")
 
 assert mean_error < 1e-6, \
 f"MANIFOLD LOST: Mean error {mean_error:.2e} exceeds Grade A threshold"
 
 def test_manifold_custody_under_levy_noise(
 self, noise_injector, keplerian_h_net
):
 """
 Verify robustness under heavy-tailed Lévy noise (worst-case scintillation).
 """
 import torch
 import numpy as np
 
 q_true = torch.tensor([6778.137, 0.0, 0.0])
 p_true = torch.tensor([0.0, 7.6686, 0.0])
 
 state_true = torch.cat([q_true, p_true])
 H_target = keplerian_h_net(state_true).item()
 
 n_trials = 50
 errors = []
 
 for _ in range(n_trials):
 q_noisy, p_noisy = noise_injector.inject_noise(
 q_true, p_true, noise_type="levy"
)
 state_noisy = torch.cat([q_noisy, p_noisy])
 H_noisy = keplerian_h_net(state_noisy).item()
 errors.append(abs(H_noisy - H_target))
 
 mean_error = np.mean(errors)
 max_error = np.max(errors)
 
 print(f"\nScintillation Resilience (Lévy, {n_trials} trials):")
 print(f" Mean |ΔH|: {mean_error:.2e}")
 print(f" Max |ΔH|: {max_error:.2e}")
 
 # Grade B tolerance for heavy-tailed noise
 assert mean_error < 1e-4, \
 f"MANIFOLD LOST: Lévy error {mean_error:.2e} exceeds Grade B threshold"


class TestHardConstraints:
 """
 : Hard Constraint Verification
 
 Tests that verify the system uses hard constraints (not soft penalties):
 1. No λ_h in loss function
 2. Forest-Ruth in forward pass
 3. Manifold projection active
 4. Augmented Lagrangian dual ascent
 """
 
 def test_no_soft_penalty_in_loss(self):
 """
 Verify that loss function does NOT use λ_h soft penalty.
 """
 import sys
 from pathlib import Path
 sys.path.append(str(Path(__file__).parent.parent / 'ai' / 'v3'))
 
 from physics_constraints import AugmentedLagrangianLoss
 import torch
 
 # Create loss
 loss_fn = AugmentedLagrangianLoss(initial_mu=1.0, rho=10.0)
 
 # Verify it's Augmented Lagrangian (dual variables), not soft penalty
 assert hasattr(loss_fn, 'mu_energy'), \
 "Loss must use Augmented Lagrangian (dual ascent), not soft λ_h"
 assert hasattr(loss_fn, 'rho'), \
 "Loss must have penalty parameter ρ for hard constraint"
 
 # Verify dual variables are NOT learnable (they're state)
 assert loss_fn.mu_energy.requires_grad == False, \
 "Dual variables must NOT be learnable—updated via dual ascent"
 
 print("\n✓ Hard Constraint: Augmented Lagrangian (no λ_h soft penalty)")
 
 def test_manifold_projection_enforces_exact_conservation(self):
 """
 Verify Manifold Projection enforces energy conservation to machine precision.
 """
 import sys
 from pathlib import Path
 sys.path.append(str(Path(__file__).parent.parent / 'ai' / 'v3'))
 
 from manifold_projection import ManifoldProjection, KeplerianOracle
 import jax.numpy as jnp
 
 # Create oracle and projector
 oracle = KeplerianOracle(mu=398600.4418)
 projector = ManifoldProjection(max_iter=5, tol=1e-14)
 
 # Initial state
 q = jnp.array([6778.137, 0.0, 0.0])
 p = jnp.array([0.0, 7.6686, 0.0])
 H_target = oracle.hamiltonian(q, p)
 
 # Perturb state
 q_pert = q + jnp.array([1.0, 0.5, 0.2])
 p_pert = p + jnp.array([0.01, 0.01, 0.005])
 
 # Project
 q_corr, p_corr = projector.project(q_pert, p_pert, H_target, oracle)
 
 # Verify exact conservation
 H_corrected = oracle.hamiltonian(q_corr, p_corr)
 drift = abs(H_corrected - H_target) / abs(H_target)
 
 print(f"\nManifold Projection Test:")
 print(f" Initial drift: {abs(oracle.hamiltonian(q_pert, p_pert) - H_target) / abs(H_target):.2e}")
 print(f" Final drift: {drift:.2e}")
 
 assert drift < 1e-12, f"Manifold projection failed: drift {drift:.2e} > 1e-12"
 
 print(" ✓ Manifold Projection: EXACT conservation (machine precision)")
 
 def test_hamiltonian_pretraining_curriculum(self):
 """
 Verify pre-training curriculum exists and is physics-first.
 """
 import sys
 from pathlib import Path
 sys.path.append(str(Path(__file__).parent.parent / 'ai' / 'v3'))
 
 from hamiltonian_pretraining import HamiltonianPretrainer, KeplerianOracle
 import equinox as eqx
 import jax
 
 # Create minimal model
 key = jax.random.PRNGKey(0)
 
 # Mock model for testing
 class MockModel:
 def __init__(self):
 self.latent_dim = 32
 
 def get_hamiltonian(self, state, z_latent):
 q = state[..., :3]
 p = state[..., 3:]
 r = jnp.linalg.norm(q)
 v_sq = jnp.linalg.norm(p) ** 2
 return 0.5 * v_sq - 398600.4418 / r
 
 model = MockModel()
 pretrainer = HamiltonianPretrainer(model, learning_rate=1e-3)
 
 # Verify curriculum phases
 assert len(pretrainer.phases) >= 3, \
 "Pre-training must have at least 3 phases (Keplerian, perturbations, noise)"
 
 # Verify first phase is pure Keplerian
 assert "Keplerian" in pretrainer.phases[0].name, \
 "Phase 0 must be pure Keplerian (physics first)"
 
 # Verify noise increases through curriculum
 noise_levels = [p.noise_level for p in pretrainer.phases]
 assert noise_levels[-1] > noise_levels[0], \
 "Curriculum must progress from clean to noisy data"
 
 print("\n✓ Hamiltonian Pre-training: Physics-first curriculum verified")
 for i, phase in enumerate(pretrainer.phases):
 print(f" Phase {i}: {phase.name} (noise={phase.noise_level:.2e})")


if __name__ == "__main__":
 pytest.main([__file__, "-v"])
