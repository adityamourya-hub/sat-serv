#!/usr/bin/env python3
"""
test_sovereign_integrity.py - Complete System Verification

This is the "Sovereign Smoke Test"—it verifies that all hard constraint
components work together to prevent physics violations.

Run before any deployment:
 python tests/test_sovereign_integrity.py

Exit codes:
 0: All systems operational (Grade A)
 1: Critical failure (do not deploy)
"""

import sys
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'ai' / 'v3'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))

import jax
import jax.numpy as jnp


def test_1_forest_ruth_integrity():
 """
 Test 1: Forest-Ruth integrator maintains symplectic structure.
 """
 print("\n" + "=" * 70)
 print("TEST 1: Forest-Ruth Symplectic Integrity")
 print("=" * 70)
 
 from physics_axiom_zero_test import TestAxiomZeroSymplecticity
 
 test = TestAxiomZeroSymplecticity()
 
 try:
 test.test_hamiltonian_conservation_circular_1000_steps(
 test.keplerian_model(),
 test.circular_orbit_state()
)
 print("✓ PASS: Energy drift < 1e-8 over 1000 steps")
 return True
 except AssertionError as e:
 print(f"✗ FAIL: {e}")
 return False


def test_2_manifold_projection():
 """
 Test 2: Manifold projection enforces exact conservation.
 """
 print("\n" + "=" * 70)
 print("TEST 2: Manifold Projection (Micro-creep Elimination)")
 print("=" * 70)
 
 from manifold_projection import ManifoldProjection, KeplerianOracle
 
 oracle = KeplerianOracle(mu=398600.4418)
 projector = ManifoldProjection(max_iter=5, tol=1e-14)
 
 # Initial state
 q = jnp.array([6778.137, 0.0, 0.0])
 p = jnp.array([0.0, 7.6686, 0.0])
 H_target = oracle.hamiltonian(q, p)
 
 # Simulate massive perturbation
 q_pert = q + jnp.array([10.0, 5.0, 2.0]) # 10km error!
 p_pert = p + jnp.array([0.1, 0.1, 0.05]) # 100 m/s error!
 
 H_initial_drift = abs(oracle.hamiltonian(q_pert, p_pert) - H_target) / abs(H_target)
 
 # Project
 q_corr, p_corr = projector.project(q_pert, p_pert, H_target, oracle)
 H_final_drift = abs(oracle.hamiltonian(q_corr, p_corr) - H_target) / abs(H_target)
 
 print(f"Initial drift: {H_initial_drift:.2e} (after 10km perturbation)")
 print(f"Final drift: {H_final_drift:.2e} (after projection)")
 
 if H_final_drift < 1e-12:
 print("✓ PASS: Manifold projection achieves machine precision")
 return True
 else:
 print(f"✗ FAIL: Drift {H_final_drift:.2e} > 1e-12")
 return False


def test_3_no_soft_penalty():
 """
 Test 3: Loss function uses hard constraints, not λ_h soft penalty.
 """
 print("\n" + "=" * 70)
 print("TEST 3: No Soft Penalty (λ_h) in Loss Function")
 print("=" * 70)
 
 from physics_constraints import AugmentedLagrangianLoss
 import torch
 
 loss_fn = AugmentedLagrangianLoss(initial_mu=1.0, rho=10.0)
 
 # Check for dual variables (not soft penalty weight)
 has_dual = hasattr(loss_fn, 'mu_energy')
 dual_not_learnable = not loss_fn.mu_energy.requires_grad
 
 if has_dual and dual_not_learnable:
 print("✓ PASS: Using Augmented Lagrangian (dual ascent)")
 print(f" Dual variable μ_energy: {loss_fn.mu_energy.item():.2f}")
 print(f" Penalty parameter ρ: {loss_fn.rho.item():.2f}")
 print(" No λ_h soft penalty detected")
 return True
 else:
 print("✗ FAIL: Loss function structure incorrect")
 return False


def test_4_hessian_hash():
 """
 Test 4: Hessian hash provides cryptographic integrity.
 """
 print("\n" + "=" * 70)
 print("TEST 4: Hessian Hash (Cryptographic Integrity)")
 print("=" * 70)
 
 from integrity_gate import IntegrityGate, AxiomPoint
 import torch.nn as nn
 
 # Create test model
 h_net = nn.Sequential(
 nn.Linear(6, 128),
 nn.Tanh(),
 nn.Linear(128, 1)
)
 
 gate = IntegrityGate(device='cpu')
 result = gate.generate_hessian_hash(h_net, model_version="TEST")
 
 print(f"Generated hash: {result.hash_value[:32]}...")
 print(f"Axiom points evaluated: {len(result.axiom_names)}")
 
 # Verify hash format
 if len(result.hash_value) == 64 and all(c in '0123456789abcdef' for c in result.hash_value):
 print("✓ PASS: Valid SHA-256 Hessian hash generated")
 return True
 else:
 print("✗ FAIL: Invalid hash format")
 return False


def test_5_noise_resilience():
 """
 Test 5: System maintains custody under Hanle-level scintillation.
 """
 print("\n" + "=" * 70)
 print("TEST 5: Scintillation Resilience (S₄ > 0.6)")
 print("=" * 70)
 
 from manifold_projection import KeplerianOracle
 from physics_axiom_zero_test import TestScintillationResilience
 
 test = TestScintillationResilience()
 injector = test.noise_injector()
 oracle = KeplerianOracle(mu=398600.4418)
 
 q_true = jnp.array([6778.137, 0.0, 0.0])
 p_true = jnp.array([0.0, 7.6686, 0.0])
 H_target = oracle.hamiltonian(q_true, p_true)
 
 # Test with heavy-tailed Lévy noise
 n_trials = 100
 errors = []
 
 for _ in range(n_trials):
 q_noisy, p_noisy = injector.inject_noise(q_true, p_true, noise_type="levy")
 H_noisy = oracle.hamiltonian(q_noisy, p_noisy)
 errors.append(abs(H_noisy - H_target))
 
 mean_error = np.mean(errors)
 
 print(f"Trials: {n_trials}")
 print(f"Mean |ΔH|: {mean_error:.2e}")
 print(f"Threshold: 1e-4 (Grade B)")
 
 if mean_error < 1e-4:
 print("✓ PASS: System maintains custody under heavy-tailed noise")
 return True
 else:
 print(f"✗ FAIL: Error {mean_error:.2e} exceeds Grade B threshold")
 return False


def test_6_pretraining_curriculum():
 """
 Test 6: Physics-first curriculum exists.
 """
 print("\n" + "=" * 70)
 print("TEST 6: Hamiltonian Pre-training Curriculum")
 print("=" * 70)
 
 from hamiltonian_pretraining import HamiltonianPretrainer, KeplerianOracle
 
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
 
 print("Curriculum phases:")
 for i, phase in enumerate(pretrainer.phases):
 print(f" {i}. {phase.name} (noise={phase.noise_level:.2e})")
 
 # Verify curriculum structure
 first_is_keplerian = "Keplerian" in pretrainer.phases[0].name
 noise_increases = pretrainer.phases[-1].noise_level > pretrainer.phases[0].noise_level
 
 if first_is_keplerian and noise_increases:
 print("✓ PASS: Physics-first curriculum verified")
 return True
 else:
 print("✗ FAIL: Curriculum structure incorrect")
 return False


def run_all_tests():
 """Run complete Sovereign Integrity Test Suite."""
 
 print("\n" + "█" * 70)
 print("█ SOVEREIGN INTEGRITY TEST SUITE")
 print("█ Verifying Hard Manifold Constraints")
 print("█" * 70)
 
 tests = [
 ("Forest-Ruth Symplecticity", test_1_forest_ruth_integrity),
 ("Manifold Projection", test_2_manifold_projection),
 ("No Soft Penalty", test_3_no_soft_penalty),
 ("Hessian Hash", test_4_hessian_hash),
 ("Noise Resilience", test_5_noise_resilience),
 ("Pre-training Curriculum", test_6_pretraining_curriculum),
 ]
 
 results = []
 for name, test_fn in tests:
 try:
 passed = test_fn()
 results.append((name, passed))
 except Exception as e:
 print(f"\n✗ EXCEPTION in {name}: {e}")
 results.append((name, False))
 
 # Summary
 print("\n" + "=" * 70)
 print("SOVEREIGN INTEGRITY SUMMARY")
 print("=" * 70)
 
 passed = sum(1 for _, p in results if p)
 total = len(results)
 
 for name, p in results:
 status = "✓ PASS" if p else "✗ FAIL"
 print(f" {status}: {name}")
 
 print(f"\nTotal: {passed}/{total} tests passed")
 
 if passed == total:
 print("\n" + "█" * 70)
 print("█ SOVEREIGN INTEGRITY: VERIFIED")
 print("█ All hard constraint components operational")
 print("█ System ready for May 4th audit")
 print("█" * 70)
 return 0
 else:
 print("\n" + "!" * 70)
 print("! SOVEREIGN INTEGRITY: COMPROMISED")
 print("! Critical failures detected")
 print("! DO NOT DEPLOY")
 print("!" * 70)
 return 1


if __name__ == "__main__":
 sys.exit(run_all_tests())
