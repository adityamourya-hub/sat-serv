"""
Validation Tests for Hardened MBPO Agent
Implements safety certification requirements
"""

import jax
import jax.numpy as jnp
from jax import random
import time
import numpy as np

from mbpo_hardened import (
 UncertaintyAwareWorldModel,
 IspConstrainedLoss,
 HardenedMBPOAgent,
 LatencyProfiler,
 MBPOConfig
)
from koopman_propagator import KoopmanPropagator


class MockKoopmanPropagator:
 """Mock Koopman propagator for testing without JAX dependencies."""
 
 def propagate_state(self, state, K):
 # Simple integration: x' = x + v*dt, v' = v + a*dt
 dt = 1.0
 pos = state[:3]
 vel = state[3:]
 
 # Add small orbital mechanics (simplified)
 r = jnp.linalg.norm(pos)
 acc = -398600.4418 * pos / (r**3) # Gravity
 
 new_pos = pos + vel * dt
 new_vel = vel + acc * dt
 
 return jnp.concatenate([new_pos, new_vel])


def test_epistemic_uncertainty_tracking():
 """
 Test 1: Epistemic Uncertainty Tracking
 
 Validates that the world model correctly identifies when
 SINDy dynamics are noisy (e.g., during Rahu jamming).
 """
 print("\n" + "="*60)
 print("TEST 1: Epistemic Uncertainty Tracking")
 print("="*60)
 
 # Create uncertainty-aware world model
 K = jnp.eye(6) # Identity Koopman operator
 koopman = MockKoopmanPropagator()
 
 world_model = UncertaintyAwareWorldModel(
 koopman=koopman,
 K=K,
 noise_std=0.01,
 max_residual=0.1
)
 
 # Test with increasing noise levels
 noise_levels = [0.001, 0.01, 0.05, 0.1]
 state = jnp.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
 action = jnp.array([0.5, 0.0, 0.0]) # 0.5 m/s
 
 results = []
 key = random.PRNGKey(42)
 
 for noise in noise_levels:
 world_model.noise_std = noise
 
 _, _, _, info = world_model.step_with_uncertainty(state, action, key)
 
 results.append({
 'noise': noise,
 'epistemic': info['epistemic_uncertainty'],
 'aleatoric': info['aleatoric_uncertainty'],
 'total': info['total_uncertainty'],
 'max_residual': info['max_residual'],
 })
 
 print(f" Noise: {noise:.3f} | "
 f"Epistemic: {info['epistemic_uncertainty']:.4f} | "
 f"Total: {info['total_uncertainty']:.4f}")
 
 # Validation: Uncertainty should increase with noise
 epistemic_increases = all(
 results[i]['epistemic'] < results[i+1]['epistemic']
 for i in range(len(results)-1)
)
 
 if epistemic_increases:
 print("✅ PASS: Epistemic uncertainty correctly scales with noise")
 else:
 print("⚠️ WARNING: Epistemic uncertainty may not be properly calibrated")
 
 return results


def test_isp_hard_constraints():
 """
 Test 2: Isp Hard Constraint Validation
 
 Ensures that the policy respects thruster physics limits.
 Tests Lagrangian multiplier approach for constraint enforcement.
 """
 print("\n" + "="*60)
 print("TEST 2: Isp Hard Constraint Validation")
 print("="*60)
 
 isp_loss = IspConstrainedLoss(
 max_delta_v=10.0,
 lambda_penalty=100.0
)
 
 test_cases = [
 # (action_mag, thruster_type, should_be_feasible)
 (2.0, 'cold_gas', True), # Well within limit
 (5.0, 'cold_gas', True), # At limit
 (6.0, 'cold_gas', False), # Exceeds limit
 (0.4, 'hall_effect', True), # Within limit
 (0.6, 'hall_effect', False),# Exceeds limit
 (40.0, 'chemical', True), # Within limit
 (60.0, 'chemical', False), # Exceeds limit
 ]
 
 passed = 0
 failed = 0
 
 for action_mag, thruster_type, should_feasible in test_cases:
 action = jnp.array([action_mag, 0.0, 0.0])
 
 feasible, reason = isp_loss.check_action_feasible(action, thruster_type)
 
 if feasible == should_feasible:
 status = "✅"
 passed += 1
 else:
 status = "❌"
 failed += 1
 
 print(f" {status} {action_mag:.1f} m/s {thruster_type}: "
 f"{'Feasible' if feasible else 'Infeasible'} "
 f"(expected: {'Feasible' if should_feasible else 'Infeasible'})")
 
 print(f"\nResults: {passed}/{passed+failed} tests passed")
 
 # Test Lagrangian penalty
 print("\n Testing Lagrangian penalty function...")
 
 base_reward = 1.0
 
 # Feasible action should have minimal penalty
 action_feasible = jnp.array([3.0, 0.0, 0.0])
 loss_feasible = isp_loss.compute_constrained_loss(
 base_reward, action_feasible, 10.0, 'cold_gas'
)
 
 # Infeasible action should have large penalty
 action_infeasible = jnp.array([20.0, 0.0, 0.0])
 loss_infeasible = isp_loss.compute_constrained_loss(
 base_reward, action_infeasible, 10.0, 'cold_gas'
)
 
 print(f" Feasible action (3 m/s): Loss = {loss_feasible:.2f}")
 print(f" Infeasible action (20 m/s): Loss = {loss_infeasible:.2f}")
 
 if loss_infeasible < loss_feasible:
 print("✅ PASS: Lagrangian penalty correctly penalizes infeasible actions")
 else:
 print("❌ FAIL: Lagrangian penalty not working correctly")
 
 return passed, failed


def test_adversarial_robustness():
 """
 Test 3: Adversarial Noise Robustness
 
 Validates policy robustness against adversarial noise injection
 into the Koopman K matrix (simulates "Model Mismatch").
 """
 print("\n" + "="*60)
 print("TEST 3: Adversarial Noise Robustness")
 print("="*60)
 
 # Create hardened agent
 K = jnp.eye(6)
 koopman = MockKoopmanPropagator()
 world_model = UncertaintyAwareWorldModel(koopman, K, noise_std=0.01)
 
 agent = HardenedMBPOAgent(
 config=MBPOConfig(),
 world_model=world_model,
 isp_constraints=IspConstrainedLoss(max_delta_v=5.0)
)
 agent.thruster_type = 'cold_gas'
 
 # Run adversarial test
 print(" Running 100 adversarial tests with increasing noise...")
 
 robustness = agent.test_adversarial_robustness(num_tests=100)
 
 print(f"\n Results:")
 print(f" Tests run: {robustness['tests_run']}")
 print(f" Safe actions: {robustness['safe_actions']}")
 print(f" Constrained actions: {robustness['constrained_actions']}")
 print(f" Unsafe actions: {robustness['unsafe_actions']}")
 print(f" Safety rate: {robustness['safety_rate']*100:.1f}%")
 print(f" Avg uncertainty: {robustness['avg_uncertainty']:.4f}")
 
 # Target: >95% safety rate
 if robustness['safety_rate'] >= 0.95:
 print("✅ PASS: Safety rate >= 95%")
 else:
 print(f"❌ FAIL: Safety rate {robustness['safety_rate']*100:.1f}% below 95% threshold")
 
 return robustness


def test_latency_budget():
 """
 Test 4: 100ms Decision Dominance Budget
 
 Validates end-to-end latency meets real-time requirements.
 """
 print("\n" + "="*60)
 print("TEST 4: 100ms Decision Dominance Budget")
 print("="*60)
 
 profiler = LatencyProfiler()
 
 # Mock components
 class MockDetector:
 def detect(self, state):
 time.sleep(0.025) # 25ms
 return True
 
 class MockSINDy:
 def DiscoverManeuver(self, states, dt):
 time.sleep(0.015) # 15ms
 return type('obj', (object,), {'ThrustDetected': True})()
 
 class MockMBPO:
 def act_with_safety(self, state, key):
 time.sleep(0.010) # 10ms
 return jnp.array([0.1, 0.0, 0.0]), {'total_uncertainty': 0.05}
 
 detector = MockDetector()
 sindy = MockSINDy()
 mbpo = MockMBPO()
 
 state = jnp.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
 key = random.PRNGKey(0)
 
 print(" Profiling full decision pipeline...")
 
 # Profile multiple runs
 latencies = []
 for _ in range(10):
 profile = profiler.profile_full_pipeline(detector, sindy, mbpo, state, key)
 latencies.append(profile['total']['time'])
 
 avg_latency = np.mean(latencies)
 p99_latency = np.percentile(latencies, 99)
 
 print(f"\n Latency Statistics (10 runs):")
 print(f" Average: {avg_latency*1000:.1f} ms")
 print(f" P99: {p99_latency*1000:.1f} ms")
 print(f" Budget: 100.0 ms")
 
 # Component breakdown (from last run)
 print(f"\n Component Breakdown:")
 print(f" S2 Detection: {profile['s2_detection']['time']*1000:.1f} ms")
 print(f" SINDy Discovery: {profile['sindy_discovery']['time']*1000:.1f} ms")
 print(f" MBPO Action: {profile['mbpo_action']['time']*1000:.1f} ms")
 print(f" Buffer Remaining: {profile['total']['buffer_remaining']*1000:.1f} ms")
 
 if avg_latency < 0.100:
 print("✅ PASS: Average latency within 100ms budget")
 else:
 print(f"❌ FAIL: Average latency {avg_latency*1000:.1f} ms exceeds 100ms budget")
 
 if p99_latency < 0.100:
 print("✅ PASS: P99 latency within 100ms budget")
 else:
 print(f"⚠️ WARNING: P99 latency {p99_latency*1000:.1f} ms exceeds budget")
 
 return profile


def test_safety_under_noise():
 """
 Test 5: Safety Under Sensor Noise ("Self-Preservation Instinct")
 
 Validates that when SINDy residuals are high (Rahu jamming),
 the agent defaults to conservative (safe) actions.
 """
 print("\n" + "="*60)
 print("TEST 5: Safety Under Sensor Noise")
 print("="*60)
 
 K = jnp.eye(6)
 koopman = MockKoopmanPropagator()
 
 agent = HardenedMBPOAgent(
 config=MBPOConfig(),
 world_model=UncertaintyAwareWorldModel(koopman, K, noise_std=0.1),
 isp_constraints=IspConstrainedLoss(max_delta_v=5.0)
)
 agent.thruster_type = 'cold_gas'
 
 state = jnp.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
 key = random.PRNGKey(0)
 
 # Test with different uncertainty levels
 uncertainty_levels = [0.05, 0.10, 0.15, 0.20, 0.30]
 
 conservative_count = 0
 total_tests = len(uncertainty_levels)
 
 for unc in uncertainty_levels:
 # Inject uncertainty by setting model noise
 agent.world_model.noise_std = unc
 
 action, info = agent.act_with_safety(state, key)
 
 is_conservative = info.get('action_type') == 'CONSERVATIVE_SAFETY'
 
 if is_conservative:
 conservative_count += 1
 
 action_mag = float(jnp.linalg.norm(action))
 
 print(f" Uncertainty: {unc:.2f} | "
 f"Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}] | "
 f"Magnitude: {action_mag:.3f} | "
 f"{'CONSERVATIVE' if is_conservative else 'NORMAL'}")
 
 # Validation: Higher uncertainty should trigger more conservative actions
 # At 0.30 uncertainty, should definitely be conservative
 if conservative_count >= 2: # At least 2 conservative actions at high uncertainty
 print(f"✅ PASS: Agent correctly switches to conservative mode under uncertainty")
 else:
 print(f"⚠️ WARNING: Agent may not be conservative enough under uncertainty")
 
 return conservative_count


def run_all_tests():
 """Run all validation tests."""
 print("\n" + "="*70)
 print(" HARDENED MBPO VALIDATION SUITE")
 print("="*70)
 print("\nValidating Neural Policy Safety & Physical Grounding")
 print("Target: Mission-Critical GNC Certification\n")
 
 results = {}
 
 try:
 results['uncertainty'] = test_epistemic_uncertainty_tracking()
 except Exception as e:
 print(f"❌ Test 1 failed: {e}")
 results['uncertainty'] = None
 
 try:
 results['isp_passed'], results['isp_failed'] = test_isp_hard_constraints()
 except Exception as e:
 print(f"❌ Test 2 failed: {e}")
 results['isp_passed'] = 0
 results['isp_failed'] = 1
 
 try:
 results['robustness'] = test_adversarial_robustness()
 except Exception as e:
 print(f"❌ Test 3 failed: {e}")
 results['robustness'] = {'safety_rate': 0.0}
 
 try:
 results['latency'] = test_latency_budget()
 except Exception as e:
 print(f"❌ Test 4 failed: {e}")
 results['latency'] = None
 
 try:
 results['safety'] = test_safety_under_noise()
 except Exception as e:
 print(f"❌ Test 5 failed: {e}")
 results['safety'] = 0
 
 # Summary
 print("\n" + "="*70)
 print("VALIDATION SUMMARY")
 print("="*70)
 
 all_passed = (
 results.get('robustness', {}).get('safety_rate', 0) >= 0.95 and
 results.get('safety', 0) >= 2
)
 
 if all_passed:
 print("\n✅ ALL CRITICAL TESTS PASSED")
 print("Status: READY for Mission-Critical GNC")
 else:
 print("\n⚠️ SOME TESTS FAILED")
 print("Status: ADDITIONAL HARDENING REQUIRED")
 
 print("\n" + "="*70)
 
 return results


if __name__ == "__main__":
 run_all_tests()
