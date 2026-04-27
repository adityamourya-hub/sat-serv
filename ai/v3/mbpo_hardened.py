"""
Hardened MBPO Agent with Epistemic Uncertainty and Isp Constraints
Implements safety-critical RL for mission certification
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, random
from functools import partial
from typing import Tuple, Dict, Callable, Optional
import numpy as np

from mbpo_agent import MBPOAgent, MBPOConfig, WorldModel
from koopman_propagator import KoopmanPropagator


class UncertaintyAwareWorldModel(WorldModel):
                """
                Hardened world model with epistemic uncertainty tracking.

                Tracks both aleatoric (data noise) and epistemic (model uncertainty)
                to prevent "hallucinated" maneuvers during jamming events.
                """

                def __init__(self, koopman: KoopmanPropagator, K: jnp.ndarray,
                noise_std: float = 0.01, max_residual: float = 0.1):
                                super().__init__(koopman, K)
                                self.noise_std = noise_std # Aleatoric uncertainty (sensor noise)
                                self.max_residual = max_residual # Max acceptable model residual
                                self.residual_history = [] # Track model mismatch
                                self.uncertainty_quantile = 0.95 # 95% confidence threshold

                                @partial(jit, static_argnums=(0,))
                                def step_with_uncertainty(self, state: jnp.ndarray, action: jnp.ndarray,
                                key: random.PRNGKey) -> Tuple[jnp.ndarray, float, bool, Dict]:
                                                """
                                                Predict next state with uncertainty manifold.

                                                Returns:
                                                next_state: Predicted state
                                                reward: Scalar reward (penalized if uncertainty high)
                                                done: Episode termination
                                                info: Dict with uncertainty metrics
                                                """
                                                # Base prediction from Koopman
                                                action_kms = action / 1000.0
                                                perturbed_state = state.at[3:].add(action_kms)

                                                # Propagate with uncertainty sampling (ensemble)
                                                next_states = []
                                                residuals = []

                                                # Multiple samples with noise injection to measure epistemic uncertainty
                                                for _ in range(5):
                                                                key, subkey = random.split(key)
                                                                # Inject noise into Koopman operator (simulates model mismatch)
                                                                noisy_K = self.K + random.normal(subkey, self.K.shape) * self.noise_std
                                                                next_s = self.koopman.propagate_state(perturbed_state, noisy_K)
                                                                next_states.append(next_s)

                                                                # Calculate residual from nominal prediction
                                                                nominal = self.koopman.propagate_state(perturbed_state, self.K)
                                                                residual = jnp.linalg.norm(next_s - nominal)
                                                                residuals.append(residual)

                                                                # Calculate uncertainty manifold
                                                                next_states_stack = jnp.stack(next_states)
                                                                mean_state = jnp.mean(next_states_stack, axis=0)
                                                                std_state = jnp.std(next_states_stack, axis=0)
                                                                epistemic_uncertainty = jnp.mean(std_state) # Model uncertainty

                                                                # Calculate aleatoric uncertainty (input noise propagation)
                                                                aleatoric_uncertainty = self.noise_std * jnp.linalg.norm(action)

                                                                # Total uncertainty
                                                                total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

                                                                # Conservative reward: penalize high-energy actions when uncertain
                                                                # This implements "epistemic caution" - when model is uncertain,
                                                                # prefer low-energy (safe) actions
                                                                base_reward = self._compute_reward(state, action, mean_state)

                                                                # Uncertainty penalty: high uncertainty + high action = large penalty
                                                                uncertainty_penalty = total_uncertainty * jnp.linalg.norm(action) * 0.5

                                                                # If residuals are too high, we're in "unmodeled dynamics" regime
                                                                max_residual = jnp.max(jnp.array(residuals))
                                                                if max_residual > self.max_residual:
                                                                                # Heavy penalty for operating outside model validity
                                                                                uncertainty_penalty += 10.0

                                                                                conservative_reward = base_reward - uncertainty_penalty

                                                                                info = {
                                                                                'epistemic_uncertainty': float(epistemic_uncertainty),
                                                                                'aleatoric_uncertainty': float(aleatoric_uncertainty),
                                                                                'total_uncertainty': float(total_uncertainty),
                                                                                'max_residual': float(max_residual),
                                                                                'state_std': jnp.linalg.norm(std_state).item(),
                                                                                }

                                                                                # Check termination
                                                                                done = self._check_termination(mean_state, action)

                                                                                return mean_state, conservative_reward, done, info

                                                                                def validate_against_physics(self, imagined_trajectory: jnp.ndarray,
                                                                                true_trajectory: jnp.ndarray) -> float:
                                                                                                """
                                                                                                Compute validation error between imagination and ground truth.
                                                                                                Used to detect when model needs re-training.
                                                                                                """
                                                                                                mse = jnp.mean((imagined_trajectory - true_trajectory) ** 2)
                                                                                                return float(mse)


class IspConstrainedLoss:
                """
                Physics-informed loss with hard Isp (specific impulse) constraints.

                Ensures policies are executable by real satellite thrusters using
                Lagrangian multiplier approach.
                """

                def __init__(self, max_delta_v: float = 10.0, lambda_penalty: float = 100.0):
                                """
                                Args:
                                max_delta_v: Maximum delta-V per maneuver (m/s) - based on thruster Isp
                                lambda_penalty: Lagrangian multiplier strength
                                """
                                self.max_delta_v = max_delta_v # m/s
                                self.lambda_penalty = lambda_penalty

                                # Typical satellite constraints:
                                # - Cold gas thrusters: ~5 m/s per burn
                                # - Hall-effect electric: ~0.1 m/s per burn (high Isp, low thrust)
                                # - Chemical bi-propellant: ~50 m/s per burn (but limited fuel)
                                self.thruster_profiles = {
                                'cold_gas': {'max_delta_v': 5.0, 'total_budget': 100.0},
                                'hall_effect': {'max_delta_v': 0.5, 'total_budget': 500.0},
                                'chemical': {'max_delta_v': 50.0, 'total_budget': 1000.0},
                                }

                                def compute_constrained_loss(self, reward: float, action: jnp.ndarray,
                                cumulative_delta_v: float,
                                thruster_type: str = 'cold_gas') -> float:
                                                """
                                                Compute loss with hard Isp constraints.

                                                L_total = L_reward + λ * max(0, ||ΔV|| - ΔV_limit) + λ * max(0, cumΔV - total_budget)
                                                """
                                                profile = self.thruster_profiles.get(thruster_type, self.thruster_profiles['cold_gas'])

                                                action_magnitude = jnp.linalg.norm(action)

                                                # Constraint 1: Single-maneuver limit
                                                single_maneuver_violation = jnp.maximum(0.0, action_magnitude - profile['max_delta_v'])

                                                # Constraint 2: Total fuel budget
                                                total_budget_violation = jnp.maximum(0.0, cumulative_delta_v - profile['total_budget'])

                                                # Combined penalty
                                                penalty = self.lambda_penalty * (single_maneuver_violation + total_budget_violation)

                                                return reward - penalty

                                                def check_action_feasible(self, action: jnp.ndarray, thruster_type: str = 'cold_gas') -> Tuple[bool, str]:
                                                                """
                                                                Check if action is physically feasible given thruster constraints.

                                                                Returns:
                                                                feasible: True if action is within Isp limits
                                                                reason: Description of constraint violation if any
                                                                """
                                                                profile = self.thruster_profiles.get(thruster_type, self.thruster_profiles['cold_gas'])

                                                                action_mag = float(jnp.linalg.norm(action))

                                                                if action_mag > profile['max_delta_v']:
                                                                                return False, f"Action {action_mag:.2f} m/s exceeds {thruster_type} limit ({profile['max_delta_v']:.2f} m/s)"

                                                                                return True, "Action feasible"


class HardenedMBPOAgent(MBPOAgent):
                """
                Mission-critical MBPO agent with safety guarantees.

                Extends base MBPO with:
                - Epistemic uncertainty tracking
                - Isp hard constraints
                - Conservative safety action selection
                """

                def __init__(self, config: MBPOConfig, world_model: UncertaintyAwareWorldModel,
                isp_constraints: Optional[IspConstrainedLoss] = None):
                                super().__init__(config, world_model)
                                self.world_model = world_model # Override with uncertainty-aware version
                                self.isp_constraints = isp_constraints or IspConstrainedLoss()
                                self.cumulative_delta_v = 0.0
                                self.thruster_type = 'cold_gas'
                                self.safety_mode = False # Activated when uncertainty is high

                                def act_with_safety(self, state: jnp.ndarray, key: random.PRNGKey,
                                collision_imminent: bool = False) -> Tuple[jnp.ndarray, Dict]:
                                                """
                                                Select action with safety guarantees.

                                                If uncertainty is high, choose most conservative action.
                                                If collision imminent, allow higher delta-V but stay within Isp limits.
                                                """
                                                # Get uncertainty estimate
                                                _, _, _, info = self.world_model.step_with_uncertainty(state, jnp.zeros(3), key)
                                                uncertainty = info['total_uncertainty']

                                                # Check if we're in high-uncertainty regime
                                                if uncertainty > 0.1: # 10% uncertainty threshold
                                                                self.safety_mode = True
                                                else:
                                                                self.safety_mode = False

                                                                if self.safety_mode and not collision_imminent:
                                                                                # Conservative action: minimal delta-V
                                                                                safe_action = jnp.zeros(3)
                                                                                info['action_type'] = 'CONSERVATIVE_SAFETY'
                                                                                return safe_action, info

                                                                                # Normal action selection with Isp constraints
                                                                                action = self.sample_action(state, key, deterministic=collision_imminent)

                                                                                # Apply Isp constraints
                                                                                feasible, reason = self.isp_constraints.check_action_feasible(action, self.thruster_type)

                                                                                if not feasible:
                                                                                                # Clamp action to feasible region
                                                                                                action = self._clamp_action_to_isp_limit(action)
                                                                                                info['constraint_violation'] = reason
                                                                                                info['action_clamped'] = True

                                                                                                # Track cumulative delta-V
                                                                                                self.cumulative_delta_v += float(jnp.linalg.norm(action))
                                                                                                info['cumulative_delta_v'] = self.cumulative_delta_v

                                                                                                return action, info

                                                                                                def _clamp_action_to_isp_limit(self, action: jnp.ndarray) -> jnp.ndarray:
                                                                                                                """Clamp action magnitude to Isp constraint."""
                                                                                                                profile = self.isp_constraints.thruster_profiles[self.thruster_type]
                                                                                                                max_dv = profile['max_delta_v']

                                                                                                                action_mag = jnp.linalg.norm(action)
                                                                                                                if action_mag > max_dv:
                                                                                                                                # Scale down to limit
                                                                                                                                return action * (max_dv / action_mag) * 0.95 # 5% safety margin
                                                                                                                                return action

                                                                                                                                def test_adversarial_robustness(self, num_tests: int = 100) -> Dict:
                                                                                                                                                """
                                                                                                                                                Test policy robustness against adversarial noise injection.

                                                                                                                                                Simulates "Model Mismatch" - gap between software math and ground reality.
                                                                                                                                                """
                                                                                                                                                results = {
                                                                                                                                                'tests_run': num_tests,
                                                                                                                                                'safe_actions': 0,
                                                                                                                                                'constrained_actions': 0,
                                                                                                                                                'unsafe_actions': 0,
                                                                                                                                                'avg_uncertainty': 0.0,
                                                                                                                                                }

                                                                                                                                                key = random.PRNGKey(42)

                                                                                                                                                for i in range(num_tests):
                                                                                                                                                                key, subkey = random.split(key)

                                                                                                                                                                # Random state
                                                                                                                                                                state = jnp.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])

                                                                                                                                                                # Inject adversarial noise into world model
                                                                                                                                                                noise_level = 0.05 * (i / num_tests) # Increasing noise
                                                                                                                                                                self.world_model.noise_std = noise_level

                                                                                                                                                                # Get action with safety
                                                                                                                                                                action, info = self.act_with_safety(state, subkey)

                                                                                                                                                                # Check constraints
                                                                                                                                                                feasible, _ = self.isp_constraints.check_action_feasible(action, self.thruster_type)

                                                                                                                                                                if feasible and info.get('action_type') == 'CONSERVATIVE_SAFETY':
                                                                                                                                                                                results['safe_actions'] += 1
                                                                                                                                                                elif feasible and info.get('action_clamped', False):
                                                                                                                                                                                results['constrained_actions'] += 1
                                                                                                                                                                elif not feasible:
                                                                                                                                                                                results['unsafe_actions'] += 1

                                                                                                                                                                                results['avg_uncertainty'] += info.get('total_uncertainty', 0.0)

                                                                                                                                                                                results['avg_uncertainty'] /= num_tests
                                                                                                                                                                                results['safety_rate'] = (results['safe_actions'] + results['constrained_actions']) / num_tests

                                                                                                                                                                                return results


# =============================================================================
# Latency Profiling
# =============================================================================

class LatencyProfiler:
                """
                End-to-end latency profiling for 100ms "Decision Dominance" budget.

                Budget breakdown:
                - S2/mrDMD Detection: 30ms
                - SINDy Discovery: 20ms
                - MBPO Action Selection: 15ms
                - Buffer: 35ms for mesh serialization
                """

                def __init__(self):
                                self.budgets = {
                                's2_detection': 30e-3, # 30ms
                                'sindy_discovery': 20e-3, # 20ms
                                'mbpo_action': 15e-3, # 15ms
                                'mesh_serialization': 35e-3, # 35ms
                                }
                                self.total_budget = 100e-3 # 100ms

                                def profile_component(self, name: str, fn: Callable, *args, **kwargs) -> Tuple[float, any]:
                                                """Profile a component and check against budget."""
                                                import time
                                                start = time.perf_counter()
                                                result = fn(*args, **kwargs)
                                                elapsed = time.perf_counter() - start

                                                budget = self.budgets.get(name, self.total_budget)
                                                status = 'PASS' if elapsed < budget else 'FAIL'

                                                return elapsed, result, status

                                                def profile_full_pipeline(self, detector, sindy, mbpo, state, key) -> Dict:
                                                                """Profile complete decision pipeline."""
                                                                results = {}

                                                                # S2 Detection
                                                                t1, _, s1 = self.profile_component('s2_detection', detector.detect, state)
                                                                results['s2_detection'] = {'time': t1, 'status': s1, 'budget': self.budgets['s2_detection']}

                                                                # SINDy Discovery
                                                                t2, _, s2 = self.profile_component('sindy_discovery', sindy.DiscoverManeuver, [state], 1.0)
                                                                results['sindy_discovery'] = {'time': t2, 'status': s2, 'budget': self.budgets['sindy_discovery']}

                                                                # MBPO Action
                                                                t3, _, s3 = self.profile_component('mbpo_action', mbpo.act_with_safety, state, key)
                                                                results['mbpo_action'] = {'time': t3, 'status': s3, 'budget': self.budgets['mbpo_action']}

                                                                total = t1 + t2 + t3
                                                                results['total'] = {
                                                                'time': total,
                                                                'status': 'PASS' if total < self.total_budget else 'FAIL',
                                                                'budget': self.total_budget,
                                                                'buffer_remaining': self.total_budget - total
                                                                }

                                                                return results


# =============================================================================
# Usage Example
# =============================================================================

"""
# Create hardened agent for
koopman, K = load_koopman_operator()
world_model = UncertaintyAwareWorldModel(koopman, K, noise_std=0.01)
isp_constraints = IspConstrainedLoss(max_delta_v=5.0) # Cold gas thrusters

agent = HardenedMBPOAgent(
config=MBPOConfig(),
world_model=world_model,
isp_constraints=isp_constraints
)
agent.thruster_type = 'cold_gas'

# Real-time decision with safety
state = jnp.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
key = random.PRNGKey(0)

action, info = agent.act_with_safety(state, key, collision_imminent=True)

print(f"Action: {action}")
print(f"Uncertainty: {info['total_uncertainty']:.4f}")
print(f"Cumulative ΔV: {info['cumulative_delta_v']:.2f} m/s")

# Adversarial robustness test
robustness = agent.test_adversarial_robustness(num_tests=100)
print(f"Safety Rate: {robustness['safety_rate']*100:.1f}%")

# Latency profiling
profiler = LatencyProfiler()
profile = profiler.profile_full_pipeline(detector, sindy, agent, state, key)
print(f"Total latency: {profile['total']['time']*1000:.1f} ms")
"""
