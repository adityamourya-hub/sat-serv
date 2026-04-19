# Advanced Exploration Strategies for ADITI 4.0 RL Policy Head

## Overview

This document describes the physics-constrained exploration strategies implemented for the ADITI 4.0 RL Policy Head. Unlike standard ε-greedy approaches that are too "noisy" for the orbital domain, these strategies ensure exploration is physically correlated with the orbital environment and respects hardware constraints.

## 1. Physics-Constrained Entropy Bonus

### Problem
Standard entropy bonuses use a constant curiosity factor, leading to over-exploration in stable orbits and under-exploration during uncertainty events.

### Solution
Scale the entropy weight β by Kalman Filter uncertainty (position covariance trace):

$$\beta(t) = \beta_0 \cdot \exp \left( \frac{\text{Trace}(\Sigma_{pos})}{k} \right)$$

### Implementation
**File:** `ai/rl/policy_bridge.py`

```python
def compute_dynamic_entropy_weight(self, kalman_uncertainty: torch.Tensor) -> torch.Tensor:
    scaled_uncertainty = kalman_uncertainty / self.uncertainty_scale
    entropy_weight = self.base_entropy_weight * torch.exp(scaled_uncertainty)
    entropy_weight = torch.clamp(entropy_weight, max=self.max_entropy_weight)
    return entropy_weight
```

### Behavior
- **High Uncertainty** (sleeper activation, solar flare): β increases → forces exploration
- **Low Uncertainty** (stable catalog match): β decreases → exploits learned policy
- **Prevents Over-fitting**: Scales exploration to physical ignorance

### Parameters
- `base_entropy_weight = 0.01` (β₀)
- `uncertainty_scale = 1.0` (k)
- `max_entropy_weight = 0.1` (cap)

---

## 2. Ornstein-Uhlenbeck (OU) Noise

### Problem
Random ε-greedy exploration causes "jittery" action selection that loses target custody and wastes Δv.

### Solution
Use OU noise for temporally correlated exploration that respects hardware inertia:

$$dx_t = \theta (\mu - x_t) dt + \sigma dW_t$$

### Implementation
**File:** `ai/rl/rl_trainer.py`

```python
class OUNoise:
    def __init__(self, action_dim: int = 6, theta: float = 0.15, sigma: float = 0.2):
        self.theta = theta  # Mean reversion rate
        self.sigma = sigma  # Volatility
        self.x = np.ones(action_dim) * mu
    
    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.x) * self.dt
        dw = np.random.normal(0, self.sigma * np.sqrt(self.dt))
        self.x = self.x + dx + dw
        return self.x
```

### Behavior
- **Temporal Correlation**: If exploring, continue exploring for several time-steps
- **Hardware Respect**: Smooth action transitions (INDI mount inertia)
- **No Oscillation**: Prevents "flickering" sensor tasking

### Parameters
- `theta = 0.15` (mean reversion rate)
- `sigma = 0.2` (volatility)
- `dt = 1.0` (time step)

---

## 3. Dynamic Curiosity Threshold

### Problem
Fixed thresholds treat quiet and contested skies equally, missing rare maneuvers in quiet sectors.

### Solution
Scale curiosity threshold C_T by UCT density and innovation residuals:

$$C_T(\rho_{UCT}, \nu) = C_{base} + \alpha \cdot \log(1 + \rho_{UCT}) + \beta \cdot \|\nu\|$$

### Implementation
**File:** `ai/rl/rl_trainer.py`

```python
class CuriosityGovernor:
    def update_threshold(self, uct_count: float, innovation_residuals: List[float]):
        uct_factor = np.log1p(uct_count)
        physics_factor = np.clip(np.mean(innovation_residuals), 0, 5.0)
        dynamic_eps = self.base_epsilon + self.alpha * uct_factor + self.beta * physics_factor
        self.current_threshold = np.min(dynamic_eps, self.max_epsilon)
```

### Behavior
- **Low UCT Density**: Agent stays in "High-Efficiency" mode (exploitation)
- **High UCT Density**: Agent enters "State of Discovery" (exploration)
- **Saturation Attack**: Instant recognition of broken orbital model

### Parameters
- `base_epsilon = 0.1` (minimum exploration)
- `alpha = 0.05` (UCT sensitivity)
- `beta = 0.1` (physics surprise factor)
- `max_epsilon = 0.5` (chaos cap)

### Expert Insight
This mimics biological "arousal" response: dark alley → pupils dilate, well-lit room → relax.

---

## 4. Coordinate Randomization

### Problem
Agent learns "targets always appear at dawn/dusk" (sun-synchronous overfitting).

### Solution
Apply random ECI frame rotation during training to vary lighting conditions.

### Implementation
**File:** `pkg/sim/siliguri.go`

```go
func (ss *SiliguriSimulator) ApplyCoordinateRandomization() {
    phi := ss.rng.Float64() * 2 * math.Pi
    // Generate random rotation axis
    // Apply Rodrigues' rotation formula to all states
}
```

### Behavior
- **ECI Rotation**: Random axis and angle (0-2π)
- **Position & Velocity**: Both rotated to maintain orbital dynamics
- **Training Only**: Disabled during deployment

### Effect
Forces agent to learn mechanistic relationships rather than memorize specific lighting patterns.

---

## 5. Feature Masking

### Problem
Agent relies on absolute altitude (altitude-invariant feature) instead of relative metrics.

### Solution
Occasionally mask altitude and intent to force reliance on HCW residuals and relative velocity.

### Implementation
**File:** `pkg/sim/siliguri.go`

```go
func (ss *SiliguriSimulator) ApplyFeatureMasking(states []SimState, maskAltitude bool, maskIntent bool) {
    if maskAltitude {
        stateData["z"] = 0.0  // Zero out radial component
        stateData["altitude_masked"] = true
    }
    if maskIntent {
        stateData["intent"] = -1  // Unknown intent
        stateData["intent_masked"] = true
    }
}
```

### Behavior
- **Altitude Masking**: Zero out radial component (z-axis)
- **Intent Masking**: Set intent to -1 (unknown)
- **Forces**: HCW residuals, relative velocity, thrust signatures

### Effect
Prevents altitude-dependent shortcuts, ensures robustness across orbital regimes.

---

## Integration with PPOTrainer

### Action Selection Flow

```python
def select_action_with_exploration(self, state, kalman_uncertainty, uct_count, innovation_residuals):
    # 1. Update curiosity threshold
    self.curiosity_governor.update_threshold(uct_count, innovation_residuals)
    
    # 2. Get base action from policy
    action, action_prob, value = self.policy.get_action(state)
    
    # 3. Apply OU noise if curiosity governor says explore
    if self.curiosity_governor.should_explore():
        action = self.ou_noise.get_action_with_noise(action)
        explored = True
    
    return action, action_prob, value, explored
```

### Training Pipeline

1. **Generate Intent Scenario**: `sim.GenerateIntentScenario()`
2. **Apply Coordinate Randomization**: `sim.ApplyCoordinateRandomization()`
3. **Apply Feature Masking**: `sim.ApplyFeatureMasking(states, maskAltitude=True)`
4. **Collect Experience**: `trainer.collect_experience(env)`
5. **Update with Physics-Constrained Entropy**: `trainer.update(experiences, kalman_uncertainty)`

---

## Comparison of Exploration Strategies

| Strategy | Mechanism | Expert Verdict |
| :--- | :--- | :--- |
| **ε-Greedy** | Uniformly random actions | **Fail**: Too jittery; loses custody |
| **Entropy Bonus (Static)** | Constant reward for high variance | **Medium**: Risks over-exploration |
| **Physics-Constrained** | Entropy scaled by Kalman uncertainty | **Optimal**: Exploration driven by "physical ignorance" |
| **OU Noise** | Temporally correlated exploration | **Optimal**: Respects hardware inertia |
| **Dynamic C_T** | Scaled by UCT density | **Optimal**: Calm in peace, hyper-aware in conflict |

---

## Strategic Payoff for ADITI 4.0

### 1. Physics-Grounded Exploration
- Agent learns cheap Keplerian vs expensive adversarial maneuvers
- Prevents "black box" hallucinations
- Exploration scales to actual orbital uncertainty

### 2. Hardware Respect
- OU noise respects INDI mount inertia
- No oscillating sensor tasking
- Maintains target custody during exploration

### 3. Robustness
- Coordinate randomization prevents sun-synchronous overfitting
- Feature masking ensures altitude-invariant learning
- Dynamic curiosity adapts to environmental chaos

### 4. Autonomous Kill-Chain
```
[High UCT Density] → Curiosity Governor → Exploration Mode → Active Scan + EM Geo
[Low UCT Density] → Curiosity Governor → Exploitation Mode → Passive Track
[High Kalman Uncertainty] → Entropy Bonus → Force Exploration → Resolve Ambiguity
```

---

## Usage Examples

### 1. Enable Advanced Exploration in Training

```python
from rl_trainer import RLTrainer

trainer = RLTrainer(
    lr=3e-4,
    use_ou_noise=True,
    use_curiosity_governor=True
)
```

### 2. Enable Coordinate Randomization in Simulation

```go
sim := sim.NewSiliguriSimulator()
sim.EnableCoordinateRandomization(true)

states := sim.GenerateIntentScenario()
sim.ApplyCoordinateRandomization()
```

### 3. Apply Feature Masking for Training Data

```go
masked_states := sim.ApplyFeatureMasking(
    states,
    maskAltitude=true,   # Force reliance on relative metrics
    maskIntent=true,      # Force reliance on physical signatures
)
```

### 4. Select Action with Exploration

```python
action, prob, value, explored = trainer.select_action_with_exploration(
    state=state_vector,
    kalman_uncertainty=kalman_cov_trace,
    uct_count=num_ucts,
    innovation_residuals=residuals
)

if explored:
    print(f"Exploring due to UCT density: {trainer.curiosity_governor.uct_density}")
```

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Exploration Correlation | > 0.7 | OU noise temporal correlation |
| Curiosity Response Time | < 1s | UCT density update latency |
| Entropy Scaling Range | 0.01-0.1 | Prevents over/under-exploration |
| Coordinate Randomization | Every epoch | Prevents sun-synchronous overfit |
| Feature Masking Rate | 20% of batches | Forces robust learning |

---

## Summary

The advanced exploration strategies transform ADITI 4.0 from a reactive system to a **proactive, physics-aware autonomous agent**:

- **Physics-Constrained Entropy**: Exploration driven by orbital uncertainty
- **OU Noise**: Temporally correlated exploration respecting hardware
- **Dynamic Curiosity**: Scales to environmental chaos (UCT density)
- **Coordinate Randomization**: Prevents sun-synchronous overfitting
- **Feature Masking**: Forces reliance on relative metrics

This ensures the agent is **"Calm in Peace, Hyper-Aware in Conflict"**—the ideal behavior for sovereign space domain awareness.
