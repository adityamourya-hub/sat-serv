# RL Policy Head with Intent Labeling for ADITI 4.0

## Overview

This implementation provides Actor-Critic Reinforcement Learning (A2C/PPO) architecture for autonomous SSA decision-making, transforming ADITI 4.0 from "Passive Observation" to "Autonomous Agency."

## Architecture

### 1. Go-Python ZMQ Bridge

**Go Side: `pkg/rl/command_bridge.go`**
- `CommandBridge`: ZMQ REQ socket for policy requests
- `TacticalAgent`: High-level RL-driven decision making
- `StateVector`: 15-dimensional input (position, velocity, intent, threat indicators)
- `Decide()`: Selects and executes optimal tactical action
- `CalculateReward()`: Physics-informed reward computation

**Python Side: `ai/rl/policy_bridge.py`**
- `PolicyHead`: Actor-Critic neural network (PyTorch)
- Actor: Outputs action probability distribution
- Critic: Estimates state value
- ZMQ REP server at `ipc:///tmp/policy_bridge.ipc`

### 2. Action Space (6 Discrete Actions)

| Index | Command | Description |
|-------|---------|-------------|
| 0 | PASSIVE_TRACK | Standard custody, minimal sensor draw |
| 1 | ACTIVE_SCAN | High frame rate, MGE + Spectroscopy |
| 2 | FOLLOW_TARGET | INDI mount lock on specific target |
| 3 | EMITTER_GEO | TDOA geolocation for jammer detection |
| 4 | TACTICAL_ALERT | Broadcast FCH to mesh |
| 5 | FUEL_OPTIMAL_EVADE | Execute escape maneuver (on-orbit) |

### 3. Intent Labels (4 Classes)

| Label | Physics Signature | Tactical Context |
|-------|-------------------|------------------|
| PASSIVE (0) | Pure Keplerian, minimal Δv | Standard mission |
| RPO_INSPECTION (1) | HCW-locked, circumnavigation | Stalking/inspection |
| ELECTRONIC_WARFARE (2) | Nadir-pointing, thermal spikes | Jamming/disruption |
| KINETIC_THREAT (3) | High-thrust, non-Keplerian | ASAT/intercept |

### 4. Siliguri Intent Simulator

**Updated `pkg/sim/siliguri.go` with:**

**Behavioral Profiles:**
- **RPO Inspector**: HCW circumnavigation around Siliguri
  - Low thrust (0.001 m/s²) for station-keeping
  - Elliptical relative orbit (x₀=0.5km, z₀=1.0km)
  
- **EW Platform**: Nadir-pointing thermal jitter
  - Random thrust perturbations
  - Variable brightness (IR signature)
  
- **Kinetic Threat**: High-thrust intercept arc
  - 10 km/s intercept velocity (faster than LEO circular)
  - 0.5 m/s² thrust toward target
  - 500 m/s recent delta-V (intercept burn)
  
- **Passive**: Standard Keplerian orbit
  - Zero thrust
  - No recent maneuvers

**Methods:**
- `GenerateIntentScenario()`: Creates all 4 intent types
- `PropagateWithThrust()`: Non-Keplerian dynamics simulation
- `GetIntentProfileSummary()`: Training data analysis

### 5. Physics-Informed Reward Function

**Formula:**
$$R = w_1 \cdot \text{Custody} - w_2 \cdot \sum|\Delta v| - w_3 \cdot \text{ThreatMiss} + w_4 \cdot \text{IntentBonus}$$

**Weights:**
- $w_1 = 1.0$: Custody time reward (frames in lock)
- $w_2 = 0.5$: Fuel penalty (Yoshida-4 residuals)
- $w_3 = 10.0$: Threat miss penalty (heavy for kinetic)
- $w_4 = 2.0$: Intent-action matching bonus

**Components:**
- **Custody Reward**: Linear with consecutive frames
- **Fuel Penalty**: Proportional to recent delta-V
- **Threat Miss**: Massive negative for missed KINETIC threats
- **Intent Bonus**: Reward for correct action-intent pairs
  - EW → EMITTER_GEO (+2.0)
  - RPO → FOLLOW_TARGET (+1.0)
  - KINETIC → TACTICAL_ALERT/FUEL_OPTIMAL_EVADE (no penalty)

### 6. PPO Training Loop

**File: `ai/rl/rl_trainer.py`**

**Key Components:**
- `PolicyHead`: Actor-Critic network
- `PPOTrainer`: Proximal Policy Optimization
- `ReplayBuffer`: Experience collection
- `PhysicsInformedReward`: Reward shaping

**Training Features:**
- Generalized Advantage Estimation (GAE)
- PPO clipping (ε = 0.2)
- Gradient clipping (max_norm = 0.5)
- Checkpointing every 100 episodes

**Hyperparameters:**
- Learning rate: 3e-4
- Discount factor (γ): 0.99
- GAE lambda: 0.95
- Value coefficient: 0.5
- Entropy coefficient: 0.01

## Integration Flow

```
[Go Propagation Loop @ 100Hz]
         ↓
[Siliguri Simulator] → Intent-labeled states
         ↓
[StateVector] → Position, Velocity, Intent, Threat
         ↓
[ZMQ REQ] → ipc:///tmp/policy_bridge.ipc
         ↓
[Python PolicyHead] → Actor-Critic inference
         ↓
[Action + Value] → ZMQ REP
         ↓
[Go TacticalAgent] → Execute action
         ↓
[Environment] → Next state + Reward
         ↓
[PPO Update] → Policy gradient
```

## Usage Examples

### 1. Initialize Policy Bridge (Go)

```go
bridge := rl.NewCommandBridge("ipc:///tmp/policy_bridge.ipc")
bridge.Connect()

agent := rl.NewTacticalAgent(bridge)
```

### 2. Make Tactical Decision

```go
state := rl.StateVector{
    Position:  [3]float64{x, y, z},
    Velocity:  [3]float64{vx, vy, vz},
    Intent:    int(rl.IntentKineticThreat),
    Confidence: 0.85,
    RiskScore: 75.0,
}

action, err := agent.Decide(state)
// Returns: TACTICAL_ALERT (for kinetic threat)
```

### 3. Generate Intent Training Data

```go
sim := sim.NewSiliguriSimulator()
states := sim.GenerateIntentScenario()

// Returns: [RPO, EW, KINETIC, PASSIVE]
for _, state := range states {
    fmt.Printf("Sat %d: %s (Δv=%.1f m/s)\n", 
        state.SatID,
        sim.IntentNames[state.Intent],
        state.DeltaV)
}
```

### 4. Run Python Policy Server

```bash
python ai/rl/policy_bridge.py --endpoint ipc:///tmp/policy_bridge.ipc
```

### 5. Train RL Policy

```python
from rl_trainer import RLTrainer

trainer = RLTrainer(lr=3e-4)
trainer.train(env, num_episodes=1000)
```

## Technical Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| ZMQ Latency | < 0.5ms | IPC socket round-trip |
| Policy Inference | < 10ms | M2 Air PyTorch CPU |
| Physics Loop | 100Hz | Deterministic propagation |
| Convergence | < 500 episodes | Intent-labeled data |
| Critic Value | Real-time | Risk meter visualization |

## Strategic Payoff

### 1. Actor-Critic Explainability

- **Actor (Pilot)**: Maps state to actions in real-time
- **Critic (Strategist)**: Estimates state value
- **Risk Meter**: Critic value drops → human intervention prompt

### 2. Physics-Grounded RL

- Reward function incorporates Yoshida-4 residuals
- Agent learns cheap Keplerian vs expensive adversarial maneuvers
- Prevents "black box" hallucinations

### 3. Sovereign Training

- Air-gapped training on M2 Air
- No cloud dependencies
- 10k+ intent-labeled scenarios from Siliguri digital twin

### 4. Autonomous Kill-Chain

1. **Detect** (Go) → Physical anomaly via HCW logic
2. **Characterize** (Python/MGE) → Array-to-bus ratio analysis
3. **Decide** (RL Policy) → Actor-Critic action selection
4. **Act** (pyindi) → Telescope slewing + mesh broadcast

## Files Created/Updated

```
pkg/rl/
└── command_bridge.go          # (new) Go-Python ZMQ bridge

pkg/sim/
└── siliguri.go                # (updated) Intent simulation

ai/rl/
├── policy_bridge.py           # (new) Actor-Critic server
├── rl_trainer.py              # (new) PPO training loop
└── reward_function.py         # (new) Physics-informed rewards

docs/
└── RL_POLICY_INTEGRATION.md   # (new) This documentation
```

## Dependencies

**Go:**
- No new dependencies (uses existing math, time packages)

**Python:**
```bash
pip install torch numpy pyzmq
```

## Next Steps

1. **Train Policy**: Run `rl_trainer.py` on Siliguri intent data
2. **Validate**: Test action selection for each intent type
3. **Integrate**: Connect to pyindi telescope control
4. **Demo**: End-to-end autonomous response at Siliguri Corridor

## Summary

The RL Policy Head transforms ADITI 4.0 into an **Autonomous Sentinel**:

- **Actor-Critic Architecture**: Differentiable decisions with value estimation
- **Intent-Aware State**: Combines physics + neural inference
- **Physics-Informed Rewards**: Yoshida-4 residuals constrain agent
- **Zero-Copy Bridge**: ZMQ IPC for Go-Python communication
- **Air-Gapped Training**: M2 Air optimized, no cloud required

This completes the autonomous kill-chain: **Detect → Characterize → Decide → Act**.
