# Generative Space Domain Training Simulator (Feature 4)

Implements ADITI Feature 4 with:
- Synthetic scenario generator (10k+ labeled conjunction/threat records)
- Conditional diffusion-style denoiser model (PyTorch -> ONNX)
- Small conditional GAN generator (PyTorch -> ONNX)
- Autonomous collision-avoidance RL agent in a custom Gym environment (PPO or SAC)
- Go runtime package for scenario generation + training environment simulation

## Go Package

- [engine/pkg/ml/simulator/simulator.go](engine/pkg/ml/simulator/simulator.go): scenario schema and synthetic generator
- [engine/pkg/ml/simulator/env.go](engine/pkg/ml/simulator/env.go): gym-like environment for agent training logic
- [engine/pkg/ml/simulator/simulator_test.go](engine/pkg/ml/simulator/simulator_test.go): unit tests

## Python Assets

- [ai/simulator/generate_scenarios.py](ai/simulator/generate_scenarios.py)
- [ai/simulator/train_conditional_diffusion.py](ai/simulator/train_conditional_diffusion.py)
- [ai/simulator/train_small_gan.py](ai/simulator/train_small_gan.py)
- [ai/simulator/train_collision_avoidance_rl.py](ai/simulator/train_collision_avoidance_rl.py)
- [ai/simulator/requirements.txt](ai/simulator/requirements.txt)

## Quickstart

python3 -m venv .venv
source .venv/bin/activate
pip install -r ai/simulator/requirements.txt

### 1) Generate 10k scenarios

python ai/simulator/generate_scenarios.py --count 10000 --seed 42

Artifacts:
- ai/simulator/data/scenarios_10000.jsonl
- ai/simulator/data/scenarios_10000.json

### 2) Train conditional diffusion and export ONNX

python ai/simulator/train_conditional_diffusion.py

Artifacts:
- ai/simulator/data/diffusion_denoiser.onnx
- ai/simulator/data/diffusion_metrics.json

### 3) Train small GAN and export generator ONNX

python ai/simulator/train_small_gan.py

Artifacts:
- ai/simulator/data/small_gan_generator.onnx
- ai/simulator/data/gan_metrics.json

### 4) Train PPO operator policy

python ai/simulator/train_collision_avoidance_rl.py --algo ppo --timesteps 120000

### 5) Train SAC operator policy

python ai/simulator/train_collision_avoidance_rl.py --algo sac --timesteps 150000

Artifacts:
- ai/simulator/data/ppo_collision_avoidance.zip
- ai/simulator/data/sac_collision_avoidance.zip
- ai/simulator/data/rl_metrics_ppo.json
- ai/simulator/data/rl_metrics_sac.json
- ai/simulator/data/cognitive_trace_ppo.jsonl
- ai/simulator/data/cognitive_trace_sac.jsonl

## Notes

- This module is physics-first and data-centric: generated labels include risk, priority, recommended action, and operator difficulty.
- RL environment is stateful (multi-step) and optimizes risk-fuel-confidence-time-pressure tradeoffs.
- Cognitive demonstration artifacts (`rl_metrics_*.json`, `cognitive_trace_*.jsonl`) show policy behavior for IAF ops briefings.
- ONNX exports are CPU-friendly and designed for later runtime integration in Go adapters.
