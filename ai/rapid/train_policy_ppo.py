#!/usr/bin/env python3
"""
Optional PPO policy trainer for evasive action recommendation.
This script creates a lightweight synthetic gym environment and trains PPO.

Outputs:
- ai/rapid/data/ppo_policy.zip
- ai/rapid/data/policy_notes.json
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO


class RapidThreatEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # monitor, retask, prepare, evasive_now
        self.state = None

    def _sample_state(self):
        return np.array([
            np.random.uniform(0, 1),  # normalized score
            np.random.uniform(0, 1),  # confidence
            np.random.uniform(0, 1),  # reachability
            np.random.uniform(0, 1),  # hypersonic likelihood
            np.random.uniform(0, 1),  # time pressure
            np.random.uniform(0, 1),  # sensor quality
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._sample_state()
        return self.state, {}

    def step(self, action):
        score, conf, reach, hyp, pressure, sensor = self.state

        optimal = 0
        if reach < 0.2:
            optimal = 0
        elif score > 0.8 and hyp > 0.6:
            optimal = 3
        elif score > 0.65 or pressure > 0.7:
            optimal = 2
        elif sensor < 0.4:
            optimal = 1

        reward = 1.0 if action == optimal else -0.3
        self.state = self._sample_state()
        done = True
        truncated = False
        return self.state, reward, done, truncated, {}


def main():
    env = RapidThreatEnv()
    model = PPO("MlpPolicy", env, verbose=0, n_steps=512, batch_size=64, learning_rate=3e-4, gamma=0.99)
    model.learn(total_timesteps=50000)

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "ppo_policy"
    model.save(str(model_path))

    notes = {
        "model": "PPO MlpPolicy",
        "actions": {
            "0": "monitor_only",
            "1": "sensor_retask",
            "2": "prepare_evasive",
            "3": "evasive_maneuver_now",
        },
        "note": "Export to ONNX can be done via torch.onnx export of extracted policy network when deployment requires ORT-only runtime.",
    }
    with (out_dir / "policy_notes.json").open("w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)

    print(f"saved={model_path}.zip")


if __name__ == "__main__":
    main()
