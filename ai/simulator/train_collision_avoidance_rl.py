#!/usr/bin/env python3
"""
Autonomous collision-avoidance RL trainer for IAF SSA operations.

Implements a custom Gymnasium environment with two policy families:
- PPO (discrete action policy)
- SAC (continuous control mapped to maneuver actions)

Outputs:
- ai/simulator/data/{algo}_collision_avoidance.zip
- ai/simulator/data/rl_metrics_{algo}.json
- ai/simulator/data/cognitive_trace_{algo}.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO, SAC


ACTION_LABELS = {
    0: "monitor_only",
    1: "sensor_retask",
    2: "small_burn",
    3: "large_burn",
}


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


class SSACollisionAvoidanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, continuous_actions: bool = False, max_steps: int = 24):
        super().__init__()
        self.continuous_actions = continuous_actions
        self.max_steps = max_steps
        # state = [risk_proxy, rel_vel, tca_pressure, fuel, track_conf, threat_severity]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        if continuous_actions:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)

        self.state: np.ndarray | None = None
        self.steps = 0

    def _sample_state(self) -> np.ndarray:
        return np.array(
            [
                np.random.uniform(0.45, 0.98),
                np.random.uniform(0.15, 0.95),
                np.random.uniform(0.25, 0.98),
                np.random.uniform(0.25, 1.00),
                np.random.uniform(0.25, 0.95),
                np.random.uniform(0.20, 0.98),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self.state = self._sample_state()
        self.steps = 0
        return self.state, {}

    def _discretize_action(self, action) -> int:
        if not self.continuous_actions:
            return int(action)
        scalar = float(np.asarray(action).reshape(-1)[0])
        if scalar < -0.5:
            return 0
        if scalar < 0.0:
            return 1
        if scalar < 0.5:
            return 2
        return 3

    def _recommended_action(self, risk: float, fuel: float, conf: float, severity: float, tca_pressure: float) -> int:
        if risk > 0.80 and severity > 0.68 and fuel > 0.20 and tca_pressure > 0.55:
            return 3
        if risk > 0.62 and fuel > 0.10 and tca_pressure > 0.35:
            return 2
        if conf < 0.45:
            return 1
        return 0

    def step(self, action):
        self.steps += 1
        assert self.state is not None
        risk, rel_vel, tca_pressure, fuel, conf, severity = [float(v) for v in self.state]

        action_id = self._discretize_action(action)
        ideal_action = self._recommended_action(risk, fuel, conf, severity, tca_pressure)

        if action_id == 0:
            risk = clip01(risk + 0.035 * severity - 0.015 * conf)
            fuel = clip01(fuel)
            conf = clip01(conf + 0.01)
            tca_pressure = clip01(tca_pressure + 0.02)
            burn_cost = 0.0
        elif action_id == 1:
            risk = clip01(risk - 0.04 * conf)
            fuel = clip01(fuel - 0.02)
            conf = clip01(conf + 0.13)
            tca_pressure = clip01(tca_pressure + 0.01)
            burn_cost = 0.02
        elif action_id == 2:
            risk = clip01(risk - (0.16 + 0.06 * conf))
            rel_vel = clip01(rel_vel - 0.10)
            fuel = clip01(fuel - 0.08)
            conf = clip01(conf - 0.01)
            tca_pressure = clip01(tca_pressure - 0.10)
            burn_cost = 0.08
        else:
            risk = clip01(risk - (0.26 + 0.08 * conf))
            rel_vel = clip01(rel_vel - 0.18)
            fuel = clip01(fuel - 0.18)
            conf = clip01(conf - 0.03)
            tca_pressure = clip01(tca_pressure - 0.17)
            burn_cost = 0.18

        severity = clip01(severity + np.random.uniform(-0.03, 0.03))

        action_match_bonus = 0.9 if action_id == ideal_action else -0.25
        safety_bonus = 1.1 if (risk < 0.22 and tca_pressure < 0.40) else 0.0
        catastrophe_penalty = 2.0 if (risk > 0.75 and tca_pressure > 0.70 and fuel < 0.08) else 0.0
        reward = (0.7 * (0.60 - risk)) + action_match_bonus + safety_bonus - (0.55 * burn_cost) - catastrophe_penalty

        self.state = np.array([risk, rel_vel, tca_pressure, fuel, conf, severity], dtype=np.float32)

        terminated = bool(risk < 0.10 and tca_pressure < 0.25)
        truncated = self.steps >= self.max_steps
        info = {
            "ideal_action": float(ideal_action),
            "chosen_action": float(action_id),
            "risk": float(risk),
            "fuel": float(fuel),
        }
        return self.state, float(reward), terminated, truncated, info


def evaluate_policy(model, env: SSACollisionAvoidanceEnv, episodes: int = 50) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    rewards: List[float] = []
    terminal_risk: List[float] = []
    fuel_left: List[float] = []
    action_counts = {k: 0 for k in ACTION_LABELS}
    trace_rows: List[Dict[str, float]] = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        step_idx = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_id = env._discretize_action(action)
            action_counts[action_id] += 1
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += float(reward)
            trace_rows.append(
                {
                    "episode": float(episode),
                    "step": float(step_idx),
                    "risk": float(info["risk"]),
                    "fuel": float(info["fuel"]),
                    "chosen_action": float(info["chosen_action"]),
                    "ideal_action": float(info["ideal_action"]),
                }
            )
            step_idx += 1

        rewards.append(episode_reward)
        assert env.state is not None
        terminal_risk.append(float(env.state[0]))
        fuel_left.append(float(env.state[3]))

    total_actions = max(1, sum(action_counts.values()))
    action_mix = {ACTION_LABELS[k]: float(v) / float(total_actions) for k, v in action_counts.items()}
    metrics = {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_terminal_risk": float(np.mean(terminal_risk)),
        "mean_terminal_fuel": float(np.mean(fuel_left)),
        "safe_episode_rate": float(np.mean(np.array(terminal_risk) < 0.22)),
    }
    metrics.update({f"action_mix_{k}": v for k, v in action_mix.items()})
    return metrics, trace_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train autonomous collision-avoidance RL policy")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo", help="RL algorithm")
    parser.add_argument("--timesteps", type=int, default=120000, help="training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--eval_episodes", type=int, default=60, help="evaluation episodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.algo == "ppo":
        env = SSACollisionAvoidanceEnv(continuous_actions=False)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=args.seed,
            n_steps=1024,
            batch_size=128,
            learning_rate=3e-4,
            gamma=0.995,
        )
    else:
        env = SSACollisionAvoidanceEnv(continuous_actions=True)
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            seed=args.seed,
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.995,
            train_freq=1,
            gradient_steps=1,
        )

    model.learn(total_timesteps=max(2000, args.timesteps))

    artifact_base = out_dir / f"{args.algo}_collision_avoidance"
    model.save(str(artifact_base))

    eval_env = SSACollisionAvoidanceEnv(continuous_actions=(args.algo == "sac"))
    eval_metrics, trace_rows = evaluate_policy(model, eval_env, episodes=max(5, args.eval_episodes))

    metrics = {
        "module": "autonomous_collision_avoidance_rl",
        "algorithm": args.algo.upper(),
        "timesteps": int(args.timesteps),
        "seed": int(args.seed),
        "artifact": str(artifact_base) + ".zip",
        "actions": ACTION_LABELS,
        "iaf_cognitive_claim": {
            "decision_basis": "risk-fuel-confidence-time pressure tradeoff",
            "policy_type": "model-free reinforcement learning",
            "ops_alignment": "collision-avoidance recommendation for SSA ops cells",
        },
        "evaluation": eval_metrics,
    }

    metrics_path = out_dir / f"rl_metrics_{args.algo}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    trace_path = out_dir / f"cognitive_trace_{args.algo}.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(row) + "\n")

    print(f"saved={artifact_base}.zip")
    print(f"saved={metrics_path}")
    print(f"saved={trace_path}")


if __name__ == "__main__":
    main()
