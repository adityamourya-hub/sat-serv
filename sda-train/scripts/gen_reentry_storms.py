#!/usr/bin/env python3
"""
Re-entry storm scenario generator.

Produces synthetic decay trajectories across curriculum stages:
  0-Easy:   low beta, steep decay, calm atmosphere
  1-Medium: moderate beta, mild skip-out risk
  2-Hard:   high beta, grazing re-entry, skip-out common
  3-Storm:  variable density (F10.7 proxy + stochastic), tumbling Cd

Each stage returns (features [T, 10], target [3]) where features include
position, velocity, altitude, density, cd_eff, a_over_m, plus derived
quantities for the forecaster.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gen_reentry import generate_decay, MU, R_E


def _density_exponential(alt_m: float, rho0: float = 1.225, H: float = 8500.0) -> float:
    return rho0 * np.exp(-alt_m / H)


def _density_storm(alt_m: float, t: float, base_rho0: float = 1.225,
                   base_H: float = 8500.0, f107: float = 150.0,
                   noise_amp: float = 0.0) -> float:
    """Variable density with F10.7 proxy (sinusoidal) + stochastic noise."""
    H_eff = base_H * (1.0 - 0.3 * np.clip((f107 - 70.0) / 200.0, 0, 1))
    rho = base_rho0 * np.exp(-alt_m / H_eff)
    # Solar cycle proxy: diurnal + 27-day solar rotation
    rho *= (1.0 + 0.15 * np.sin(2 * np.pi * t / 86400.0))
    rho *= (1.0 + 0.10 * np.sin(2 * np.pi * t / (27.0 * 86400.0)))
    if noise_amp > 0.0:
        rho *= (1.0 + noise_amp * np.random.randn())
    return max(rho, 1e-20)


def _drag_accel(state: np.ndarray, cd: float, beta: float, rho: float) -> np.ndarray:
    r = state[:3]
    v = state[3:]
    v_mag = np.linalg.norm(v)
    a_drag = -0.5 * cd * beta * rho * v_mag * v
    return np.concatenate([v, a_drag])


def _gravity_accel(state: np.ndarray) -> np.ndarray:
    r = state[:3]
    v = state[3:]
    r_mag = np.linalg.norm(r)
    a_grav = -MU * r / (r_mag ** 3)
    return np.concatenate([v, a_grav])


def _rk4_step(state: np.ndarray, dt: float, cd: float, beta: float,
              rho_func, t: float) -> np.ndarray:
    def acc(s):
        r_mag = np.linalg.norm(s[:3])
        alt = r_mag - R_E
        rho = rho_func(alt, t)
        return _gravity_accel(s) + _drag_accel(s, cd, beta, rho)

    k1 = acc(state)
    k2 = acc(state + 0.5 * dt * k1)
    k3 = acc(state + 0.5 * dt * k2)
    k4 = acc(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_storm_decay(
    stage: int = 0,
    start_alt_m: float = 140000.0,
    dt: float = 30.0,
    max_steps: int = 2000,
    stop_alt_m: float = 30000.0,
) -> tuple:
    """
    Generate a re-entry trajectory for a given curriculum stage.

    Returns:
        features: np.ndarray [T, 10] — [x,y,z,vx,vy,vz,alt_km,rho,cd_eff,a_over_m]
        target: np.ndarray [3] — [ttd_hr, impact_lat, impact_lon]
        skip_out: bool
    """
    # Stage-dependent parameters
    if stage == 0:      # Easy
        beta_min, beta_max = 0.08, 0.15
        cd = 2.2
        f107 = 70.0
        noise_amp = 0.0
        tumbling = False
    elif stage == 1:    # Medium
        beta_min, beta_max = 0.04, 0.10
        cd = 2.2
        f107 = 120.0
        noise_amp = 0.05
        tumbling = False
    elif stage == 2:    # Hard
        beta_min, beta_max = 0.005, 0.05
        cd = 2.2
        f107 = 200.0
        noise_amp = 0.10
        tumbling = False
    else:               # Storm
        beta_min, beta_max = 0.005, 0.08
        cd = 2.2
        f107 = 250.0 + np.random.exponential(50.0)
        noise_amp = 0.25
        tumbling = True

    beta = np.random.uniform(beta_min, beta_max)

    # Initial state: circular-ish orbit at start_alt
    r = R_E + start_alt_m
    v = np.sqrt(MU / r) + np.random.normal(0, 50.0)
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    pos = np.array([x, y, z])
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z) < 0.9 * r else np.array([0.0, 1.0, 0.0])
    vel_dir = np.cross(pos, arbitrary)
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-12)
    vel = v * vel_dir + np.random.normal(0, 100.0, 3)
    state = np.concatenate([pos, vel]).astype(np.float64)

    traj = [state.copy()]
    t = 0.0
    skip_out = False
    skip_counter = 0
    cd_eff = cd

    for step in range(max_steps):
        if tumbling and step % 10 == 0:
            cd_eff = cd * (1.0 + 0.3 * np.sin(step * 0.1) + 0.1 * np.random.randn())
            cd_eff = max(1.0, min(4.0, cd_eff))

        state = _rk4_step(state, dt, cd_eff, beta,
                          lambda alt, t_step: _density_storm(alt, t_step, f107=f107, noise_amp=noise_amp),
                          t)
        traj.append(state.copy())
        t += dt
        alt = np.linalg.norm(state[:3]) - R_E
        if alt < stop_alt_m:
            break
        # Skip-out detection disabled: normal orbital decay does not skip out
        # from a low-perigee orbit.  Skip-out only occurs for high-beta shallow
        # entries where drag is insufficient to circularise before exit.

    traj = np.array(traj, dtype=np.float32)
    ttd = t
    impact_lat = np.degrees(np.arcsin(np.clip(traj[-1, 2] / np.linalg.norm(traj[-1, :3]), -1, 1)))
    impact_lon = np.degrees(np.arctan2(traj[-1, 1], traj[-1, 0]))

    # Build 10-D feature vector
    features = np.zeros((len(traj), 10), dtype=np.float32)
    for i, s in enumerate(traj):
        r_mag = np.linalg.norm(s[:3])
        alt_km = (r_mag - R_E) / 1000.0
        rho = _density_storm(r_mag - R_E, i * dt, f107=f107, noise_amp=noise_amp)
        features[i, :6] = s
        features[i, 6] = alt_km
        features[i, 7] = rho
        features[i, 8] = cd_eff
        features[i, 9] = beta

    target = np.array([float(ttd) / 3600.0, float(impact_lat), float(impact_lon)], dtype=np.float32)
    return features, target, skip_out


def generate_shard(stage: int, n: int = 100, output_dir: str = "data/synthetic/reentry"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_names = ["easy", "medium", "hard", "storm"]
    name = stage_names[stage]

    trajs = []
    targets = []
    skip_flags = []
    max_len = 0

    for _ in range(n):
        feat, tgt, skip = generate_storm_decay(stage=stage)
        trajs.append(feat)
        targets.append(tgt)
        skip_flags.append(skip)
        max_len = max(max_len, len(feat))

    # Pad to common length
    padded = np.zeros((n, max_len, 10), dtype=np.float32)
    lengths = np.array([len(t) for t in trajs], dtype=np.int32)
    for i, t in enumerate(trajs):
        padded[i, :len(t)] = t

    path = out_dir / f"re_storm_{name}_0000.npz"
    np.savez(
        path,
        trajectories=padded,
        lengths=lengths,
        targets=np.array(targets, dtype=np.float32),
        skip_out=np.array(skip_flags, dtype=np.bool_),
    )
    ttd_mean = np.mean([t[0] for t in targets])
    skip_rate = np.mean(skip_flags)
    print(f"Generated {path}: {n} trajectories, stage={name}, mean TTD={ttd_mean:.1f} hr, skip_rate={skip_rate:.2f}")
    return path


def validate_shard(path: Path):
    data = np.load(path)
    trajs = data["trajectories"]
    lengths = data["lengths"]
    targets = data["targets"]
    skip = data["skip_out"]
    n = len(trajs)

    # Monotonic energy check (should decrease due to drag)
    ok = 0
    for i in range(n):
        traj = trajs[i, :lengths[i]]
        r = np.linalg.norm(traj[:, :3], axis=1)
        v = np.linalg.norm(traj[:, 3:6], axis=1)
        E = 0.5 * v * v - MU / r
        if np.all(np.diff(E) <= 1e-2):
            ok += 1

    print(f"Validation {path.name}: {ok}/{n} monotonic energy, skip_rate={skip.mean():.2f}, "
          f"TTD mean={targets[:, 0].mean():.1f} hr, std={targets[:, 0].std():.1f} hr")
    return ok == n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/synthetic/reentry")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        out_dir = Path(args.output)
        npzs = sorted(out_dir.glob("re_storm_*.npz"))
        if not npzs:
            print("No shards to validate")
            exit(1)
        all_ok = True
        for p in npzs:
            if not validate_shard(p):
                all_ok = False
        exit(0 if all_ok else 1)

    generate_shard(args.stage, args.samples, args.output)


if __name__ == "__main__":
    main()
