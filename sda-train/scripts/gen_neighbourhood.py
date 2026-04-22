#!/usr/bin/env python3
"""Neighbourhood Watch encounter scenario generator."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

J2 = 1.08263e-3  # Earth's J2
MU = 3.986004418e14  # m^3/s^2
R_E = 6371000.0  # m


def _j2_accel(state: np.ndarray) -> np.ndarray:
    """Add J2 perturbation to 2-body acceleration. state: [x,y,z,vx,vy,vz] m, m/s."""
    r = state[:3]
    v = state[3:]
    r_mag = np.linalg.norm(r)
    z2 = r[2] ** 2
    factor = 1.5 * J2 * MU * (R_E ** 2) / (r_mag ** 5)
    accel = -MU * r / (r_mag ** 3)
    accel += factor * np.array([
        r[0] * (5 * z2 / (r_mag ** 2) - 1),
        r[1] * (5 * z2 / (r_mag ** 2) - 1),
        r[2] * (5 * z2 / (r_mag ** 2) - 3),
    ])
    return np.concatenate([v, accel])


def _rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = _j2_accel(state)
    k2 = _j2_accel(state + 0.5 * dt * k1)
    k3 = _j2_accel(state + 0.5 * dt * k2)
    k4 = _j2_accel(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _propagate(state: np.ndarray, duration: float = 120.0, dt: float = 1.0) -> np.ndarray:
    steps = int(duration / dt)
    traj = np.zeros((steps, 6), dtype=np.float32)
    traj[0] = state
    for i in range(1, steps):
        traj[i] = _rk4_step(traj[i - 1], dt)
    return traj


def _random_leo_state(alt_min: float = 200e3, alt_max: float = 1200e3):
    alt = np.random.uniform(alt_min, alt_max)
    r = R_E + alt
    # Circular orbit in random plane
    inc = np.random.uniform(0, np.pi)
    raan = np.random.uniform(0, 2 * np.pi)
    v = np.sqrt(MU / r)
    # Simple random position on sphere
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    # Velocity perpendicular to position (simplified)
    pos = np.array([x, y, z])
    # Random orthogonal velocity direction
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z) < 0.9 * r else np.array([0.0, 1.0, 0.0])
    vel_dir = np.cross(pos, arbitrary)
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-12)
    vel = v * vel_dir
    return np.array([x, y, z, vel[0], vel[1], vel[2]], dtype=np.float64)


def generate_encounter(duration: float = 120.0, dt: float = 1.0, noise_sigma_pos: float = 50.0, noise_sigma_vel: float = 0.5):
    """Generate one encounter window."""
    s1 = _random_leo_state()
    s2 = _random_leo_state()

    traj1 = _propagate(s1, duration, dt)
    traj2 = _propagate(s2, duration, dt)

    rel_pos = traj2[:, :3] - traj1[:, :3]
    rel_vel = traj2[:, 3:] - traj1[:, 3:]
    ranges = np.linalg.norm(rel_pos, axis=1)
    approach_rates = np.sum(rel_pos * rel_vel, axis=1) / (ranges + 1e-6)

    min_range = ranges.min()
    min_idx = int(ranges.argmin())
    tti = min_idx * dt

    # Features per timestep: [rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz, approach_rate, miss_dist, solar_phase, illum_flag]
    # Simplified: solar_phase = random, illum_flag = 1 always
    features = np.zeros((int(duration / dt), 10), dtype=np.float32)
    features[:, :3] = rel_pos + np.random.normal(0, noise_sigma_pos, rel_pos.shape)
    features[:, 3:6] = rel_vel + np.random.normal(0, noise_sigma_vel, rel_vel.shape)
    features[:, 6] = approach_rates
    features[:, 7] = min_range  # miss distance (constant across window for label leakage prevention, actually we shouldn't do that - but for baseline it's fine)
    # Better: use instantaneous range as a proxy feature, but don't leak min_range
    features[:, 7] = ranges
    features[:, 8] = np.random.uniform(0, np.pi, size=features.shape[0])
    features[:, 9] = 1.0

    # Label
    if min_range < 50000.0 and approach_rates[min_idx] < 0:
        intent = 1
    else:
        intent = 0
        tti = -1.0

    return features, int(intent), float(tti), float(min_range)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/synthetic/neighbourhood")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.validate:
        # Check class balance from existing shards
        npzs = sorted(out_dir.glob("nbh_*.npz"))
        if not npzs:
            print("No shards to validate")
            exit(1)
        intents = []
        for p in npzs:
            data = np.load(p)
            intents.extend(data["intent"])
        intents = np.array(intents)
        print(f"Samples: {len(intents)}, Intercept ratio: {intents.mean():.4f}")
        exit(0 if 0.05 <= intents.mean() <= 0.20 else 1)

    features_list = []
    intents = []
    ttis = []
    miss_dists = []

    for i in range(args.samples):
        feat, intent, tti, miss = generate_encounter()
        features_list.append(feat)
        intents.append(intent)
        ttis.append(tti)
        miss_dists.append(miss)

    path = out_dir / "nbh_0000.npz"
    np.savez(
        path,
        features=np.stack(features_list),
        intent=np.array(intents, dtype=np.int32),
        tti=np.array(ttis, dtype=np.float32),
        miss_dist=np.array(miss_dists, dtype=np.float32),
    )
    print(f"Generated {path}: {args.samples} windows, intercept ratio={np.mean(intents):.3f}")


if __name__ == "__main__":
    main()
