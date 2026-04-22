#!/usr/bin/env python3
"""Monte Carlo conjunction scenario generator + empirical Pc labels."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MU = 3.986004418e14
R_E = 6371000.0


def _random_state_near(alt_m: float = 400e3):
    r = R_E + alt_m + np.random.normal(0, 5e3)
    v = math.sqrt(MU / r) + np.random.normal(0, 20.0)
    theta = np.random.uniform(0, 2 * math.pi)
    phi = np.random.uniform(0, math.pi)
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    pos = np.array([x, y, z])
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z) < 0.9 * r else np.array([0.0, 1.0, 0.0])
    vel_dir = np.cross(pos, arbitrary)
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-12)
    vel = v * vel_dir + np.random.normal(0, 5.0, 3)
    return np.concatenate([pos, vel]).astype(np.float64)


def _propagate_linear(state: np.ndarray, dt: float = 1.0, steps: int = 100):
    """Simple linear propagation for MC sampling (fast)."""
    traj = np.zeros((steps, 6), dtype=np.float64)
    traj[0] = state
    for i in range(1, steps):
        # 2-body only for speed
        r = traj[i - 1, :3]
        v = traj[i - 1, 3:]
        r_mag = np.linalg.norm(r)
        a = -MU * r / (r_mag ** 3)
        traj[i, :3] = r + v * dt + 0.5 * a * dt * dt
        traj[i, 3:] = v + a * dt
    return traj


def _compute_cpa(primary: np.ndarray, secondary: np.ndarray):
    """Find closest approach geometry from two propagated trajectories."""
    rel = secondary[:, :3] - primary[:, :3]
    rel_v = secondary[:, 3:] - primary[:, 3:]
    ranges = np.linalg.norm(rel, axis=1)
    min_idx = int(np.argmin(ranges))
    min_range = float(ranges[min_idx])
    rel_vel = np.linalg.norm(rel_v[min_idx])
    approach_rate = float(np.dot(rel[min_idx], rel_v[min_idx]) / (ranges[min_idx] + 1e-12))
    ttc = float(min_idx)
    return min_range, rel_vel, approach_rate, ttc, rel[min_idx], rel_v[min_idx]


def _empirical_pc(primary: np.ndarray, secondary: np.ndarray,
                   sigma_pos: float = 100.0, sigma_vel: float = 1.0,
                   combined_radius: float = 20.0, mc_samples: int = 10000):
    """Monte Carlo miss distance for empirical Pc."""
    # Use CPA point only for speed (sufficient for baseline)
    _, _, _, _, rel_pos, rel_vel = _compute_cpa(primary, secondary)
    # Sample position uncertainty sphere + velocity uncertainty
    pos_samples = np.random.normal(0, sigma_pos, (mc_samples, 3)) + rel_pos
    vel_samples = np.random.normal(0, sigma_vel, (mc_samples, 3)) + rel_vel
    # Project onto collision plane (approximate: plane perpendicular to relative velocity)
    v_hat = rel_vel / (np.linalg.norm(rel_vel) + 1e-12)
    proj = pos_samples - np.outer(np.dot(pos_samples, v_hat), v_hat)
    miss_distances = np.linalg.norm(proj, axis=1)
    hits = np.sum(miss_distances < combined_radius)
    return float(hits / mc_samples)


def _analytic_pc(miss_distance: float, combined_radius: float = 20.0,
                 sigma_pos: float = 100.0):
    """Simplified 2D Gaussian Pc (Alfriend 1999 approximation)."""
    u = miss_distance / sigma_pos
    return float(math.exp(-0.5 * u * u))


def generate_encounter(mc_samples: int = 10000, combined_radius: float = 20.0):
    primary = _random_state_near()
    secondary = _random_state_near()
    primary_traj = _propagate_linear(primary)
    secondary_traj = _propagate_linear(secondary)

    miss_dist, rel_speed, approach_rate, ttc, rel_pos, rel_vel = _compute_cpa(
        primary_traj, secondary_traj
    )

    emp_pc = _empirical_pc(primary_traj, secondary_traj,
                           combined_radius=combined_radius, mc_samples=mc_samples)
    ana_pc = _analytic_pc(miss_dist, combined_radius=combined_radius)

    size_p = np.random.uniform(1.0, 5.0)
    size_s = np.random.uniform(1.0, 5.0)
    aspect = np.random.uniform(1.0, 5.0)
    solar_angle = np.random.uniform(0, math.pi)
    inclination = math.acos(np.clip(primary[2] / np.linalg.norm(primary[:3]), -1, 1))

    record = {
        "features": [
            math.log10(max(miss_dist, 1.0)),
            math.log10(max(ana_pc, 1e-15)),
            rel_speed,
            size_p,
            size_s,
            aspect,
            100.0,  # pos_uncertainty
            1.0,    # vel_uncertainty
            solar_angle,
            ttc,
            math.sin(inclination),
            math.cos(inclination),
        ],
        "empirical_pc": emp_pc,
        "analytic_pc": ana_pc,
        "miss_distance_m": miss_dist,
        "time_to_cpa_s": ttc,
    }
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/synthetic/conjunction")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.validate:
        jsonls = sorted(out_dir.glob("conj_*.jsonl"))
        if not jsonls:
            print("No shards to validate")
            exit(1)
        pcs = []
        for p in jsonls:
            with open(p) as f:
                for line in f:
                    rec = json.loads(line)
                    pcs.append(rec["empirical_pc"])
        pcs = np.array(pcs)
        print(f"Samples: {len(pcs)}, High-Pc (>1e-4): {(pcs > 1e-4).mean():.4f}")
        exit(0 if 0.005 <= (pcs > 1e-4).mean() <= 0.20 else 1)

    path = out_dir / "conj_0000.jsonl"
    with open(path, "w") as f:
        for i in range(args.samples):
            rec = generate_encounter()
            f.write(json.dumps(rec) + "\n")

    print(f"Generated {path}: {args.samples} encounters")


if __name__ == "__main__":
    main()
