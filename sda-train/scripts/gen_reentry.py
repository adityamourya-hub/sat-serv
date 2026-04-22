#!/usr/bin/env python3
"""Re-entry decay scenario generator."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MU = 3.986004418e14
R_E = 6371000.0
G0 = 9.81


def _density_exponential(alt_m: float, rho0: float = 1.225, H: float = 8500.0) -> float:
    return rho0 * np.exp(-alt_m / H)


def _drag_accel(state: np.ndarray, cd: float, beta: float) -> np.ndarray:
    """state: [x,y,z,vx,vy,vz]."""
    r = state[:3]
    v = state[3:]
    r_mag = np.linalg.norm(r)
    alt = r_mag - R_E
    v_mag = np.linalg.norm(v)
    rho = _density_exponential(alt)
    a_drag = -0.5 * cd * beta * rho * v_mag * v
    return np.concatenate([v, a_drag])


def _gravity_accel(state: np.ndarray) -> np.ndarray:
    r = state[:3]
    v = state[3:]
    r_mag = np.linalg.norm(r)
    a_grav = -MU * r / (r_mag ** 3)
    return np.concatenate([v, a_grav])


def _total_accel(state: np.ndarray, cd: float, beta: float) -> np.ndarray:
    return _gravity_accel(state) + _drag_accel(state, cd, beta)


def _rk4_step(state: np.ndarray, dt: float, cd: float, beta: float) -> np.ndarray:
    k1 = _total_accel(state, cd, beta)
    k2 = _total_accel(state + 0.5 * dt * k1, cd, beta)
    k3 = _total_accel(state + 0.5 * dt * k2, cd, beta)
    k4 = _total_accel(state + dt * k3, cd, beta)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_decay(
    start_alt_m: float = 140000.0,
    cd: float = 2.2,
    beta_min: float = 0.005,
    beta_max: float = 0.1,
    dt: float = 1.0,
    max_steps: int = 50000,
    stop_alt_m: float = 30000.0,
):
    """Generate one re-entry trajectory."""
    beta = np.random.uniform(beta_min, beta_max)
    # Start with circular-ish orbit at start_alt
    r = R_E + start_alt_m
    v = np.sqrt(MU / r) + np.random.normal(0, 50.0)
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    pos = np.array([x, y, z])
    # Velocity mostly tangential
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z) < 0.9 * r else np.array([0.0, 1.0, 0.0])
    vel_dir = np.cross(pos, arbitrary)
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-12)
    vel = v * vel_dir + np.random.normal(0, 100.0, 3)
    state = np.concatenate([pos, vel]).astype(np.float64)

    traj = [state.copy()]
    t = 0.0
    for _ in range(max_steps):
        state = _rk4_step(state, dt, cd, beta)
        traj.append(state.copy())
        t += dt
        alt = np.linalg.norm(state[:3]) - R_E
        if alt < stop_alt_m:
            break
        # Skip-out detection: radial velocity positive and alt increasing
        r_mag = np.linalg.norm(state[:3])
        v_radial = np.dot(state[3:], state[:3]) / (r_mag + 1e-12)
        if alt > start_alt_m and v_radial > 0:
            break

    traj = np.array(traj, dtype=np.float32)
    ttd = t  # time to demise from first step
    impact_lat = np.degrees(np.arcsin(np.clip(traj[-1, 2] / np.linalg.norm(traj[-1, :3]), -1, 1)))
    impact_lon = np.degrees(np.arctan2(traj[-1, 1], traj[-1, 0]))
    return traj, ttd, impact_lat, impact_lon, beta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/synthetic/reentry")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.validate:
        npzs = sorted(out_dir.glob("re_*.npz"))
        if not npzs:
            print("No shards to validate")
            exit(1)
        monotonic = 0
        issues = 0
        for p in npzs:
            data = np.load(p)
            for traj in data["trajectories"]:
                # Check specific orbital energy decreases monotonically (or stays within tolerance)
                r = np.linalg.norm(traj[:, :3], axis=1)
                v = np.linalg.norm(traj[:, 3:], axis=1)
                E = 0.5 * v * v - MU / r
                if np.all(np.diff(E) <= 1e-3):
                    monotonic += 1
                else:
                    issues += 1
        print(f"Reentry validation: {monotonic} monotonic, {issues} issues")
        exit(0 if issues == 0 else 1)

    trajs = []
    ttds = []
    lats = []
    lons = []
    betas = []

    for i in range(args.samples):
        traj, ttd, lat, lon, beta = generate_decay()
        trajs.append(traj)
        ttds.append(ttd)
        lats.append(lat)
        lons.append(lon)
        betas.append(beta)

    # Pad trajectories to common length
    max_len = max(len(t) for t in trajs)
    padded = np.zeros((args.samples, max_len, 6), dtype=np.float32)
    lengths = np.array([len(t) for t in trajs], dtype=np.int32)
    for i, t in enumerate(trajs):
        padded[i, :len(t)] = t

    path = out_dir / "re_0000.npz"
    np.savez(
        path,
        trajectories=padded,
        lengths=lengths,
        ttd=np.array(ttds, dtype=np.float32),
        impact_lat=np.array(lats, dtype=np.float32),
        impact_lon=np.array(lons, dtype=np.float32),
        beta=np.array(betas, dtype=np.float32),
    )
    print(f"Generated {path}: {args.samples} trajectories, mean TTD={np.mean(ttds):.1f}s")


if __name__ == "__main__":
    main()
