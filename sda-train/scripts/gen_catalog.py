#!/usr/bin/env python3
"""Catalog chip + kinematics scenario generator."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.render_sim import PixelRenderer

MU = 3.986004418e14
R_E = 6371000.0


def _random_orbital_elements():
    a = R_E + np.random.uniform(200e3, 2000e3)
    e = np.random.uniform(0, 0.3)
    i = np.random.uniform(0, np.pi)
    raan = np.random.uniform(0, 2 * np.pi)
    arg_p = np.random.uniform(0, 2 * np.pi)
    M = np.random.uniform(0, 2 * np.pi)
    period = 2 * np.pi * np.sqrt(a ** 3 / MU)
    return {
        "a": float(a),
        "e": float(e),
        "i": float(i),
        "raan": float(raan),
        "arg_perigee": float(arg_p),
        "mean_anomaly": float(M),
        "period": float(period),
    }


def _elements_to_state(elements: dict):
    # Simplified: only semi-major axis used for circular approximation
    a = elements["a"]
    v = np.sqrt(MU / a)
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    x = a * np.sin(phi) * np.cos(theta)
    y = a * np.sin(phi) * np.sin(theta)
    z = a * np.cos(phi)
    pos = np.array([x, y, z])
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z) < 0.9 * a else np.array([0.0, 1.0, 0.0])
    vel_dir = np.cross(pos, arbitrary)
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-12)
    vel = v * vel_dir
    return np.concatenate([pos, vel]).astype(np.float64)


def _render_chips_for_states(states: list, renderer: PixelRenderer, chip_size: int = 64):
    """Render a frame and crop chips around projected object positions."""
    # Project to 2D (same logic as render_sim.py)
    frame = np.zeros(renderer.resolution)
    height, width = frame.shape
    projection_scale = 100.0

    for s in states:
        px = int((s["y"] / projection_scale) % width)
        py = int((s["z"] / projection_scale) % height)
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        brightness = 10 ** ((10 - s["mag"]) / 2.5)
        frame[py, px] += brightness

    frame = renderer.apply_psf(frame)
    frame = renderer.add_noise(frame, noise_type="mixed")
    frame = np.clip(frame, 0, 65535)

    chips = []
    for s in states:
        px = int((s["y"] / projection_scale) % width)
        py = int((s["z"] / projection_scale) % height)
        half = chip_size // 2
        x0, x1 = max(0, px - half), min(width, px + half)
        y0, y1 = max(0, py - half), min(height, py + half)
        chip = frame[y0:y1, x0:x1]
        if chip.shape[0] < chip_size or chip.shape[1] < chip_size:
            pad_y = chip_size - chip.shape[0]
            pad_x = chip_size - chip.shape[1]
            chip = np.pad(chip, ((0, pad_y), (0, pad_x)), mode="constant")
        chips.append(chip)
    return np.stack(chips) if chips else np.zeros((0, chip_size, chip_size))


class CatalogScenarioGenerator:
    def __init__(self, output_dir: str = "data/synthetic/catalog", chip_size: int = 64):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chip_size = chip_size
        self.renderer = PixelRenderer(resolution=(1024, 1024))

    def generate_shard(self, num_objects: int = 100):
        states = []
        elements_list = []
        labels = []
        for _ in range(num_objects):
            elems = _random_orbital_elements()
            state_vec = _elements_to_state(elems)
            mag = np.random.uniform(3.0, 12.0)
            bus_type = int(np.random.choice([0, 1, 2, 3]))  # 4 bus types
            mission = int(np.random.choice([0, 1, 2, 3, 4]))  # 5 mission classes
            size_class = int(np.random.choice([0, 1, 2]))  # small/medium/large
            attitude_stable = int(np.random.rand() > 0.3)
            rcs = np.random.uniform(0.1, 10.0)
            states.append({
                "sat_id": np.random.randint(100000, 999999),
                "x": float(state_vec[0]),
                "y": float(state_vec[1]),
                "z": float(state_vec[2]),
                "vx": float(state_vec[3]),
                "vy": float(state_vec[4]),
                "vz": float(state_vec[5]),
                "mag": float(mag),
            })
            elements_list.append(elems)
            labels.append([bus_type, mission, size_class, attitude_stable, rcs])

        chips = _render_chips_for_states(states, self.renderer, self.chip_size)
        kinematics = np.array([
            [
                e["a"] / 1e6,
                e["e"],
                e["i"] / np.pi,
                e["raan"] / (2 * np.pi),
                e["arg_perigee"] / (2 * np.pi),
                e["mean_anomaly"] / (2 * np.pi),
                e["period"] / 9000.0,
                l[4] / 10.0,  # rcs normalized
            ]
            for e, l in zip(elements_list, labels)
        ], dtype=np.float32)

        path = self.output_dir / "cat_0000.npz"
        np.savez(
            path,
            chips=chips.astype(np.float32),
            kinematics=kinematics,
            labels=np.array(labels, dtype=np.int32),
        )
        return path

    def validate(self):
        npzs = sorted(self.output_dir.glob("cat_*.npz"))
        if not npzs:
            print("No shards to validate")
            return False
        ok = 0
        for p in npzs:
            data = np.load(p)
            if data["chips"].shape[-2:] != (self.chip_size, self.chip_size):
                print(f"{p.name}: wrong chip shape")
                continue
            if data["kinematics"].shape[1] != 8:
                print(f"{p.name}: wrong kinematic dim")
                continue
            ok += 1
        print(f"Catalog validation: {ok} OK")
        return ok > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/synthetic/catalog")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    gen = CatalogScenarioGenerator(output_dir=args.output)
    if args.validate:
        exit(0 if gen.validate() else 1)

    p = gen.generate_shard(args.objects)
    print(f"Generated {p}")


if __name__ == "__main__":
    main()
