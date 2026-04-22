#!/usr/bin/env python3
"""Volumetric scan scenario generator for BEV training."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.render_sim import PixelRenderer


class VolumetricScenarioGenerator:
    """Generates BEV occupancy grids + 3D labels for volumetric scanning."""

    def __init__(
        self,
        output_dir: str = "data/synthetic/volumetric",
        bev_size: int = 256,
        volume_km: float = 500.0,
        resolution: tuple = (1024, 1024),
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bev_size = bev_size
        self.volume_km = volume_km
        self.resolution = resolution
        self.renderer = PixelRenderer(resolution=resolution)
        self.cell_km = volume_km / bev_size

    def _generate_truth_states(self, num_objects: int = 20, level: int = 1):
        """Generate truth states around a central LEO point."""
        states = []
        # Central orbit: 400 km altitude
        r = 6371000.0 + 400000.0
        v = 7500.0
        for i in range(num_objects):
            # Perturb around central point in a cube of side volume_km
            offset_km = self.volume_km / 2.0
            px = np.random.uniform(-offset_km, offset_km) * 1000.0
            py = np.random.uniform(-offset_km, offset_km) * 1000.0
            pz = np.random.uniform(-offset_km, offset_km) * 1000.0
            mag = 5.0 + np.random.rand() * 6.0  # 5–11
            states.append({
                "sat_id": i + 1,
                "x": r + px,
                "y": py,
                "z": pz,
                "vx": np.random.randn() * 10.0,
                "vy": v + np.random.randn() * 50.0,
                "vz": np.random.randn() * 10.0,
                "mag": mag,
            })
        return states

    def _states_to_bev(self, states: list):
        """Project states into a 2D BEV grid with 4 channels."""
        bev = np.zeros((self.bev_size, self.bev_size, 4), dtype=np.float32)
        labels = []
        half = self.volume_km / 2.0
        for s in states:
            # Project x,y into BEV coordinates (ignore z for BEV, store in channel 3)
            ix = int((s["y"] / 1000.0 + half) / self.cell_km)
            iy = int((s["z"] / 1000.0 + half) / self.cell_km)
            ix = np.clip(ix, 0, self.bev_size - 1)
            iy = np.clip(iy, 0, self.bev_size - 1)

            # Channel 0: objectness / count
            bev[iy, ix, 0] += 1.0
            # Channel 1–3: mean velocity components (m/s) scaled by 1e-3
            bev[iy, ix, 1] = (bev[iy, ix, 1] * (bev[iy, ix, 0] - 1) + s["vx"] * 1e-3) / bev[iy, ix, 0]
            bev[iy, ix, 2] = (bev[iy, ix, 2] * (bev[iy, ix, 0] - 1) + s["vy"] * 1e-3) / bev[iy, ix, 0]
            bev[iy, ix, 3] = (bev[iy, ix, 3] * (bev[iy, ix, 0] - 1) + s["vz"] * 1e-3) / bev[iy, ix, 0]

            labels.append([
                s["x"], s["y"], s["z"],
                s["vx"], s["vy"], s["vz"],
                1 if s["mag"] < 8.0 else 2 if s["mag"] < 10.0 else 3,  # 1=sat, 2=debris, 3=unknown
            ])

        # Normalize objectness channel
        if bev[..., 0].max() > 0:
            bev[..., 0] = np.clip(bev[..., 0] / 5.0, 0.0, 1.0)

        return bev, np.array(labels, dtype=np.float32)

    def generate_shard(self, shard_id: int, num_scenes: int = 100, level: int = 1):
        """Generate one NPZ shard."""
        bevs = []
        label_batches = []
        for i in range(num_scenes):
            states = self._generate_truth_states(num_objects=20 + level * 20, level=level)
            bev, labels = self._states_to_bev(states)
            bevs.append(bev)
            label_batches.append(labels)

        # Pad labels to common length for batching
        max_objs = max(len(lb) for lb in label_batches)
        padded_labels = np.zeros((num_scenes, max_objs, 7), dtype=np.float32)
        for i, lb in enumerate(label_batches):
            n = len(lb)
            padded_labels[i, :n] = lb

        path = self.output_dir / f"vol_{shard_id:04d}_l{level}.npz"
        np.savez(
            path,
            bev=np.stack(bevs),
            labels=padded_labels,
            num_labels=np.array([len(lb) for lb in label_batches], dtype=np.int32),
            timestamp=time.time(),
        )
        return path

    def validate(self):
        """Validate generated shards."""
        ok = 0
        issues = []
        for npz in sorted(self.output_dir.glob("vol_*.npz")):
            data = np.load(npz)
            if "bev" not in data or "labels" not in data:
                issues.append(f"{npz.name}: missing keys")
                continue
            if data["bev"].shape[-1] != 4:
                issues.append(f"{npz.name}: wrong channel count {data['bev'].shape[-1]}")
                continue
            ok += 1
        print(f"Volumetric validation: {ok} OK, {len(issues)} issues")
        for issue in issues[:10]:
            print(f"  {issue}")
        return len(issues) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--scenes-per-shard", type=int, default=100)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    gen = VolumetricScenarioGenerator()
    if args.validate:
        exit(0 if gen.validate() else 1)

    for sid in range(args.shards):
        p = gen.generate_shard(sid, args.scenes_per_shard, args.level)
        print(f"Generated {p}")


if __name__ == "__main__":
    main()
