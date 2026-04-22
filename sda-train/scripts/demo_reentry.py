#!/usr/bin/env python3
"""
End-to-end Re-entry Decay Forecaster Demo
1. Generate synthetic decay trajectories via gen_reentry.py
2. Train ReentryForecaster for 3 epochs (toy)
3. Export to ONNX
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gen_reentry import generate_decay, R_E as ALT_LIMIT
from pkg.reentry.decay_forecaster import ReentryForecaster, ReentryForecasterTrainer


def build_batch(trajectories: list, targets: list, batch_size: int = 8):
    idx = np.random.choice(len(trajectories), min(batch_size, len(trajectories)), replace=False)
    feats = np.stack([trajectories[i] for i in idx])
    targs = np.stack([targets[i] for i in idx])
    return {
        "features": torch.from_numpy(feats).float(),
        "target": torch.from_numpy(targs).float(),
    }


def main():
    print("=" * 60)
    print(" Re-entry Decay Forecaster Demo")
    print("=" * 60)

    print("\n[1] Generating synthetic decay trajectories...")
    trajectories = []
    targets = []
    for _ in range(100):
        states, ttd, lat, lon, beta = generate_decay()
        # Downsample to 120 timesteps if longer
        if len(states) > 120:
            idx = np.linspace(0, len(states) - 1, 120, dtype=int)
            states = states[idx]
        elif len(states) < 120:
            pad = np.zeros((120 - len(states), 6))
            states = np.concatenate([states, pad], axis=0)
        trajectories.append(states.astype(np.float32))
        # Target: [ttd_hr, impact_lat, impact_lon]
        targets.append([float(ttd) / 3600.0, float(lat), float(lon)])
    targets = np.array(targets, dtype=np.float32)
    print(f"   Trajectories: {len(trajectories)}, Avg steps: {np.mean([len(t) for t in trajectories]):.0f}")
    print(f"   Avg TTD: {targets[:, 0].mean():.1f} hr, Min: {targets[:, 0].min():.1f} hr")

    print("\n[2] Training ReentryForecaster (3 toy epochs)...")
    trainer = ReentryForecasterTrainer(device="cpu", lr=1e-3)
    for epoch in range(3):
        batch = build_batch(trajectories, targets, batch_size=8)
        metrics = trainer.train_step(batch)
        print(f"   Epoch {epoch+1}: loss={metrics['loss']:.4f}")

    print("\n[3] Exporting ONNX...")
    onnx_path = trainer.export_onnx("ai/training/models/reentry_forecaster_v1.onnx")
    size_kb = Path(onnx_path).stat().st_size / 1024
    print(f"   ONNX: {onnx_path} ({size_kb:.1f} KB)")

    print("\n[4] Inference on sample trajectory...")
    model = trainer.model
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(trajectories[0]).unsqueeze(0).float()
        pred = model(x)[0].numpy()
    print(f"   Ground truth: ttd={targets[0, 0]:.1f} hr, lat={targets[0, 1]:.2f}, lon={targets[0, 2]:.2f}")
    print(f"   Predicted:    ttd={pred[0]:.1f} hr, lat={pred[1]:.2f}, lon={pred[2]:.2f}")

    print("\n" + "=" * 60)
    print(" Re-entry Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
