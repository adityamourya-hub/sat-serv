#!/usr/bin/env python3
"""
End-to-end Conjunction Pc Calibrator Demo
1. Generate synthetic MC encounter shards via gen_conjunction.py
2. Train PcCalibrator for 5 epochs (toy)
3. Export to ONNX
4. Validate monotonicity and calibration
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gen_conjunction import generate_encounter
from pkg.conjunction.pc_calibrator import PcCalibratorTrainer, load_jsonl_shard


def generate_shard(path: Path, n: int = 2000):
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(path, "w") as f:
        for _ in range(n):
            rec = generate_encounter(mc_samples=1000)
            f.write(json.dumps(rec) + "\n")
    print(f"Generated {path}: {n} encounters")


def main():
    print("=" * 60)
    print(" Conjunction Pc Calibrator Demo")
    print("=" * 60)

    shard_path = Path("data/synthetic/conjunction/conj_demo.jsonl")
    generate_shard(shard_path, n=2000)

    print("\n[1] Loading data...")
    data = load_jsonl_shard(shard_path)
    print(f"   Features: {data['features'].shape}, Targets: {data['target'].shape}")
    print(f"   High-Pc (>1e-4): {(data['target'] > 1e-4).float().mean():.4f}")

    print("\n[2] Training PcCalibrator (5 toy epochs)...")
    trainer = PcCalibratorTrainer(device="cpu", lr=1e-3)
    # Simple mini-batch
    batch_size = 512
    n = len(data["features"])
    for epoch in range(5):
        perm = torch.randperm(n)
        total_loss = 0.0
        num_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch = {"features": data["features"][idx], "target": data["target"][idx]}
            metrics = trainer.train_step(batch)
            total_loss += metrics["loss"]
            num_batches += 1
        print(f"   Epoch {epoch+1}: loss={total_loss/num_batches:.6f}")

    print("\n[3] Exporting ONNX...")
    onnx_path = trainer.export_onnx("ai/training/models/pc_calibrator_v1.onnx")
    size_kb = Path(onnx_path).stat().st_size / 1024
    print(f"   ONNX: {onnx_path} ({size_kb:.1f} KB)")

    print("\n[4] Validation: monotonicity + calibration...")
    model = trainer.model
    model.eval()
    with torch.no_grad():
        # Monotonicity grid
        grid = torch.linspace(-2, 6, 20).unsqueeze(1)
        base = torch.tensor([[-4.0, 7.0, 2.0, 2.0, 2.0, 100.0, 1.0, 1.0, 100.0, 0.5, 0.5]])
        base = base.repeat(20, 1)
        x_grid = torch.cat([grid, base], dim=1)
        pc_grid = model(x_grid)
        mono_violations = torch.sum(torch.diff(pc_grid) > 1e-4).item()
        print(f"   Monotonicity violations: {mono_violations}/19")

        # ECE on full data
        preds = model(data["features"])
        ece = trainer._ece_loss(preds, data["target"])
        print(f"   ECE: {ece:.4f}")
        print(f"   Mean Pc_pred: {preds.mean():.2e}, Mean Pc_true: {data['target'].mean():.2e}")

    print("\n" + "=" * 60)
    print(" Conjunction Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
