#!/usr/bin/env python3
"""
End-to-end Volumetric v1 Demo
1. Generate synthetic BEV shards via gen_volumetric.py
2. Train BEV encoder for 3 epochs (toy)
3. Export to ONNX
4. Simulate Plasma frame ingestion via ZMQ topic
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gen_volumetric import VolumetricScenarioGenerator
from pkg.volumetric.bev_encoder import BEVModel, BEVTrainer


def build_batch_from_shard(shard_path: Path, batch_size: int = 8):
    data = np.load(shard_path)
    bevs = data["bev"]
    labels = data["labels"]
    num_labels = data["num_labels"]

    indices = np.random.choice(len(bevs), min(batch_size, len(bevs)), replace=False)
    batch_bev = torch.from_numpy(bevs[indices]).permute(0, 3, 1, 2).float()  # [B, 4, H, W]

    # Build heatmap targets: 3 classes (sat, debris, unknown)
    B = len(indices)
    H, W = bevs.shape[1:3]
    heatmap = torch.zeros(B, 3, H, W)
    reg_mask = torch.zeros(B, H, W)
    reg = torch.zeros(B, 6, H, W)
    wh = torch.zeros(B, 2, H, W)

    half = 500.0 / 2.0  # volume_km / 2
    cell_km = 500.0 / 256.0

    for b_idx, idx in enumerate(indices):
        lbls = labels[idx, : num_labels[idx]]
        for obj in lbls:
            x, y, z, vx, vy, vz, cls = obj
            # Project y,z to BEV grid
            ix = int((y / 1000.0 + half) / cell_km)
            iy = int((z / 1000.0 + half) / cell_km)
            ix = np.clip(ix, 0, W - 1)
            iy = np.clip(iy, 0, H - 1)
            cls_idx = max(0, min(2, int(cls) - 1))
            heatmap[b_idx, cls_idx, iy, ix] = 1.0
            reg_mask[b_idx, iy, ix] = 1.0
            # Store normalized offset + velocity
            reg[b_idx, 0, iy, ix] = float((x - (6371000.0 + 400000.0)) / 1e6)
            reg[b_idx, 1, iy, ix] = float(y / 1e6)
            reg[b_idx, 2, iy, ix] = float(z / 1e6)
            reg[b_idx, 3, iy, ix] = float(vx / 1e3)
            reg[b_idx, 4, iy, ix] = float(vy / 1e3)
            reg[b_idx, 5, iy, ix] = float(vz / 1e3)
            wh[b_idx, 0, iy, ix] = 1.0
            wh[b_idx, 1, iy, ix] = 1.0

    return {
        "bev": batch_bev,
        "heatmap": heatmap,
        "reg_mask": reg_mask,
        "reg": reg,
        "wh": wh,
    }


def simulate_plasma_ingestion(zmq_sender=None):
    """Simulate a Plasma frame arriving and being converted to BEV."""
    gen = VolumetricScenarioGenerator(output_dir="data/synthetic/volumetric_demo")
    states = gen._generate_truth_states(num_objects=15, level=1)
    bev, labels = gen._states_to_bev(states)

    # Simulate ZMQ topic message
    msg = {
        "topic": "volumetric.scan",
        "frame_id": f"demo_{int(time.time()*1000)}",
        "bev_shape": list(bev.shape),
        "num_objects": len(labels),
        "timestamp": time.time(),
    }
    if zmq_sender:
        zmq_sender.send_string(f"volumetric.scan|{json.dumps(msg)}")
    else:
        print(f"[PLASMA] Simulated frame: {msg['frame_id']} objects={msg['num_objects']}")
    return bev, labels


def main():
    print("=" * 60)
    print(" Volumetric v1 End-to-End Demo")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n[1] Generating synthetic BEV shards...")
    gen = VolumetricScenarioGenerator(output_dir="data/synthetic/volumetric")
    shard_path = gen.generate_shard(shard_id=0, num_scenes=50, level=1)
    print(f"   Shard: {shard_path}")

    # 2. Build batch targets and train for 3 epochs
    print("\n[2] Training BEV encoder (3 toy epochs)...")
    trainer = BEVTrainer(device="cpu", lr=1e-3)
    for epoch in range(3):
        batch = build_batch_from_shard(Path(shard_path), batch_size=8)
        metrics = trainer.train_step(batch)
        print(f"   Epoch {epoch+1}: loss={metrics['loss']:.4f} "
              f"hm={metrics['loss_hm']:.4f} reg={metrics['loss_reg']:.4f} wh={metrics['loss_wh']:.4f}")

    # 3. ONNX export
    print("\n[3] Exporting to ONNX...")
    onnx_path = trainer.export_onnx("ai/training/models/bev_volumetric_v1.onnx")
    size_kb = Path(onnx_path).stat().st_size / 1024
    print(f"   ONNX: {onnx_path} ({size_kb:.1f} KB)")

    # 4. Simulate Plasma ingestion
    print("\n[4] Simulating Plasma frame ingestion...")
    bev, labels = simulate_plasma_ingestion()
    print(f"   BEV shape: {bev.shape}, Labels: {len(labels)}")

    # 5. Inference on simulated frame
    print("\n[5] Inference on simulated frame...")
    model = trainer.model
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).float()
        hm, reg, wh = model(x)
    print(f"   heatmap: {hm.shape}, reg: {reg.shape}, wh: {wh.shape}")

    # Extract top-5 detections from heatmap
    hm_cls = hm[0].max(dim=0)[0]
    flat = hm_cls.view(-1)
    top5_vals, top5_idx = torch.topk(flat, 5)
    H, W = hm_cls.shape
    print(f"   Top-5 detection scores: {top5_vals.numpy().round(3)}")

    print("\n" + "=" * 60)
    print(" Volumetric v1 Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
