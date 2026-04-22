#!/usr/bin/env python3
"""
End-to-end Neighbourhood Watch Intent Demo
1. Generate synthetic encounter windows via gen_neighbourhood.py
2. Train IntentEncoder for 3 epochs (toy)
3. Export to ONNX
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gen_neighbourhood import generate_encounter
from pkg.neighbourhood.intent_encoder import IntentEncoder, IntentEncoderTrainer


def generate_shard(n: int = 500, window: int = 120, features: int = 10):
    data = []
    for _ in range(n):
        feat, intent, tti, miss = generate_encounter()
        data.append({
            "features": feat.astype(np.float32),
            "intent": int(intent),
            "tti": float(tti),
            "miss_dist": float(miss),
        })
    return data


def build_batch(records: list, batch_size: int = 8):
    idx = np.random.choice(len(records), batch_size, replace=False)
    feats = np.stack([records[i]["features"] for i in idx])
    intents = np.array([records[i]["intent"] for i in idx], dtype=np.float32)
    ttis = np.array([records[i]["tti"] for i in idx], dtype=np.float32)
    return {
        "features": torch.from_numpy(feats),
        "intent": torch.from_numpy(intents),
        "tti": torch.from_numpy(ttis),
    }


def main():
    print("=" * 60)
    print(" Neighbourhood Watch Intent Demo")
    print("=" * 60)

    print("\n[1] Generating synthetic encounter windows...")
    records = generate_shard(n=500)
    intents = np.array([r["intent"] for r in records])
    print(f"   Samples: {len(records)}, Intercept ratio: {intents.mean():.3f}")

    print("\n[2] Training IntentEncoder (3 toy epochs)...")
    trainer = IntentEncoderTrainer(device="cpu", lr=1e-3)
    for epoch in range(3):
        batch = build_batch(records, batch_size=16)
        metrics = trainer.train_step(batch)
        print(f"   Epoch {epoch+1}: loss={metrics['loss']:.4f} "
              f"intent={metrics['loss_intent']:.4f} tti={metrics['loss_tti']:.4f} acc={metrics['acc']:.3f}")

    print("\n[3] Exporting to ONNX...")
    onnx_path = trainer.export_onnx("ai/training/models/intent_encoder_v1.onnx")
    size_kb = Path(onnx_path).stat().st_size / 1024
    print(f"   ONNX: {onnx_path} ({size_kb:.1f} KB)")

    print("\n[4] Inference on single window...")
    model = trainer.model
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(records[0]["features"]).unsqueeze(0).float()
        intent_logits, tti = model(x)
        intent_prob = torch.sigmoid(intent_logits).item()
    print(f"   Ground truth: intent={records[0]['intent']}, tti={records[0]['tti']:.1f}s")
    print(f"   Predicted: intent_prob={intent_prob:.3f}, tti={tti.item():.1f}s")

    print("\n" + "=" * 60)
    print(" Neighbourhood Intent Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
