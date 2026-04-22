#!/usr/bin/env python3
"""
End-to-end Catalog Characterization Demo
1. Generate synthetic chips + kinematics via gen_catalog.py
2. Contrastive pre-training (2 epochs)
3. Supervised fine-tuning (2 epochs)
4. Export ONNX + build FAISS index
5. k-NN query against catalog
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gen_catalog import CatalogScenarioGenerator
from pkg.catalog.dual_encoder import CatalogDualEncoder, CatalogDualEncoderTrainer


def build_batch(data: dict, batch_size: int = 16):
    n = min(batch_size, len(data["chips"]))
    idx = np.random.choice(len(data["chips"]), n, replace=False)
    chips = torch.from_numpy(data["chips"][idx]).unsqueeze(1).float()  # [B, 1, 64, 64]
    kin = torch.from_numpy(data["kinematics"][idx]).float()
    labels = torch.from_numpy(data["labels"][idx]).long()
    return {
        "chips": chips,
        "kinematics": kin,
        "bus_type": labels[:, 0],
        "mission": labels[:, 1],
        "size_class": labels[:, 2],
        "stability": labels[:, 3].float(),
    }


def main():
    print("=" * 60)
    print(" Catalog Characterization Demo")
    print("=" * 60)

    print("\n[1] Generating synthetic catalog shards...")
    gen = CatalogScenarioGenerator(output_dir="data/synthetic/catalog")
    shard_path = gen.generate_shard(num_objects=200)
    data = np.load(shard_path)
    print(f"   Chips: {data['chips'].shape}, Kinematics: {data['kinematics'].shape}, Labels: {data['labels'].shape}")

    print("\n[2] Contrastive pre-training (2 toy epochs)...")
    trainer = CatalogDualEncoderTrainer(device="cpu", lr=1e-3)
    for epoch in range(2):
        batch = build_batch(data, batch_size=16)
        print(f"   DEBUG: chips shape in batch = {batch['chips'].shape}")
        metrics = trainer.train_step(batch, stage="contrastive")
        print(f"   Epoch {epoch+1}: contrastive_loss={metrics['loss']:.4f}")

    print("\n[3] Supervised fine-tuning (2 toy epochs)...")
    for epoch in range(2):
        batch = build_batch(data, batch_size=16)
        metrics = trainer.train_step(batch, stage="supervised")
        print(f"   Epoch {epoch+1}: loss={metrics['loss']:.4f} "
              f"bus={metrics['loss_bus']:.3f} mission={metrics['loss_mission']:.3f} "
              f"size={metrics['loss_size']:.3f} stability={metrics['loss_stability']:.3f} "
              f"bus_acc={metrics['bus_acc']:.3f}")

    print("\n[4] Exporting ONNX...")
    onnx_path = trainer.export_onnx("ai/training/models/catalog_dual_encoder_v1.onnx")
    size_kb = Path(onnx_path).stat().st_size / 1024
    print(f"   ONNX: {onnx_path} ({size_kb:.1f} KB)")

    print("\n[5] Building embeddings + FAISS index...")
    trainer.model.eval()
    with torch.no_grad():
        all_chips = torch.from_numpy(data["chips"]).unsqueeze(1).float()
        all_kin = torch.from_numpy(data["kinematics"]).float()
        embeddings = []
        for i in range(0, len(all_chips), 32):
            emb = trainer.model(all_chips[i:i+32], all_kin[i:i+32])["embedding"]
            embeddings.append(emb.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
    print(f"   Embeddings shape: {embeddings.shape}")

    index = trainer.build_faiss_index(embeddings, "ai/training/models/catalog_index.faiss")

    print("\n[6] k-NN query (k=5)...")
    query_idx = 0
    query_emb = embeddings[query_idx:query_idx+1]
    if index is not None:
        import faiss
        D, I = index.search(query_emb, 5)
        print(f"   Query obj {query_idx}: top-5 matches = {I[0]}")
        print(f"   Inner products: {D[0].round(3)}")
    else:
        # Brute-force fallback
        sim = query_emb @ embeddings.T
        top5 = np.argsort(-sim[0])[:5]
        print(f"   Query obj {query_idx}: top-5 matches = {top5}")
        print(f"   Inner products: {sim[0, top5].round(3)}")

    print("\n" + "=" * 60)
    print(" Catalog Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
