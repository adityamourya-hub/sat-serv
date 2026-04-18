#!/usr/bin/env python3
"""
Train a compact Transformer association model for noisy/fused TLE + radar track association.
Exports ONNX model for Go integration.

Outputs:
- ai/tracker/data/association_transformer.onnx
- ai/tracker/data/association_metrics.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class AssocDataset(Dataset):
    def __init__(self, n: int, n_candidates: int = 8):
        self.n = n
        self.n_candidates = n_candidates

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # observation feature vector from fused sensors
        obs = np.zeros((6,), dtype=np.float32)
        x_true, y_true, z_true = np.random.uniform(-200, 200, size=3)
        radar_sigma = np.random.uniform(0.3, 1.2)
        tle_sigma = np.random.uniform(1.0, 3.5)

        radar = np.array([
            x_true + np.random.normal(0, radar_sigma),
            y_true + np.random.normal(0, radar_sigma),
            z_true + np.random.normal(0, radar_sigma),
        ])
        tle = np.array([
            x_true + np.random.normal(0, tle_sigma),
            y_true + np.random.normal(0, tle_sigma),
            z_true + np.random.normal(0, tle_sigma),
        ])
        obs[:3] = (radar * (1.0 / radar_sigma**2) + tle * (1.0 / tle_sigma**2)) / ((1.0 / radar_sigma**2) + (1.0 / tle_sigma**2))
        obs[3] = radar_sigma
        obs[4] = tle_sigma
        obs[5] = np.random.uniform(0, 1)  # track quality proxy

        candidates = np.zeros((self.n_candidates, 5), dtype=np.float32)
        target = np.zeros((self.n_candidates,), dtype=np.float32)

        true_idx = np.random.randint(0, self.n_candidates)
        for i in range(self.n_candidates):
            if i == true_idx:
                pos = np.array([x_true, y_true, z_true]) + np.random.normal(0, 0.8, size=3)
                target[i] = 1.0
            else:
                pos = np.random.uniform(-220, 220, size=3)
                if np.linalg.norm(pos - np.array([x_true, y_true, z_true])) < 5:
                    pos += np.array([20, -15, 10])
            candidates[i, :3] = pos
            candidates[i, 3] = np.random.uniform(0.1, 0.9)  # recency/quality
            d = np.linalg.norm(pos - obs[:3])
            candidates[i, 4] = d

        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(candidates, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


class AssocTransformer(nn.Module):
    def __init__(self, obs_dim=6, cand_dim=5, d_model=64, heads=4, layers=2):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.cand_proj = nn.Linear(cand_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.score_head = nn.Linear(d_model, 1)
        self.conf_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, obs, candidates):
        # obs: [B, Fo], candidates: [B, N, Fc]
        obs_tok = self.obs_proj(obs).unsqueeze(1)
        cand_tok = self.cand_proj(candidates)
        seq = torch.cat([obs_tok, cand_tok], dim=1)
        enc = self.encoder(seq)

        cand_enc = enc[:, 1:, :]
        scores = self.score_head(cand_enc).squeeze(-1)
        attn = torch.softmax(scores, dim=-1)
        pooled = (cand_enc * attn.unsqueeze(-1)).sum(dim=1)
        conf = self.conf_head(pooled)
        return scores, conf, attn


def train():
    ds = AssocDataset(14000, n_candidates=8)
    dl = DataLoader(ds, batch_size=128, shuffle=True)

    model = AssocTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    best = math.inf
    for epoch in range(10):
        losses = []
        for obs, cands, y in dl:
            scores, conf, _ = model(obs, cands)
            loss_assoc = bce(scores, y)
            y_conf = (y.max(dim=1).values).unsqueeze(1)
            loss_conf = nn.functional.mse_loss(conf, y_conf)
            loss = loss_assoc + 0.2 * loss_conf

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val = float(np.mean(losses))
        best = min(best, val)
        print(f"epoch={epoch+1} loss={val:.6f}")

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "association_transformer.onnx"

    model.eval()
    dummy_obs = torch.zeros(1, 6, dtype=torch.float32)
    dummy_cand = torch.zeros(1, 8, 5, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy_obs, dummy_cand),
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["observation", "candidates"],
        output_names=["association_scores", "model_confidence", "attention"],
        dynamic_axes={
            "observation": {0: "batch"},
            "candidates": {0: "batch", 1: "num_candidates"},
            "association_scores": {0: "batch", 1: "num_candidates"},
            "model_confidence": {0: "batch"},
            "attention": {0: "batch", 1: "num_candidates"},
        },
    )

    metrics = {
        "model": "transformer_association",
        "best_loss": best,
        "onnx": str(onnx_path),
    }
    with (out_dir / "association_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"onnx_export={onnx_path}")


if __name__ == "__main__":
    train()
