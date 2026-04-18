#!/usr/bin/env python3
"""
Train a compact conditional diffusion-style denoiser for tabular SSA scenarios and export ONNX.
This is a practical approximation suitable for CPU deployment pipelines.

Outputs:
- ai/simulator/data/diffusion_denoiser.onnx
- ai/simulator/data/diffusion_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def load_or_make_dataset(path: Path, n: int = 10000):
    if not path.exists():
        x = np.random.normal(0, 1, size=(n, 12)).astype(np.float32)
        c = np.random.uniform(0, 1, size=(n, 4)).astype(np.float32)
        return x, c

    rows = []
    cond = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append([
                rec["miss_distance_km"],
                rec["relative_velocity_km_s"],
                rec["time_to_tca_seconds"] / 7200.0,
                rec["sigma_radial_km"],
                rec["sigma_intrack_km"],
                rec["sigma_crosstrack_km"],
                rec["distance_to_asset_km"] / 3200.0,
                rec["mach"] / 13.0,
                rec["altitude_km"] / 350.0,
                rec["burnout_velocity_km_s"] / 7.0,
                rec["radar_cross_section"] / 6.0,
                rec["risk_label"],
            ])
            cond.append([
                1.0 if rec["type"] == "conjunction" else 0.0,
                1.0 if rec["threat_class"] == "hypersonic" else 0.0,
                rec["priority_label"] / 300.0,
                rec["operator_difficulty"],
            ])

    x = np.asarray(rows, dtype=np.float32)
    c = np.asarray(cond, dtype=np.float32)
    return x, c


class Denoiser(nn.Module):
    def __init__(self, x_dim: int, c_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + c_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, x_dim),
        )

    def forward(self, noisy_x, cond, t):
        return self.net(torch.cat([noisy_x, cond, t], dim=-1))


def main() -> None:
    base = Path(__file__).resolve().parent
    data_path = base / "data" / "scenarios_10000.jsonl"
    x_np, c_np = load_or_make_dataset(data_path)

    x = torch.tensor(x_np)
    c = torch.tensor(c_np)
    ds = TensorDataset(x, c)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    model = Denoiser(x_dim=x.shape[1], c_dim=c.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    best = 1e9
    for epoch in range(8):
        losses = []
        for xb, cb in dl:
            t = torch.rand(xb.shape[0], 1)
            noise = torch.randn_like(xb)
            noisy = (1 - t) * xb + t * noise
            pred = model(noisy, cb, t)
            loss = mse(pred, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val = float(np.mean(losses))
        best = min(best, val)
        print(f"epoch={epoch+1} loss={val:.6f}")

    out_dir = base / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "diffusion_denoiser.onnx"

    model.eval()
    dummy_x = torch.zeros(1, x.shape[1], dtype=torch.float32)
    dummy_c = torch.zeros(1, c.shape[1], dtype=torch.float32)
    dummy_t = torch.zeros(1, 1, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_x, dummy_c, dummy_t),
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["noisy_x", "condition", "time_step"],
        output_names=["denoised_x"],
        dynamic_axes={
            "noisy_x": {0: "batch"},
            "condition": {0: "batch"},
            "time_step": {0: "batch"},
            "denoised_x": {0: "batch"},
        },
    )

    metrics = {
        "model": "conditional_diffusion_denoiser",
        "best_loss": best,
        "feature_dim": int(x.shape[1]),
        "condition_dim": int(c.shape[1]),
        "onnx": str(onnx_path),
    }
    with (out_dir / "diffusion_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"onnx_export={onnx_path}")


if __name__ == "__main__":
    main()
