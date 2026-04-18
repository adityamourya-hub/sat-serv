#!/usr/bin/env python3
"""
Train a small conditional GAN for scenario vector generation and export generator ONNX.
Outputs:
- ai/simulator/data/small_gan_generator.onnx
- ai/simulator/data/gan_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def load_data(path: Path):
    if not path.exists():
        x = np.random.normal(0, 1, (10000, 12)).astype(np.float32)
        c = np.random.uniform(0, 1, (10000, 4)).astype(np.float32)
        return x, c

    rows, cond = [], []
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
    return np.asarray(rows, dtype=np.float32), np.asarray(cond, dtype=np.float32)


class Generator(nn.Module):
    def __init__(self, z_dim=16, c_dim=4, out_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + c_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=-1))


class Discriminator(nn.Module):
    def __init__(self, x_dim=12, c_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + c_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=-1))


def main() -> None:
    base = Path(__file__).resolve().parent
    x_np, c_np = load_data(base / "data" / "scenarios_10000.jsonl")
    x = torch.tensor(x_np)
    c = torch.tensor(c_np)

    dl = DataLoader(TensorDataset(x, c), batch_size=256, shuffle=True)

    gen = Generator()
    disc = Discriminator()
    g_opt = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    z_dim = 16
    g_last, d_last = 0.0, 0.0
    for epoch in range(8):
        for xb, cb in dl:
            b = xb.shape[0]
            real_y = torch.ones(b, 1)
            fake_y = torch.zeros(b, 1)

            z = torch.randn(b, z_dim)
            fake_x = gen(z, cb)

            d_real = disc(xb, cb)
            d_fake = disc(fake_x.detach(), cb)
            d_loss = bce(d_real, real_y) + bce(d_fake, fake_y)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            z2 = torch.randn(b, z_dim)
            g_fake = gen(z2, cb)
            g_logits = disc(g_fake, cb)
            g_loss = bce(g_logits, real_y)
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            g_last, d_last = g_loss.item(), d_loss.item()

        print(f"epoch={epoch+1} g_loss={g_last:.5f} d_loss={d_last:.5f}")

    out_dir = base / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "small_gan_generator.onnx"

    gen.eval()
    dummy_z = torch.zeros(1, z_dim)
    dummy_c = torch.zeros(1, 4)
    torch.onnx.export(
        gen,
        (dummy_z, dummy_c),
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["noise", "condition"],
        output_names=["scenario_vector"],
        dynamic_axes={"noise": {0: "batch"}, "condition": {0: "batch"}, "scenario_vector": {0: "batch"}},
    )

    with (out_dir / "gan_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"generator_loss": g_last, "discriminator_loss": d_last, "onnx": str(onnx_path)}, f, indent=2)

    print(f"onnx_export={onnx_path}")


if __name__ == "__main__":
    main()
