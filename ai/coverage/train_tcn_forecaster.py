#!/usr/bin/env python3
"""
Train a compact Temporal Convolutional Network forecaster for blind-spot ratio prediction.
Forecast horizon targets 24-72h and exports ONNX.

Outputs:
- ai/coverage/data/blindspot_tcn.onnx
- ai/coverage/data/tcn_metrics.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class BlindSpotDataset(Dataset):
    def __init__(self, n=18000, seq_len=48, horizon=24, seed=42):
        self.n = n
        self.seq_len = seq_len
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        t = np.arange(self.seq_len + self.horizon)
        base = 0.35 + 0.04 * np.sin(2 * np.pi * (t % 24) / 24.0)
        trend = self.rng.normal(0.0, 0.0008) * t
        orbital = self.rng.uniform(0.0, 1.0, size=t.shape)
        sensor = self.rng.uniform(0.0, 1.0, size=t.shape)
        noise = self.rng.normal(0.0, 0.012, size=t.shape)

        ratio = np.clip(base + trend + 0.04 * orbital - 0.03 * sensor + noise, 0.02, 0.98)

        hist = np.stack([
            ratio[: self.seq_len],
            orbital[: self.seq_len],
            sensor[: self.seq_len],
        ], axis=1).astype(np.float32)
        target = ratio[self.seq_len : self.seq_len + self.horizon].astype(np.float32)
        return torch.tensor(hist), torch.tensor(target)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        y = y[..., : x.shape[-1]]
        res = x if self.down is None else self.down(x)
        return self.relu(y + res)


class TCNForecaster(nn.Module):
    def __init__(self, in_feat=3, channels=32, horizon=24):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(in_feat, channels, dilation=1),
            TCNBlock(channels, channels, dilation=2),
            TCNBlock(channels, channels, dilation=4),
        )
        self.head = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        z = self.net(x)
        pooled = z[:, :, -1]
        return self.head(pooled)


def main():
    seq_len = 48
    horizon = 24

    ds = BlindSpotDataset(n=18000, seq_len=seq_len, horizon=horizon)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    model = TCNForecaster(in_feat=3, channels=32, horizon=horizon)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    best = math.inf
    for epoch in range(10):
        losses = []
        for x, y in dl:
            pred = model(x)
            loss = mse(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val = float(np.mean(losses))
        best = min(best, val)
        print(f"epoch={epoch+1} loss={val:.6f}")

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "blindspot_tcn.onnx"

    model.eval()
    dummy = torch.zeros(1, seq_len, 3, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["history_series"],
        output_names=["forecast_ratio"],
        dynamic_axes={"history_series": {0: "batch", 1: "time"}, "forecast_ratio": {0: "batch"}},
    )

    with (out_dir / "tcn_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"best_loss": best, "horizon": horizon, "onnx": str(onnx_path)}, f, indent=2)

    print(f"onnx_export={onnx_path}")


if __name__ == "__main__":
    main()
