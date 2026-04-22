#!/usr/bin/env python3
"""
Baseline Pc Calibrator for Conjunction Analysis.

Tiny MLP (<=10k params, <5ms CPU) that calibrates analytic Pc
using empirical Monte-Carlo labels.

Loss: Brier score + ECE regularizer + monotonicity penalty.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PcCalibrator(nn.Module):
    """Tiny MLP: 12 features -> calibrated Pc."""

    def __init__(self, in_dim: int = 12, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PcCalibratorTrainer:
    def __init__(self, model: PcCalibrator = None, device: str = "cpu", lr: float = 1e-3):
        self.device = torch.device(device)
        self.model = model or PcCalibrator().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    @staticmethod
    def _brier_score(pred, target):
        return torch.mean((pred - target) ** 2)

    @staticmethod
    def _ece_loss(pred, target, n_bins: int = 10):
        """Expected Calibration Error."""
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=pred.device)
        ece = 0.0
        total = 0
        for i in range(n_bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (pred >= low) & (pred < high)
            if i == n_bins - 1:
                mask = (pred >= low) & (pred <= high)
            if mask.sum() == 0:
                continue
            bin_acc = target[mask].float().mean()
            bin_conf = pred[mask].mean()
            ece += mask.sum().item() * torch.abs(bin_acc - bin_conf)
            total += mask.sum().item()
        return ece / max(total, 1)

    @staticmethod
    def _monotonicity_loss(model, device, n_grid: int = 20):
        """
        Finite-difference penalty: dPc/d(log_miss_dist) <= 0.
        Construct grid over log_miss_dist, fix other features at median.
        """
        grid = torch.linspace(-2, 6, n_grid, device=device).unsqueeze(1)
        # dummy features: [log_miss, log_analytic_pc= -4, rel_speed=7, sizes=2,2, aspect=2,
        #                 pos_unc=100, vel_unc=1, solar=1, ttc=100, sin_inc=0.5, cos_inc=0.5]
        base = torch.tensor([[-4.0, 7.0, 2.0, 2.0, 2.0, 100.0, 1.0, 1.0, 100.0, 0.5, 0.5]], device=device)
        base = base.repeat(n_grid, 1)
        x_grid = torch.cat([grid, base], dim=1)
        with torch.no_grad():
            pc_grid = model(x_grid)
        # Penalize increases as miss distance increases
        diff = torch.diff(pc_grid)
        penalty = torch.relu(diff).sum()
        return penalty

    def train_step(self, batch: dict) -> dict:
        features = batch["features"].to(self.device)
        target = batch["target"].to(self.device)

        pred = self.model(features)

        loss_brier = self._brier_score(pred, target)
        loss_ece = self._ece_loss(pred, target)
        loss_mono = self._monotonicity_loss(self.model, self.device)

        loss = loss_brier + 0.1 * loss_ece + 0.01 * loss_mono

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "brier": loss_brier.item(),
            "ece": loss_ece.item(),
            "mono": loss_mono.item(),
        }

    def export_onnx(self, path: str = "ai/training/models/pc_calibrator_v1.onnx"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(1, 12).to(self.device)
        torch.onnx.export(
            self.model,
            dummy,
            path,
            input_names=["features"],
            output_names=["pc_calibrated"],
            dynamic_axes={"features": {0: "batch"}},
            opset_version=11,
        )
        print(f"ONNX exported to {path}")
        return path

    def save_checkpoint(self, path: str):
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


def load_jsonl_shard(shard_path: Path):
    records = []
    with open(shard_path) as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
    features = torch.tensor([r["features"] for r in records], dtype=torch.float32)
    targets = torch.tensor([r["empirical_pc"] for r in records], dtype=torch.float32)
    return {"features": features, "target": targets}


if __name__ == "__main__":
    # Sanity check
    model = PcCalibrator()
    x = torch.randn(4, 12)
    out = model(x)
    assert out.shape == (4,)
    assert torch.all((out >= 0) & (out <= 1))
    print("PcCalibrator model OK")
