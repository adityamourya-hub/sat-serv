#!/usr/bin/env python3
"""
Re-entry Decay Forecaster.

Given a trajectory sequence (position, velocity, altitude, density),
predicts:
  - time_to_demise (hours)
  - impact_lat, impact_lon (degrees)

Architecture: 1D CNN + LSTM over trajectory windows.
Input: [B, T, 10] = [x, y, z, vx, vy, vz, alt_km, rho, cd_eff, a_over_m]
Output: [B, 3] = [ttd_hr, impact_lat, impact_lon]

ONNX-exportable, <10ms CPU inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict


class ReentryForecaster(nn.Module):
    def __init__(self, in_features: int = 6, window: int = 120,
                 hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(64, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),  # ttd, lat, lon
        )

    def forward(self, x):
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        c = self.cnn(x)  # [B, 64, T/2]
        c = c.transpose(1, 2)  # [B, T/2, 64]
        out, _ = self.lstm(c)
        pooled = out[:, -1, :]  # last timestep
        return self.head(pooled)


class ReentryForecasterTrainer:
    def __init__(self, model: ReentryForecaster = None, device: str = "cpu", lr: float = 1e-3):
        self.device = torch.device(device)
        self.model = model or ReentryForecaster().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        x = batch["features"].to(self.device)      # [B, T, 10]
        target = batch["target"].to(self.device)     # [B, 3]
        pred = self.model(x)
        loss = self.mse(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def export_onnx(self, path: str = "ai/training/models/reentry_forecaster_v1.onnx"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(1, 120, 6).to(self.device)
        torch.onnx.export(
            self.model,
            dummy,
            path,
            input_names=["trajectory"],
            output_names=["forecast"],
            dynamic_axes={"trajectory": {0: "batch"}},
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


if __name__ == "__main__":
    model = ReentryForecaster()
    x = torch.zeros(4, 120, 6)
    out = model(x)
    assert out.shape == (4, 3)
    print("ReentryForecaster model OK")
