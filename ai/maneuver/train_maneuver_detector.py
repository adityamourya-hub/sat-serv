#!/usr/bin/env python3
"""
Train an LSTM + Multi-Head Attention maneuver detector and export ONNX.
Outputs:
- ai/maneuver/data/maneuver_detector.onnx
- ai/maneuver/data/metrics.json
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class TrainConfig:
    seq_len: int = 32
    train_samples: int = 12000
    val_samples: int = 2000
    batch_size: int = 128
    epochs: int = 12
    lr: float = 1e-3
    hidden_size: int = 64
    lstm_layers: int = 2
    heads: int = 4
    dropout: float = 0.1


class SyntheticManeuverDataset(Dataset):
    """Synthetic residual + delta-v windows inspired by resources in docmd/ai.md.

    Feature 0: residual series
    Feature 1: observed delta-v estimate series

    Labels:
      y_prob: maneuver probability target (0/1)
      y_type: 0=impulsive, 1=continuous
      y_dv: maneuver delta-v target
      y_conf: confidence target (heuristic quality score)
    """

    def __init__(self, n: int, seq_len: int):
        self.n = n
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.n

    def _sample(self) -> Tuple[np.ndarray, float, int, float, float]:
        x = np.zeros((self.seq_len, 2), dtype=np.float32)

        base_noise = np.random.normal(0.0, 0.004, size=self.seq_len)
        residual = base_noise.copy()
        dv_series = np.abs(np.random.normal(0.002, 0.001, size=self.seq_len))

        has_maneuver = np.random.rand() < 0.55
        m_type = 0
        dv_target = float(np.clip(np.abs(np.random.normal(0.005, 0.002)), 0.0, 0.2))

        if has_maneuver:
            center = np.random.randint(self.seq_len // 3, self.seq_len - 3)
            is_impulsive = np.random.rand() < 0.5
            if is_impulsive:
                m_type = 0
                amp = np.clip(np.random.normal(0.08, 0.02), 0.03, 0.18)
                residual[center : center + 2] += amp
                dv_series[center] += amp
                dv_target = float(amp)
            else:
                m_type = 1
                amp = np.clip(np.random.normal(0.02, 0.008), 0.008, 0.08)
                tail = np.exp(-np.linspace(0, 2.2, self.seq_len - center)) * amp
                residual[center:] += tail
                dv_series[center:] += tail * 0.65
                dv_target = float(amp)

        x[:, 0] = residual.astype(np.float32)
        x[:, 1] = dv_series.astype(np.float32)

        quality = float(np.clip(1.0 - np.std(base_noise) * 60.0, 0.2, 1.0))
        y_prob = float(has_maneuver)
        y_type = int(m_type)
        y_conf = float(quality)
        return x, y_prob, y_type, dv_target, y_conf

    def __getitem__(self, idx: int):
        x, y_prob, y_type, y_dv, y_conf = self._sample()
        return (
            torch.from_numpy(x),
            torch.tensor([y_prob], dtype=torch.float32),
            torch.tensor(y_type, dtype=torch.long),
            torch.tensor([y_dv], dtype=torch.float32),
            torch.tensor([y_conf], dtype=torch.float32),
        )


class LSTMAttentionDetector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.prob_head = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.type_head = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 2))
        self.dv_head = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.conf_head = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x):
        # x: [B, T, F]
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        z = self.norm(lstm_out + attn_out)
        pooled = z[:, -1, :]

        prob = torch.sigmoid(self.prob_head(pooled))
        type_logits = self.type_head(pooled)
        dv = torch.relu(self.dv_head(pooled))
        conf = torch.sigmoid(self.conf_head(pooled))
        return prob, type_logits, dv, conf


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    losses = []
    with torch.no_grad():
        for x, y_prob, y_type, y_dv, y_conf in loader:
            x = x.to(device)
            y_prob = y_prob.to(device)
            y_type = y_type.to(device)
            y_dv = y_dv.to(device)
            y_conf = y_conf.to(device)

            p, t, d, c = model(x)
            loss = bce(p, y_prob) + ce(t, y_type) + 0.7 * mse(d, y_dv) + 0.2 * mse(c, y_conf)
            losses.append(loss.item())

    return float(np.mean(losses)) if losses else math.inf


def train(config: TrainConfig, out_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SyntheticManeuverDataset(config.train_samples, config.seq_len)
    val_ds = SyntheticManeuverDataset(config.val_samples, config.seq_len)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = LSTMAttentionDetector(
        input_size=2,
        hidden_size=config.hidden_size,
        layers=config.lstm_layers,
        heads=config.heads,
        dropout=config.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    best_val = math.inf
    best_state = None
    patience = 3
    stale = 0

    for epoch in range(config.epochs):
        model.train()
        running = []

        for x, y_prob, y_type, y_dv, y_conf in train_loader:
            x = x.to(device)
            y_prob = y_prob.to(device)
            y_type = y_type.to(device)
            y_dv = y_dv.to(device)
            y_conf = y_conf.to(device)

            p, t, d, c = model(x)
            loss = bce(p, y_prob) + ce(t, y_type) + 0.7 * mse(d, y_dv) + 0.2 * mse(c, y_conf)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running.append(loss.item())

        val_loss = evaluate(model, val_loader, device)
        train_loss = float(np.mean(running)) if running else math.inf
        print(f"epoch={epoch+1} train={train_loss:.5f} val={val_loss:.5f}")

        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print("early stopping")
                break

        # Periodic checkpoint every 10 epochs (survives VM reboot)
        if epoch % 10 == 0 and epoch > 0:
            checkpoint_path = out_dir / f"maneuver_checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_val': best_val,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "maneuver_detector.onnx"

    dummy = torch.zeros(1, config.seq_len, 2, dtype=torch.float32).to(device)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["maneuver_window"],
        output_names=["maneuver_probability", "maneuver_type_logits", "delta_v_estimate", "confidence_score"],
        dynamic_axes={
            "maneuver_window": {0: "batch", 1: "time"},
            "maneuver_probability": {0: "batch"},
            "maneuver_type_logits": {0: "batch"},
            "delta_v_estimate": {0: "batch"},
            "confidence_score": {0: "batch"},
        },
    )

    metrics = {
        "best_val_loss": best_val,
        "seq_len": config.seq_len,
        "model": "LSTM + MultiHeadAttention",
        "outputs": [
            "maneuver_probability",
            "maneuver_type_logits",
            "delta_v_estimate",
            "confidence_score",
        ],
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"onnx_export={onnx_path}")


if __name__ == "__main__":
    cfg = TrainConfig()
    root = Path(__file__).resolve().parent / "data"
    train(cfg, root)
