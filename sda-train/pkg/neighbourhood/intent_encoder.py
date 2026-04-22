#!/usr/bin/env python3
"""
Neighbourhood Watch Intent Encoder.

1D CNN over 120-second encounter windows (10 features @ 1 Hz).
Dual heads:
  - intent: binary classification (benign vs intercept)
  - tti: time-to-intercept regression (-1 if benign)

ONNX-exportable, <5ms CPU inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict


class IntentEncoder(nn.Module):
    """1D CNN for encounter window classification."""

    def __init__(self, in_features: int = 10, window: int = 120,
                 channels: list = [32, 64, 128], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = in_features
        for ch in channels:
            layers += [
                nn.Conv1d(prev, ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ]
            prev = ch
        self.backbone = nn.Sequential(*layers)

        # Global average pooling + classification
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.intent_head = nn.Sequential(
            nn.Linear(prev, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.tti_head = nn.Sequential(
            nn.Linear(prev, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        feat = self.backbone(x)
        pooled = self.gap(feat).squeeze(-1)  # [B, C]
        intent_logits = self.intent_head(pooled).squeeze(-1)
        tti = self.tti_head(pooled).squeeze(-1)
        return intent_logits, tti


class IntentEncoderTrainer:
    def __init__(self, model: IntentEncoder = None, device: str = "cpu", lr: float = 1e-3):
        self.device = torch.device(device)
        self.model = model or IntentEncoder().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.intent_criterion = nn.BCEWithLogitsLoss()
        self.tti_criterion = nn.SmoothL1Loss(beta=5.0)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        x = batch["features"].to(self.device)          # [B, T, F]
        intent_target = batch["intent"].to(self.device).float()  # [B]
        tti_target = batch["tti"].to(self.device).float()          # [B]
        mask = (intent_target == 1.0).float()                     # only regress TTI for intercepts

        intent_logits, tti_pred = self.model(x)

        loss_intent = self.intent_criterion(intent_logits, intent_target)
        # TTI loss: ignore benign samples (tti = -1)
        tti_masked = tti_pred * mask
        tti_target_masked = tti_target * mask
        loss_tti = self.tti_criterion(tti_masked, tti_target_masked)
        # avoid division by zero if all benign
        if mask.sum() > 0:
            loss_tti = loss_tti / (mask.sum() / mask.numel())

        loss = loss_intent + 0.1 * loss_tti

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            intent_prob = torch.sigmoid(intent_logits)
            pred_label = (intent_prob > 0.5).float()
            acc = (pred_label == intent_target).float().mean()

        return {
            "loss": loss.item(),
            "loss_intent": loss_intent.item(),
            "loss_tti": loss_tti.item(),
            "acc": acc.item(),
        }

    def export_onnx(self, path: str = "ai/training/models/intent_encoder_v1.onnx"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(1, 120, 10).to(self.device)
        torch.onnx.export(
            self.model,
            dummy,
            path,
            input_names=["window"],
            output_names=["intent_logits", "tti"],
            dynamic_axes={"window": {0: "batch"}},
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
    model = IntentEncoder()
    x = torch.zeros(4, 120, 10)
    intent_logits, tti = model(x)
    assert intent_logits.shape == (4,)
    assert tti.shape == (4,)
    print("IntentEncoder model OK")
