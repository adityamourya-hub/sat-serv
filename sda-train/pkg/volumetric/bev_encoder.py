#!/usr/bin/env python3
"""
BEV (Bird's Eye View) Encoder for Volumetric Scanning.

Lightweight CenterHead-style model:
- Input: 256 x 256 x 4 BEV grid (density, vx, vy, vz)
- Backbone: 3-layer CNN with residual connections
- Heads:
  - heatmap: objectness probability per cell
  - reg: 3D center offset + velocity residual (vx, vy, vz)
  - wh: object size proxy (2 channels: log_w, log_h)

Designed for <20ms CPU inference, ONNX exportable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class BEVBackbone(nn.Module):
    def __init__(self, in_ch: int = 4, base_ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock(base_ch, base_ch)
        self.layer2 = ResBlock(base_ch, base_ch * 2, stride=2)
        self.layer3 = ResBlock(base_ch * 2, base_ch * 4, stride=2)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        return x


class CenterHead(nn.Module):
    def __init__(self, in_ch: int = 32, num_classes: int = 3):
        super().__init__()
        self.heatmap = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, num_classes, 1),
        )
        self.reg = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 6, 1),  # dx, dy, dz, dvx, dvy, dvz
        )
        self.wh = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 2, 1),  # log_w, log_h
        )

    def forward(self, x):
        hm = torch.sigmoid(self.heatmap(x))
        reg = self.reg(x)
        wh = torch.exp(torch.clamp(self.wh(x), max=3.0))  # prevent explosion
        return hm, reg, wh


class BEVModel(nn.Module):
    def __init__(self, in_ch: int = 4, base_ch: int = 32, num_classes: int = 3):
        super().__init__()
        self.backbone = BEVBackbone(in_ch, base_ch)
        self.head = CenterHead(base_ch, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


class BEVTrainer:
    def __init__(self, model: BEVModel = None, device: str = "cpu", lr: float = 1e-3):
        self.device = torch.device(device)
        self.model = model or BEVModel().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def _focal_loss(self, pred_hm: torch.Tensor, gt_hm: torch.Tensor, alpha: float = 2.0, beta: float = 4.0):
        """Focal loss for heatmap classification."""
        pos_mask = gt_hm.eq(1).float()
        neg_mask = gt_hm.lt(1).float()
        neg_weight = torch.pow(1 - gt_hm, beta)
        pos_loss = torch.log(pred_hm + 1e-6) * torch.pow(1 - pred_hm, alpha) * pos_mask
        neg_loss = torch.log(1 - pred_hm + 1e-6) * torch.pow(pred_hm, alpha) * neg_weight * neg_mask
        num_pos = pos_mask.sum().clamp(min=1)
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos

    def _reg_loss(self, pred_reg: torch.Tensor, gt_reg: torch.Tensor, mask: torch.Tensor):
        """L1 loss for regression, masked by positive cells."""
        mask = mask.unsqueeze(1).float()
        loss = F.l1_loss(pred_reg * mask, gt_reg * mask, reduction="sum")
        return loss / (mask.sum() + 1e-6)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        batch keys:
          bev: [B, 4, H, W]
          heatmap: [B, C, H, W] one-hot per class at center cells
          reg_mask: [B, H, W] bool mask for cells with objects
          reg: [B, 6, H, W] center offset + velocity
          wh: [B, 2, H, W] size
        """
        bev = batch["bev"].to(self.device)
        gt_hm = batch["heatmap"].to(self.device)
        reg_mask = batch["reg_mask"].to(self.device)
        gt_reg = batch["reg"].to(self.device)
        gt_wh = batch["wh"].to(self.device)

        pred_hm, pred_reg, pred_wh = self.model(bev)

        loss_hm = self._focal_loss(pred_hm, gt_hm)
        loss_reg = self._reg_loss(pred_reg, gt_reg, reg_mask)
        loss_wh = self._reg_loss(pred_wh, gt_wh, reg_mask)

        loss = loss_hm + 0.1 * loss_reg + 0.1 * loss_wh

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "loss_hm": loss_hm.item(),
            "loss_reg": loss_reg.item(),
            "loss_wh": loss_wh.item(),
        }

    def export_onnx(self, path: str = "ai/training/models/bev_volumetric_v1.onnx"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(1, 4, 256, 256).to(self.device)
        torch.onnx.export(
            self.model,
            dummy,
            path,
            input_names=["bev"],
            output_names=["heatmap", "reg", "wh"],
            dynamic_axes={"bev": {0: "batch"}},
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
    # Sanity check
    model = BEVModel()
    x = torch.zeros(2, 4, 256, 256)
    hm, reg, wh = model(x)
    print(f"heatmap: {hm.shape}, reg: {reg.shape}, wh: {wh.shape}")
    assert hm.shape == (2, 3, 256, 256)
    assert reg.shape == (2, 6, 256, 256)
    assert wh.shape == (2, 2, 256, 256)
    print("BEV model OK")
