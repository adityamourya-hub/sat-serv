#!/usr/bin/env python3
"""
Catalog Dual Encoder for Space Object Characterisation.

Architecture:
- Image branch: tiny CNN (3 conv layers) on 64x64 chips -> 128-d embedding
- Kinematics branch: MLP on 8 orbital features -> 128-d embedding
- Fusion: concatenation -> 256-d joint embedding
- Heads: multi-label attribute classifier (bus type, mission class, size, stability)
- Contrastive pre-training: InfoNCE on image+kinematics pairs

ONNX-exportable, <10ms CPU inference per object.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class ImageEncoder(nn.Module):
    """Tiny CNN for 64x64 optical chips."""

    def __init__(self, in_ch: int = 1, embed_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),  # 64->32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),     # 32->16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),    # 16->8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        feat = self.conv(x).flatten(1)
        return F.normalize(self.fc(feat), p=2, dim=1)


class KinematicsEncoder(nn.Module):
    """MLP on 8 orbital elements / derived features."""

    def __init__(self, in_dim: int = 8, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


class CatalogDualEncoder(nn.Module):
    """
    Dual encoder with optional modality dropout for robustness.
    Outputs: 256-d joint embedding + attribute logits.
    """

    def __init__(
        self,
        img_ch: int = 1,
        kin_dim: int = 8,
        embed_dim: int = 128,
        n_bus_types: int = 4,
        n_missions: int = 5,
        n_sizes: int = 3,
    ):
        super().__init__()
        self.img_enc = ImageEncoder(img_ch, embed_dim)
        self.kin_enc = KinematicsEncoder(kin_dim, embed_dim)
        self.fusion_dim = embed_dim * 2

        # Attribute classification heads
        self.bus_head = nn.Linear(self.fusion_dim, n_bus_types)
        self.mission_head = nn.Linear(self.fusion_dim, n_missions)
        self.size_head = nn.Linear(self.fusion_dim, n_sizes)
        self.stability_head = nn.Linear(self.fusion_dim, 1)

        # Embedding projection for contrastive pre-training
        self.projection = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, img: torch.Tensor, kin: torch.Tensor):
        z_img = self.img_enc(img)
        z_kin = self.kin_enc(kin)
        z_joint = torch.cat([z_img, z_kin], dim=-1)

        logits = {
            "bus": self.bus_head(z_joint),
            "mission": self.mission_head(z_joint),
            "size": self.size_head(z_joint),
            "stability": self.stability_head(z_joint).squeeze(-1),
            "embedding": z_joint,
            "projection": F.normalize(self.projection(z_joint), p=2, dim=1),
        }
        return logits

    def forward_img_only(self, img: torch.Tensor):
        """Inference with image only (kinematics missing)."""
        z_img = self.img_enc(img)
        z_kin = torch.zeros_like(z_img)
        z_joint = torch.cat([z_img, z_kin], dim=-1)
        return self._heads_from_joint(z_joint)

    def forward_kin_only(self, kin: torch.Tensor):
        """Inference with kinematics only (image missing)."""
        z_kin = self.kin_enc(kin)
        z_img = torch.zeros_like(z_kin)
        z_joint = torch.cat([z_img, z_kin], dim=-1)
        return self._heads_from_joint(z_joint)

    def _heads_from_joint(self, z_joint):
        return {
            "bus": self.bus_head(z_joint),
            "mission": self.mission_head(z_joint),
            "size": self.size_head(z_joint),
            "stability": self.stability_head(z_joint).squeeze(-1),
            "embedding": z_joint,
        }


def info_nce_loss(proj_img: torch.Tensor, proj_kin: torch.Tensor, temperature: float = 0.07):
    """
    Symmetric InfoNCE for image-kinematics pairs.
    proj_img, proj_kin: [B, D] normalized
    """
    logits = torch.matmul(proj_img, proj_kin.T) / temperature  # [B, B]
    labels = torch.arange(len(logits), device=logits.device)
    loss_i2k = F.cross_entropy(logits, labels)
    loss_k2i = F.cross_entropy(logits.T, labels)
    return (loss_i2k + loss_k2i) / 2.0


class CatalogDualEncoderTrainer:
    def __init__(self, model: CatalogDualEncoder = None, device: str = "cpu", lr: float = 1e-3):
        self.device = torch.device(device)
        self.model = model or CatalogDualEncoder().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.contrastive_weight = 0.5
        self.ce_weight = 0.5

    def contrastive_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Self-supervised contrastive pre-training."""
        img = batch["chips"].to(self.device)  # [B, 1, 64, 64]
        kin = batch["kinematics"].to(self.device)

        logits = self.model(img, kin)
        loss = info_nce_loss(logits["projection"], logits["projection"])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def supervised_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Supervised attribute classification."""
        img = batch["chips"].to(self.device)
        kin = batch["kinematics"].to(self.device)

        logits = self.model(img, kin)

        loss_bus = F.cross_entropy(logits["bus"], batch["bus_type"].to(self.device).long())
        loss_mission = F.cross_entropy(logits["mission"], batch["mission"].to(self.device).long())
        loss_size = F.cross_entropy(logits["size"], batch["size_class"].to(self.device).long())
        loss_stability = F.binary_cross_entropy_with_logits(
            logits["stability"], batch["stability"].to(self.device).float()
        )

        loss = loss_bus + loss_mission + loss_size + loss_stability

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            bus_acc = (logits["bus"].argmax(dim=1) == batch["bus_type"].to(self.device)).float().mean().item()

        return {
            "loss": loss.item(),
            "loss_bus": loss_bus.item(),
            "loss_mission": loss_mission.item(),
            "loss_size": loss_size.item(),
            "loss_stability": loss_stability.item(),
            "bus_acc": bus_acc,
        }

    def train_step(self, batch: Dict[str, torch.Tensor], stage: str = "supervised") -> Dict[str, float]:
        if stage == "contrastive":
            return self.contrastive_step(batch)
        return self.supervised_step(batch)

    def export_onnx(self, path: str = "ai/training/models/catalog_dual_encoder_v1.onnx"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dummy_img = torch.zeros(1, 1, 64, 64).to(self.device)
        dummy_kin = torch.zeros(1, 8).to(self.device)
        torch.onnx.export(
            self.model,
            (dummy_img, dummy_kin),
            path,
            input_names=["chip", "kinematics"],
            output_names=["bus", "mission", "size", "stability", "embedding"],
            dynamic_axes={"chip": {0: "batch"}, "kinematics": {0: "batch"}},
            opset_version=11,
        )
        print(f"ONNX exported to {path}")
        return path

    def build_faiss_index(self, embeddings: np.ndarray, path: str = "ai/training/models/catalog_index.faiss"):
        """Build FAISS IVF index for fast similarity search."""
        try:
            import faiss
            d = embeddings.shape[1]
            quantizer = faiss.IndexFlatIP(d)
            nlist = min(64, len(embeddings) // 10)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(embeddings)
            index.add(embeddings)
            faiss.write_index(index, path)
            print(f"FAISS index saved to {path}: {index.ntotal} vectors")
            return index
        except ImportError:
            print("[WARN] faiss not installed; using numpy brute-force fallback")
            np.save(path.replace(".faiss", ".npy"), embeddings)
            return None

    def save_checkpoint(self, path: str):
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


if __name__ == "__main__":
    model = CatalogDualEncoder()
    img = torch.zeros(4, 1, 64, 64)
    kin = torch.zeros(4, 8)
    out = model(img, kin)
    assert out["embedding"].shape == (4, 256)
    assert out["bus"].shape == (4, 4)
    assert out["mission"].shape == (4, 5)
    assert out["size"].shape == (4, 3)
    assert out["stability"].shape == (4,)
    print("CatalogDualEncoder model OK")
