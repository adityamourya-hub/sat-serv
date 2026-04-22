#!/usr/bin/env python3
"""
Re-entry Curriculum Trainer.

Trains ReentryForecaster in 4 stages of increasing difficulty:
  Stage 0 (Easy):     Low beta, calm atmosphere, steep decay
  Stage 1 (Medium):   Moderate beta, mild skip-out, F10.7=120
  Stage 2 (Hard):     High beta, grazing re-entry, skip-out common
  Stage 3 (Storm):    Variable density + tumbling Cd, extreme F10.7

At each stage:
  1. Generate synthetic storm shard
  2. Train for N epochs
  3. Validate on next-stage held-out data
  4. Advance if validation MSE < threshold

ONNX export after final stage.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional

from .decay_forecaster import ReentryForecaster, ReentryForecasterTrainer


class ReentryCurriculumTrainer:
    """Curriculum-aware trainer with stage progression."""

    STAGE_CONFIG = {
        0: {"name": "easy",   "epochs": 5, "threshold_mse": 5.0,   "lr": 1e-3},
        1: {"name": "medium", "epochs": 5, "threshold_mse": 10.0,  "lr": 5e-4},
        2: {"name": "hard",   "epochs": 5, "threshold_mse": 20.0,  "lr": 3e-4},
        3: {"name": "storm",  "epochs": 5, "threshold_mse": 50.0,  "lr": 1e-4},
    }

    def __init__(self, model: ReentryForecaster = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model or ReentryForecaster(in_features=10, window=120)
        self.model = self.model.to(self.device)
        self.current_stage = 0
        self.history: List[Dict] = []

    def _prepare_batch(self, trajectories: np.ndarray, lengths: np.ndarray,
                       targets: np.ndarray, batch_size: int = 8):
        """Pad/truncate trajectories to [B, 120, 10] and return batch dict."""
        n = len(trajectories)
        idx = np.random.choice(n, min(batch_size, n), replace=False)

        B = len(idx)
        x = np.zeros((B, 120, 10), dtype=np.float32)
        for i, j in enumerate(idx):
            seq = trajectories[j, :lengths[j]]
            L = min(len(seq), 120)
            x[i, :L] = seq[:L]

        y = targets[idx]
        return {
            "features": torch.from_numpy(x).float(),
            "target": torch.from_numpy(y).float(),
        }

    def train_stage(self, shard_path: Path, stage: int, batch_size: int = 8) -> Dict[str, float]:
        """Train on a single curriculum shard."""
        cfg = self.STAGE_CONFIG[stage]
        data = np.load(shard_path)
        trajs = data["trajectories"]
        lengths = data["lengths"]
        targets = data["targets"]

        trainer = ReentryForecasterTrainer(
            model=self.model,
            device=str(self.device),
            lr=cfg["lr"],
        )

        n = len(trajs)
        losses = []
        for epoch in range(cfg["epochs"]):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch = self._prepare_batch(trajs[idx], lengths[idx], targets[idx], batch_size)
                metrics = trainer.train_step(batch)
                epoch_loss += metrics["loss"]
                num_batches += 1
            avg_loss = epoch_loss / max(num_batches, 1)
            losses.append(avg_loss)
            print(f"  Stage {stage} ({cfg['name']}) Epoch {epoch+1}/{cfg['epochs']}: loss={avg_loss:.4f}")

        final_loss = losses[-1]
        self.history.append({
            "stage": stage,
            "name": cfg["name"],
            "final_loss": final_loss,
            "loss_curve": losses,
        })
        return {"stage": stage, "final_loss": final_loss, "history": losses}

    def validate_on_shard(self, shard_path: Path, batch_size: int = 8) -> Dict[str, float]:
        """Compute MSE on a held-out shard without training."""
        data = np.load(shard_path)
        trajs = data["trajectories"]
        lengths = data["lengths"]
        targets = data["targets"]

        self.model.eval()
        n = len(trajs)
        total_mse = 0.0
        num_batches = 0
        preds_all = []
        targs_all = []

        with torch.no_grad():
            for i in range(0, n, batch_size):
                idx = np.arange(i, min(i + batch_size, n))
                batch = self._prepare_batch(trajs[idx], lengths[idx], targets[idx], batch_size)
                x = batch["features"].to(self.device)
                pred = self.model(x).cpu().numpy()
                tgt = batch["target"].cpu().numpy()
                mse = np.mean((pred - tgt) ** 2)
                total_mse += mse
                num_batches += 1
                preds_all.append(pred)
                targs_all.append(tgt)

        avg_mse = total_mse / max(num_batches, 1)
        preds_all = np.concatenate(preds_all, axis=0)
        targs_all = np.concatenate(targs_all, axis=0)

        # Per-metric breakdown
        ttd_mse = np.mean((preds_all[:, 0] - targs_all[:, 0]) ** 2)
        lat_mse = np.mean((preds_all[:, 1] - targs_all[:, 1]) ** 2)
        lon_mse = np.mean((preds_all[:, 2] - targs_all[:, 2]) ** 2)

        return {
            "mse": float(avg_mse),
            "ttd_mse": float(ttd_mse),
            "lat_mse": float(lat_mse),
            "lon_mse": float(lon_mse),
            "preds": preds_all,
            "targets": targs_all,
        }

    def should_advance(self, stage: int, val_metrics: Dict) -> bool:
        """Check if validation MSE is below stage threshold."""
        threshold = self.STAGE_CONFIG[stage]["threshold_mse"]
        return val_metrics["mse"] < threshold

    def export_onnx(self, path: str = "ai/training/models/reentry_forecaster_v1.onnx"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(1, 120, 10).to(self.device)
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
        torch.save({
            "model": self.model.state_dict(),
            "stage": self.current_stage,
            "history": self.history,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.current_stage = ckpt.get("stage", 0)
        self.history = ckpt.get("history", [])


if __name__ == "__main__":
    print("ReentryCurriculumTrainer module OK")
