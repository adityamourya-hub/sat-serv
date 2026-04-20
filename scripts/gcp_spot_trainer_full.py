#!/usr/bin/env python3
"""
SSIS GCP Spot Instance Trainer with GCS Integration

Features:
- Downloads datasets from GCS bucket (training-data/ folder)
- Uploads trained weights to GCS bucket (weights/ folder)
- Automatic checkpoint/resume for spot preemption
- Handles SIGTERM gracefully

GCS Bucket Structure:
    gs://your-bucket/
    ├── datasets/           # Download training data from here
    │   ├── A-TLES-Ephemeris/
    │   ├── B-Conjuction-Collision-dataset/
    │   ├── C-NAVIC-anamoly/
    │   └── D-Rentry-data/
    ├── checkpoints/        # Training checkpoints saved here
    └── weights/            # Final trained models uploaded here

Usage:
    # First time: upload datasets to GCS
    gsutil -m cp -r training-data/ gs://your-bucket/datasets/

    # Run training on spot VM
    python3 scripts/gcp_spot_trainer_full.py \
        --component all \
        --gcs-bucket your-bucket \
        --epochs 100 \
        --checkpoint-every 5

    # If VM terminated, restart same command - auto-resumes from checkpoint
"""

import argparse
import os
import sys
import signal
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import tempfile

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ai" / "v3"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ai" / "collision"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ai" / "maneuver"))

try:
    from google.cloud import storage
    from google.api_core.exceptions import NotFound
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("⚠️ google-cloud-storage not installed. Run: pip3 install google-cloud-storage")

import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# GCS Manager - Handles all GCS operations
# =============================================================================

class GCSManager:
    """Manages dataset download, checkpoint sync, and weight upload to GCS."""

    def __init__(self, bucket_name: str, local_base: str = "/ssis"):
        self.bucket_name = bucket_name
        self.local_base = Path(local_base)
        self.client = None
        self.bucket = None

        if GCS_AVAILABLE and bucket_name:
            try:
                self.client = storage.Client()
                self.bucket = self.client.bucket(bucket_name)
                print(f"☁️  GCS connected: gs://{bucket_name}")
            except Exception as e:
                print(f"⚠️ GCS connection failed: {e}")

    def download_datasets(self, force: bool = False) -> bool:
        """Download training datasets from GCS if not present locally."""
        if not self.bucket:
            print("⚠️ No GCS bucket configured, skipping dataset download")
            return False

        local_data = self.local_base / "training-data"
        if local_data.exists() and not force:
            print(f"📂 Datasets already exist at {local_data}")
            return True

        print(f"\n📥 Downloading datasets from gs://{self.bucket_name}/datasets/")
        local_data.mkdir(parents=True, exist_ok=True)

        try:
            blobs = list(self.bucket.list_blobs(prefix="datasets/"))
            if not blobs:
                print(f"⚠️ No datasets found in gs://{self.bucket_name}/datasets/")
                return False

            for blob in blobs:
                # Skip directories
                if blob.name.endswith('/'):
                    continue

                # Compute local path
                relative_path = blob.name.replace("datasets/", "")
                local_path = local_data / relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Skip if exists and size matches
                if local_path.exists():
                    if local_path.stat().st_size == blob.size:
                        print(f"  ✓ {relative_path} (cached)")
                        continue

                print(f"  📥 {relative_path} ({blob.size / 1024 / 1024:.1f} MB)")
                blob.download_to_filename(str(local_path))

            print(f"✅ Datasets downloaded to {local_data}")
            return True

        except Exception as e:
            print(f"❌ Dataset download failed: {e}")
            return False

    def upload_weights(self, weights_dir: Path, component: str) -> bool:
        """Upload trained weights to GCS weights/ folder."""
        if not self.bucket:
            print("⚠️ No GCS bucket configured, skipping weight upload")
            return False

        print(f"\n📤 Uploading weights to gs://{self.bucket_name}/weights/{component}/")

        uploaded = []
        for ext in ['*.onnx', '*.pt', '*.pth']:
            for weight_file in weights_dir.glob(ext):
                dest_path = f"weights/{component}/{weight_file.name}"
                try:
                    blob = self.bucket.blob(dest_path)
                    blob.upload_from_filename(str(weight_file))
                    uploaded.append(weight_file.name)
                    print(f"  ☁️  {weight_file.name}")
                except Exception as e:
                    print(f"  ⚠️ Failed to upload {weight_file.name}: {e}")

        # Upload training metadata
        metadata = {
            "component": component,
            "timestamp": datetime.now().isoformat(),
            "files": uploaded,
            "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown"
        }
        meta_blob = self.bucket.blob(f"weights/{component}/metadata.json")
        meta_blob.upload_from_string(json.dumps(metadata, indent=2))

        print(f"✅ Uploaded {len(uploaded)} files to GCS")
        return True

    def list_available_weights(self) -> Dict[str, list]:
        """List all trained weights available in GCS."""
        if not self.bucket:
            return {}

        weights = {}
        try:
            blobs = list(self.bucket.list_blobs(prefix="weights/"))
            for blob in blobs:
                parts = blob.name.split('/')
                if len(parts) >= 2 and parts[0] == "weights":
                    component = parts[1]
                    if component not in weights:
                        weights[component] = []
                    weights[component].append(blob.name)
        except Exception as e:
            print(f"⚠️ Failed to list weights: {e}")

        return weights


# =============================================================================
# Checkpoint Manager - Handles spot preemption
# =============================================================================

class SpotCheckpointManager:
    """Manages checkpoints with GCS sync for spot instance survival."""

    def __init__(self, gcs_manager: GCSManager, component: str, local_dir: str = "/tmp/checkpoints"):
        self.gcs = gcs_manager
        self.component = component
        self.local_dir = Path(local_dir) / component
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.preemption_requested = False
        self._setup_signal_handlers()

        # Last checkpoint tracking for liveness probe
        self.last_checkpoint_file = self.local_dir / "last_checkpoint.txt"

    def _setup_signal_handlers(self):
        """Handle SIGTERM from GCP when spot instance is being reclaimed."""
        def on_sigterm(signum, frame):
            print(f"\n⚠️ SIGTERM received! Spot instance being reclaimed...")
            print("💾 Emergency checkpoint will be saved...")
            self.preemption_requested = True

        signal.signal(signal.SIGTERM, on_sigterm)
        signal.signal(signal.SIGINT, on_sigterm)

        # GCP gives 30 seconds before hard termination
        print("🔔 SIGTERM handler ready (30s grace period on spot preemption)")

    def should_stop(self) -> bool:
        return self.preemption_requested

    def save(self, state: Dict[str, Any], epoch: int, is_emergency: bool = False) -> str:
        """Save checkpoint locally and sync to GCS."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{self.component}_{timestamp}_epoch_{epoch:04d}.pt"
        local_path = self.local_dir / filename

        # Save locally
        torch.save({
            'epoch': epoch,
            'timestamp': time.time(),
            'component': self.component,
            'state': state,
            'is_emergency': is_emergency,
        }, local_path)

        # Update last checkpoint marker
        self.last_checkpoint_file.write_text(f"{timestamp}\n{epoch}\n{local_path}\n")

        print(f"💾 Checkpoint: {local_path.name} ({local_path.stat().st_size / 1024**2:.1f} MB)")

        # Sync to GCS
        if self.gcs.bucket:
            try:
                gcs_path = f"checkpoints/{self.component}/{filename}"
                blob = self.gcs.bucket.blob(gcs_path)
                blob.upload_from_filename(str(local_path))
                print(f"☁️  Synced to gs://{self.gcs.bucket_name}/{gcs_path}")
            except Exception as e:
                print(f"⚠️ GCS sync failed: {e}")

        return str(local_path)

    def load_latest(self) -> Optional[Dict]:
        """Load latest checkpoint from local or GCS."""
        # Try local first (faster)
        local_checkpoints = sorted(self.local_dir.glob("checkpoint_*.pt"))
        if local_checkpoints:
            latest = local_checkpoints[-1]
            print(f"🔄 Resuming from local: {latest.name}")
            return torch.load(latest, map_location='cpu')

        # Try GCS
        if self.gcs.bucket:
            try:
                prefix = f"checkpoints/{self.component}/checkpoint_"
                blobs = list(self.gcs.bucket.list_blobs(prefix=prefix))
                if blobs:
                    latest = max(blobs, key=lambda b: b.time_created or b.updated)
                    filename = latest.name.split('/')[-1]
                    local_path = self.local_dir / filename

                    print(f"🔄 Downloading checkpoint from GCS: {filename}")
                    latest.download_to_filename(str(local_path))

                    return torch.load(local_path, map_location='cpu')
            except Exception as e:
                print(f"⚠️ GCS checkpoint download failed: {e}")

        print("🆕 No checkpoint found, starting from scratch")
        return None

    def cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints to save disk space."""
        checkpoints = sorted(self.local_dir.glob("checkpoint_*.pt"))
        if len(checkpoints) > keep_last:
            for old in checkpoints[:-keep_last]:
                old.unlink()
                print(f"🗑️  Removed old checkpoint: {old.name}")


# =============================================================================
# Component Trainers
# =============================================================================

class ComponentTrainer:
    """Base class for training components with checkpoint support."""

    def __init__(self, component: str, gcs_manager: GCSManager, args):
        self.component = component
        self.gcs = gcs_manager
        self.args = args
        self.ckpt = SpotCheckpointManager(gcs_manager, component)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*60}")
        print(f"Training: {component.upper()}")
        print(f"Device: {self.device}")
        print(f"Epochs: {args.epochs}")
        print(f"{'='*60}")

    def train(self) -> bool:
        """Override in subclass. Returns True on success."""
        raise NotImplementedError

    def save_weights(self, output_dir: Path):
        """Save final weights and upload to GCS."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        if hasattr(self, 'model'):
            pt_path = output_dir / f"{self.component}_model.pt"
            torch.save(self.model.state_dict(), pt_path)
            print(f"✅ Saved PyTorch weights: {pt_path}")

        # Upload to GCS
        self.gcs.upload_weights(output_dir, self.component)


class CollisionTrainer(ComponentTrainer):
    """Train collision surrogate GNN."""

    def train(self) -> bool:
        try:
            from train_collision_surrogate_spot import (
                TrainConfig, CollisionGraphDataset, MultiSensorFusionGNN
            )
            from torch.utils.data import DataLoader
        except ImportError as e:
            print(f"❌ Failed to import collision training: {e}")
            return False

        cfg = TrainConfig()
        cfg.epochs = self.args.epochs

        # Model
        model = MultiSensorFusionGNN(
            node_dim=cfg.feat_dim, edge_dim=2, hidden=cfg.hidden, heads=cfg.heads
        ).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.model = model

        # Resume from checkpoint
        start_epoch = 0
        checkpoint = self.ckpt.load_latest()
        if checkpoint:
            state = checkpoint['state']
            model.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"🔄 Resumed from epoch {start_epoch}")

        # Dataset
        data_path = Path("training-data/B-Conjuction-Collision-dataset/train_data")
        if not data_path.exists():
            print(f"❌ Dataset not found: {data_path}")
            print("   Run with --download-datasets first")
            return False

        # Preprocess if needed
        graph_path = Path("ai/simulator/data/scenarios_graph_v2.jsonl")
        if not graph_path.exists():
            print("📊 Preprocessing collision data...")
            subprocess.run([sys.executable, "scripts/preprocess_training_data.py", "--dataset", "B"], check=False)

        dataset = CollisionGraphDataset(str(graph_path), cfg.max_nodes, cfg.feat_dim)
        if len(dataset) == 0:
            print("❌ No training data available")
            return False

        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        print(f"📊 Dataset: {len(dataset)} scenarios")

        # Training loop
        print(f"\n🏃 Training from epoch {start_epoch} to {cfg.epochs}")
        try:
            for epoch in range(start_epoch, cfg.epochs):
                if self.ckpt.should_stop():
                    print("🛑 Preemption detected! Emergency save...")
                    self.ckpt.save({
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, epoch, is_emergency=True)
                    return False  # Will be resumed on new VM

                model.train()
                total_loss = 0.0

                for batch_idx, batch in enumerate(loader):
                    if self.ckpt.should_stop():
                        break

                    # Unpack batch (adjust based on your dataset)
                    if len(batch) == 5:
                        nodes, adj, iono, y_conj, y_man = batch
                    else:
                        continue

                    nodes = nodes.to(self.device)
                    adj = adj.to(self.device)
                    iono = iono.to(self.device)
                    y_conj = y_conj.to(self.device)
                    y_man = y_man.to(self.device)

                    risk_alphas, man_logits = model(nodes, adj, iono)

                    # Dirichlet loss for evidential learning
                    alpha0 = risk_alphas.sum(dim=1, keepdim=True)
                    conj_probs = risk_alphas / alpha0
                    loss_conj = -(y_conj * conj_probs.log().clamp(min=-100)).sum(dim=1).mean()

                    # Cross-entropy for maneuver
                    ce_loss = nn.CrossEntropyLoss()
                    loss_man = ce_loss(man_logits, y_man)

                    loss = loss_conj + 0.5 * loss_man

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(loader) if len(loader) > 0 else 0

                if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                    print(f"  Epoch {epoch+1}/{cfg.epochs} - Loss: {avg_loss:.4f}")

                # Checkpoint
                if (epoch + 1) % self.args.checkpoint_every == 0:
                    self.ckpt.save({
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, epoch + 1)
                    self.ckpt.cleanup_old_checkpoints(keep_last=3)

        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            self.ckpt.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, epoch if 'epoch' in locals() else 0, is_emergency=True)
            raise

        print("\n✅ Collision training complete!")

        # Export ONNX
        self.export_onnx(model, cfg)
        return True

    def export_onnx(self, model, cfg):
        """Export trained model to ONNX format."""
        print("\n📦 Exporting to ONNX...")
        model.eval()

        dummy_nodes = torch.zeros(1, cfg.max_nodes, cfg.feat_dim).to(self.device)
        dummy_adj = torch.eye(cfg.max_nodes).unsqueeze(0).to(self.device)
        dummy_iono = torch.zeros(1, 16).to(self.device)

        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = output_dir / "collision_multi_task_v2.onnx"
        torch.onnx.export(
            model, (dummy_nodes, dummy_adj, dummy_iono), str(onnx_path),
            input_names=["nodes", "adj", "iono"],
            output_names=["risk_alphas", "maneuver_logits"],
            dynamic_axes={
                "nodes": {0: "batch_size"},
                "risk_alphas": {0: "batch_size"},
                "maneuver_logits": {0: "batch_size"}
            },
            opset_version=11
        )
        print(f"✅ ONNX exported: {onnx_path}")


class NavICTrainer(ComponentTrainer):
    """Train NavIC anomaly detector."""

    def train(self) -> bool:
        try:
            from navic_anomaly_detector import NavICAnomalyDetector, train_navic_detector
        except ImportError as e:
            print(f"❌ Failed to import NavIC training: {e}")
            return False

        # Load preprocessed data or use synthetic
        data_path = Path("training-data/C-NAVIC-anamoly-and-GNSS-spoofing-jamming-dataset")
        npz_path = Path("ai/v3/data/navic_training.npz")

        if not npz_path.exists() and data_path.exists():
            print("📊 Preprocessing NavIC data...")
            subprocess.run([sys.executable, "scripts/preprocess_training_data.py", "--dataset", "C"], check=False)

        if npz_path.exists():
            data = np.load(npz_path)
            n_samples = len(data['X'])
            print(f"📂 Loaded {n_samples} samples from {npz_path}")
        else:
            n_samples = 10000
            print(f"⚠️ Using synthetic data: {n_samples} samples")

        # Train
        self.model = train_navic_detector(
            n_epochs=self.args.epochs,
            n_samples=n_samples,
            device=str(self.device)
        )

        print("\n✅ NavIC training complete!")
        return True


class ReentryTrainer(ComponentTrainer):
    """Train re-entry predictor."""

    def train(self) -> bool:
        try:
            from reentry_predictor import train_reentry_predictor
        except ImportError as e:
            print(f"❌ Failed to import reentry training: {e}")
            return False

        # Load preprocessed data
        npz_path = Path("ai/v3/data/reentry_training.npz")
        if not npz_path.exists():
            print("📊 Preprocessing reentry data...")
            subprocess.run([sys.executable, "scripts/preprocess_training_data.py", "--dataset", "D"], check=False)

        n_samples = 5000
        if npz_path.exists():
            data = np.load(npz_path)
            n_samples = len(data['X'])

        self.model = train_reentry_predictor(
            n_samples=n_samples,
            epochs=self.args.epochs,
            device=str(self.device)
        )

        print("\n✅ Re-entry training complete!")
        return True


class ManeuverTrainer(ComponentTrainer):
    """Train maneuver detector."""

    def train(self) -> bool:
        try:
            from train_maneuver_detector import train
        except ImportError as e:
            print(f"❌ Failed to import maneuver training: {e}")
            return False

        # Preprocess TLE data if needed
        jsonl_path = Path("ai/maneuver/data/tle_residuals.jsonl")
        if not jsonl_path.exists():
            print("📊 Preprocessing TLE data...")
            subprocess.run([sys.executable, "scripts/preprocess_training_data.py", "--dataset", "A"], check=False)

        # The existing train() function handles everything
        train()

        print("\n✅ Maneuver training complete!")
        return True


# =============================================================================
# Main Entry Point
# =============================================================================

def create_parser():
    parser = argparse.ArgumentParser(
        description="SSIS GCP Spot Instance Trainer with GCS Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Upload datasets to GCS (run once locally)
  gsutil -m cp -r training-data/ gs://your-bucket/datasets/

  # Step 2: Start training on spot VM (downloads data, trains, uploads weights)
  python3 scripts/gcp_spot_trainer_full.py \\
      --gcs-bucket your-bucket \\
      --component collision \\
      --epochs 100 \\
      --checkpoint-every 5

  # Step 3: If VM terminated, restart (auto-resumes from checkpoint)
  python3 scripts/gcp_spot_trainer_full.py \\
      --gcs-bucket your-bucket \\
      --component collision \\
      --epochs 100

  # Train all components sequentially
  python3 scripts/gcp_spot_trainer_full.py \\
      --gcs-bucket your-bucket \\
      --component all \\
      --epochs 50

  # Quick test mode (fewer epochs)
  python3 scripts/gcp_spot_trainer_full.py \\
      --gcs-bucket your-bucket \\
      --component navic \\
      --epochs 10 \\
      --quick-test
        """
    )

    parser.add_argument(
        "--gcs-bucket",
        required=True,
        help="GCS bucket name (datasets/ and weights/ will be created here)"
    )
    parser.add_argument(
        "--component",
        choices=['collision', 'navic', 'reentry', 'maneuver', 'all'],
        default='collision',
        help="Which component to train"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--output-dir",
        default='/ssis/ai/training/models',
        help="Local directory for model outputs"
    )
    parser.add_argument(
        "--download-datasets",
        action='store_true',
        help="Download datasets from GCS before training"
    )
    parser.add_argument(
        "--quick-test",
        action='store_true',
        help="Quick test with reduced epochs (10)"
    )
    parser.add_argument(
        "--list-weights",
        action='store_true',
        help="List all trained weights in GCS and exit"
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.epochs = 10
        print("🧪 Quick test mode: epochs=10")

    # Initialize GCS manager
    gcs = GCSManager(args.gcs_bucket)

    # List weights and exit
    if args.list_weights:
        weights = gcs.list_available_weights()
        print("\n📦 Trained weights in GCS:")
        for component, files in weights.items():
            print(f"\n  {component}:")
            for f in files[-5:]:  # Show last 5
                print(f"    - {f.split('/')[-1]}")
        return

    print("\n" + "="*70)
    print("  SSIS GCP Spot Instance Trainer")
    print("="*70)
    print(f"  GCS Bucket:    gs://{args.gcs_bucket}")
    print(f"  Component:     {args.component}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Checkpoint:    every {args.checkpoint_every} epochs")
    print(f"  Output:        {args.output_dir}")
    print("="*70)

    # Download datasets if requested or if not present
    if args.download_datasets or not Path("training-data").exists():
        success = gcs.download_datasets()
        if not success and args.component != 'navic':  # NavIC can use synthetic
            print("\n⚠️ Dataset download failed. Some components may not work.")
            response = input("Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                return

    # Train components
    components = ['collision', 'navic', 'reentry', 'maneuver'] if args.component == 'all' else [args.component]

    results = {}
    for comp in components:
        trainer_class = {
            'collision': CollisionTrainer,
            'navic': NavICTrainer,
            'reentry': ReentryTrainer,
            'maneuver': ManeuverTrainer,
        }.get(comp)

        if not trainer_class:
            print(f"❌ Unknown component: {comp}")
            continue

        trainer = trainer_class(comp, gcs, args)

        try:
            success = trainer.train()
            results[comp] = "success" if success else "failed"

            if success:
                trainer.save_weights(Path(args.output_dir))

        except Exception as e:
            print(f"\n❌ {comp} training failed: {e}")
            import traceback
            traceback.print_exc()
            results[comp] = f"error: {str(e)}"

    # Summary
    print("\n" + "="*70)
    print("  Training Summary")
    print("="*70)
    for comp, status in results.items():
        icon = "✅" if status == "success" else "❌"
        print(f"  {icon} {comp}: {status}")

    # List available weights
    weights = gcs.list_available_weights()
    if weights:
        print("\n  📦 Available weights in GCS:")
        for comp in weights:
            print(f"     - {comp}: {len(weights[comp])} files")

    print("="*70)

    # Exit code based on results
    if all(s == "success" for s in results.values()):
        print("\n🎉 All components trained successfully!")
        sys.exit(0)
    else:
        print("\n⚠️ Some components failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
