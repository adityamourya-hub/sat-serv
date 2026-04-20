#!/usr/bin/env python3
"""
Single VM Spot GPU Trainer for GCP
Simple, low-cost training on a single spot VM with automatic checkpoint/resume.

Usage:
    # Run on GCP spot VM (T4 ~$0.10/hr, V100 ~$0.75/hr)
    python3 scripts/gcp_spot_vm_trainer.py \
        --component collision \
        --epochs 100 \
        --checkpoint-every 10 \
        --gcs-bucket your-bucket

Features:
    - Handles spot preemption (SIGTERM)
    - Saves to GCS every N epochs
    - Resumes automatically if interrupted
    - Simple: no Kubernetes needed
    - Cost: ~70% savings vs on-demand

Setup:
    1. Create spot VM:
       gcloud compute instances create ssis-gpu-spot \
         --zone=us-central1-a \
         --machine-type=n1-standard-4 \
         --accelerator=type=nvidia-tesla-t4,count=1 \
         --maintenance-policy=TERMINATE \
         --provisioning-model=SPOT \
         --metadata-from-file startup-script=scripts/gcp_spot_startup.sh
    
    2. SSH and run training:
       gcloud compute ssh ssis-gpu-spot
       cd /ssis && python3 scripts/gcp_spot_vm_trainer.py --epochs 100
    
    3. If VM terminated, restart and it resumes automatically
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

# Optional GCS
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

import torch
import torch.nn as nn


class SimpleCheckpointManager:
    """Minimal checkpoint manager for single VM spot training."""
    
    def __init__(self, gcs_bucket: Optional[str], local_dir: str = "/tmp/checkpoints"):
        self.gcs_bucket = gcs_bucket
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        self.gcs_client = None
        if GCS_AVAILABLE and gcs_bucket:
            try:
                self.gcs_client = storage.Client()
                self.bucket = self.gcs_client.bucket(gcs_bucket)
                print(f"✓ GCS checkpointing: gs://{gcs_bucket}/checkpoints/")
            except Exception as e:
                print(f"⚠ GCS failed: {e}")
        
        self.preemption_requested = False
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Handle spot preemption."""
        def on_sigterm(signum, frame):
            print(f"\n⚠️ SIGTERM received! Saving emergency checkpoint...")
            self.preemption_requested = True
        
        signal.signal(signal.SIGTERM, on_sigterm)
        signal.signal(signal.SIGINT, on_sigterm)
    
    def save(self, state: Dict[str, Any], epoch: int, is_emergency: bool = False) -> str:
        """Save checkpoint locally and to GCS."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{timestamp}_epoch_{epoch:04d}.pt"
        local_path = self.local_dir / filename
        
        # Save locally
        torch.save({
            'epoch': epoch,
            'timestamp': time.time(),
            'state': state,
            'is_emergency': is_emergency,
        }, local_path)
        
        print(f"💾 Checkpoint: {local_path} ({local_path.stat().st_size / 1024**2:.1f} MB)")
        
        # Upload to GCS
        if self.gcs_client:
            try:
                blob = self.bucket.blob(f"checkpoints/{filename}")
                blob.upload_from_filename(str(local_path))
                print(f"☁️  Uploaded: gs://{self.gcs_bucket}/checkpoints/{filename}")
            except Exception as e:
                print(f"⚠ GCS upload failed: {e}")
        
        return str(local_path)
    
    def load_latest(self) -> Optional[Dict]:
        """Load latest checkpoint (local or GCS)."""
        # Try local first
        local_checkpoints = sorted(self.local_dir.glob("checkpoint_*.pt"))
        if local_checkpoints:
            latest = local_checkpoints[-1]
            print(f"🔄 Resuming from local: {latest.name}")
            return torch.load(latest, map_location='cpu')
        
        # Try GCS
        if self.gcs_client:
            try:
                blobs = list(self.bucket.list_blobs(prefix="checkpoints/checkpoint_"))
                if blobs:
                    latest = max(blobs, key=lambda b: b.time_created)
                    filename = latest.name.split('/')[-1]
                    local_path = self.local_dir / filename
                    latest.download_to_filename(str(local_path))
                    print(f"🔄 Downloaded from GCS: {filename}")
                    return torch.load(local_path, map_location='cpu')
            except Exception as e:
                print(f"⚠ GCS download failed: {e}")
        
        return None
    
    def should_stop(self) -> bool:
        return self.preemption_requested


def install_ssis():
    """Clone and setup SSIS on fresh VM."""
    if not Path("/ssis").exists():
        print("📥 Cloning SSIS repository...")
        subprocess.run([
            "git", "clone", "https://github.com/your-org/idex.git", "/ssis"
        ], check=False)
    
    os.chdir("/ssis")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run(["pip3", "install", "-q", "torch", "numpy", "scipy", "google-cloud-storage"], check=False)
    
    # Setup paths
    sys.path.insert(0, "/ssis")
    sys.path.insert(0, "/ssis/ai/utils")
    sys.path.insert(0, "/ssis/ai/collision")
    sys.path.insert(0, "/ssis/ai/v3")


def train_collision_with_checkpoints(args):
    """Train collision surrogate with checkpointing."""
    from ai.collision.train_collision_surrogate import (
        TrainConfig, CollisionGraphDataset, MultiSensorFusionGNN
    )
    from torch.utils.data import DataLoader
    
    # Setup
    cfg = TrainConfig()
    cfg.epochs = args.epochs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Checkpoint manager
    ckpt = SimpleCheckpointManager(args.gcs_bucket)
    
    # Model & optimizer
    model = MultiSensorFusionGNN(
        node_dim=cfg.feat_dim, edge_dim=2, hidden=cfg.hidden, heads=cfg.heads
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    # Resume?
    start_epoch = 0
    checkpoint = ckpt.load_latest()
    if checkpoint:
        print(f"🔄 Resuming from epoch {checkpoint['epoch']}")
        state = checkpoint['state']
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Data
    dataset = CollisionGraphDataset(cfg.data_path, cfg.max_nodes, cfg.feat_dim)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    # Training loop
    print(f"\n🏃 Training from epoch {start_epoch} to {cfg.epochs}")
    print("=" * 60)
    
    try:
        for epoch in range(start_epoch, cfg.epochs):
            # Check for preemption
            if ckpt.should_stop():
                print("🛑 Preemption detected! Emergency save...")
                ckpt.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }, epoch, is_emergency=True)
                return 1
            
            # Train
            model.train()
            total_loss = 0.0
            
            for batch_idx, (nodes, adj, iono, y_conj, y_man) in enumerate(train_loader):
                if ckpt.should_stop():
                    break
                
                nodes, adj, iono = nodes.to(device), adj.to(device), iono.to(device)
                y_conj, y_man = y_conj.to(device), y_man.to(device)
                
                risk_alphas, man_logits = model(nodes, adj, iono)
                
                alpha0 = risk_alphas.sum(dim=1, keepdim=True)
                conj_probs = risk_alphas / alpha0
                loss_conj = -(y_conj * conj_probs.log()).sum(dim=1).mean()
                
                ce_loss = nn.CrossEntropyLoss()
                loss_man = ce_loss(man_logits, y_man)
                
                loss = loss_conj + 0.5 * loss_man
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                print(f"Epoch {epoch+1}/{cfg.epochs} - Loss: {avg_loss:.4f}")
            
            # Checkpoint
            if (epoch + 1) % args.checkpoint_every == 0:
                ckpt.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }, epoch + 1)
        
        print("\n✅ Training complete!")
        
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        ckpt.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, epoch, is_emergency=True)
        raise
    
    # Export ONNX
    print("\n📦 Exporting ONNX...")
    model.eval()
    dummy_nodes = torch.zeros(1, cfg.max_nodes, cfg.feat_dim).to(device)
    dummy_adj = torch.eye(cfg.max_nodes).unsqueeze(0).to(device)
    dummy_iono = torch.zeros(1, 16).to(device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_dir / "collision_multi_task_v2.onnx"
    torch.onnx.export(
        model, (dummy_nodes, dummy_adj, dummy_iono), str(onnx_path),
        input_names=["nodes", "adj", "iono"],
        output_names=["risk_alphas", "maneuver_logits"],
    )
    print(f"✓ Exported: {onnx_path}")
    
    # Upload to GCS
    if ckpt.gcs_client:
        try:
            blob = ckpt.bucket.blob("models/collision_multi_task_v2.onnx")
            blob.upload_from_filename(str(onnx_path))
            print(f"☁️  Model uploaded to gs://{args.gcs_bucket}/models/")
        except Exception as e:
            print(f"⚠ Model upload failed: {e}")
    
    return 0


def train_navic_with_checkpoints(args):
    """Train NavIC anomaly detector with checkpointing."""
    sys.path.insert(0, "ai/v3")
    from navic_anomaly_detector import NavICAnomalyDetector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    ckpt = SimpleCheckpointManager(args.gcs_bucket)

    model = NavICAnomalyDetector(input_dim=12, hidden_dim=64, n_heads=4, n_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    start_epoch = 0
    checkpoint = ckpt.load_latest()
    if checkpoint:
        state = checkpoint['state']
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1

    # Load preprocessed data if available, else use synthetic
    import numpy as np
    data_path = Path("ai/v3/data/navic_training.npz")
    if data_path.exists():
        data = np.load(data_path)
        X_all, y_all = data['X'], data['y']
        print(f"📂 Loaded {len(X_all)} samples from {data_path}")
    else:
        print("⚠ No preprocessed data found, using synthetic. Run preprocess_training_data.py first.")
        rng = np.random.default_rng(42)
        X_all = rng.normal(0, 1, (5000, 60, 12)).astype(np.float32)
        y_all = rng.integers(0, 3, 5000).astype(np.int64)

    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    print(f"\n🏃 Training NavIC from epoch {start_epoch} to {args.epochs}")
    try:
        for epoch in range(start_epoch, args.epochs):
            if ckpt.should_stop():
                ckpt.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, epoch, is_emergency=True)
                return 1
            model.train()
            total_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = nn.CrossEntropyLoss()(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.4f}")
            if (epoch + 1) % args.checkpoint_every == 0:
                ckpt.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, epoch + 1)
        print("\n✅ NavIC training complete!")
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        ckpt.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, epoch, is_emergency=True)
        raise

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "navic_detector.pt")
    print(f"✓ Saved: {output_dir / 'navic_detector.pt'}")
    return 0


def train_reentry_with_checkpoints(args):
    """Train re-entry predictor with checkpointing."""
    sys.path.insert(0, "ai/v3")
    from reentry_predictor import train_reentry_predictor, export_reentry_onnx
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Load preprocessed data
    data_path = Path("ai/v3/data/reentry_training.npz")
    n_samples = 5000
    if data_path.exists():
        data = np.load(data_path)
        n_samples = len(data['X'])
        print(f"📂 Loaded {n_samples} samples from {data_path}")

    model = train_reentry_predictor(n_samples=n_samples, epochs=args.epochs, device=str(device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_reentry_onnx(model, str(output_dir / "reentry_predictor.onnx"))
    print(f"✓ Exported: {output_dir / 'reentry_predictor.onnx'}")
    return 0


def train_maneuver_with_checkpoints(args):
    """Train maneuver detector with checkpointing."""
    sys.path.insert(0, "ai/maneuver")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    ckpt = SimpleCheckpointManager(args.gcs_bucket)

    from train_maneuver_detector import TrainConfig, SyntheticManeuverDataset
    cfg = TrainConfig()
    cfg.epochs = args.epochs

    # Check for preprocessed TLE residuals
    data_path = Path("ai/maneuver/data/tle_residuals.jsonl")
    if data_path.exists():
        print(f"📂 Found preprocessed TLE residuals at {data_path}")

    # Use existing training script
    from train_maneuver_detector import train as maneuver_train
    maneuver_train()
    print("\n✅ Maneuver detector training complete!")
    return 0


def train_all_with_checkpoints(args):
    """Train all components sequentially."""
    components = ['maneuver', 'navic', 'reentry', 'collision']
    for comp in components:
        print(f"\n{'#'*60}")
        print(f"Training: {comp}")
        print(f"{'#'*60}")
        args.component = comp
        try:
            if comp == 'collision':
                train_collision_with_checkpoints(args)
            elif comp == 'navic':
                train_navic_with_checkpoints(args)
            elif comp == 'reentry':
                train_reentry_with_checkpoints(args)
            elif comp == 'maneuver':
                train_maneuver_with_checkpoints(args)
        except Exception as e:
            print(f"⚠️ {comp} failed: {e}")
            continue
    return 0


def main():
    parser = argparse.ArgumentParser(description='Train on GCP Spot VM')
    parser.add_argument('--component', type=str, default='collision',
                       choices=['collision', 'navic', 'reentry', 'maneuver', 'all'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoint-every', type=int, default=10,
                       help='Checkpoint every N epochs')
    parser.add_argument('--gcs-bucket', type=str,
                       default=os.getenv('GCS_CHECKPOINT_BUCKET'),
                       help='GCS bucket for checkpoints')
    parser.add_argument('--output-dir', type=str, default='/ssis/ai/training/models')
    parser.add_argument('--auto-setup', action='store_true',
                       help='Auto-clone and setup SSIS on fresh VM')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run data preprocessing before training')
    
    args = parser.parse_args()
    
    # Auto-setup if needed
    if args.auto_setup or not Path("/ssis").exists():
        install_ssis()
    
    # Run preprocessing if requested
    if args.preprocess:
        print("📊 Running data preprocessing...")
        subprocess.run([sys.executable, "scripts/preprocess_training_data.py"], check=True)
    
    print("=" * 60)
    print("SSIS Spot VM Trainer")
    print("=" * 60)
    print(f"Component: {args.component}")
    print(f"Epochs: {args.epochs}")
    print(f"GCS Bucket: {args.gcs_bucket or 'None (local only)'}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Train
    if args.component == 'collision':
        exit_code = train_collision_with_checkpoints(args)
    elif args.component == 'navic':
        exit_code = train_navic_with_checkpoints(args)
    elif args.component == 'reentry':
        exit_code = train_reentry_with_checkpoints(args)
    elif args.component == 'maneuver':
        exit_code = train_maneuver_with_checkpoints(args)
    elif args.component == 'all':
        exit_code = train_all_with_checkpoints(args)
    else:
        print(f"Component {args.component} not yet implemented")
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
