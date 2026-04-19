#!/bin/bash
# GCP Spot VM Startup Script
# This runs automatically when VM starts. Handles:
# 1. GPU driver installation
# 2. Python environment setup
# 3. SSIS cloning/setup
# 4. Automatic training with resume

set -e

LOGFILE="/var/log/ssis-training.log"
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "SSIS Spot VM Startup - $(date)"
echo "=========================================="

# Configuration (set via metadata)
GCS_BUCKET=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket" -H "Metadata-Flavor: Google" || echo "")
COMPONENT=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/component" -H "Metadata-Flavor: Google" || echo "collision")
EPOCHS=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/epochs" -H "Metadata-Flavor: Google" || echo "100")
CHECKPOINT_EVERY=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/checkpoint-every" -H "Metadata-Flavor: Google" || echo "10")
GITHUB_REPO=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/github-repo" -H "Metadata-Flavor: Google" || echo "")

echo "Config:"
echo "  GCS_BUCKET: ${GCS_BUCKET:-None}"
echo "  COMPONENT: $COMPONENT"
echo "  EPOCHS: $EPOCHS"
echo "  CHECKPOINT_EVERY: $CHECKPOINT_EVERY"

# 1. Install GPU drivers
echo ""
echo "[1/5] Installing NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    
    apt-get update
    apt-get install -y linux-headers-$(uname -r)
    apt-get install -y nvidia-driver-470  # Stable driver
    
    # Reboot to load driver
    echo "Rebooting to load NVIDIA driver..."
    reboot
    exit 0
fi

nvidia-smi
echo "✓ GPU driver ready"

# 2. Install Python & dependencies
echo ""
echo "[2/5] Installing Python dependencies..."
apt-get update
apt-get install -y python3-pip python3-dev git wget

pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install numpy scipy pandas scikit-learn tqdm
pip3 install google-cloud-storage onnx onnxruntime-gpu

echo "✓ Python packages installed"

# 3. Setup SSIS
echo ""
echo "[3/5] Setting up SSIS..."

if [ ! -d "/ssis" ]; then
    if [ -n "$GITHUB_REPO" ]; then
        echo "Cloning from $GITHUB_REPO..."
        git clone "$GITHUB_REPO" /ssis
    else
        echo "Creating SSIS structure..."
        mkdir -p /ssis
        # Download scripts directly
        cd /ssis
        mkdir -p ai/collision ai/v3 ai/utils scripts
    fi
fi

cd /ssis

# Create the training script if not exists
if [ ! -f "scripts/gcp_spot_vm_trainer.py" ]; then
    cat > scripts/gcp_spot_vm_trainer.py << 'TRAINER_EOF'
#!/usr/bin/env python3
"""Simple spot VM trainer - embedded version"""
import os, sys, time, signal, argparse, subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

import torch
import torch.nn as nn

class SimpleCheckpointManager:
    def __init__(self, gcs_bucket: Optional[str], local_dir: str = "/tmp/checkpoints"):
        self.gcs_bucket = gcs_bucket
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.gcs_client = None
        if GCS_AVAILABLE and gcs_bucket:
            try:
                self.gcs_client = storage.Client()
                self.bucket = self.gcs_client.bucket(gcs_bucket)
                print(f"✓ GCS: gs://{gcs_bucket}/checkpoints/")
            except Exception as e:
                print(f"⚠ GCS failed: {e}")
        self.preemption_requested = False
        signal.signal(signal.SIGTERM, lambda s, f: self._on_preempt())
        signal.signal(signal.SIGINT, lambda s, f: self._on_preempt())
    
    def _on_preempt(self):
        print("\n⚠️ SIGTERM! Saving emergency checkpoint...")
        self.preemption_requested = True
    
    def save(self, state: Dict, epoch: int, is_emergency: bool = False):
        filename = f"ckpt_epoch_{epoch:04d}_{int(time.time())}.pt"
        local_path = self.local_dir / filename
        torch.save({'epoch': epoch, 'timestamp': time.time(), 'state': state, 'is_emergency': is_emergency}, local_path)
        print(f"💾 {local_path.name}")
        if self.gcs_client:
            try:
                self.bucket.blob(f"checkpoints/{filename}").upload_from_filename(str(local_path))
                print(f"☁️  gs://{self.gcs_bucket}/checkpoints/{filename}")
            except Exception as e:
                print(f"⚠ Upload failed: {e}")
        return str(local_path)
    
    def load_latest(self):
        local_ckpts = sorted(self.local_dir.glob("ckpt_*.pt"))
        if local_ckpts:
            print(f"🔄 Local: {local_ckpts[-1].name}")
            return torch.load(local_ckpts[-1], map_location='cpu')
        if self.gcs_client:
            try:
                blobs = list(self.bucket.list_blobs(prefix="checkpoints/ckpt_"))
                if blobs:
                    latest = max(blobs, key=lambda b: b.time_created)
                    local_path = self.local_dir / latest.name.split('/')[-1]
                    latest.download_to_filename(str(local_path))
                    print(f"🔄 GCS: {latest.name}")
                    return torch.load(local_path, map_location='cpu')
            except Exception as e:
                print(f"⚠ GCS download: {e}")
        return None
    
    def should_stop(self): return self.preemption_requested

sys.path.insert(0, "/ssis")
sys.path.insert(0, "/ssis/ai/collision")

def train_collision(args):
    from train_collision_surrogate import TrainConfig, CollisionGraphDataset, MultiSensorFusionGNN
    from torch.utils.data import DataLoader
    
    cfg = TrainConfig()
    cfg.epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  {device}")
    if device.type == "cuda":
        print(f"   {torch.cuda.get_device_name(0)}")
    
    ckpt = SimpleCheckpointManager(args.gcs_bucket)
    model = MultiSensorFusionGNN(node_dim=cfg.feat_dim, edge_dim=2, hidden=cfg.hidden, heads=cfg.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    start_epoch = 0
    checkpoint = ckpt.load_latest()
    if checkpoint:
        print(f"🔄 Resuming epoch {checkpoint['epoch']}")
        model.load_state_dict(checkpoint['state']['model_state'])
        optimizer.load_state_dict(checkpoint['state']['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
    
    dataset = CollisionGraphDataset(cfg.data_path, cfg.max_nodes, cfg.feat_dim)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    print(f"\n🏃 {start_epoch} → {cfg.epochs}")
    for epoch in range(start_epoch, cfg.epochs):
        if ckpt.should_stop():
            ckpt.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, epoch, True)
            return 1
        
        model.train()
        total_loss = 0.0
        for nodes, adj, iono, y_conj, y_man in train_loader:
            if ckpt.should_stop(): break
            nodes, adj, iono = nodes.to(device), adj.to(device), iono.to(device)
            y_conj, y_man = y_conj.to(device), y_man.to(device)
            risk_alphas, man_logits = model(nodes, adj, iono)
            alpha0 = risk_alphas.sum(dim=1, keepdim=True)
            loss_conj = -(y_conj * (risk_alphas/alpha0).log()).sum(dim=1).mean()
            loss_man = nn.CrossEntropyLoss()(man_logits, y_man)
            loss = loss_conj + 0.5 * loss_man
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"E{epoch+1}/{cfg.epochs} Loss: {total_loss/len(train_loader):.4f}")
        
        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, epoch + 1)
    
    # Export
    model.eval()
    dummy = (torch.zeros(1, cfg.max_nodes, cfg.feat_dim).to(device),
             torch.eye(cfg.max_nodes).unsqueeze(0).to(device),
             torch.zeros(1, 16).to(device))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    onnx_path = Path(args.output_dir) / "collision_multi_task_v2.onnx"
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["nodes", "adj", "iono"],
                      output_names=["risk_alphas", "maneuver_logits"])
    print(f"✓ {onnx_path}")
    
    if ckpt.gcs_client:
        try:
            ckpt.bucket.blob("models/collision_multi_task_v2.onnx").upload_from_filename(str(onnx_path))
            print(f"☁️  Model uploaded")
        except Exception as e:
            print(f"⚠ Upload: {e}")
    return 0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--component', default='collision')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--checkpoint-every', type=int, default=10)
    p.add_argument('--gcs-bucket', default=os.getenv('GCS_CHECKPOINT_BUCKET'))
    p.add_argument('--output-dir', default='/ssis/ai/training/models')
    args = p.parse_args()
    
    print("="*60)
    print("SSIS Spot VM Trainer")
    print(f"Component: {args.component} | Epochs: {args.epochs}")
    print(f"GCS: {args.gcs_bucket or 'None'}")
    print("="*60)
    
    sys.exit(train_collision(args) if args.component == 'collision' else 1)
TRAINER_EOF
    chmod +x scripts/gcp_spot_vm_trainer.py
fi

echo "✓ SSIS ready at /ssis"

# 4. Setup GCS credentials if bucket specified
if [ -n "$GCS_BUCKET" ]; then
    echo ""
    echo "[4/5] Setting up GCS access..."
    
    # Try to get credentials from metadata
    SERVICE_ACCOUNT=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")
    
    if [ -n "$SERVICE_ACCOUNT" ]; then
        echo "Using instance service account: $SERVICE_ACCOUNT"
        export GOOGLE_APPLICATION_DEFAULT=$SERVICE_ACCOUNT
    fi
fi

# 5. Start training
echo ""
echo "[5/5] Starting training..."
cd /ssis

export GCS_CHECKPOINT_BUCKET="$GCS_BUCKET"
export PYTHONUNBUFFERED=1

# Run training with auto-resume loop
MAX_RETRIES=3
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    echo ""
    echo "=== Training attempt $((RETRY+1))/$MAX_RETRIES ==="
    
    if python3 scripts/gcp_spot_vm_trainer.py \
        --component "$COMPONENT" \
        --epochs "$EPOCHS" \
        --checkpoint-every "$CHECKPOINT_EVERY" \
        --gcs-bucket "$GCS_BUCKET"; then
        
        echo "✅ Training completed successfully!"
        
        # Signal completion via metadata
        if [ -n "$GCS_BUCKET" ]; then
            echo "Training complete at $(date)" | gsutil cp - gs://$GCS_BUCKET/status/training-complete-$(hostname).txt
        fi
        
        break
    else
        EXIT_CODE=$?
        echo "⚠️ Training exited with code $EXIT_CODE"
        
        if [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 1 ]; then
            # Interrupted (SIGTERM/SIGINT) - likely spot preemption
            echo "Detected interruption. Will retry after delay..."
            RETRY=$((RETRY+1))
            
            if [ $RETRY -lt $MAX_RETRIES ]; then
                echo "Waiting 30 seconds before retry..."
                sleep 30
                echo "Resuming from checkpoint..."
            fi
        else
            # Other error - don't retry
            echo "Non-recoverable error. Exiting."
            break
        fi
    fi
done

if [ $RETRY -ge $MAX_RETRIES ]; then
    echo "⚠️ Max retries exceeded. Check logs: $LOGFILE"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training Complete - $(date)"
echo "Check GCS for models: gs://$GCS_BUCKET/models/"
echo "=========================================="
