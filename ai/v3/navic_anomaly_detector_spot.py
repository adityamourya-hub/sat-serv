#!/usr/bin/env python3
"""
GKE Spot-Instance Compatible Training for NavIC Anomaly Detector.

Usage:
    GCS_CHECKPOINT_BUCKET=your-bucket \
    python3 ai/v3/navic_anomaly_detector_spot.py \
        --epochs 50 \
        --checkpoint-every 10
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import checkpoint manager
try:
    from gke_checkpoint_manager import GKECheckpointManager, CheckpointConfig, SpotInstanceHandler
except ImportError:
    GKECheckpointManager = None
    CheckpointConfig = None
    SpotInstanceHandler = None


class NavICAnomalyDetector(nn.Module):
    """BiLSTM + MultiHeadAttention for GNSS spoofing/jamming detection."""
    
    def __init__(self, input_dim=12, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 3)  # [normal, jamming, spoofing]
        self.confidence = nn.Linear(64, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_out, dim=1)
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        logits = self.fc2(x)
        confidence = torch.sigmoid(self.confidence(x))
        return logits, confidence


def generate_navic_data(n_samples=10000, seq_len=60, input_dim=12, device='cpu'):
    """Generate synthetic NavIC pseudorange residual data."""
    X = torch.randn(n_samples, seq_len, input_dim, device=device)
    
    # Anomaly injection (30% of samples)
    anomaly_mask = torch.rand(n_samples) < 0.3
    n_anomalies = anomaly_mask.sum().item()
    
    # Jamming: correlated noise spikes
    jamming_idx = anomaly_mask.nonzero()[:n_anomalies // 2]
    for idx in jamming_idx:
        spike_pos = torch.randint(20, seq_len - 20, (1,))
        X[idx, spike_pos:spike_pos+10, :] += torch.randn(1, 10, input_dim) * 5.0
    
    # Spoofing: gradual drift then jump
    spoofing_idx = anomaly_mask.nonzero()[n_anomalies // 2:]
    for idx in spoofing_idx:
        drift = torch.linspace(0, 3, seq_len // 2).unsqueeze(1).expand(-1, input_dim)
        X[idx, seq_len//2:, :] += drift.to(device)
        X[idx, seq_len//2:, :] += torch.randn(1, input_dim).to(device) * 2.0
    
    # Labels: 0=normal, 1=jamming, 2=spoofing
    y = torch.zeros(n_samples, dtype=torch.long, device=device)
    y[jamming_idx] = 1
    y[spoofing_idx] = 2
    
    return X, y


def train_with_checkpoints(args):
    """Training with spot instance support."""
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"🖥  Device: {device}")
    print(f"🎯 Epochs: {args.epochs}")
    
    # Initialize checkpoint manager
    ckpt_mgr = None
    if GKECheckpointManager and args.checkpoint:
        config = CheckpointConfig(
            gcs_bucket=args.gcs_bucket or os.getenv("GCS_CHECKPOINT_BUCKET"),
            gcs_prefix=f"ssis-checkpoints/navic/{args.job_name}",
            local_dir=args.local_checkpoint_dir,
            checkpoint_every_epochs=args.checkpoint_every,
            checkpoint_every_minutes=args.checkpoint_minutes,
            keep_last_n_checkpoints=args.keep_last_n,
            enable_spot_handler=True,
        )
        
        if config.gcs_bucket:
            ckpt_mgr = GKECheckpointManager(config, job_name=args.job_name)
            print(f"💾 Checkpointing to gs://{config.gcs_bucket}/{config.gcs_prefix}")
    
    # Model and optimizer
    model = NavICAnomalyDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Check for resume
    start_epoch = 0
    if ckpt_mgr and (args.resume or args.auto_resume):
        resume_info = ckpt_mgr.get_resume_info()
        if resume_info['can_resume']:
            print(f"🔄 Resuming from epoch {resume_info['start_epoch']}")
            state = resume_info['state']
            if 'model_state' in state:
                model.load_state_dict(state['model_state'])
            if 'optimizer_state' in state:
                optimizer.load_state_dict(state['optimizer_state'])
            start_epoch = resume_info['start_epoch']
    
    # Training data
    X_train, y_train = generate_navic_data(n_samples=args.n_samples, device=device)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Training loop
    print(f"\n🏃 Starting training from epoch {start_epoch}")
    print("=" * 60)
    
    try:
        handler = SpotInstanceHandler(ckpt_mgr) if ckpt_mgr else None
        with handler if handler else None:
            for epoch in range(start_epoch, args.epochs):
                if handler and handler.should_stop():
                    print("\n🛑 Spot preemption! Saving checkpoint...")
                    if ckpt_mgr:
                        ckpt_mgr.emergency_save({
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                        }, epoch)
                    return 1
                
                model.train()
                total_loss = 0.0
                correct = 0
                total = 0
                
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    logits, confidence = model(X_batch)
                    
                    loss = F.cross_entropy(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = logits.max(1)
                    correct += predicted.eq(y_batch).sum().item()
                    total += y_batch.size(0)
                
                acc = 100.0 * correct / total
                avg_loss = total_loss / len(train_loader)
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.2f}%")
                
                # Checkpoint
                if ckpt_mgr and ckpt_mgr.should_checkpoint(epoch):
                    ckpt_mgr.save_checkpoint({
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, epoch, metrics={'loss': avg_loss, 'accuracy': acc})
        
        print("\n✅ Training complete")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        if ckpt_mgr:
            ckpt_mgr.emergency_save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, epoch)
        return 130
    
    # Export
    print("\n📦 Exporting model...")
    model.eval()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PT
    pt_path = output_dir / "navic_detector.pt"
    torch.save(model.state_dict(), pt_path)
    
    # Export ONNX
    dummy = torch.randn(1, 60, 12).to(device)
    onnx_path = output_dir / "navic_detector.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["navic_history"],
        output_names=["logits", "confidence"],
    )
    
    print(f"✓ Saved: {pt_path}")
    print(f"✓ Exported: {onnx_path}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Train NavIC Detector with GKE Spot Support')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output-dir', type=str, default='ai/training/models')
    
    # Checkpoint params
    parser.add_argument('--checkpoint', action='store_true', default=True)
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--checkpoint-minutes', type=int, default=30)
    parser.add_argument('--keep-last-n', type=int, default=3)
    parser.add_argument('--local-checkpoint-dir', type=str, default='/tmp/checkpoints')
    
    # GCS params
    parser.add_argument('--gcs-bucket', type=str, default=os.getenv('GCS_CHECKPOINT_BUCKET'))
    parser.add_argument('--job-name', type=str, default='navic-detector')
    
    # Resume
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--auto-resume', action='store_true', default=True)
    
    args = parser.parse_args()
    
    if args.no_checkpoint:
        args.checkpoint = False
    
    exit_code = train_with_checkpoints(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
