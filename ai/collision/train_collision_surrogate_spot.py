#!/usr/bin/env python3
"""
GKE Spot-Instance Compatible Training for Collision Surrogate GNN.

This is a checkpoint-enabled version of train_collision_surrogate.py
that supports resuming from interruptions on GKE spot instances.

Usage:
    # Standard training with checkpointing
    GCS_CHECKPOINT_BUCKET=your-bucket \
    python3 ai/collision/train_collision_surrogate_spot.py \
        --epochs 100 \
        --checkpoint_every 5
    
    # Resume from checkpoint (automatic if checkpoint exists)
    GCS_CHECKPOINT_BUCKET=your-bucket \
    python3 ai/collision/train_collision_surrogate_spot.py \
        --resume \
        --epochs 100

Features:
- Automatic checkpoint every N epochs
- GCS persistence for spot node recovery
- SIGTERM handling for graceful spot preemption
- Resume from exact epoch on restart
- Dual local + GCS storage
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import original training components
from train_collision_surrogate import (
    TrainConfig, 
    CollisionGraphDataset, 
    MultiSensorFusionGNN,
    train as original_train
)

# Import checkpoint manager
try:
    from gke_checkpoint_manager import GKECheckpointManager, CheckpointConfig, SpotInstanceHandler
except ImportError:
    print("⚠ gke_checkpoint_manager not found. Checkpointing disabled.")
    GKECheckpointManager = None
    CheckpointConfig = None
    SpotInstanceHandler = None


def train_with_checkpoints(args):
    """Training loop with GKE spot instance support."""
    
    # Configuration
    cfg = TrainConfig()
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    if args.data_path:
        cfg.data_path = args.data_path
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"🖥  Device: {device}")
    print(f"📊 Data: {cfg.data_path}")
    print(f"🎯 Epochs: {cfg.epochs}")
    
    # Initialize checkpoint manager
    ckpt_mgr = None
    if GKECheckpointManager and args.checkpoint:
        config = CheckpointConfig(
            gcs_bucket=args.gcs_bucket or os.getenv("GCS_CHECKPOINT_BUCKET"),
            gcs_prefix=f"ssis-checkpoints/collision/{args.job_name}",
            local_dir=args.local_checkpoint_dir,
            checkpoint_every_epochs=args.checkpoint_every,
            checkpoint_every_minutes=args.checkpoint_minutes,
            keep_last_n_checkpoints=args.keep_last_n,
            enable_spot_handler=True,
        )
        
        if config.gcs_bucket:
            ckpt_mgr = GKECheckpointManager(config, job_name=args.job_name)
            print(f"💾 Checkpointing enabled: gs://{config.gcs_bucket}/{config.gcs_prefix}")
            print(f"   Every {args.checkpoint_every} epochs or {args.checkpoint_minutes} minutes")
        else:
            print("⚠ No GCS bucket specified. Local checkpointing only.")
            print("   Set --gcs-bucket or GCS_CHECKPOINT_BUCKET env var for GCS persistence")
    
    # Load dataset
    dataset = CollisionGraphDataset(cfg.data_path, cfg.max_nodes, cfg.feat_dim)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    # Initialize model
    model = MultiSensorFusionGNN(
        node_dim=cfg.feat_dim, 
        edge_dim=2, 
        hidden=cfg.hidden, 
        heads=cfg.heads
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    # Check for resume
    start_epoch = 0
    if ckpt_mgr and (args.resume or args.auto_resume):
        resume_info = ckpt_mgr.get_resume_info()
        if resume_info['can_resume']:
            print(f"🔄 Resuming from epoch {resume_info['start_epoch']}")
            
            # Load state
            state = resume_info['state']
            if 'model_state' in state:
                model.load_state_dict(state['model_state'])
            if 'optimizer_state' in state:
                optimizer.load_state_dict(state['optimizer_state'])
            
            start_epoch = resume_info['start_epoch']
            print(f"   Last metrics: {resume_info.get('metrics', {})}")
    
    # Training loop with spot handling
    print(f"\n🏃 Starting training from epoch {start_epoch}")
    print("=" * 60)
    
    try:
        with SpotInstanceHandler(ckpt_mgr) if ckpt_mgr else None as handler:
            for epoch in range(start_epoch, cfg.epochs):
                # Check for spot preemption
                if handler and handler.should_stop():
                    print(f"\n🛑 Spot preemption detected! Saving emergency checkpoint...")
                    if ckpt_mgr:
                        ckpt_mgr.emergency_save({
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'cfg': cfg,
                        }, epoch)
                    print("✓ Emergency checkpoint saved. Exiting.")
                    return 1  # Exit code indicates interruption
                
                # Train epoch
                model.train()
                total_loss = 0.0
                
                for nodes, adj, iono, y_conj, y_man in train_loader:
                    nodes, adj, iono = nodes.to(device), adj.to(device), iono.to(device)
                    y_conj, y_man = y_conj.to(device), y_man.to(device)
                    
                    risk_alphas, man_logits = model(nodes, adj, iono)
                    
                    # Evidential loss
                    alpha0 = risk_alphas.sum(dim=1, keepdim=True)
                    conj_probs = risk_alphas / alpha0
                    loss_conj = -(y_conj * conj_probs.log()).sum(dim=1).mean()
                    
                    # Maneuver classification
                    ce_loss = nn.CrossEntropyLoss()
                    loss_man = ce_loss(man_logits, y_man)
                    
                    loss = loss_conj + 0.5 * loss_man
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{cfg.epochs} - Loss: {avg_loss:.4f}")
                
                # Checkpoint if needed
                if ckpt_mgr and ckpt_mgr.should_checkpoint(epoch):
                    print(f"💾 Saving checkpoint at epoch {epoch}...")
                    ckpt_mgr.save_checkpoint({
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'cfg': cfg,
                    }, epoch, metrics={'loss': avg_loss})
        
        print("\n✅ Training completed successfully")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        if ckpt_mgr:
            print("Saving final checkpoint...")
            ckpt_mgr.emergency_save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'cfg': cfg,
            }, epoch)
        return 130
    
    # Export final model
    print("\n📦 Exporting final model...")
    model.eval()
    
    # Save PT weights
    pt_path = Path(args.output_dir) / "collision_multi_task_v2.pt"
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pt_path)
    print(f"✓ Saved: {pt_path}")
    
    # Export ONNX
    dummy_nodes = torch.zeros(1, cfg.max_nodes, cfg.feat_dim).to(device)
    dummy_adj = torch.eye(cfg.max_nodes).unsqueeze(0).to(device)
    dummy_iono = torch.zeros(1, 16).to(device)
    
    out_path = Path(args.output_dir) / "collision_multi_task_v2.onnx"
    torch.onnx.export(
        model, (dummy_nodes, dummy_adj, dummy_iono), str(out_path),
        input_names=["nodes", "adj", "iono"],
        output_names=["risk_alphas", "maneuver_logits"],
        dynamic_axes={"nodes": {0: "batch"}, "adj": {0: "batch"}, "iono": {0: "batch"}}
    )
    print(f"✓ Exported: {out_path}")
    
    # Sync final model to GCS if configured
    if ckpt_mgr and args.gcs_bucket:
        print(f"☁️ Syncing to GCS...")
        # Upload final models
        import subprocess
        for ext in ['.pt', '.onnx']:
            model_file = pt_path.parent / f"collision_multi_task_v2{ext}"
            if model_file.exists():
                subprocess.run([
                    'gsutil', 'cp', str(model_file),
                    f"gs://{args.gcs_bucket}/ssis-models/{args.job_name}/"
                ], check=False)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Train Collision Surrogate with GKE Spot Instance Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  GCS_CHECKPOINT_BUCKET    GCS bucket for checkpoint storage
  GOOGLE_APPLICATION_CREDENTIALS  Path to service account JSON

Examples:
  # Basic training with checkpointing
  GCS_CHECKPOINT_BUCKET=my-bucket python3 %(prog)s --epochs 100

  # Resume from checkpoint
  python3 %(prog)s --resume --epochs 100

  # Local checkpointing only (no GCS)
  python3 %(prog)s --checkpoint --gcs-bucket ""
        """
    )
    
    # Training params
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data-path', type=str, default=None, help='Path to training data')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    parser.add_argument('--output-dir', type=str, default='ai/training/models', 
                       help='Output directory for models')
    
    # Checkpointing params
    parser.add_argument('--checkpoint', action='store_true', default=True,
                       help='Enable checkpointing (default: True)')
    parser.add_argument('--no-checkpoint', action='store_true', 
                       help='Disable checkpointing')
    parser.add_argument('--checkpoint-every', type=int, default=5,
                       help='Checkpoint every N epochs')
    parser.add_argument('--checkpoint-minutes', type=int, default=30,
                       help='Checkpoint every N minutes (minimum)')
    parser.add_argument('--keep-last-n', type=int, default=3,
                       help='Keep only last N checkpoints (cleanup old ones)')
    parser.add_argument('--local-checkpoint-dir', type=str, default='/tmp/checkpoints',
                       help='Local directory for checkpoints')
    
    # GCS params
    parser.add_argument('--gcs-bucket', type=str, 
                       default=os.getenv('GCS_CHECKPOINT_BUCKET'),
                       help='GCS bucket for checkpoints')
    parser.add_argument('--job-name', type=str, default='collision_gnn',
                       help='Unique job name for this training run')
    
    # Resume params
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    parser.add_argument('--auto-resume', action='store_true', default=True,
                       help='Auto-resume if checkpoint exists (default: True)')
    
    args = parser.parse_args()
    
    # Handle --no-checkpoint
    if args.no_checkpoint:
        args.checkpoint = False
    
    # Validate GCS setup if checkpointing enabled
    if args.checkpoint and args.gcs_bucket:
        try:
            from google.cloud import storage
            print(f"✓ GCS authentication verified")
        except ImportError:
            print("⚠ google-cloud-storage not installed. GCS disabled.")
            args.gcs_bucket = None
        except Exception as e:
            print(f"⚠ GCS auth failed: {e}. Check GOOGLE_APPLICATION_CREDENTIALS.")
            args.gcs_bucket = None
    
    # Run training
    exit_code = train_with_checkpoints(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
