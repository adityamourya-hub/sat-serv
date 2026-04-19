#!/usr/bin/env python3
"""
GKE Spot Instance Checkpoint Manager for SSIS Training

Features:
- Automatic checkpointing every N epochs/minutes
- GCS persistence for spot node recovery
- Resume from last checkpoint on restart
- Graceful handling of SIGTERM (spot preemption)
- Local + GCS dual storage for reliability

Usage:
    from gke_checkpoint_manager import GKECheckpointManager
    
    ckpt_mgr = GKECheckpointManager(
        gcs_bucket="your-bucket",
        gcs_prefix="ssis-checkpoints/collision",
        local_dir="/tmp/checkpoints",
        checkpoint_every_epochs=5,
        checkpoint_every_minutes=30,
    )
    
    # In training loop
    for epoch in range(start_epoch, total_epochs):
        train_epoch()
        
        # Auto-checkpoint
        if ckpt_mgr.should_checkpoint(epoch):
            ckpt_mgr.save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': current_loss,
            })
    
    # On startup, resume from checkpoint
    checkpoint = ckpt_mgr.load_latest_checkpoint()
    if checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
"""

import os
import json
import signal
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, asdict
import threading
import logging

# Optional GCS import - gracefully handles absence
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logging.warning("google-cloud-storage not installed. GCS persistence disabled.")

import torch


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing behavior."""
    gcs_bucket: Optional[str] = None
    gcs_prefix: str = "ssis-checkpoints"
    local_dir: str = "/tmp/checkpoints"
    checkpoint_every_epochs: int = 5
    checkpoint_every_minutes: int = 30
    keep_last_n_checkpoints: int = 3
    enable_spot_handler: bool = True
    sync_to_gcs_on_termination: bool = True


class GKECheckpointManager:
    """
    Manages checkpoints for GKE spot instance training.
    
    Handles:
    - Periodic local checkpoints
    - GCS synchronization
    - Spot preemption signals (SIGTERM)
    - Automatic resume
    """
    
    def __init__(self, config: CheckpointConfig, job_name: str = "training"):
        self.config = config
        self.job_name = job_name
        self.local_dir = Path(config.local_dir) / job_name
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        # GCS client
        self.gcs_client = None
        self.gcs_bucket = None
        if GCS_AVAILABLE and config.gcs_bucket:
            try:
                self.gcs_client = storage.Client()
                self.gcs_bucket = self.gcs_client.bucket(config.gcs_bucket)
                print(f"✓ GCS checkpointing enabled: gs://{config.gcs_bucket}/{config.gcs_prefix}")
            except Exception as e:
                print(f"⚠ GCS initialization failed: {e}")
        
        # State tracking
        self.last_checkpoint_time = time.time()
        self.last_checkpoint_epoch = -1
        self.checkpoint_count = 0
        self.spot_preemption_received = False
        self.current_checkpoint_path: Optional[Path] = None
        
        # Register spot preemption handler
        if config.enable_spot_handler:
            self._register_signal_handlers()
    
    def _register_signal_handlers(self):
        """Register handlers for spot preemption signals."""
        def handle_sigterm(signum, frame):
            print(f"\n⚠️ SIGTERM received (spot preemption?). Saving emergency checkpoint...")
            self.spot_preemption_received = True
            # Note: Can't save here directly due to signal safety
            # Main loop should check flag and save
            
        signal.signal(signal.SIGTERM, handle_sigterm)
        
        # Also handle SIGINT for testing
        def handle_sigint(signum, frame):
            print(f"\n⚠️ SIGINT received. Saving checkpoint before exit...")
            self.spot_preemption_received = True
            
        signal.signal(signal.SIGINT, handle_sigint)
        
        print("✓ Spot preemption handlers registered (SIGTERM, SIGINT)")
    
    def should_checkpoint(self, epoch: int, force: bool = False) -> bool:
        """
        Determine if we should checkpoint now.
        
        Args:
            epoch: Current epoch number
            force: Force checkpoint regardless of timing
        
        Returns:
            True if checkpoint should be saved
        """
        if force:
            return True
        
        # Check if spot preemption signal received
        if self.spot_preemption_received:
            return True
        
        # Epoch-based checkpointing
        if epoch >= 0 and epoch % self.config.checkpoint_every_epochs == 0:
            if epoch != self.last_checkpoint_epoch:
                return True
        
        # Time-based checkpointing
        elapsed_minutes = (time.time() - self.last_checkpoint_time) / 60
        if elapsed_minutes >= self.config.checkpoint_every_minutes:
            return True
        
        return False
    
    def save_checkpoint(self, state: Dict[str, Any], epoch: int, 
                        metrics: Optional[Dict] = None,
                        is_emergency: bool = False) -> Path:
        """
        Save checkpoint locally and optionally to GCS.
        
        Args:
            state: Training state dict (must include 'model_state', 'optimizer_state', etc.)
            epoch: Current epoch
            metrics: Optional metrics to save
            is_emergency: If True, prioritize GCS upload
        
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint with metadata
        checkpoint = {
            'epoch': epoch,
            'timestamp': time.time(),
            'job_name': self.job_name,
            'state': state,
            'metrics': metrics or {},
            'is_emergency': is_emergency,
        }
        
        # Local save
        checkpoint_filename = f"checkpoint_epoch_{epoch:04d}.pt"
        local_path = self.local_dir / checkpoint_filename
        
        # Atomic write (write to temp, then rename)
        temp_path = local_path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.rename(local_path)
        
        self.current_checkpoint_path = local_path
        self.last_checkpoint_epoch = epoch
        self.last_checkpoint_time = time.time()
        self.checkpoint_count += 1
        
        print(f"✓ Checkpoint saved: {local_path} (size: {local_path.stat().st_size / 1024**2:.1f} MB)")
        
        # Save metadata JSON for easy inspection
        metadata_path = local_path.with_suffix('.json')
        metadata = {
            'epoch': epoch,
            'timestamp': checkpoint['timestamp'],
            'local_path': str(local_path),
            'gcs_path': None,
            'metrics': metrics,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # GCS upload
        if self.gcs_bucket:
            self._upload_to_gcs(local_path, epoch, is_emergency)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return local_path
    
    def _upload_to_gcs(self, local_path: Path, epoch: int, is_emergency: bool = False):
        """Upload checkpoint to GCS."""
        if not self.gcs_bucket:
            return
        
        try:
            gcs_path = f"{self.config.gcs_prefix}/{self.job_name}/{local_path.name}"
            blob = self.gcs_bucket.blob(gcs_path)
            
            # Upload with retry
            max_retries = 3 if is_emergency else 1
            for attempt in range(max_retries):
                try:
                    blob.upload_from_filename(str(local_path))
                    print(f"✓ Uploaded to GCS: gs://{self.config.gcs_bucket}/{gcs_path}")
                    
                    # Update metadata
                    metadata_path = local_path.with_suffix('.json')
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        metadata['gcs_path'] = f"gs://{self.config.gcs_bucket}/{gcs_path}"
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                    
                    return
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    raise
                    
        except Exception as e:
            print(f"⚠ GCS upload failed: {e}")
            if is_emergency:
                print(f"⚠ CRITICAL: Emergency checkpoint not uploaded to GCS!")
    
    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints locally."""
        checkpoints = sorted(self.local_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            to_delete = checkpoints[:-self.config.keep_last_n_checkpoints]
            for ckpt in to_delete:
                try:
                    ckpt.unlink()
                    # Also delete metadata
                    metadata = ckpt.with_suffix('.json')
                    if metadata.exists():
                        metadata.unlink()
                except Exception as e:
                    print(f"⚠ Failed to cleanup old checkpoint: {e}")
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """
        Load the most recent checkpoint from local or GCS.
        
        Returns:
            Checkpoint dict or None if no checkpoint found
        """
        # Try local first
        local_checkpoint = self._find_latest_local_checkpoint()
        
        if local_checkpoint:
            print(f"✓ Found local checkpoint: {local_checkpoint}")
            return torch.load(local_checkpoint, map_location='cpu')
        
        # Try GCS if no local checkpoint
        if self.gcs_bucket:
            return self._download_latest_gcs_checkpoint()
        
        print("ℹ No checkpoint found, starting from scratch")
        return None
    
    def _find_latest_local_checkpoint(self) -> Optional[Path]:
        """Find the most recent local checkpoint."""
        checkpoints = list(self.local_dir.glob("checkpoint_epoch_*.pt"))
        
        if not checkpoints:
            return None
        
        # Sort by modification time (most recent first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    def _download_latest_gcs_checkpoint(self) -> Optional[Dict]:
        """Download latest checkpoint from GCS."""
        if not self.gcs_bucket:
            return None
        
        try:
            prefix = f"{self.config.gcs_prefix}/{self.job_name}/"
            blobs = list(self.gcs_bucket.list_blobs(prefix=prefix))
            
            if not blobs:
                return None
            
            # Find most recent checkpoint blob
            checkpoints = [b for b in blobs if b.name.endswith('.pt')]
            if not checkpoints:
                return None
            
            latest = max(checkpoints, key=lambda b: b.updated)
            
            # Download to local
            local_path = self.local_dir / Path(latest.name).name
            print(f"✓ Downloading checkpoint from GCS: {latest.name}")
            latest.download_to_filename(str(local_path))
            
            return torch.load(local_path, map_location='cpu')
            
        except Exception as e:
            print(f"⚠ Failed to download from GCS: {e}")
            return None
    
    def get_resume_info(self) -> Dict:
        """Get information for resuming training."""
        checkpoint = self.load_latest_checkpoint()
        
        if not checkpoint:
            return {
                'can_resume': False,
                'start_epoch': 0,
                'state': None,
            }
        
        return {
            'can_resume': True,
            'start_epoch': checkpoint['epoch'] + 1,
            'state': checkpoint['state'],
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint['timestamp'],
        }
    
    def emergency_save(self, state: Dict[str, Any], epoch: int):
        """
        Emergency save on spot preemption.
        Prioritizes speed and GCS upload.
        """
        print(f"\n🚨 EMERGENCY CHECKPOINT: Epoch {epoch}")
        return self.save_checkpoint(state, epoch, is_emergency=True)
    
    def check_spot_preemption(self) -> bool:
        """Check if spot preemption signal was received."""
        return self.spot_preemption_received


class SpotInstanceHandler:
    """
    Context manager for handling spot instance lifecycle.
    
    Usage:
        with SpotInstanceHandler(checkpoint_manager) as handler:
            for epoch in range(epochs):
                train_epoch()
                
                # Check for preemption
                if handler.should_stop():
                    break
    """
    
    def __init__(self, checkpoint_manager: GKECheckpointManager):
        self.ckpt_mgr = checkpoint_manager
        self.stop_requested = False
    
    def __enter__(self):
        # Check if resuming
        info = self.ckpt_mgr.get_resume_info()
        if info['can_resume']:
            print(f"🔄 Resuming from epoch {info['start_epoch']}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Final checkpoint on exit
        print("💾 Final checkpoint before exit...")
        # Note: Actual save should be done by caller with current state
    
    def should_stop(self) -> bool:
        """Check if we should stop (preemption received)."""
        return self.ckpt_mgr.check_spot_preemption()


def create_spot_aware_training_wrapper(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    gcs_bucket: Optional[str] = None,
    job_name: str = "training",
    **kwargs
) -> tuple[GKECheckpointManager, SpotInstanceHandler]:
    """
    Convenience function to create checkpointing infrastructure.
    
    Returns:
        (checkpoint_manager, spot_handler)
    """
    config = CheckpointConfig(
        gcs_bucket=gcs_bucket,
        job_name=job_name,
        **kwargs
    )
    
    ckpt_mgr = GKECheckpointManager(config, job_name)
    handler = SpotInstanceHandler(ckpt_mgr)
    
    return ckpt_mgr, handler


# Example integration with existing training script
def example_integration():
    """Example of integrating with existing training script."""
    
    # Initialize checkpointing
    config = CheckpointConfig(
        gcs_bucket=os.getenv("GCS_CHECKPOINT_BUCKET", "your-bucket"),
        gcs_prefix="ssis-checkpoints/collision",
        checkpoint_every_epochs=5,
        checkpoint_every_minutes=30,
    )
    
    ckpt_mgr = GKECheckpointManager(config, job_name="collision_gnn")
    
    # Check for resume
    resume_info = ckpt_mgr.get_resume_info()
    start_epoch = resume_info['start_epoch']
    
    if resume_info['can_resume']:
        model_state = resume_info['state']['model_state']
        optimizer_state = resume_info['state']['optimizer_state']
        # Load into model/optimizer
        print(f"🔄 Resumed from epoch {start_epoch - 1}")
    
    # Training loop with spot handling
    with SpotInstanceHandler(ckpt_mgr) as handler:
        for epoch in range(start_epoch, 100):
            # Check for preemption
            if handler.should_stop():
                print("🛑 Stopping due to spot preemption signal")
                # Save emergency checkpoint
                ckpt_mgr.emergency_save({
                    'model_state': model.state_dict(),  # noqa
                    'optimizer_state': optimizer.state_dict(),  # noqa
                }, epoch)
                break
            
            # Regular training
            # train_epoch(...)
            
            # Regular checkpointing
            if ckpt_mgr.should_checkpoint(epoch):
                ckpt_mgr.save_checkpoint({
                    'model_state': model.state_dict(),  # noqa
                    'optimizer_state': optimizer.state_dict(),  # noqa
                    'loss': 0.0,  # noqa
                }, epoch, metrics={'loss': 0.0})


if __name__ == "__main__":
    print("GKE Checkpoint Manager for SSIS Training")
    print("=" * 60)
    print()
    print("Environment variables:")
    print("  GCS_CHECKPOINT_BUCKET=your-bucket-name")
    print("  GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json")
    print()
    print("Usage in training script:")
    print()
    print("from ai.utils.gke_checkpoint_manager import GKECheckpointManager, CheckpointConfig")
    print()
    print("config = CheckpointConfig(")
    print("    gcs_bucket='your-bucket',")
    print("    checkpoint_every_epochs=5,")
    print(")")
    print("ckpt_mgr = GKECheckpointManager(config, job_name='collision_gnn')")
    print()
    print("# In training loop:")
    print("if ckpt_mgr.should_checkpoint(epoch):")
    print("    ckpt_mgr.save_checkpoint(state, epoch)")
