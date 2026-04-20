"""
H100 Foundation Training with W&B Mission Control
ADITI 4.0 Sovereign Distillation Pipeline

This script orchestrates the 72-hour H100 training run with real-time physics monitoring.
Integrates Lie-Group PHS training with Weights & Biases dashboard for audit trail.

72-Hour Phase Strategy:
1. Warmup (4h): Latent disentanglement - SupCon loss tracking
2. Foundation (48h): Dynamics convergence - sub-10m residual target
3. Refinement (20h): SINDy sparsity lock - >85% zeros for explainability

Physics Metrics Monitored:
- Energy drift: Normalized |H(60s) - H(0)| / H(0) < 1e-7
- Manifold orthogonality: ||R^T R - I||_F + ||q|| - 1| < 1e-9
- SINDy sparsity: % of zero coefficients in control matrix Xi
- Validation residual: L2 tracking error in meters

Symbolic Certificate Integration:
- Dashboard state hashed and embedded in certificate.Provenance
- Direct link from M2 Air inference to H100 training proof
- Audit trail from live telemetry to sovereign model

Usage:
    # On H100 head node
    python train_with_monitoring.py \
        --phase foundation \
        --nodes 8 \
        --batch-size 4096 \
        --wandb-project iDEX-ADITI-4.0-SDA

    # Resume from checkpoint
    python train_with_monitoring.py \
        --resume run_id:abc123 \
        --checkpoint checkpoints/epoch_500.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
from pathlib import Path

# Import Lie-Group components
from lie_integrator import unified_lie_step
from lie_phs import LiePHSLayer
from rodrigues import verify_so3_properties

# Import W&B monitoring
from wandb_setup import init_sovereign_mission_control, get_certificate_hash

# Import training utilities
import sys
sys.path.append('..')
from neural_ode.train import (
    CanonicalTrajectoryDataset, 
    collate_fn,
    SymplecticAugmentedLoss
)


class SovereignTrainer:
    """
    H100 Foundation trainer with real-time physics monitoring.
    
    Combines Lie-Group PHS training with W&B Mission Control dashboard.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = LiePHSLayer(
            latent_dim=config["latent_dim"],
            hidden_dim=config["hidden_dim"]
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # Loss function
        self.criterion = SymplecticAugmentedLoss(
            lambda_symp_init=config.get("symplectic_lambda", 1e-5),
            symp_ramp_epochs=50
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_residual = float("inf")
        
        # Initialize W&B logger
        self.logger = init_sovereign_mission_control(
            run_name=config.get("run_name"),
            config_overrides=config,
            resume=config.get("resume", False)
        )
        
        print(f"🚀 Sovereign Trainer initialized on {self.device}")
        print(f"   Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"   Precision: {config.get('precision', 'float32')}")
        print(f"   H100 Nodes: {config.get('h100_nodes', 1)}")
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """
        Train for one epoch with Lie-Group symplectic integration.
        
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            x0 = batch['initial_state'].to(self.device).requires_grad_(True)
            true_states = batch['states'].to(self.device)
            times = batch['times'][0].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Lie-Group Symplectic Integration
            dt = times[1] - times[0]
            current_x = x0
            preds = [current_x]
            
            # Integration loop with unified Lie-Group step
            regime = "S2"  # HGV uses S² manifold
            
            for step in range(1, true_states.shape[1]):
                current_x = unified_lie_step(
                    self.model,
                    current_x,
                    z_latent=None,  # Would come from SupCon
                    dt=dt,
                    regime=regime
                )
                preds.append(current_x)
                
                # Periodic physics logging (every 100 steps)
                if step % 100 == 0:
                    self.logger.log_physics_step(
                        self.current_epoch * len(train_loader) + batch_idx,
                        self.model,
                        current_x
                    )
            
            pred_states = torch.stack(preds, dim=1)
            
            # Compute loss
            loss, loss_metrics = self.criterion(
                self.model, pred_states, true_states, x0, dt
            )
            
            # Backprop
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.criterion.ramp_symplectic_weight()
            
            # Track metrics
            total_loss += loss.item() * x0.shape[0]
            total_samples += x0.shape[0]
            
            # Log batch progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.6f}")
        
        avg_loss = total_loss / total_samples
        
        return {
            "train/loss_total": avg_loss,
            "train/loss_mse": loss_metrics.get("obs", 0),
            "train/loss_physics": loss_metrics.get("energy_violation", 0),
        }
    
    def validate(self, val_loader: DataLoader) -> dict:
        """
        Validate on held-out trajectories.
        
        Returns:
            metrics: Dictionary of validation metrics including L2 residual
        """
        self.model.eval()
        total_residual = 0
        total_samples = 0
        max_residual = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x0 = batch['initial_state'].to(self.device)
                true_states = batch['states'].to(self.device)
                times = batch['times'][0].to(self.device)
                
                dt = times[1] - times[0]
                current_x = x0
                
                # Predict trajectory
                regime = "S2"
                for step in range(1, true_states.shape[1]):
                    current_x = unified_lie_step(
                        self.model, current_x, None, dt, regime
                    )
                
                # Compute L2 residual in meters
                # Convert from canonical units (DU = 6378.137 km)
                DU_METERS = 6378137.0
                residual = torch.norm(current_x - true_states[:, -1], dim=-1)
                residual_meters = residual * DU_METERS
                
                total_residual += residual_meters.sum().item()
                max_residual = max(max_residual, residual_meters.max().item())
                total_samples += x0.shape[0]
        
        avg_residual = total_residual / total_samples
        
        return {
            "val/l2_residual_meters": avg_residual,
            "val/max_error_meters": max_residual,
        }
    
    def check_manifold_integrity(self) -> dict:
        """
        Run manifold integrity check and log results.
        
        Returns:
            report: Manifold integrity report dictionary
        """
        # Mock initial state for check
        initial_state = {
            'q': torch.tensor([[1.0, 0.0, 0.0]], device=self.device),
            'p': torch.tensor([[0.0, 1.0, 0.0]], device=self.device),
            'R': torch.eye(3, device=self.device).unsqueeze(0),
            'omega': torch.tensor([[0.0, 0.0, 0.1]], device=self.device)
        }
        
        # Run quick integrity check (100 steps instead of 10k)
        from check_manifold import check_manifold_integrity
        
        report = check_manifold_integrity(
            model=self.model,
            initial_state=initial_state,
            steps=100,  # Quick check during training
            dt=0.01,
            tol=1e-7,
            log_interval=100,
            device=self.device,
            save_report=False
        )
        
        return report.to_dict()
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """
        Full training loop with W&B monitoring.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Total training epochs
        """
        print(f"\n{'='*70}")
        print(f"  H100 FOUNDATION TRAINING - ADITI 4.0")
        print(f"  Epochs: {epochs} | Target: sub-10m residual")
        print(f"{'='*70}\n")
        
        for epoch in range(self.current_epoch, epochs):
            epoch_start = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase (every 5 epochs)
            if epoch % 5 == 0:
                val_metrics = self.validate(val_loader)
                
                # Check manifold integrity
                manifold_report = self.check_manifold_integrity()
                
                # Combine all metrics
                all_metrics = {**train_metrics, **val_metrics}
                
                # Log to W&B
                self.logger.log_epoch(epoch, train_metrics["train/loss_total"], 
                                     self.model, None, self.optimizer)
                self.logger.log_manifold_integrity(epoch, manifold_report)
                
                # Update best model
                if val_metrics["val/l2_residual_meters"] < self.best_val_residual:
                    self.best_val_residual = val_metrics["val/l2_residual_meters"]
                    self.save_checkpoint(epoch, is_best=True)
                    
                    print(f"\n🎯 New best model! Residual: {self.best_val_residual:.2f}m")
            
            # Regular checkpoint
            if epoch % 50 == 0 and epoch > 0:
                self.save_checkpoint(epoch)
            
            # Progress
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{epochs} complete in {epoch_time:.1f}s")
            print(f"  Train Loss: {train_metrics['train/loss_total']:.6f}")
            
            self.current_epoch = epoch + 1
        
        # Finalize
        self.finalize()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            path = checkpoint_dir / "best_model.pt"
        else:
            path = checkpoint_dir / f"epoch_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_residual': self.best_val_residual,
            'config': self.config,
        }, path)
        
        print(f"  Checkpoint saved: {path}")
    
    def finalize(self):
        """Finalize training and generate certificate hash."""
        # Save final model
        final_path = Path(self.config.get("output_dir", "models")) / "sovereign_phv_final.pt"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'final_epoch': self.current_epoch,
            'best_val_residual': self.best_val_residual,
        }, final_path)
        
        # Generate certificate hash
        certificate_hash = get_certificate_hash(self.logger.logger.run.id if hasattr(self.logger, 'logger') else 'mock')
        
        # Finalize W&B
        self.logger.finalize(str(final_path), certificate_hash)
        
        print(f"\n{'='*70}")
        print(f"  TRAINING COMPLETE")
        print(f"  Final Model: {final_path}")
        print(f"  Certificate Hash: {certificate_hash}")
        print(f"  Best Residual: {self.best_val_residual:.2f}m")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="H100 Foundation Training - ADITI 4.0 Sovereign Distillation"
    )
    
    # Data args
    parser.add_argument("--train-data", type=str, required=True,
                       help="Path to training trajectory data")
    parser.add_argument("--val-data", type=str, required=True,
                       help="Path to validation trajectory data")
    
    # Model args
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    
    # Training args
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dt", type=float, default=0.01,
                       help="Integration time step (100Hz = 0.01s)")
    
    # Physics args
    parser.add_argument("--symplectic-lambda", type=float, default=1e-5)
    parser.add_argument("--manifold-tol", type=float, default=1e-9)
    parser.add_argument("--energy-tol", type=float, default=1e-7)
    
    # System args
    parser.add_argument("--h100-nodes", type=int, default=8)
    parser.add_argument("--precision", type=str, default="bfloat16-mixed",
                       choices=["float32", "float16", "bfloat16-mixed"])
    
    # W&B args
    parser.add_argument("--wandb-project", type=str, default="iDEX-ADITI-4.0-SDA")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    
    # Output args
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="models")
    
    args = parser.parse_args()
    
    # Build config
    config = vars(args)
    config["target_l2_residual_m"] = 10.0
    config["target_energy_drift"] = 1e-7
    config["sindy_sparsity_target"] = 0.85
    
    # Create data loaders (mock for now)
    # In production, these would load actual trajectory data
    print("⚠️  Mock data loaders - replace with actual trajectory data")
    
    # Create trainer
    trainer = SovereignTrainer(config)
    
    # Mock loaders for demonstration
    mock_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 6),  # initial states
        torch.randn(100, 100, 6)  # trajectories
    )
    mock_loader = DataLoader(mock_dataset, batch_size=config["batch_size"])
    
    # Train
    trainer.train(mock_loader, mock_loader, config["epochs"])


if __name__ == "__main__":
    main()
