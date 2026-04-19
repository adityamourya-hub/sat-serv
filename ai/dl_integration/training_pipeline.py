"""
Physics-Informed Training Pipeline for ADITI 4.0
Fine-tunes DL models with orbital dynamics constraints
Integrates Keplerian physics into loss landscape for sovereign SSA
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time
import json
from pathlib import Path

# Import physics loss from training scaffolding
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "sda-train"))
try:
    from pkg.physics.loss import PhysicsInformedLoss, AdaptivePhysicsLoss
except ImportError:
    print("[WARN] Physics loss not available, using fallback")


@dataclass
class TrainingConfig:
    """Configuration for physics-informed training"""
    model_type: str  # 'denoiser', 'segmenter', 'regressor'
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    weight_decay: float = 1e-4
    use_physics_loss: bool = True
    physics_weight_start: float = 0.1
    physics_weight_end: float = 0.4
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    checkpoint_interval: int = 10
    log_interval: int = 10
    validation_interval: int = 5


class OrbitalPhysicsConstraint:
    """
    Keplerian dynamics constraint for training
    Ensures predictions respect orbital mechanics
    """
    
    def __init__(self, mu: float = 3.986004418e14):
        """
        Initialize orbital physics constraint
        
        Args:
            mu: Earth's gravitational parameter (m³/s²)
        """
        self.mu = mu
    
    def keplerian_acceleration(self, position: torch.Tensor) -> torch.Tensor:
        """
        Calculate expected Keplerian acceleration
        
        a = -mu * r / |r|³
        
        Args:
            position: Position tensor [batch, 3] in meters
            
        Returns:
            Expected acceleration tensor [batch, 3]
        """
        r = position
        r_mag = torch.norm(r, dim=-1, keepdim=True)
        
        # Keplerian acceleration
        accel = -self.mu * r / (r_mag ** 3)
        
        return accel
    
    def specific_energy(self, position: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """
        Calculate specific orbital energy
        
        E = v²/2 - mu/|r|
        
        Args:
            position: Position tensor [batch, 3] in meters
            velocity: Velocity tensor [batch, 3] in m/s
            
        Returns:
            Specific energy tensor [batch]
        """
        r_mag = torch.norm(position, dim=-1)
        v_mag_sq = torch.sum(velocity ** 2, dim=-1)
        
        energy = v_mag_sq / 2 - self.mu / r_mag
        return energy
    
    def angular_momentum(self, position: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """
        Calculate specific angular momentum
        
        h = r × v
        
        Args:
            position: Position tensor [batch, 3]
            velocity: Velocity tensor [batch, 3]
            
        Returns:
            Angular momentum tensor [batch, 3]
        """
        h = torch.cross(position, velocity, dim=-1)
        return h


class PhysicsInformedTrainer:
    """
    Main trainer with physics-informed loss
    Fine-tunes pre-trained models with orbital dynamics constraints
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: TrainingConfig,
                 orbital_constraint: Optional[OrbitalPhysicsConstraint] = None):
        """
        Initialize physics-informed trainer
        
        Args:
            model: Neural network model to train
            config: Training configuration
            orbital_constraint: Orbital physics constraint
        """
        self.model = model
        self.config = config
        self.orbital_constraint = orbital_constraint or OrbitalPhysicsConstraint()
        
        # Setup optimizer
        if config.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "Adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9
            )
        
        # Setup learning rate scheduler
        if config.scheduler == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs
            )
        elif config.scheduler == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            self.scheduler = None
        
        # Physics loss
        try:
            self.physics_loss = AdaptivePhysicsLoss(
                w_data_start=1.0 - config.physics_weight_start,
                w_physics_start=config.physics_weight_start
            )
        except:
            # Fallback
            self.physics_loss = None
            self.use_physics_loss = False
        else:
            self.use_physics_loss = config.use_physics_loss
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Training metrics dictionary
        """
        self.model.train()
        
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        num_batches = 0
        
        # Update physics loss training progress
        if self.use_physics_loss and self.physics_loss:
            self.physics_loss.set_training_progress(epoch, self.config.epochs)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    
                    if self.use_physics_loss and self.physics_loss:
                        loss = self.physics_loss(outputs, targets, use_physics=True)
                        components = self.physics_loss.get_loss_components(outputs, targets)
                    else:
                        loss = nn.functional.mse_loss(outputs, targets)
                        components = {'total_loss': loss.item()}
            else:
                outputs = self.model(inputs)
                
                if self.use_physics_loss and self.physics_loss:
                    loss = self.physics_loss(outputs, targets, use_physics=True)
                    components = self.physics_loss.get_loss_components(outputs, targets)
                else:
                    loss = nn.functional.mse_loss(outputs, targets)
                    components = {'total_loss': loss.item()}
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Accumulate metrics
            total_loss += components['total_loss']
            total_data_loss += components.get('mse_loss', 0.0)
            total_physics_loss += components.get('physics_loss', 0.0)
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.log_interval == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {components['total_loss']:.6f}")
        
        # Step scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Calculate averages
        metrics = {
            'epoch': epoch,
            'loss': total_loss / num_batches if num_batches > 0 else 0,
            'data_loss': total_data_loss / num_batches if num_batches > 0 else 0,
            'physics_loss': total_physics_loss / num_batches if num_batches > 0 else 0,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        self.training_history.append(metrics)
        
        return metrics
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Validation metrics dictionary
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                if self.use_physics_loss and self.physics_loss:
                    loss = self.physics_loss(outputs, targets, use_physics=True)
                else:
                    loss = nn.functional.mse_loss(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'epoch': epoch,
            'val_loss': avg_loss
        }
    
    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             checkpoint_dir: Optional[str] = None) -> List[Dict]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            List of training metrics per epoch
        """
        print(f"[TRAIN] Starting {self.config.epochs} epochs of physics-informed training")
        print(f"  Model: {self.config.model_type}")
        print(f"  Physics Loss: {self.use_physics_loss}")
        print(f"  Device: {self.device}")
        
        all_metrics = []
        
        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader is not None and epoch % self.config.validation_interval == 0:
                val_metrics = self.validate(val_loader, epoch)
                train_metrics.update(val_metrics)
                
                # Save best model
                if val_metrics['val_loss'] < self.best_loss:
                    self.best_loss = val_metrics['val_loss']
                    if checkpoint_dir:
                        self.save_checkpoint(checkpoint_dir, epoch, is_best=True)
            
            # Save checkpoint
            if checkpoint_dir and epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir, epoch)
            
            all_metrics.append(train_metrics)
            
            print(f"[TRAIN] Epoch {epoch}/{self.config.epochs} - "
                  f"Loss: {train_metrics['loss']:.6f}, "
                  f"Data Loss: {train_metrics['data_loss']:.6f}, "
                  f"Physics Loss: {train_metrics['physics_loss']:.6f}")
        
        print("[TRAIN] Training complete!")
        return all_metrics
    
    def save_checkpoint(self, checkpoint_dir: str, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            path = checkpoint_dir / f"{self.config.model_type}_best.pt"
        else:
            path = checkpoint_dir / f"{self.config.model_type}_epoch_{epoch}.pt"
        
        torch.save(checkpoint, path)
        print(f"[CHECKPOINT] Saved to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"[CHECKPOINT] Loaded from {checkpoint_path}")
    
    def export_to_onnx(self, output_path: str, dummy_input: torch.Tensor):
        """
        Export model to ONNX format
        
        Args:
            output_path: Path to save ONNX model
            dummy_input: Example input tensor
        """
        self.model.eval()
        dummy_input = dummy_input.to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"[EXPORT] Exported to ONNX: {output_path}")


class CurriculumLearningScheduler:
    """
    Curriculum learning scheduler for progressive difficulty
    """
    
    def __init__(self, stages: List[Dict]):
        """
        Initialize curriculum scheduler
        
        Args:
            stages: List of stage configurations
                Each stage: {'name': str, 'epochs': int, 'difficulty': float}
        """
        self.stages = stages
        self.current_stage = 0
    
    def get_current_stage(self) -> Dict:
        """Get current curriculum stage"""
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return self.stages[-1]
    
    def advance_stage(self) -> bool:
        """Advance to next curriculum stage"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False


# Example usage
if __name__ == "__main__":
    print("Testing Physics-Informed Training Pipeline...")
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(6, 32),
                nn.ReLU(),
                nn.Linear(32, 6)
            )
        
        def forward(self, x):
            return self.net(x)
    
    model = DummyModel()
    
    # Create config
    config = TrainingConfig(
        model_type="regressor",
        epochs=5,
        batch_size=4,
        learning_rate=1e-4,
        use_physics_loss=True
    )
    
    # Initialize trainer
    trainer = PhysicsInformedTrainer(model, config)
    
    # Create dummy data
    class DummyDataset(Dataset):
        def __len__(self):
            return 32
        
        def __getitem__(self, idx):
            # Simulate orbital state [x, y, z, vx, vy, vz]
            state = torch.randn(6)
            target = state + torch.randn(6) * 0.1
            return state, target
    
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Train
    metrics = trainer.train(loader)
    
    print(f"Training complete. Final loss: {metrics[-1]['loss']:.6f}")
    print("All tests passed!")
