"""
Training pipeline for Hybrid Neural ODE Propagator.

Uses TLE history to train physics-informed neural network.
Expected: 2 orders of magnitude improvement over SGP4 alone.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime

from hybrid_propagator import (
    HybridNeuralODE, 
    PhysicsInformedLoss,
    export_to_onnx,
    MU, RE
)
from sgp4.api import Satrec, jday

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TLETrajectoryDataset(Dataset):
    """
    Dataset of satellite trajectories from TLE history.
    
    Each sample: (initial_state, time_points, observed_states)
    """
    
    def __init__(self, 
                 tle_history_file: str,
                 sequence_length: int = 10,
                 min_observations: int = 5,
                 max_time_span_hours: float = 24.0):
        self.sequence_length = sequence_length
        self.samples = []
        
        logger.info(f"Loading TLE history from {tle_history_file}")
        self._load_data(tle_history_file, min_observations, max_time_span_hours)
        
        logger.info(f"Created {len(self.samples)} training samples")
    
    def _load_data(self, filepath: str, min_obs: int, max_hours: float):
        """Parse TLE history and create training samples."""
        
        # Load TLEs from file (format: NORAD_ID, EPOCH, LINE1, LINE2)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Group by satellite
        sat_groups = {}
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
            
            norad_id = parts[0]
            epoch = float(parts[1])
            line1 = parts[2]
            line2 = parts[3]
            
            if norad_id not in sat_groups:
                sat_groups[norad_id] = []
            
            sat_groups[norad_id].append({
                'epoch': epoch,
                'line1': line1,
                'line2': line2
            })
        
        # Create samples for each satellite
        for norad_id, tles in sat_groups.items():
            if len(tles) < min_obs:
                continue
            
            # Sort by epoch
            tles = sorted(tles, key=lambda x: x['epoch'])
            
            # Create sliding window samples
            for i in range(len(tles) - 1):
                tle_current = tles[i]
                tle_future = tles[i+1:]
                
                # Parse initial state
                x0 = self._tle_to_state(tle_current)
                
                # Get future observations
                obs_times = []
                obs_states = []
                
                for tle_next in tle_future:
                    dt_hours = (tle_next['epoch'] - tle_current['epoch']) * 24
                    
                    # Skip if too far in future
                    if dt_hours > max_hours:
                        break
                    
                    # Propagate using SGP4 (ground truth for training)
                    state = self._propagate_sgp4(tle_current, tle_next['epoch'])
                    
                    obs_times.append(dt_hours * 3600)  # Convert to seconds
                    obs_states.append(state)
                    
                    if len(obs_times) >= self.sequence_length:
                        break
                
                if len(obs_times) >= min_obs:
                    self.samples.append({
                        'sat_id': norad_id,
                        'initial_state': x0,
                        'obs_times': np.array(obs_times, dtype=np.float32),
                        'obs_states': np.array(obs_states, dtype=np.float32)
                    })
    
    def _tle_to_state(self, tle: Dict) -> np.ndarray:
        """Convert TLE to Cartesian state vector."""
        sat = Satrec.twoline2rv(tle['line1'], tle['line2'])
        
        jd, fr = jday(2024, 1, 1, 0, 0, 0)  # Reference date doesn't matter for SGP4
        fr = tle['epoch'] - jd
        
        e, r, v = sat.sgp4(jd, fr)
        
        if e != 0:
            # Failed propagation - skip
            return np.zeros(6, dtype=np.float32)
        
        return np.array([r[0], r[1], r[2], v[0], v[1], v[2]], dtype=np.float32)
    
    def _propagate_sgp4(self, tle_from: Dict, epoch_to: float) -> np.ndarray:
        """Propagate from tle_from epoch to epoch_to using SGP4."""
        sat = Satrec.twoline2rv(tle_from['line1'], tle_from['line2'])
        
        jd, fr = jday(2024, 1, 1, 0, 0, 0)
        fr = epoch_to - jd
        
        e, r, v = sat.sgp4(jd, fr)
        
        return np.array([r[0], r[1], r[2], v[0], v[1], v[2]], dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_variable_length(batch: List[Dict]) -> Dict:
    """
    Collate function for variable-length trajectories.
    Pads sequences to maximum length in batch.
    """
    # Find max length
    max_len = max(len(s['obs_times']) for s in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    x0 = torch.stack([torch.tensor(s['initial_state']) for s in batch])
    
    # Pad observation sequences
    obs_times = torch.zeros(batch_size, max_len)
    obs_states = torch.zeros(batch_size, max_len, 6)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, sample in enumerate(batch):
        n = len(sample['obs_times'])
        lengths[i] = n
        obs_times[i, :n] = torch.tensor(sample['obs_times'])
        obs_states[i, :n] = torch.tensor(sample['obs_states'])
    
    return {
        'initial_state': x0,
        'obs_times': obs_times,
        'obs_states': obs_states,
        'lengths': lengths,
        'sat_ids': [s['sat_id'] for s in batch]
    }


class Trainer:
    """Training manager for Hybrid Neural ODE."""
    
    def __init__(self, 
                 model: HybridNeuralODE,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        self.criterion = PhysicsInformedLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.best_loss = float('inf')
        self.train_history = []
        
        logger.info(f"Trainer initialized on {device}")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_obs_loss = 0
        total_energy_loss = 0
        total_momentum_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            x0 = batch['initial_state'].to(self.device)
            obs_times = batch['obs_times'].to(self.device)
            obs_states = batch['obs_states'].to(self.device)
            lengths = batch['lengths']
            
            batch_size, max_len = obs_times.shape
            
            self.optimizer.zero_grad()
            
            # Forward propagation for each sample
            batch_loss = 0
            
            for i in range(batch_size):
                n = int(lengths[i].item())
                if n < 2:
                    continue
                
                t_span = obs_times[i, :n]
                
                # Propagate
                trajectory = self.model(
                    x0[i].unsqueeze(0),
                    t_span,
                    method='dopri5',
                    rtol=1e-7,
                    atol=1e-9
                )
                
                # Compute loss
                pred = trajectory[0]  # [n, 6]
                true = obs_states[i, :n]
                
                loss, loss_dict = self.criterion(
                    pred.unsqueeze(0),
                    true.unsqueeze(0),
                    t_span
                )
                
                batch_loss += loss
                total_obs_loss += loss_dict['obs'].item()
                total_energy_loss += loss_dict['energy'].item()
                total_momentum_loss += loss_dict['momentum'].item()
            
            # Backward
            if batch_size > 0:
                batch_loss = batch_loss / batch_size
                batch_loss.backward()
                
                # Gradient clipping (critical for ODE stability)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        
        return {
            'loss': avg_loss,
            'obs_loss': total_obs_loss / n_batches if n_batches > 0 else 0,
            'energy_loss': total_energy_loss / n_batches if n_batches > 0 else 0,
            'momentum_loss': total_momentum_loss / n_batches if n_batches > 0 else 0
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on held-out data."""
        self.model.eval()
        
        total_loss = 0
        total_position_error = 0
        total_velocity_error = 0
        n_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x0 = batch['initial_state'].to(self.device)
                obs_times = batch['obs_times'].to(self.device)
                obs_states = batch['obs_states'].to(self.device)
                lengths = batch['lengths']
                
                batch_size, max_len = obs_times.shape
                
                for i in range(batch_size):
                    n = int(lengths[i].item())
                    if n < 2:
                        continue
                    
                    t_span = obs_times[i, :n]
                    
                    trajectory = self.model(
                        x0[i].unsqueeze(0),
                        t_span,
                        method='dopri5'
                    )
                    
                    pred = trajectory[0]  # [n, 6]
                    true = obs_states[i, :n]
                    
                    # MSE loss
                    loss = torch.nn.functional.mse_loss(pred, true)
                    total_loss += loss.item()
                    
                    # Position error (km)
                    pos_error = torch.norm(pred[:, :3] - true[:, :3], dim=1).mean()
                    total_position_error += pos_error.item()
                    
                    # Velocity error (km/s)
                    vel_error = torch.norm(pred[:, 3:] - true[:, 3:], dim=1).mean()
                    total_velocity_error += vel_error.item()
                    
                    n_samples += 1
        
        return {
            'val_loss': total_loss / n_samples if n_samples > 0 else 0,
            'val_pos_error_km': total_position_error / n_samples if n_samples > 0 else 0,
            'val_vel_error_kms': total_velocity_error / n_samples if n_samples > 0 else 0
        }
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'train_history': self.train_history
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint.get('train_history', [])
        logger.info(f"Loaded checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid Neural ODE')
    parser.add_argument('--data', type=str, required=True, help='TLE history CSV file')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=100)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--output-dir', type=str, default='models/orbital')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    logger.info("Loading datasets...")
    full_dataset = TLETrajectoryDataset(args.data)
    
    # Split train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable_length,
        num_workers=4
    )
    
    # Create model
    model = HybridNeuralODE(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_adjoint=True
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, device=args.device, lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        logger.info(f"Train: loss={train_metrics['loss']:.6f}, "
                   f"obs={train_metrics['obs_loss']:.6f}, "
                   f"energy={train_metrics['energy_loss']:.6f}")
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        logger.info(f"Val: loss={val_metrics['val_loss']:.6f}, "
                   f"pos_error={val_metrics['val_pos_error_km']:.3f}km, "
                   f"vel_error={val_metrics['val_vel_error_kms']:.6f}km/s")
        
        # Update scheduler
        trainer.scheduler.step(val_metrics['val_loss'])
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            
            checkpoint_path = output_dir / 'best_model.pt'
            trainer.save_checkpoint(str(checkpoint_path), epoch, val_metrics)
            
            # Export to ONNX
            onnx_path = output_dir / 'neural_ode.onnx'
            export_to_onnx(model, str(onnx_path))
            
            logger.info(f"New best model! Val loss: {best_val_loss:.6f}")
        
        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch{epoch+1}.pt'
            trainer.save_checkpoint(str(checkpoint_path), epoch, val_metrics)
    
    # Final evaluation
    logger.info("\n" + "="*50)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to: {output_dir}")
    
    # Save training history
    history_path = output_dir / 'train_history.json'
    with open(history_path, 'w') as f:
        json.dump(trainer.train_history, f, indent=2)


if __name__ == "__main__":
    main()
