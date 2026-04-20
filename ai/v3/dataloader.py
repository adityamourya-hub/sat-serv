#!/usr/bin/env python3
"""
Unified Physics Dataloader for Foundation Model Training

Implements balanced 40/30/30 mix of HGV/Orbital/Ballistic regimes
to prevent overfitting to nonlinear lift while preserving Keplerian laws.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict


class UnifiedPhysicsDataset(Dataset):
    """
    Joint dataloader ensuring model sees balanced diet of manifolds in every batch.
    
    Regime Distribution:
    - 40% HGV (hypersonic skip-glide, nonlinear lift)
    - 30% Orbital (Keplerian dynamics, vacuum)
    - 30% Ballistic (re-entry, drag-dominated)
    
    This ensures the H100s don't overfit to the HGV's "glamour" data
    while forgetting basic Keplerian laws.
    """
    
    def __init__(self, 
                 orbital_data: List[np.ndarray],
                 ballistic_data: List[np.ndarray],
                 hgv_data: List[np.ndarray]):
        """
        Initialize unified dataset with regime-balanced sampling.
        
        Args:
            orbital_data: List of orbital trajectory arrays (shape: [T, 6])
            ballistic_data: List of ballistic trajectory arrays
            hgv_data: List of HGV trajectory arrays
        """
        # Store regime data separately for balanced sampling
        self.orbital_data = orbital_data
        self.ballistic_data = ballistic_data
        self.hgv_data = hgv_data
        
        # Calculate sampling weights (40/30/30)
        total_samples = len(orbital_data) + len(ballistic_data) + len(hgv_data)
        self.orbital_weight = 0.30
        self.ballistic_weight = 0.30
        self.hgv_weight = 0.40
        
        # Create balanced index list
        self.indices = self._create_balanced_indices()
        
        # Regime label mapping
        self.regime_to_id = {"Orbital": 0, "Ballistic": 1, "HGV": 2}
        self.id_to_regime = {v: k for k, v in self.regime_to_id.items()}
    
    def _create_balanced_indices(self) -> List[Tuple[int, str]]:
        """
        Create balanced sampling indices with 40/30/30 distribution.
        
        Returns:
            List of (data_index, regime_label) tuples
        """
        indices = []
        
        # Determine target counts based on smallest dataset
        min_count = min(
            len(self.orbital_data),
            len(self.ballistic_data),
            len(self.hgv_data)
        )
        
        # Sample from each regime with target proportions
        orbital_count = int(min_count * (self.orbital_weight / 0.30))
        ballistic_count = int(min_count * (self.ballistic_weight / 0.30))
        hgv_count = int(min_count * (self.hgv_weight / 0.40))
        
        # Add orbital samples
        for i in range(min(orbital_count, len(self.orbital_data))):
            indices.append((i, "Orbital"))
        
        # Add ballistic samples
        for i in range(min(ballistic_count, len(self.ballistic_data))):
            indices.append((i, "Ballistic"))
        
        # Add HGV samples
        for i in range(min(hgv_count, len(self.hgv_data))):
            indices.append((i, "HGV"))
        
        # Shuffle to mix regimes in each batch
        np.random.shuffle(indices)
        
        return indices
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample with regime label.
        
        Args:
            idx: Index into balanced indices
            
        Returns:
            trajectory_tensor: Shape [T, 6] (pos[3], vel[3])
            regime_id: 0=Orbital, 1=Ballistic, 2=HGV
        """
        data_idx, regime_label = self.indices[idx]
        
        # Retrieve trajectory based on regime
        if regime_label == "Orbital":
            traj = self.orbital_data[data_idx]
        elif regime_label == "Ballistic":
            traj = self.ballistic_data[data_idx]
        else:  # HGV
            traj = self.hgv_data[data_idx]
        
        # Convert to tensor
        traj_tensor = torch.tensor(traj, dtype=torch.float32)
        
        # Get regime ID
        regime_id = self.regime_to_id[regime_label]
        
        return traj_tensor, regime_id
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def get_regime_distribution(self) -> Dict[str, int]:
        """Return count of samples per regime."""
        counts = {"Orbital": 0, "Ballistic": 0, "HGV": 0}
        for _, regime in self.indices:
            counts[regime] += 1
        return counts


def create_unified_dataloader(
    orbital_data: List[np.ndarray],
    ballistic_data: List[np.ndarray],
    hgv_data: List[np.ndarray],
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create DataLoader with unified physics dataset.
    
    Args:
        orbital_data: Orbital trajectories
        ballistic_data: Ballistic trajectories
        hgv_data: HGV trajectories
        batch_size: Batch size for training
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader with balanced regime sampling
    """
    dataset = UnifiedPhysicsDataset(orbital_data, ballistic_data, hgv_data)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    # Log distribution
    dist = dataset.get_regime_distribution()
    total = sum(dist.values())
    print(f"Unified Dataset Distribution:")
    print(f"  Orbital: {dist['Orbital']} ({100*dist['Orbital']/total:.1f}%)")
    print(f"  Ballistic: {dist['Ballistic']} ({100*dist['Ballistic']/total:.1f}%)")
    print(f"  HGV: {dist['HGV']} ({100*dist['HGV']/total:.1f}%)")
    
    return dataloader


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing UnifiedPhysicsDataset...")
    
    # Generate synthetic trajectories
    np.random.seed(42)
    
    orbital_data = [np.random.randn(100, 6) for _ in range(50)]
    ballistic_data = [np.random.randn(100, 6) for _ in range(50)]
    hgv_data = [np.random.randn(100, 6) for _ in range(50)]
    
    # Create dataset
    dataset = UnifiedPhysicsDataset(orbital_data, ballistic_data, hgv_data)
    print(f"Dataset size: {len(dataset)}")
    print(f"Regime distribution: {dataset.get_regime_distribution()}")
    
    # Test dataloader
    dataloader = create_unified_dataloader(
        orbital_data, ballistic_data, hgv_data,
        batch_size=8, num_workers=0
    )
    
    # Test iteration
    for batch, labels in dataloader:
        print(f"Batch shape: {batch.shape}, Labels: {labels}")
        break
    
    print("UnifiedPhysicsDataset test passed!")
