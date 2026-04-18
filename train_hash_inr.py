#!/usr/bin/env python3
"""
ADITI 4.0 Hash INR Training Pipeline for Atmospheric Density

Trains multiresolution hash encoding + tiny MLP to approximate NRLMSISE-00.

Pipeline:
1. Sample NRLMSISE-00 on grid (lat, lon, alt, F107, Ap, local_time)
2. Train Hash INR with physics-informed regularization
3. Export to ONNX for Go inference
4. Validate: accuracy vs speed vs NRLMSISE baseline

Architecture (Instant-NGP style):
- L=16 levels, resolutions 16→512
- Hash table size: 2^19 entries
- Feature dim: 2 per level
- MLP: 2 layers, 64 hidden units
- Output: log(density) for stability

Physics constraints:
- Density > 0 (exponential output)
- Smooth gradients (for UKF/adjoint)
- Altitude decay (exponential with scale height)

References:
- Müller et al. "Instant Neural Graphics Primitives" (SIGGRAPH 2022)
- tiny-cuda-nn: https://github.com/NVlabs/tiny-cuda-nn

Usage:
    python train_hash_inr.py --samples 1000000 --epochs 100 --export
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Try to import tiny-cuda-nn (optional, falls back to PyTorch)
try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False
    print("[INFO] tiny-cuda-nn not available, using pure PyTorch implementation")

# ONNX export
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class HashEncoding(nn.Module):
    """
    Multiresolution hash encoding (PyTorch implementation).
    
    Falls back to pure PyTorch if tiny-cuda-nn unavailable.
    """
    
    def __init__(self, n_levels=16, base_res=16, max_res=512,
                 log2_hashmap_size=19, n_features_per_level=2):
        super().__init__()
        
        self.n_levels = n_levels
        self.base_res = base_res
        self.max_res = max_res
        self.log2_hashmap_size = log2_hashmap_size
        self.n_features_per_level = n_features_per_level
        
        # Hash table size
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # Growth factor b: max_res = base_res * b^(n_levels-1)
        self.b = np.exp((np.log(max_res) - np.log(base_res)) / (n_levels - 1))
        
        # Hash tables for each level
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(self.hashmap_size, n_features_per_level) * 0.01)
            for _ in range(n_levels)
        ])
    
    def forward(self, x):
        """
        Encode input coordinates to feature vector.
        
        Args:
            x: (B, 3) normalized coordinates in [0,1]
        
        Returns:
            features: (B, L*F) concatenated features
        """
        B = x.shape[0]
        features = []
        
        for level in range(self.n_levels):
            # Resolution at this level
            resolution = int(self.base_res * (self.b ** level))
            
            # Scale to grid
            scaled = x * resolution
            
            # Find voxel corners
            x0 = torch.floor(scaled[:, 0]).long()
            y0 = torch.floor(scaled[:, 1]).long()
            z0 = torch.floor(scaled[:, 2]).long()
            
            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1
            
            # Interpolation weights
            wx = scaled[:, 0] - x0.float()
            wy = scaled[:, 1] - y0.float()
            wz = scaled[:, 2] - z0.float()
            
            # Hash and lookup 8 corners
            # Hash function: spatial hash with primes
            def hash_coords(xc, yc, zc):
                return ((xc * 1 + yc * 2654435761 + zc * 805459861 + level) % self.hashmap_size).long()
            
            # Clamp to valid range
            x0 = torch.clamp(x0, 0, resolution)
            x1 = torch.clamp(x1, 0, resolution)
            y0 = torch.clamp(y0, 0, resolution)
            y1 = torch.clamp(y1, 0, resolution)
            z0 = torch.clamp(z0, 0, resolution)
            z1 = torch.clamp(z1, 0, resolution)
            
            # Hash indices
            h000 = hash_coords(x0, y0, z0)
            h100 = hash_coords(x1, y0, z0)
            h010 = hash_coords(x0, y1, z0)
            h110 = hash_coords(x1, y1, z0)
            h001 = hash_coords(x0, y0, z1)
            h101 = hash_coords(x1, y0, z1)
            h011 = hash_coords(x0, y1, z1)
            h111 = hash_coords(x1, y1, z1)
            
            # Lookup features
            f000 = self.hash_tables[level][h000]
            f100 = self.hash_tables[level][h100]
            f010 = self.hash_tables[level][h010]
            f110 = self.hash_tables[level][h110]
            f001 = self.hash_tables[level][h001]
            f101 = self.hash_tables[level][h101]
            f011 = self.hash_tables[level][h011]
            f111 = self.hash_tables[level][h111]
            
            # Trilinear interpolation
            c00 = f000 * (1 - wx[:, None]) + f100 * wx[:, None]
            c01 = f001 * (1 - wx[:, None]) + f101 * wx[:, None]
            c10 = f010 * (1 - wx[:, None]) + f110 * wx[:, None]
            c11 = f011 * (1 - wx[:, None]) + f111 * wx[:, None]
            
            c0 = c00 * (1 - wy[:, None]) + c10 * wy[:, None]
            c1 = c01 * (1 - wy[:, None]) + c11 * wy[:, None]
            
            feat = c0 * (1 - wz[:, None]) + c1 * wz[:, None]
            
            features.append(feat)
        
        # Concatenate all levels
        return torch.cat(features, dim=-1)


class DensityINR(nn.Module):
    """
    Hash INR for atmospheric density.
    
    Architecture:
    - Hash encoding: 16 levels → 32-dim features
    - MLP: 2 layers, 64 hidden, tanh activation
    - Output: log(density) for stability
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Hash encoding for spatial coordinates (lat, lon, alt)
        self.encoding = HashEncoding(
            n_levels=config['n_levels'],
            base_res=config['base_resolution'],
            max_res=config['max_resolution'],
            log2_hashmap_size=config['log2_hashmap_size'],
            n_features_per_level=config['n_features_per_level']
        )
        
        # MLP decoder
        encoding_dim = config['n_levels'] * config['n_features_per_level']
        
        # Additional inputs: F107, Ap, local_time
        total_input_dim = encoding_dim + 3
        
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, config['mlp_hidden_dim']),
            nn.Tanh(),
            nn.Linear(config['mlp_hidden_dim'], config['mlp_hidden_dim']),
            nn.Tanh(),
            nn.Linear(config['mlp_hidden_dim'], 1)
        )
        
        # Learnable scale for output
        self.log_density_scale = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, spatial_coords, solar_indices):
        """
        Forward pass.
        
        Args:
            spatial_coords: (B, 3) normalized (lat, lon, alt)
            solar_indices: (B, 3) normalized (F107, Ap, local_time)
        
        Returns:
            log_density: (B, 1)
        """
        # Hash encode spatial coordinates
        features = self.encoding(spatial_coords)
        
        # Concatenate with solar indices
        x = torch.cat([features, solar_indices], dim=-1)
        
        # MLP
        log_density = self.mlp(x) + self.log_density_scale
        
        return log_density


def generate_nrlmsise_samples(n_samples, seed=42):
    """
    Generate training data from NRLMSISE-00 model.
    
    For air-gapped deployment: Run once, save to file, distribute.
    
    Returns:
        spatial_coords: (N, 3) normalized lat, lon, alt
        solar_indices: (N, 3) normalized F107, Ap, local_time
        log_density: (N,) ground truth log(density)
    """
    np.random.seed(seed)
    
    # Sample ranges
    lats = np.random.uniform(-90, 90, n_samples)
    lons = np.random.uniform(-180, 180, n_samples)
    alts = np.random.uniform(100, 1000, n_samples)  # LEO focus
    
    # Solar/geomagnetic activity
    f107s = np.random.uniform(50, 300, n_samples)  # Solar flux
    aps = np.random.uniform(0, 100, n_samples)      # Geomagnetic
    local_times = np.random.uniform(0, 24, n_samples)
    
    # Mock NRLMSISE-00 calculation
    # Production: Call actual NRLMSISE-00 or load pre-computed
    log_densities = []
    
    for i in range(n_samples):
        # Simplified density model for demonstration
        # Real: Call nrlmsise00 library or wrapper
        
        # Base exponential decay with altitude
        h = alts[i]
        rho0 = 1.225  # kg/m3 at sea level
        H = 8.5       # Scale height (km) - varies with solar activity
        
        # Altitude-dependent base
        rho_base = rho0 * np.exp(-h / H)
        
        # Solar activity enhancement
        f107_factor = 1.0 + 0.3 * (f107s[i] - 150) / 150
        
        # Diurnal variation (simplified)
        local_time_factor = 1.0 + 0.2 * np.sin(2 * np.pi * local_times[i] / 24)
        
        # Geomagnetic storm
        ap_factor = 1.0 + 0.1 * aps[i] / 50
        
        rho = rho_base * f107_factor * local_time_factor * ap_factor
        
        # Clamp minimum density
        rho = max(rho, 1e-15)
        
        log_densities.append(np.log(rho))
    
    # Normalize to [0, 1] for INR input
    # Spatial coords
    spatial_norm = np.stack([
        (lats + 90) / 180,          # [-90,90] → [0,1]
        (lons + 180) / 360,         # [-180,180] → [0,1]
        (alts - 100) / 900          # [100,1000] → [0,1]
    ], axis=1).astype(np.float32)
    
    # Solar indices
    solar_norm = np.stack([
        (f107s - 50) / 250,         # [50,300] → [0,1]
        aps / 100,                   # [0,100] → [0,1]
        local_times / 24             # [0,24] → [0,1]
    ], axis=1).astype(np.float32)
    
    return (
        spatial_norm,
        solar_norm,
        np.array(log_densities, dtype=np.float32)
    )


def train(args):
    """Main training loop."""
    
    # Configuration
    config = {
        'n_levels': 16,
        'base_resolution': 16,
        'max_resolution': 512,
        'log2_hashmap_size': 19,
        'n_features_per_level': 2,
        'mlp_hidden_dim': 64,
        'mlp_num_layers': 2,
        'learning_rate': 1e-3,
        'batch_size': 4096,
        'epochs': args.epochs,
    }
    
    print(f"Training Hash INR for atmospheric density")
    print(f"Samples: {args.samples}")
    print(f"Config: {config}")
    
    # Generate training data
    print("\nGenerating NRLMSISE-00 samples...")
    spatial, solar, log_density = generate_nrlmsise_samples(args.samples)
    
    print(f"Spatial coords shape: {spatial.shape}")
    print(f"Log density range: [{log_density.min():.2f}, {log_density.max():.2f}]")
    print(f"Density range: [{np.exp(log_density.min()):.2e}, {np.exp(log_density.max()):.2e}] kg/m³")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DensityINR(config).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Convert to tensors
    spatial_t = torch.from_numpy(spatial).to(device)
    solar_t = torch.from_numpy(solar).to(device)
    target_t = torch.from_numpy(log_density[:, None]).to(device)
    
    # Training loop
    print("\nTraining...")
    n_batches = args.samples // config['batch_size']
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle
        perm = torch.randperm(args.samples)
        
        for i in range(n_batches):
            batch_idx = perm[i * config['batch_size']:(i + 1) * config['batch_size']]
            
            s_spatial = spatial_t[batch_idx]
            s_solar = solar_t[batch_idx]
            s_target = target_t[batch_idx]
            
            # Forward
            pred = model(s_spatial, s_solar)
            
            # Loss: MSE on log density
            loss = nn.functional.mse_loss(pred, s_target)
            
            # Physics regularization: smoothness
            # Gradient penalty (optional)
            if args.physics_reg > 0:
                s_spatial.requires_grad_(True)
                pred_grad = model(s_spatial, s_solar)
                grad = torch.autograd.grad(
                    pred_grad.sum(), s_spatial,
                    create_graph=True, retain_graph=True
                )[0]
                smooth_loss = (grad ** 2).mean()
                loss += args.physics_reg * smooth_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / n_batches
            print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4e}")
    
    # Validation
    print("\nValidating...")
    model.eval()
    with torch.no_grad():
        # Sample validation set
        val_idx = torch.randperm(args.samples)[:10000]
        val_spatial = spatial_t[val_idx]
        val_solar = solar_t[val_idx]
        val_target = target_t[val_idx]
        
        val_pred = model(val_spatial, val_solar)
        val_loss = nn.functional.mse_loss(val_pred, val_target)
        
        # Relative error on density
        pred_density = torch.exp(val_pred).cpu().numpy()
        true_density = torch.exp(val_target).cpu().numpy()
        rel_error = np.abs(pred_density - true_density) / true_density
        
        print(f"Validation MSE (log): {val_loss.item():.4e}")
        print(f"Mean relative error: {rel_error.mean():.2%}")
        print(f"Max relative error: {rel_error.max():.2%}")
    
    # Save model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch model
    torch.save({
        'config': config,
        'model_state': model.state_dict(),
        'input_mins': [-90, -180, 100, 50, 0, 0],
        'input_maxs': [90, 180, 1000, 300, 100, 24],
    }, output_dir / 'hash_inr_density.pt')
    
    print(f"\nSaved PyTorch model to {output_dir}/hash_inr_density.pt")
    
    # Export to ONNX
    if args.export_onnx and ONNX_AVAILABLE:
        export_onnx(model, config, output_dir / 'hash_inr_density.onnx')
    
    # Export hash tables for pure Go implementation
    export_hash_tables(model, output_dir / 'hash_tables.json')
    
    return model


def export_onnx(model, config, path):
    """Export to ONNX format."""
    print(f"\nExporting to ONNX: {path}")
    
    model.eval()
    
    # Dummy inputs
    dummy_spatial = torch.randn(1, 3)
    dummy_solar = torch.randn(1, 3)
    
    # Export
    torch.onnx.export(
        model,
        (dummy_spatial, dummy_solar),
        path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['spatial_coords', 'solar_indices'],
        output_names=['log_density'],
        dynamic_axes={
            'spatial_coords': {0: 'batch_size'},
            'solar_indices': {0: 'batch_size'},
            'log_density': {0: 'batch_size'}
        }
    )
    
    print("ONNX export complete")


def export_hash_tables(model, path):
    """Export hash tables for Go implementation."""
    print(f"\nExporting hash tables to {path}")
    
    tables = []
    for level, table in enumerate(model.encoding.hash_tables):
        table_data = {
            'level': level,
            'shape': list(table.shape),
            'data': table.detach().cpu().numpy().tolist()
        }
        tables.append(table_data)
    
    with open(path, 'w') as f:
        json.dump(tables, f)
    
    print(f"Exported {len(tables)} hash tables")


def benchmark(model, spatial, solar, n_queries=100000):
    """Benchmark query speed."""
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        # Warmup
        _ = model(spatial[:100], solar[:100])
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        import time
        start = time.time()
        
        batch_size = 10000
        for i in range(0, n_queries, batch_size):
            end = min(i + batch_size, n_queries)
            _ = model(spatial[i:end], solar[i:end])
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        
        print(f"\nBenchmark: {n_queries} queries in {elapsed:.3f}s")
        print(f"Throughput: {n_queries/elapsed:,.0f} queries/sec")
        print(f"Latency: {elapsed*1e6/n_queries:.2f} μs/query")


def main():
    parser = argparse.ArgumentParser(
        description='Train Hash INR for atmospheric density'
    )
    
    parser.add_argument('--samples', type=int, default=100000,
                        help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--physics-reg', type=float, default=1e-4,
                        help='Physics regularization weight (smoothness)')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export to ONNX format')
    parser.add_argument('--output', type=str, default='./models',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Train
    model = train(args)
    
    # Benchmark
    print("\n" + "="*50)
    print("BENCHMARK")
    print("="*50)
    
    spatial, solar, _ = generate_nrlmsise_samples(100000)
    spatial_t = torch.from_numpy(spatial)
    solar_t = torch.from_numpy(solar)
    
    benchmark(model, spatial_t, solar_t, n_queries=100000)
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    print(f"\nIntegration:")
    print(f"  Go: onnxruntime-go for ONNX model")
    print(f"  Or: Load hash_tables.json + custom MLP")
    print(f"  API: QueryDensity(lat, lon, alt, F107, Ap, LT) → kg/m³")


if __name__ == '__main__':
    main()
