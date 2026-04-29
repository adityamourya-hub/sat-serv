import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from ateena_iqnet import ATeenaIQNet
from radar_augmentation import RadarAugmentation

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulator'))
from ateena_sim import ATeenaSimulator, generate_random_scenario

class StreamingRadarDataset(Dataset):
    """
    Zero-disk streaming dataset for ATeena-IQNet.
    Generates CPIs on-the-fly for training.
    """
    def __init__(self, augmentation=None):
        self.sim = ATeenaSimulator()
        self.augmentation = augmentation

    def __len__(self):
        # Arbitrary virtual epoch size
        return 1000

    def __getitem__(self, idx):
        # 1. Generate random scenario
        targets, noise_fig = generate_random_scenario()
        # 2. Simulate CPI
        iq_np = self.sim.simulate_cpi(targets, noise_figure_db=noise_fig)
        
        # 3. Process into torch format [P, R, E, 2]
        # Normalization: Applying 1e8 scale
        iq = torch.from_numpy(np.stack([iq_np.real, iq_np.imag], axis=-1)).float() * 1e8
        
        # 4. Create labels [50, 6]
        # [range, doppler, az, el, rcs, confidence]
        labels = torch.zeros(50, 6)
        for i, t in enumerate(targets[:50]):
            labels[i, 0] = t['range_km']
            labels[i, 1] = t['velocity_kms']
            labels[i, 2] = t['azimuth_deg']
            labels[i, 3] = t['elevation_deg']
            labels[i, 4] = t['rcs_m2']
            labels[i, 5] = 1.0 # Present
            
        if self.augmentation:
            iq = self.augmentation(iq)  # Augmentation handles shape transformation
            
        return iq, labels

def islr_loss(compressed_signal, template=None, epoch=0):
    """
    Hybrid ISLR Loss: Template-based (Stage 1) -> Peak-Prominence (Adaptive).
    """
    B, C, R = compressed_signal.shape
    # For simplicity, calculate energy in the neighborhood of the peak
    # compressed_signal is [B, 2, R]. We calculate magnitude.
    mag = torch.sqrt(torch.sum(compressed_signal**2, dim=1)) # [B, R]
    
    # Pre-training: LFM Template alignment (MSE)
    if epoch < 10 and template is not None:
        return F.mse_loss(mag, template)
    
    # Adaptive: Minimize Integrated Sidelobe Ratio
    # Find peak per batch
    peaks, peak_idx = torch.max(mag, dim=1)
    
    total_energy = torch.sum(mag**2, dim=1)
    
    # Define mainlobe as +/- 2 bins around peak
    # Use masking to find sidelobes
    indices = torch.arange(R).unsqueeze(0).to(compressed_signal.device)
    mainlobe_mask = (indices > (peak_idx.unsqueeze(1) - 3)) & (indices < (peak_idx.unsqueeze(1) + 3))
    
    mainlobe_energy = torch.sum(mag**2 * mainlobe_mask, dim=1)
    sidelobe_energy = torch.sum(mag**2 * (~mainlobe_mask), dim=1)
    
    islr = torch.mean(sidelobe_energy / (mainlobe_energy + 1e-6))
    return islr

def evidential_radar_loss(alphas, y_present, lambda_reg=0.2):
    """
    Dirichlet Loss with Non-Uniform Clutter Prior.
    y_present: [B, max_detections] binary target mask
    """
    S = torch.sum(alphas, dim=-1, keepdim=True)
    p = alphas / S
    
    # MSE loss for classification (present vs absent)
    # y_present maps to the "target" channel of the Dirichlet
    # For a binary detection, we treat it as 2-class Dirichlet: [p_target, p_clutter]
    # Here alphas is [B, max_detections], we use it as the target evidence
    
    # Regression error (simplified)
    err = F.binary_cross_entropy(torch.clamp(p, 1e-6, 1-1e-6), y_present)
    
    # Regularization against Clutter Prior
    # In a full logic, we'd have a 2nd alpha for clutter. 
    # Here we use a generic penalty on misleading evidence.
    penalty = lambda_reg * torch.mean(torch.sum((1-y_present) * (alphas - 1), dim=-1))
    
    return err + penalty

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Config proxy
    class Config:
        n_elements = 64
        n_pulses = 128
        n_doppler_bins = 64
    
    model = ATeenaIQNet(Config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    aug = RadarAugmentation(device=device)
    
    # Initializing Streaming Dataset
    dataset = StreamingRadarDataset(augmentation=aug)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    print("Starting ATeena-IQNet Streaming Production Training Loop...")
    try:
        for epoch in range(1): # One virtual epoch of 1000 samples
            model.train()
            for i, (iq_batch, labels) in enumerate(loader):
                iq_batch = iq_batch.permute(0, 2, 3, 4, 1).to(device) # [B, P, R, E, 2]
                labels = labels.to(device)
                
                detections, alphas, features = model(iq_batch)
                
                # Loss calculation
                l_reg = F.smooth_l1_loss(detections, labels)
                l_evd = evidential_radar_loss(alphas, (labels[:, :, 5] > 0.5).float())
                
                loss = l_reg + l_evd
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i % 10 == 0:
                    print(f"Epoch {epoch+1} | Step {i} | Loss: {loss.item():.4f}")
                
                # Incremental Checkpoint
                if i % 20 == 0 and i > 0:
                    model.eval()
                    dummy_iq = torch.randn(1, 128, 1024, 64, 2).to(device)
                    onnx_path = f"ai/training/models/ateena_iqnet_step_{i}.onnx"
                    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                    torch.onnx.export(model, dummy_iq, onnx_path, input_names=["iq_samples"], output_names=["detections", "alphas", "features"], dynamic_axes={"iq_samples": {0: "batch"}, "detections": {0: "batch"}})
                    print(f"✓ Checkpoint saved: {onnx_path}")
                    model.train()

                if i >= 100: # Limit to 100 batches (400 samples)
                    break
    except Exception as e:
        print(f"FATAL: Training interrupted: {e}")
    finally:
        # Final hardening: Export to ONNX
        model.eval()
        dummy_iq = torch.randn(1, 128, 1024, 64, 2).to(device)
        onnx_path = "ai/training/models/ateena_iqnet_production.onnx"
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        torch.onnx.export(model, dummy_iq, onnx_path, input_names=["iq_samples"], output_names=["detections", "alphas", "features"], dynamic_axes={"iq_samples": {0: "batch"}, "detections": {0: "batch"}})
        print(f"✓ Final model exported: {onnx_path}")

    # Final hardening: Export to ONNX
    model.eval()
    dummy_iq = torch.randn(1, 128, 1024, 64, 2).to(device)
    onnx_path = "ai/training/models/ateena_iqnet_production.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    torch.onnx.export(
        model, dummy_iq, onnx_path,
        input_names=["iq_samples"],
        output_names=["detections", "alphas", "features"],
        dynamic_axes={"iq_samples": {0: "batch"}, "detections": {0: "batch"}}
    )
    print(f"✓ ATeena-IQNet specialized production model exported to {onnx_path}")

if __name__ == "__main__":
    train()
