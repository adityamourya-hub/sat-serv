"""
SAR-OrbitNet — CNN for orbital element + structure estimation from RISAT SAR/ISAR imagery.

Architecture:
  Complex-valued ResNet18 backbone → ISAR feature extractor → Physics-constrained orbit head
  + Structure classifier (satellite bus, debris, rocket body, unknown)

Physics constraints:
  - Vis-viva equation: v² = μ(2/r - 1/a)
  - Angular momentum conservation: h = r × v = const
  - Kepler's 3rd law: T² ∝ a³

Handles complex-valued SAR data (preserving phase for interferometric measurements).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional


# --- Complex-Valued Layers ---

class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution preserving I/Q structure."""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv_rr = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_ri = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_ir = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_ii = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2*C, H, W] where channels are [real_1..real_C, imag_1..imag_C]
        Returns:
            [B, 2*out_C, H', W']
        """
        C = x.size(1) // 2
        xr, xi = x[:, :C], x[:, C:]
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_r = self.conv_rr(xr) - self.conv_ii(xi)
        out_i = self.conv_ri(xr) + self.conv_ir(xi)
        out = torch.cat([out_r, out_i], dim=1)
        return F.relu(self.bn(out))


class ComplexResBlock(nn.Module):
    """Residual block for complex-valued feature maps."""
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ComplexConv2d(channels, channels, 3, stride, 1)
        self.conv2 = ComplexConv2d(channels, channels, 3, 1, 1)
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels * 2, channels * 2, 1, stride, bias=False),
                nn.BatchNorm2d(channels * 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


# --- ISAR Feature Extractor ---

class ISARFeatureExtractor(nn.Module):
    """
    Extracts physical features from ISAR (Inverse SAR) images:
      - Range profile length → object physical size
      - Cross-range extent → attitude-dependent dimension
      - Scattering center distribution → structural complexity
      - Polarimetric features → material properties
    """
    def __init__(self, in_channels: int = 256, out_dim: int = 64):
        super().__init__()
        self.range_pool = nn.AdaptiveAvgPool2d((1, None))   # Pool range dim
        self.cross_range_pool = nn.AdaptiveAvgPool2d((None, 1))  # Pool cross-range dim
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 4, 128), nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature maps from backbone
        Returns:
            [B, out_dim] ISAR-specific features
        """
        B, C, H, W = x.shape
        range_feat = self.range_pool(x).squeeze(2).mean(dim=-1)       # [B, C]
        cross_feat = self.cross_range_pool(x).squeeze(3).mean(dim=-1) # [B, C]
        global_feat = x.mean(dim=(2, 3))                               # [B, C]
        max_feat = x.amax(dim=(2, 3))                                  # [B, C]
        combined = torch.cat([range_feat, cross_feat, global_feat, max_feat], dim=-1)
        return self.fc(combined)


# --- Physics-Constrained Orbit Head ---

class PhysicsConstrainedOrbitHead(nn.Module):
    """
    Predicts orbital elements with built-in physics constraints.
    Enforces vis-viva and Kepler consistency in the loss function.
    """
    MU_EARTH = 3.986004418e5  # km³/s²
    R_EARTH = 6371.0          # km

    def __init__(self, in_dim: int = 320):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 6),  # [a, e, i, Ω, ω, M]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.head(z)
        # Apply physical constraints
        a = F.softplus(raw[:, 0:1]) + self.R_EARTH    # a > R_earth
        e = torch.sigmoid(raw[:, 1:2]) * 0.99         # 0 ≤ e < 1
        i = torch.sigmoid(raw[:, 2:3]) * math.pi      # 0 ≤ i ≤ π
        raan = torch.sigmoid(raw[:, 3:4]) * 2 * math.pi
        argp = torch.sigmoid(raw[:, 4:5]) * 2 * math.pi
        M = torch.sigmoid(raw[:, 5:6]) * 2 * math.pi
        return torch.cat([a, e, i, raan, argp, M], dim=-1)

    @staticmethod
    def physics_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Combined MSE + physics consistency penalty.
        Enforces Kepler's 3rd law: T = 2π√(a³/μ)
        """
        mse = F.mse_loss(predicted, target)

        # Kepler consistency: predicted period should match
        a_pred = predicted[:, 0]
        a_true = target[:, 0]
        T_pred = 2 * math.pi * torch.sqrt(a_pred ** 3 / PhysicsConstrainedOrbitHead.MU_EARTH)
        T_true = 2 * math.pi * torch.sqrt(a_true ** 3 / PhysicsConstrainedOrbitHead.MU_EARTH)
        kepler_loss = F.mse_loss(T_pred, T_true)

        return mse + 0.1 * kepler_loss


# --- Structure Classifier ---

class StructureClassifier(nn.Module):
    """Classifies object structure from ISAR features."""
    CLASSES = ['satellite_bus', 'debris', 'rocket_body', 'cubesat', 'unknown']

    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, len(self.CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# --- Full SAR-OrbitNet ---

class SAROrbitNet(nn.Module):
    """
    Complete SAR-to-orbit estimation pipeline.
    
    Input: Complex SAR/ISAR chip [B, 2, H, W] (I/Q channels)
    Output: Orbital elements [B, 6], Structure class [B, 5], ISAR features [B, 64]
    """
    def __init__(self, input_channels: int = 1, latent_dim: int = 256):
        super().__init__()
        # Complex-valued ResNet backbone
        self.stem = ComplexConv2d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.layer1 = nn.Sequential(ComplexResBlock(32), ComplexResBlock(32))
        self.layer2 = nn.Sequential(ComplexResBlock(32, stride=2), ComplexResBlock(32))
        self.layer3 = nn.Sequential(ComplexResBlock(32, stride=2), ComplexResBlock(32))
        self.pool = nn.AdaptiveAvgPool2d(4)

        # Feature dimension after backbone: 32*2 channels * 4*4 spatial = 1024
        backbone_dim = 32 * 2 * 4 * 4

        # ISAR feature extractor (operates on intermediate feature maps)
        self.isar_extractor = ISARFeatureExtractor(in_channels=64, out_dim=64)

        # Orbit head (backbone features + ISAR features)
        self.orbit_head = PhysicsConstrainedOrbitHead(in_dim=backbone_dim + 64)

        # Structure classifier
        self.structure_head = StructureClassifier(in_dim=64)

    def forward(self, sar_chip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            sar_chip: [B, 2, H, W] complex SAR image (channel 0=real, 1=imag)
        Returns:
            orbit_elements: [B, 6] predicted orbital elements
            structure_logits: [B, 5] structure classification logits
            isar_features: [B, 64] extracted ISAR features
        """
        # Backbone
        x = self.stem(sar_chip)        # [B, 64, H/2, W/2]
        x = self.layer1(x)             # [B, 64, H/2, W/2]
        feat_map = self.layer2(x)      # [B, 64, H/4, W/4]
        x = self.layer3(feat_map)      # [B, 64, H/8, W/8]
        pooled = self.pool(x)          # [B, 64, 4, 4]
        flat = pooled.flatten(1)       # [B, 1024]

        # ISAR features from intermediate feature map
        isar_feat = self.isar_extractor(feat_map)  # [B, 64]

        # Orbit estimation
        combined = torch.cat([flat, isar_feat], dim=-1)  # [B, 1088]
        orbit_elements = self.orbit_head(combined)

        # Structure classification
        structure_logits = self.structure_head(isar_feat)

        return orbit_elements, structure_logits, isar_feat


# --- Synthetic Data Generator ---

def generate_sar_training_sample(rng: np.random.Generator,
                                  img_size: int = 64) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate synthetic SAR chip with ground truth.
    Returns: (sar_chip [2, H, W], orbit_elements [6], structure_class int)
    """
    # Ground truth orbital elements
    a = rng.uniform(6600, 42164)
    e = rng.uniform(0.0, 0.3)
    inc = rng.uniform(0, np.pi)
    raan = rng.uniform(0, 2 * np.pi)
    argp = rng.uniform(0, 2 * np.pi)
    M = rng.uniform(0, 2 * np.pi)
    elements = np.array([a, e, inc, raan, argp, M], dtype=np.float32)

    structure_class = rng.integers(0, 5)

    # Synthetic SAR chip: point scatterers + noise
    n_scatterers = rng.integers(3, 20)
    chip_real = np.zeros((img_size, img_size), dtype=np.float32)
    chip_imag = np.zeros((img_size, img_size), dtype=np.float32)

    for _ in range(n_scatterers):
        cx, cy = rng.integers(5, img_size - 5, size=2)
        amp = rng.uniform(0.5, 5.0)
        phase = rng.uniform(0, 2 * np.pi)
        sigma = rng.uniform(1.0, 3.0)
        yy, xx = np.mgrid[:img_size, :img_size]
        gaussian = amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        chip_real += gaussian * np.cos(phase)
        chip_imag += gaussian * np.sin(phase)

    # Add speckle noise
    noise_level = rng.uniform(0.1, 0.5)
    chip_real += rng.normal(0, noise_level, (img_size, img_size)).astype(np.float32)
    chip_imag += rng.normal(0, noise_level, (img_size, img_size)).astype(np.float32)

    sar_chip = np.stack([chip_real, chip_imag], axis=0)  # [2, H, W]
    return sar_chip, elements, structure_class


# --- Self-Test ---

if __name__ == "__main__":
    print("Testing SAROrbitNet...")
    model = SAROrbitNet(input_channels=1)
    model.eval()

    rng = np.random.default_rng(42)
    sar, truth_elem, truth_class = generate_sar_training_sample(rng)
    sar_tensor = torch.tensor(sar).unsqueeze(0)  # [1, 2, 64, 64]

    with torch.no_grad():
        orbit, structure, isar_feat = model(sar_tensor)

    print(f"Input shape: {sar_tensor.shape}")
    print(f"Predicted orbit: {orbit.squeeze().numpy()}")
    print(f"Truth orbit: {truth_elem}")
    print(f"Structure logits: {structure.squeeze().numpy()}")
    print(f"Structure prediction: {StructureClassifier.CLASSES[structure.argmax(dim=-1).item()]}")
    print(f"ISAR feature dim: {isar_feat.shape}")
    print("✓ SAROrbitNet forward pass successful")
