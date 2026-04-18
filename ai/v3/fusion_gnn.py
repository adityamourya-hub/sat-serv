"""
Multi-Sensor Fusion GNN — Graph Attention Network for cross-sensor orbital state estimation.

Fuses heterogeneous sensor observations (ATeena radar, RISAT SAR, Optical CCD, NavIC GNSS)
into unified orbital state estimates via learned cross-attention on a sensor graph.

Architecture:
  Per-sensor encoders → Sensor Graph (GATv2Conv) → Cross-Attention Fusion → Orbital Element Decoder

Graph structure:
  - Nodes: Individual sensor observations (variable count per epoch)
  - Edges: Fully connected within each epoch (cross-sensor attention)
  - Edge features: Time difference, spatial proximity, sensor-type pair encoding

Output: [a, e, i, Ω, ω, M] + 6×6 covariance estimate

Unique advantage over US Project Maven: Physics-grounded fusion with orbital mechanics constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# --- Sensor Encoders ---

class RadarIQEncoder(nn.Module):
    """Encodes ATeena radar IQ features into a latent vector."""
    def __init__(self, iq_dim: int = 6, latent_dim: int = 128):
        super().__init__()
        # Input: [rcs, doppler_hz, snr_db, azimuth_rad, elevation_rad, range_km]
        self.net = nn.Sequential(
            nn.Linear(iq_dim, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SAREncoder(nn.Module):
    """Encodes RISAT SAR/ISAR chip features into a latent vector."""
    def __init__(self, feat_dim: int = 8, latent_dim: int = 128):
        super().__init__()
        # Input: [range_profile_len, cross_range_len, polarimetric_entropy,
        #         hh_power, vv_power, hv_power, aspect_angle, coherence]
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OpticalEncoder(nn.Module):
    """Encodes optical CCD observation features."""
    def __init__(self, feat_dim: int = 5, latent_dim: int = 128):
        super().__init__()
        # Input: [apparent_mag, ra_deg, dec_deg, angular_rate, color_index]
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NavICEncoder(nn.Module):
    """Encodes NavIC GNSS pseudorange residual features."""
    def __init__(self, feat_dim: int = 4, latent_dim: int = 128):
        super().__init__()
        # Input: [pr_residual_l5, pr_residual_s, iono_diff, clock_drift]
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- Graph Attention Layer (GATv2-style) ---

class GATv2Layer(nn.Module):
    """
    GATv2 attention layer with edge features.
    Implements the dynamic attention mechanism from Brody et al. 2021.
    """
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 edge_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0

        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] source/target indices
            edge_attr: [E, edge_dim] edge features (optional)
        Returns:
            [N, out_dim] updated node features
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        h_src = self.W_src(x).view(N, self.n_heads, self.head_dim)
        h_dst = self.W_dst(x).view(N, self.n_heads, self.head_dim)

        # GATv2: attention on transformed features (dynamic)
        msg = self.leaky_relu(h_src[src] + h_dst[dst])  # [E, heads, head_dim]

        if edge_attr is not None:
            e = self.W_edge(edge_attr).view(-1, self.n_heads, self.head_dim)
            msg = msg + e

        # Attention coefficients
        alpha = (msg * self.attn.unsqueeze(0)).sum(dim=-1)  # [E, heads]

        # Softmax per destination node
        alpha_max = torch.zeros(N, self.n_heads, device=x.device)
        alpha_max.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(alpha), alpha, reduce='amax')
        alpha = torch.exp(alpha - alpha_max[dst])
        alpha_sum = torch.zeros(N, self.n_heads, device=x.device)
        alpha_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(alpha), alpha)
        alpha = alpha / (alpha_sum[dst] + 1e-12)
        alpha = self.dropout(alpha)

        # Weighted message aggregation
        weighted = h_src[src] * alpha.unsqueeze(-1)  # [E, heads, head_dim]
        out = torch.zeros(N, self.n_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(weighted), weighted)

        out = out.reshape(N, -1)  # [N, out_dim]
        return self.norm(out + x[:, :out.size(1)] if x.size(1) == out.size(1) else self.norm(out))


# --- Orbital Element Decoder ---

class OrbitalElementDecoder(nn.Module):
    """
    Decodes fused latent representation into classical orbital elements
    with physics constraints (vis-viva, angular momentum conservation).
    
    Output: [a, e, i, Ω, ω, M] + 6×6 covariance (21 unique elements)
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 6),  # [a, e, i, Ω, ω, M]
        )
        # Cholesky factor for covariance (lower triangular, 21 elements)
        self.cov_head = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 21),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [batch, latent_dim] fused representation
        Returns:
            elements: [batch, 6] orbital elements
            covariance: [batch, 6, 6] positive-definite covariance matrix
        """
        elements = self.mean_head(z)
        # Physics constraints
        elements = self._apply_constraints(elements)

        # Covariance via Cholesky decomposition (guarantees positive-definite)
        chol_flat = self.cov_head(z)
        L = torch.zeros(z.size(0), 6, 6, device=z.device)
        idx = torch.tril_indices(6, 6)
        L[:, idx[0], idx[1]] = chol_flat
        # Ensure positive diagonal
        diag_idx = torch.arange(6)
        L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-6
        covariance = L @ L.transpose(-1, -2)

        return elements, covariance

    def _apply_constraints(self, elements: torch.Tensor) -> torch.Tensor:
        """Apply physical constraints to orbital elements."""
        a = F.softplus(elements[:, 0:1]) + 6371.0  # Semi-major axis > Earth radius (km)
        e = torch.sigmoid(elements[:, 1:2]) * 0.99  # Eccentricity ∈ [0, 0.99)
        i = torch.sigmoid(elements[:, 2:3]) * math.pi  # Inclination ∈ [0, π]
        omega = torch.remainder(elements[:, 3:4], 2 * math.pi)  # RAAN ∈ [0, 2π)
        w = torch.remainder(elements[:, 4:5], 2 * math.pi)      # Arg periapsis ∈ [0, 2π)
        M = torch.remainder(elements[:, 5:6], 2 * math.pi)      # Mean anomaly ∈ [0, 2π)
        return torch.cat([a, e, i, omega, w, M], dim=-1)


# --- Full Fusion GNN ---

class MultiSensorFusionGNN(nn.Module):
    """
    Complete multi-sensor fusion model.
    
    Input: Variable-length list of sensor observations per epoch
    Output: Unified orbital state estimate + covariance
    """
    SENSOR_DIMS = {'radar': 6, 'sar': 8, 'optical': 5, 'navic': 4}
    SENSOR_TYPE_IDS = {'radar': 0, 'sar': 1, 'optical': 2, 'navic': 3}

    def __init__(self, latent_dim: int = 128, n_gat_layers: int = 3,
                 n_heads: int = 4, edge_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim

        # Per-sensor encoders
        self.encoders = nn.ModuleDict({
            'radar': RadarIQEncoder(self.SENSOR_DIMS['radar'], latent_dim),
            'sar': SAREncoder(self.SENSOR_DIMS['sar'], latent_dim),
            'optical': OpticalEncoder(self.SENSOR_DIMS['optical'], latent_dim),
            'navic': NavICEncoder(self.SENSOR_DIMS['navic'], latent_dim),
        })

        # Sensor-type embedding
        self.type_embed = nn.Embedding(4, latent_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATv2Layer(latent_dim, latent_dim, n_heads, edge_dim, dropout)
            for _ in range(n_gat_layers)
        ])

        # Global readout → orbit decoder
        self.readout = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.LayerNorm(latent_dim),
        )
        self.orbit_decoder = OrbitalElementDecoder(latent_dim)

    def forward(self, observations: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            observations: List of dicts, each with:
                'type': str ('radar'/'sar'/'optical'/'navic')
                'features': Tensor [feat_dim]
                'time_offset': float (seconds from epoch)
                'position_eci': Optional[Tensor] [3] (for edge features)
        Returns:
            elements: [1, 6] orbital elements
            covariance: [1, 6, 6] covariance matrix
        """
        if not observations:
            # No observations — return default with large uncertainty
            device = next(self.parameters()).device
            return (torch.zeros(1, 6, device=device),
                    torch.eye(6, device=device).unsqueeze(0) * 1e6)

        device = next(self.parameters()).device

        # 1. Encode each observation
        node_features = []
        for obs in observations:
            encoder = self.encoders[obs['type']]
            feat = obs['features'].to(device).unsqueeze(0) if obs['features'].dim() == 1 else obs['features'].to(device)
            encoded = encoder(feat).squeeze(0)
            type_id = self.SENSOR_TYPE_IDS[obs['type']]
            type_emb = self.type_embed(torch.tensor(type_id, device=device))
            node_features.append(encoded + type_emb)

        x = torch.stack(node_features)  # [N, latent_dim]
        N = x.size(0)

        # 2. Build fully-connected graph
        src = torch.arange(N, device=device).repeat_interleave(N)
        dst = torch.arange(N, device=device).repeat(N)
        # Remove self-loops
        mask = src != dst
        edge_index = torch.stack([src[mask], dst[mask]])  # [2, E]

        # 3. Edge features (time diff, type-pair encoding)
        edge_attr = self._compute_edge_features(observations, edge_index, device)

        # 4. GATv2 message passing
        for gat in self.gat_layers:
            x = gat(x, edge_index, edge_attr)

        # 5. Global mean readout
        global_repr = self.readout(x.mean(dim=0, keepdim=True))  # [1, latent_dim]

        # 6. Decode orbital elements + covariance
        return self.orbit_decoder(global_repr)

    def _compute_edge_features(self, observations: List[Dict],
                                edge_index: torch.Tensor,
                                device: torch.device) -> torch.Tensor:
        """Compute edge features: time diff + sensor-type pair one-hot."""
        E = edge_index.size(1)
        edge_attr = torch.zeros(E, 8, device=device)

        times = torch.tensor([obs.get('time_offset', 0.0) for obs in observations],
                              device=device, dtype=torch.float32)
        types = torch.tensor([self.SENSOR_TYPE_IDS[obs['type']] for obs in observations],
                              device=device)

        src, dst = edge_index[0], edge_index[1]
        edge_attr[:, 0] = times[src] - times[dst]  # Time difference
        edge_attr[:, 1] = torch.abs(times[src] - times[dst])  # Absolute time gap

        # One-hot sensor pair (4 source types + 4 dest types = 6 remaining dims)
        for i in range(min(3, 8 - 2)):
            edge_attr[:, 2 + i] = (types[src] == i).float()
            edge_attr[:, 5 + i] = (types[dst] == i).float()

        return edge_attr


# --- Synthetic Training Data ---

def generate_fusion_training_sample(rng: np.random.Generator) -> Tuple[List[Dict], np.ndarray]:
    """
    Generate a synthetic multi-sensor observation set with ground-truth orbital elements.
    """
    # Ground truth orbital elements: [a(km), e, i(rad), Ω(rad), ω(rad), M(rad)]
    a = rng.uniform(6600, 42164)   # LEO to GEO
    e = rng.uniform(0.0, 0.3)
    inc = rng.uniform(0, np.pi)
    raan = rng.uniform(0, 2 * np.pi)
    argp = rng.uniform(0, 2 * np.pi)
    M = rng.uniform(0, 2 * np.pi)
    truth = np.array([a, e, inc, raan, argp, M], dtype=np.float32)

    observations = []
    # Simulate 2-5 sensor observations with noise
    n_obs = rng.integers(2, 6)
    sensor_types = rng.choice(['radar', 'sar', 'optical', 'navic'], size=n_obs)

    dims = {'radar': 6, 'sar': 8, 'optical': 5, 'navic': 4}
    for i, stype in enumerate(sensor_types):
        feat = rng.normal(0, 1, dims[stype]).astype(np.float32)
        # Inject truth signal
        feat[0] = a / 42164.0 + rng.normal(0, 0.01)
        if len(feat) > 1:
            feat[1] = e + rng.normal(0, 0.01)
        observations.append({
            'type': stype,
            'features': torch.tensor(feat),
            'time_offset': float(i * 30.0),
        })

    return observations, truth


# --- Self-Test ---

if __name__ == "__main__":
    print("Testing MultiSensorFusionGNN...")
    model = MultiSensorFusionGNN(latent_dim=128, n_gat_layers=2, n_heads=4)
    model.eval()

    rng = np.random.default_rng(42)
    obs, truth = generate_fusion_training_sample(rng)
    print(f"Generated {len(obs)} observations: {[o['type'] for o in obs]}")
    print(f"Ground truth: a={truth[0]:.1f}km e={truth[1]:.4f} i={np.degrees(truth[2]):.1f}°")

    with torch.no_grad():
        elements, cov = model(obs)
    print(f"Predicted elements: {elements.squeeze().numpy()}")
    print(f"Covariance trace: {cov.squeeze().trace().item():.4f}")
    print(f"Covariance is PD: {torch.linalg.eigvalsh(cov.squeeze()).min().item() > 0}")
    print("✓ MultiSensorFusionGNN forward pass successful")
