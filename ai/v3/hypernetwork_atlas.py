"""
Global Latent Physics Atlas (GLPA) - Foundation Model

Architecture:
- Atlas Encoder E_θ: Encodes (state, atmosphere, vehicle_class) → latent physics code z
- Atlas Decoder D_φ: Decodes z → DPHNO weights {W_H, W_R, W_g}

Training on H100 cluster with 10^6 synthetic trajectories.

Reference: ai/model_arch.md - HF-PGFM Dominant Architecture
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class AtlasEncoder(nn.Module):
    """
    Encodes state + atmosphere + vehicle_class into latent physics code z
    
    Maps: (state, atmosphere, vehicle_class) → z ∈ ℝ^256
    """
    
    def __init__(self, state_dim: int = 6, latent_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Atmosphere encoder (F10.7, Ap, altitude profile)
        # Atmosphere vector: 64 dimensions (simplified NRLMSISE-00 profile)
        self.atmos_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Vehicle class embedding
        # 10 vehicle classes: HGV, ballistic, satellite, debris, etc.
        self.vehicle_embedding = nn.Embedding(num_embeddings=10, embedding_dim=32)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)  # Output latent physics code z
        )
    
    def forward(self, state: torch.Tensor, atmosphere: torch.Tensor, 
                vehicle_class: torch.Tensor) -> torch.Tensor:
        """
        Encode state, atmosphere, and vehicle class into latent physics code
        
        Args:
            state: (batch, 6) state [x,y,z,vx,vy,vz]
            atmosphere: (batch, 64) atmosphere profile
            vehicle_class: (batch,) vehicle class indices
        
        Returns:
            z: (batch, 256) latent physics code
        """
        state_feat = self.state_encoder(state)
        atmos_feat = self.atmos_encoder(atmosphere)
        vehicle_feat = self.vehicle_embedding(vehicle_class)
        
        combined = torch.cat([state_feat, atmos_feat, vehicle_feat], dim=-1)
        z = self.fusion(combined)
        
        return z


class AtlasDecoder(nn.Module):
    """
    Decodes latent physics code z into DPHNO weights
    
    Maps: z → {W_H, W_R, W_g}
    
    The weights are the parameters of the DPHNO networks:
    - W_H: Hamiltonian network weights
    - W_R: Dissipation network weights
    - W_g: Control network weights
    """
    
    def __init__(self, latent_dim: int = 256, dphno_config: Dict = None):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        if dphno_config is None:
            dphno_config = {
                'state_dim': 6,
                'hidden_dim': 128
            }
        
        # Decoder outputs weight generators
        self.decoder_H = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self._count_params_H(dphno_config))
        )
        
        self.decoder_R = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self._count_params_R(dphno_config))
        )
        
        self.decoder_g = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self._count_params_g(dphno_config))
        )
    
    def _count_params_H(self, config: Dict) -> int:
        """Count parameters in Hamiltonian network (4×128 Tanh)"""
        hidden_dim = config['hidden_dim']
        state_dim = config['state_dim']
        # state_dim → hidden → hidden → hidden → hidden → 1
        return (state_dim * hidden_dim + hidden_dim + 
                hidden_dim * hidden_dim + hidden_dim +
                hidden_dim * hidden_dim + hidden_dim +
                hidden_dim * hidden_dim + hidden_dim +
                hidden_dim * 1 + 1)
    
    def _count_params_R(self, config: Dict) -> int:
        """Count parameters in dissipation network (2×128 Tanh)"""
        hidden_dim = config['hidden_dim']
        state_dim = config['state_dim']
        # state_dim → hidden → hidden → 21
        return (state_dim * hidden_dim + hidden_dim +
                hidden_dim * hidden_dim + hidden_dim +
                hidden_dim * 21 + 21)
    
    def _count_params_g(self, config: Dict) -> int:
        """Count parameters in control network (2×64 Tanh)"""
        hidden_dim = config['hidden_dim']
        state_dim = config['state_dim']
        # state_dim → 64 → 64 → state_dim
        return (state_dim * (hidden_dim // 2) + (hidden_dim // 2) +
                (hidden_dim // 2) * (hidden_dim // 2) + (hidden_dim // 2) +
                (hidden_dim // 2) * state_dim + state_dim)
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode z into DPHNO weights
        
        Args:
            z: (batch, 256) latent physics code
        
        Returns:
            weights: Dictionary with keys 'W_H', 'W_R', 'W_g'
        """
        weights_H = self.decoder_H(z)
        weights_R = self.decoder_R(z)
        weights_g = self.decoder_g(z)
        
        return {
            'W_H': weights_H,
            'W_R': weights_R,
            'W_g': weights_g
        }


class HypernetworkAtlas(nn.Module):
    """
    Complete GLPA: Encoder + Decoder
    
    Architecture:
        (state, atmosphere, vehicle_class) → z → {W_H, W_R, W_g}
    
    This enables zero-day generalization: given a new vehicle class,
    encode it to get z_k, then decode to get DPHNO weights for that class.
    """
    
    def __init__(self, state_dim: int = 6, latent_dim: int = 256):
        super().__init__()
        
        self.encoder = AtlasEncoder(state_dim, latent_dim)
        self.decoder = AtlasDecoder(latent_dim)
    
    def forward(self, state: torch.Tensor, atmosphere: torch.Tensor,
                vehicle_class: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Encode and decode to get DPHNO weights
        
        Args:
            state: (batch, 6) state
            atmosphere: (batch, 64) atmosphere profile
            vehicle_class: (batch,) vehicle class indices
        
        Returns:
            z: (batch, 256) latent physics code
            weights: Dictionary with DPHNO weights
        """
        z = self.encoder(state, atmosphere, vehicle_class)
        weights = self.decoder(z)
        return z, weights
    
    def project_vehicle_class(self, vehicle_class: int, atmosphere: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Project a single vehicle class to DPHNO weights
        
        This is the "projection" step from H100 → M2 Air:
        For each vehicle class k: z_k = E_θ(class_k), then {W_H^k, W_R^k, W_g^k} = D_φ(z_k)
        
        Args:
            vehicle_class: Single vehicle class index
            atmosphere: (batch, 64) atmosphere profile
        
        Returns:
            weights: Dictionary with DPHNO weights for this vehicle class
        """
        batch_size = atmosphere.shape[0]
        device = atmosphere.device
        
        # Create dummy state (will be ignored during projection)
        dummy_state = torch.zeros(batch_size, 6, device=device)
        vehicle_class_tensor = torch.full((batch_size,), vehicle_class, 
                                         dtype=torch.long, device=device)
        
        z, weights = self.forward(dummy_state, atmosphere, vehicle_class_tensor)
        
        return weights


class AtlasProjection(nn.Module):
    """
    Projection Layer: Distills GLPA to sparse local equations
    
    For deployment on M2 Air, we need to project the dense hypernetwork
    to sparse, interpretable equations that can run in <10ms.
    """
    
    def __init__(self, latent_dim: int = 256, num_terms: int = 10):
        super().__init__()
        
        # Learn to select most important terms from hypernetwork
        self.term_selector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_terms),
            nn.Softmax(dim=-1)  # Select top-k terms
        )
        
        # Learn coefficient for each term
        self.coefficient_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_terms)
        )
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Project latent code to sparse equation
        
        Args:
            z: (batch, 256) latent physics code
        
        Returns:
            equation: Dictionary with 'term_weights' and 'coefficients'
        """
        term_weights = self.term_selector(z)  # (batch, num_terms)
        coefficients = self.coefficient_net(z)  # (batch, num_terms)
        
        return {
            'term_weights': term_weights,
            'coefficients': coefficients
        }
