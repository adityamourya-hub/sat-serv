"""
PatchTST-JEPA Hybrid for Satellite Pattern-of-Life (PoL) Detection
Implements capability (g): Transformer-based PoL with 2-revolution latency

References:
- MIT ARCLab PoL Challenge 2024/2025 (winners used CNN-LSTM → Diffusion-Transformers)
- "Learning Satellite Pattern-of-Life Identification: A Diffusion-Transformer Approach" (arXiv 2412.10814, 2025)
- JEPA (Joint-Embedding Predictive Architecture) for self-supervised pre-training

Architecture:
- Patch Time Series Transformer for long-horizon dependencies
- JEPA for self-supervised pre-training on unlabeled TLE sequences
- 2-rev detection latency (~3 hours for LEO)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class PatchEmbedding(nn.Module):
    """Embed time series patches into latent space"""
    
    def __init__(self, patch_size: int, d_model: int, in_channels: int = 6):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.projection = nn.Linear(patch_size * in_channels, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features] where features = [pos(3), vel(3)]
        Returns:
            [batch, n_patches, d_model]
        """
        batch_size, seq_len, n_features = x.shape
        n_patches = seq_len // self.patch_size
        
        # Reshape into patches
        x = x[:, :n_patches * self.patch_size, :]  # Trim to multiple of patch_size
        x = x.reshape(batch_size, n_patches, self.patch_size * n_features)
        
        return self.projection(x)


class JEPAPredictor(nn.Module):
    """
    Joint-Embedding Predictive Architecture predictor.
    Predicts latent representation of future patches from context.
    """
    
    def __init__(self, d_model: int, n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # Context encoder (students network)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Target encoder (EMA-updated, not trained directly)
        self.target_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.target_ema_decay = 0.996
        
        # Predictor: context -> target prediction
        predictor_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.predictor = nn.TransformerDecoder(predictor_layer, num_layers=2)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000)
        
    def forward(self, 
                context_patches: torch.Tensor,
                target_patches: Optional[torch.Tensor] = None,
                mode: str = "train") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            context_patches: [batch, n_context, d_model]
            target_patches: [batch, n_target, d_model] (only for target encoding in train mode)
            mode: "train" or "inference"
        Returns:
            predictions, targets (targets only returned in train mode for loss computation)
        """
        batch_size = context_patches.size(0)
        
        # Add positional encoding
        context_patches = self.pos_encoding(context_patches)
        
        # Encode context
        context_repr = self.context_encoder(context_patches)
        
        if mode == "train" and target_patches is not None:
            # Encode targets with EMA-updated target encoder
            target_patches = self.pos_encoding(target_patches)
            with torch.no_grad():
                target_repr = self.target_encoder(target_patches)
            
            # Predict targets from context
            predictions = self.predictor(target_repr, context_repr)
            return predictions, target_repr
        else:
            # Inference: predict future without target
            # Use learned query tokens
            n_target = context_patches.size(1) // 2  # Predict half the sequence
            target_queries = torch.zeros(batch_size, n_target, self.d_model, 
                                        device=context_patches.device)
            target_queries = self.pos_encoding(target_queries)
            
            predictions = self.predictor(target_queries, context_repr)
            return predictions, None
    
    @torch.no_grad()
    def update_target_encoder(self):
        """Exponential moving average update of target encoder"""
        for param_t, param_s in zip(self.target_encoder.parameters(), 
                                       self.context_encoder.parameters()):
            param_t.data = self.target_ema_decay * param_t.data + \
                          (1 - self.target_ema_decay) * param_s.data


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :]


class PoLClassificationHead(nn.Module):
    """Classification head for PoL categories"""
    
    POL_CLASSES = [
        "station_keeper",      # Regular station-keeping maneuvers
        "deorbit_prep",        # Preparing for deorbit
        "phasing",             # Phasing maneuver
        "rendezvous",          # Approaching another object
        "evasive",             # Collision avoidance
        "asat_prep",           # Potential ASAT preparation
        "debris",              # Inactive debris
        "unknown"              # Unclassified
    ]
    
    def __init__(self, d_model: int, n_classes: int = 8):
        super().__init__()
        self.n_classes = n_classes
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes)
        )
        
    def forward(self, sequence_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_repr: [batch, d_model] (pooled representation)
        Returns:
            [batch, n_classes] logits
        """
        return self.classifier(sequence_repr)


class PatchTSTJEPA(nn.Module):
    """
    Complete PatchTST-JEPA model for Pattern-of-Life detection.
    
    Features:
    - Self-supervised pre-training via JEPA
    - Fine-tuning for 8-class PoL classification
    - 2-revolution detection latency for LEO (~3 hours)
    """
    
    def __init__(self,
                 patch_size: int = 12,           # ~2 hours of observations at 10-min intervals
                 d_model: int = 256,
                 n_encoder_layers: int = 6,
                 n_predictor_layers: int = 4,
                 n_heads: int = 8,
                 in_channels: int = 6,           # [x, y, z, vx, vy, vz]
                 n_pol_classes: int = 8):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, d_model, in_channels)
        
        # JEPA architecture
        self.jepa = JEPAPredictor(d_model, n_predictor_layers, n_heads)
        
        # Classification head (for fine-tuning)
        self.classifier = PoLClassificationHead(d_model, n_pol_classes)
        
        # Anomaly detection head (for maneuver detection)
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def pretrain_step(self, 
                     observations: torch.Tensor,
                     mask_ratio: float = 0.6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Self-supervised pre-training step with JEPA.
        
        Args:
            observations: [batch, seq_len, 6] (position + velocity)
            mask_ratio: fraction of sequence to mask/predict
        Returns:
            loss, metrics dict
        """
        # Create patches
        patches = self.patch_embed(observations)  # [batch, n_patches, d_model]
        
        # Split into context and target
        n_patches = patches.size(1)
        n_context = int(n_patches * (1 - mask_ratio))
        
        # Random split
        perm = torch.randperm(n_patches)
        context_idx = perm[:n_context].sort()[0]
        target_idx = perm[n_context:].sort()[0]
        
        context_patches = patches[:, context_idx, :]
        target_patches = patches[:, target_idx, :]
        
        # Forward pass
        predictions, targets = self.jepa(context_patches, target_patches, mode="train")
        
        # JEPA loss: prediction error in latent space
        loss = F.mse_loss(predictions, targets)
        
        # Update target encoder (EMA)
        self.jepa.update_target_encoder()
        
        metrics = {
            "jepa_loss": loss.item(),
            "context_ratio": 1 - mask_ratio
        }
        
        return loss, metrics
    
    def forward(self, observations: torch.Tensor) -> dict:
        """
        Forward pass for inference or fine-tuning.
        
        Args:
            observations: [batch, seq_len, 6] (position + velocity)
        Returns:
            dict with 'pol_logits', 'anomaly_score', 'embedding'
        """
        # Create patches
        patches = self.patch_embed(observations)
        
        # Encode full sequence
        full_repr = self.jepa.context_encoder(self.jepa.pos_encoding(patches))
        
        # Pool representation (mean over patches)
        pooled = full_repr.mean(dim=1)  # [batch, d_model]
        
        # Classification outputs
        pol_logits = self.classifier(pooled)
        anomaly_score = self.anomaly_head(pooled)
        
        return {
            "pol_logits": pol_logits,
            "pol_class": self.classifier.POL_CLASSES[pol_logits.argmax(dim=1).item()] \
                        if pol_logits.size(0) == 1 else \
                        [self.classifier.POL_CLASSES[i] for i in pol_logits.argmax(dim=1).tolist()],
            "anomaly_score": anomaly_score.squeeze(-1),
            "embedding": pooled
        }
    
    def detect_maneuver(self, 
                       tle_history: List[dict],
                       threshold: float = 0.9) -> Tuple[bool, float, str]:
        """
        Detect if a maneuver occurred in the TLE history.
        
        Args:
            tle_history: List of TLE-derived states with 'timestamp', 'position', 'velocity'
            threshold: Anomaly score threshold for maneuver detection
        Returns:
            (maneuver_detected, confidence, predicted_pol_class)
        """
        # Convert to tensor
        states = torch.tensor([
            [s['position'][0], s['position'][1], s['position'][2],
             s['velocity'][0], s['velocity'][1], s['velocity'][2]]
            for s in tle_history
        ], dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 6]
        
        with torch.no_grad():
            outputs = self.forward(states)
        
        anomaly = outputs['anomaly_score'].item()
        detected = anomaly > threshold
        
        pol_class_idx = outputs['pol_logits'].argmax(dim=1).item()
        pol_class = self.classifier.POL_CLASSES[pol_class_idx]
        
        return detected, anomaly, pol_class


def train_pol_patchtst():
    """Training orchestrator for PatchTST-JEPA"""
    # This would be called by the training pipeline
    # See ai/training/train_pol.py for full training loop
    pass


if __name__ == "__main__":
    # Quick test
    model = PatchTSTJEPA(patch_size=12, d_model=256)
    
    # Simulate 2-rev worth of observations (~12 hours at 10-min intervals = 72 obs)
    batch_size = 4
    seq_len = 72
    obs = torch.randn(batch_size, seq_len, 6)
    
    # Pre-training step
    loss, metrics = model.pretrain_step(obs, mask_ratio=0.6)
    print(f"Pre-training loss: {loss.item():.4f}")
    
    # Inference
    outputs = model(obs)
    print(f"PoL logits shape: {outputs['pol_logits'].shape}")
    print(f"Anomaly score: {outputs['anomaly_score']}")
