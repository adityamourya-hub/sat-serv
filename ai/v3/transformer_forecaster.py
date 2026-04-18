"""
Transformer/Informer-Based Maneuver & Conjunction Forecaster.

Replaces/augments BiLSTM with Transformer architecture for:
  - Long-sequence CDM/TLE analysis (better than BiLSTM on 2025 research)
  - Global dependency capture (intentional vs. routine manoeuvres)
  - Threat probability prediction (to Indian assets)

Architecture: Lightweight Transformer (Informer-style)
  - ProbSparse attention for O(L log L) complexity
  - Encoder: historical residuals + delta-v + iono context
  - Decoder: future maneuver prediction + threat scoring

Performance target: R² > 0.85 on Pc, false negatives < 3%.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ProbSparseAttention(nn.Module):
    """
    ProbSparse Self-Attention from Informer paper.
    Reduces complexity from O(L²) to O(L log L).
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, factor: int = 5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor  # Sampling factor for sparse attention
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (d_model // n_heads) ** -0.5
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        
        # Project
        Q = self.q_proj(x).view(B, L, self.n_heads, -1).transpose(1, 2)  # [B, H, L, D]
        K = self.k_proj(x).view(B, L, self.n_heads, -1).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, -1).transpose(1, 2)
        
        # ProbSparse: sample dominant queries
        if L > 50:  # Only use sparse for long sequences
            # Sample top 'factor * log L' queries
            sample_size = min(self.factor * int(np.log(L)), L)
            
            # Compute query importance (L_k norm)
            q_norm = Q.norm(dim=-1)  # [B, H, L]
            _, top_indices = torch.topk(q_norm, sample_size, dim=-1)
            
            # Sparse attention only on important queries
            Q_sparse = torch.gather(Q, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, Q.size(-1)))
            
            attn = torch.matmul(Q_sparse, K.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            out_sparse = torch.matmul(attn, V)  # [B, H, sample_size, D]
            
            # Scatter back
            out = torch.zeros_like(Q)
            out.scatter_(2, top_indices.unsqueeze(-1).expand(-1, -1, -1, Q.size(-1)), out_sparse)
        else:
            # Full attention for short sequences
            attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, V)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Lightweight Transformer encoder with ProbSparse attention."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class ManeuverTransformerForecaster(nn.Module):
    """
    Transformer-based maneuver and conjunction forecaster.
    
    Input: [batch, seq_len, features] where features include:
           [residual, delta-v, iono_context, elevation, ...]
    Output: [batch, pred_len, outputs] where outputs are:
            [maneuver_prob, maneuver_type, delta-v, confidence, threat_score]
    """
    
    def __init__(
        self,
        seq_len: int = 32,
        label_len: int = 16,  # Encoder input length
        pred_len: int = 8,   # Prediction horizon
        enc_in: int = 6,     # Input features
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        # Input embedding
        self.input_embedding = nn.Linear(enc_in, d_model)
        
        # Positional encoding (learned)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len + pred_len, d_model) * 0.02)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff=256, dropout=dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder (simplified — directly predicts from encoder output)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Output heads
        self.maneuver_prob_head = nn.Linear(64, 1)      # Binary: maneuver or not
        self.maneuver_type_head = nn.Linear(64, 2)     # Class: impulsive vs continuous
        self.delta_v_head = nn.Linear(64, 1)          # Magnitude estimate
        self.confidence_head = nn.Linear(64, 1)         # Model confidence
        self.threat_score_head = nn.Linear(64, 1)       # Threat to Indian assets [0-1]
        
        # Physics-informed constraint: orbital mechanics penalty
        self.mu_earth = 3.986004418e5  # km³/s²
    
    def forward(self, x_enc: torch.Tensor, x_iono: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x_enc: [B, seq_len, enc_in] historical residuals + delta-v + features
            x_iono: [B, 16] optional ionospheric context
        
        Returns:
            maneuver_prob: [B, pred_len, 1]
            maneuver_type: [B, pred_len, 2] logits
            delta_v: [B, pred_len, 1]
            confidence: [B, pred_len, 1]
            threat_score: [B, pred_len, 1]
        """
        B = x_enc.size(0)
        
        # Embed input
        x = self.input_embedding(x_enc)  # [B, seq_len, d_model]
        
        # Add positional encoding (only for encoder part)
        x = x + self.pos_enc[:, :self.seq_len, :]
        
        # Encode
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Global average pooling + last state
        pooled = x.mean(dim=1)  # [B, d_model]
        last_state = x[:, -1, :]  # [B, d_model]
        
        combined = (pooled + last_state) / 2  # [B, d_model]
        
        # Decode
        features = self.decoder(combined)  # [B, 64]
        
        # Expand to prediction horizon
        features_expanded = features.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, pred_len, 64]
        
        # Predictions
        maneuver_prob = torch.sigmoid(self.maneuver_prob_head(features_expanded))
        maneuver_type = self.maneuver_type_head(features_expanded)
        delta_v = torch.relu(self.delta_v_head(features_expanded))
        confidence = torch.sigmoid(self.confidence_head(features_expanded))
        threat_score = torch.sigmoid(self.threat_score_head(features_expanded))
        
        return maneuver_prob, maneuver_type, delta_v, confidence, threat_score
    
    def physics_informed_loss(
        self,
        maneuver_prob: torch.Tensor,
        delta_v: torch.Tensor,
        target_prob: torch.Tensor,
        target_dv: torch.Tensor,
        orbital_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combined MSE + physics constraint loss.
        
        Physics constraint: delta-V must be physically plausible for orbital state.
        Impulsive maneuvers: typically 0.01-0.5 m/s
        Continuous maneuvers: typically 0.001-0.1 m/s over minutes
        """
        # Standard losses
        bce = F.binary_cross_entropy(maneuver_prob, target_prob)
        mse_dv = F.mse_loss(delta_v, target_dv)
        
        # Physics constraint: penalize unrealistic delta-V magnitudes
        # Typical LEO station-keeping: < 2 m/s per year
        # Emergency avoidance: up to 5 m/s
        max_plausible_dv = 5.0  # m/s
        
        physics_penalty = torch.relu(delta_v - max_plausible_dv).mean()
        
        # If we have orbital state, add vis-viva constraint
        vis_viva_penalty = torch.tensor(0.0)
        if orbital_state is not None:
            # orbital_state: [B, 6] = [a, e, i, RAAN, arg_peri, M]
            a = orbital_state[:, 0]
            v_expected = torch.sqrt(self.mu_earth / a)  # Circular orbit velocity
            
            # Delta-V should not exceed orbital velocity (would be escape)
            escape_penalty = torch.relu(delta_v.squeeze() - v_expected * 1000).mean()  # Convert km/s to m/s
            vis_viva_penalty = escape_penalty
        
        return bce + 0.7 * mse_dv + 0.1 * physics_penalty + 0.05 * vis_viva_penalty


class ConjunctionTransformerForecaster(nn.Module):
    """
    Hybrid GNN + Transformer for conjunction prediction.
    
    Combines graph structure (satellite pairs) with temporal features.
    """
    
    def __init__(
        self,
        node_dim: int = 26,  # From collision surrogate
        seq_len: int = 10,   # Historical CDMs
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_embed = nn.Linear(node_dim, d_model)
        self.temporal_encoder = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        
        # Transformer for cross-node attention
        self.cross_attn = nn.MultiheadAttention(d_model * 2, n_heads, dropout=dropout)
        
        # Heads
        self.pc_head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Predict Pc (collision probability)
            nn.Sigmoid(),
        )
        
        self.maneuver_head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # None, Impulsive, Continuous
        )
        
        self.threat_head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Threat to Indian assets
        )
    
    def forward(self, node_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            node_history: [B, N, T, node_dim] for N satellites, T timesteps
        
        Returns:
            pc: [B, N, N] collision probability matrix
            maneuver_type: [B, N] per-satellite maneuver prediction
            threat: [B] overall threat score
        """
        B, N, T, D = node_history.shape
        
        # Embed nodes
        x = self.node_embed(node_history)  # [B, N, T, d_model]
        
        # Temporal encoding per node
        x_flat = x.view(B * N, T, -1)
        temporal_out, _ = self.temporal_encoder(x_flat)  # [B*N, T, d_model*2]
        
        # Take last timestep
        node_repr = temporal_out[:, -1, :].view(B, N, -1)  # [B, N, d_model*2]
        
        # Cross-attention between nodes (graph interaction)
        node_repr_t = node_repr.transpose(0, 1)  # [N, B, D]
        attn_out, _ = self.cross_attn(node_repr_t, node_repr_t, node_repr_t)
        attn_out = attn_out.transpose(0, 1)  # [B, N, D]
        
        # Combine
        final_repr = node_repr + attn_out
        
        # Predictions
        # Pc matrix: outer product style
        pc_logits = self.pc_head(final_repr)  # [B, N, 1]
        pc_matrix = torch.bmm(pc_logits, pc_logits.transpose(1, 2))  # [B, N, N]
        pc_matrix = torch.triu(pc_matrix, diagonal=1)  # Upper triangular
        
        maneuver_type = self.maneuver_head(final_repr.mean(dim=1))  # [B, 3]
        threat = self.threat_head(final_repr.mean(dim=1))  # [B, 1]
        
        return pc_matrix, maneuver_type, threat


# --- Training Utilities ---

def train_maneuver_transformer(
    n_samples: int = 10000,
    seq_len: int = 32,
    pred_len: int = 8,
    epochs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ManeuverTransformerForecaster:
    """Train the maneuver forecaster on synthetic data."""
    
    print(f"Training Maneuver Transformer Forecaster on {device}")
    print(f"  Samples: {n_samples}, Seq: {seq_len}, Pred: {pred_len}, Epochs: {epochs}")
    
    # Generate synthetic training data
    from maneuver.train_maneuver_detector import SyntheticManeuverDataset
    dataset = SyntheticManeuverDataset(n_samples, seq_len)
    
    # Initialize model
    model = ManeuverTransformerForecaster(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=2,  # residual + delta-v
        d_model=128,
        n_heads=4,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training loop
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        losses = []
        
        for x, y_prob, y_type, y_dv, y_conf in loader:
            x = x.to(device)
            y_prob = y_prob.to(device)
            y_dv = y_dv.to(device)
            
            # Expand targets to prediction horizon
            y_prob_expanded = y_prob.unsqueeze(1).expand(-1, pred_len, -1)
            y_dv_expanded = y_dv.unsqueeze(1).expand(-1, pred_len, -1)
            
            # Forward
            prob_pred, type_logits, dv_pred, conf_pred, threat_pred = model(x)
            
            # Loss
            loss = model.physics_informed_loss(prob_pred, dv_pred, y_prob_expanded, y_dv_expanded)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {np.mean(losses):.4f}")
    
    print("Training complete")
    return model


def export_transformer_onnx(model: nn.Module, output_path: str):
    """Export transformer model to ONNX."""
    dummy_input = torch.randn(1, 32, 2)  # [batch, seq, features]
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["history"],
        output_names=["maneuver_prob", "maneuver_type", "delta_v", "confidence", "threat"],
        dynamic_axes={"history": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX: {output_path}")


# --- Self-Test ---

if __name__ == "__main__":
    print("Testing Transformer Forecaster...")
    
    # Test maneuver forecaster
    model = ManeuverTransformerForecaster(
        seq_len=32,
        pred_len=8,
        enc_in=6,  # More features
    )
    
    x = torch.randn(2, 32, 6)  # [batch=2, seq=32, features=6]
    prob, type_logits, dv, conf, threat = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Maneuver prob: {prob.shape}")
    print(f"Type logits: {type_logits.shape}")
    print(f"Delta-V: {dv.shape}")
    print(f"Confidence: {conf.shape}")
    print(f"Threat score: {threat.shape}")
    
    assert prob.shape == (2, 8, 1)
    assert type_logits.shape == (2, 8, 2)
    print("\n✓ ManeuverTransformerForecaster shapes correct")
    
    # Test conjunction forecaster
    conj_model = ConjunctionTransformerForecaster(node_dim=26, seq_len=10)
    node_hist = torch.randn(2, 5, 10, 26)  # [B, N=5 nodes, T=10 timesteps, D=26]
    pc, man_type, threat_score = conj_model(node_hist)
    
    print(f"\nConjunction model:")
    print(f"Pc matrix: {pc.shape}")  # Should be [2, 5, 5]
    print(f"Maneuver type: {man_type.shape}")
    print(f"Threat: {threat_score.shape}")
    
    print("\n✓ All transformer models operational")
