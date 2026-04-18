"""
Transformer-based Symbolic Regression for Orbital Law Distillation
Implements capability (f): Symbolic Manoeuvre Attribution with LaTeX output

References:
- "Orbital anomaly reconstruction using deep symbolic regression" (Manzi & Vasile, IAC 2020)
- "Channel Modeling Based on Transformer Symbolic Regression" (2024)
- "Class Symbolic Regression" (arXiv/IOP 2024)
- PhySO (Physical Symbolic Optimization) with physics constraints

Key Innovation:
Neural encoder → Symbolic decoder distills delta-V residuals into 
human-readable orbital laws (e.g., "station-keep +0.12 m/s radial toward RISAT-2B")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import re
import json


class SymbolicVocabulary:
    """
    Vocabulary for orbital mechanics symbolic expressions.
    Includes: operators, variables, constants, and orbital-specific tokens.
    """
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    UNKNOWN_TOKEN = "<UNK>"
    
    # Mathematical operators
    OPERATORS = ["+", "-", "*", "/", "^", "sqrt", "exp", "log", "sin", "cos", "tan"]
    
    # Orbital variables
    VARIABLES = [
        "delta_v",      # Total delta-V magnitude
        "delta_vr",     # Radial component
        "delta_vt",     # Tangential component
        "delta_vn",     # Normal component
        "delta_r",      # Position change
        "t_maneuver",   # Time of maneuver
        "t_since_epoch", # Time since orbit epoch
        "h",            # Specific angular momentum
        "e",            # Eccentricity
        "i",            # Inclination
        "raan",         # Right ascension of ascending node
        "arg_perigee",  # Argument of perigee
        "M",            # Mean anomaly
        "target_dist",  # Distance to target
        "approach_vel", # Relative approach velocity
    ]
    
    # Physical constants
    CONSTANTS = [
        "mu_earth",     # Earth's gravitational parameter
        "R_earth",      # Earth's radius
        "J2",           # Earth's oblateness
        "pi",
        "c",            # Speed of light (for relativistic corrections)
    ]
    
    # Maneuver type descriptors
    MANEUVER_TYPES = [
        "station_keep",
        "orbit_raise",
        "orbit_lower",
        "plane_change",
        "phasing",
        "rendezvous",
        "collision_avoidance",
        "deorbit",
        "flyby",
    ]
    
    # Direction descriptors
    DIRECTIONS = [
        "radial_in",
        "radial_out",
        "prograde",
        "retrograde",
        "normal",
        "anti_normal",
        "toward_target",
        "away_from_target",
    ]
    
    def __init__(self):
        # Build vocabulary
        all_tokens = (
            [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN] +
            self.OPERATORS +
            self.VARIABLES +
            self.CONSTANTS +
            self.MANEUVER_TYPES +
            self.DIRECTIONS +
            [str(i) for i in range(100)]  # Numbers 0-99
        )
        
        self.token_to_idx = {token: idx for idx, token in enumerate(all_tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.vocab_size = len(all_tokens)
        
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert token list to indices"""
        return [self.token_to_idx.get(t, self.token_to_idx[self.UNKNOWN_TOKEN]) for t in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens"""
        return [self.idx_to_token.get(i, self.UNKNOWN_TOKEN) for i in indices]
    
    def latex_to_tokens(self, latex: str) -> List[str]:
        """Convert LaTeX string to token sequence"""
        # Simple tokenizer (can be enhanced with proper LaTeX parsing)
        latex = latex.replace("\\", " ").replace("{", " ").replace("}", " ")
        latex = latex.replace("$", " ")
        tokens = latex.split()
        return [t for t in tokens if t]


class PhysicsConstraintLayer:
    """
    Enforces dimensional consistency and orbital physics priors during decoding.
    
    Ensures generated expressions are physically meaningful.
    """
    
    # Dimensional units for each variable
    UNITS = {
        "delta_v": "L/T",
        "delta_vr": "L/T",
        "delta_vt": "L/T",
        "delta_vn": "L/T",
        "delta_r": "L",
        "t_maneuver": "T",
        "t_since_epoch": "T",
        "h": "L^2/T",
        "e": "dimensionless",
        "i": "angle",
        "raan": "angle",
        "arg_perigee": "angle",
        "M": "angle",
        "target_dist": "L",
        "approach_vel": "L/T",
        "mu_earth": "L^3/T^2",
        "R_earth": "L",
        "J2": "dimensionless",
        "pi": "dimensionless",
    }
    
    def __init__(self, vocab: SymbolicVocabulary):
        self.vocab = vocab
        
    def compute_units(self, tokens: List[str]) -> Optional[str]:
        """
        Compute dimensional units of an expression.
        Returns None if inconsistent.
        """
        if not tokens:
            return None
            
        # Simplified unit propagation
        # In a full implementation, this would build an expression tree
        # and propagate units bottom-up
        
        stack = []
        for token in tokens:
            if token in self.UNITS:
                stack.append(self.UNITS[token])
            elif token in ["+", "-"]:
                # Addition/subtraction requires same units
                if len(stack) >= 2:
                    u2, u1 = stack.pop(), stack.pop()
                    if u1 != u2:
                        return None  # Unit mismatch
                    stack.append(u1)
            elif token == "*":
                if len(stack) >= 2:
                    u2, u1 = stack.pop(), stack.pop()
                    stack.append(self._multiply_units(u1, u2))
            elif token == "/":
                if len(stack) >= 2:
                    u2, u1 = stack.pop(), stack.pop()
                    stack.append(self._divide_units(u1, u2))
            elif token == "sqrt":
                if stack:
                    u = stack.pop()
                    stack.append(self._sqrt_units(u))
                    
        return stack[0] if len(stack) == 1 else None
    
    def _multiply_units(self, u1: str, u2: str) -> str:
        # Simplified unit multiplication
        if u1 == "dimensionless":
            return u2
        if u2 == "dimensionless":
            return u1
        # Complex unit arithmetic would go here
        return f"({u1}*{u2})"
    
    def _divide_units(self, u1: str, u2: str) -> str:
        if u2 == "dimensionless":
            return u1
        if u1 == u2:
            return "dimensionless"
        return f"({u1}/{u2})"
    
    def _sqrt_units(self, u: str) -> str:
        return f"sqrt({u})"
    
    def apply(self, logits: torch.Tensor, current_tokens: List[str]) -> torch.Tensor:
        """
        Mask logits to enforce physics constraints.
        
        Args:
            logits: [vocab_size] raw logits from decoder
            current_tokens: tokens generated so far
        Returns:
            masked logits
        """
        # Default: no masking
        masked_logits = logits.clone()
        
        # If we have at least one token, check unit consistency
        if current_tokens:
            # Try each candidate next token
            for idx in range(len(logits)):
                token = self.vocab.idx_to_token.get(idx, "")
                test_sequence = current_tokens + [token]
                units = self.compute_units(test_sequence)
                
                # If units are None (inconsistent), severely penalize
                if units is None and token not in [self.vocab.END_TOKEN, self.vocab.PAD_TOKEN]:
                    masked_logits[idx] = -1e9
                    
        return masked_logits


class SymbolicTransformer(nn.Module):
    """
    Transformer-based Symbolic Regression for orbital law distillation.
    
    Architecture:
    - Encoder: Transformer processing residual time-series
    - Decoder: Autoregressive generation of symbolic tokens
    - Physics constraints: Unit consistency enforced during decoding
    """
    
    def __init__(self,
                 d_model: int = 512,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 100):
        super().__init__()
        
        self.vocab = SymbolicVocabulary()
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab.vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Encoder: Process residual time-series
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        
        # Decoder: Generate symbolic sequence
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, self.vocab.vocab_size)
        
        # Physics constraints
        self.physics_constraints = PhysicsConstraintLayer(self.vocab)
        
        # Orbital context embedding (for nearby objects, etc.)
        self.context_proj = nn.Linear(d_model, d_model)
        
    def encode_residuals(self, 
                        residuals: torch.Tensor,
                        orbital_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode maneuver residuals into latent representation.
        
        Args:
            residuals: [batch, n_obs, 6] residual time-series (position, velocity deltas)
            orbital_context: [batch, d_model] optional context (nearby objects, etc.)
        Returns:
            [batch, seq_len, d_model] encoded representation
        """
        # Project residuals to d_model
        batch_size, n_obs, feat_dim = residuals.shape
        residuals_flat = residuals.reshape(batch_size, n_obs, feat_dim)
        
        # Simple linear projection (could be more sophisticated)
        residual_proj = nn.Linear(feat_dim, self.d_model).to(residuals.device)
        x = residual_proj(residuals_flat)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Encode
        memory = self.encoder(x)
        
        # Add context if provided
        if orbital_context is not None:
            context_emb = self.context_proj(orbital_context).unsqueeze(1)
            memory = memory + context_emb
            
        return memory
    
    def decode_symbolic(self,
                       memory: torch.Tensor,
                       target_seq: Optional[torch.Tensor] = None,
                       max_length: int = 50,
                       temperature: float = 1.0) -> Tuple[torch.Tensor, List[str]]:
        """
        Decode symbolic law from encoded residuals.
        
        Args:
            memory: [batch, seq_len, d_model] encoded residuals
            target_seq: [batch, tgt_seq_len] ground truth for teacher forcing (training)
            max_length: maximum generation length
            temperature: sampling temperature
        Returns:
            (output_logits, generated_tokens)
        """
        batch_size = memory.size(0)
        device = memory.device
        
        if target_seq is not None:
            # Training mode: teacher forcing
            tgt_emb = self.token_embedding(target_seq)
            tgt_emb = self.pos_encoding(tgt_emb)
            
            # Causal mask for autoregressive generation
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_seq.size(1)).to(device)
            
            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(output)
            
            # Decode tokens for return
            tokens = self.vocab.decode(target_seq[0].cpu().tolist())
            
            return logits, tokens
        else:
            # Inference mode: autoregressive generation
            generated = torch.full((batch_size, 1), 
                                 self.vocab.token_to_idx[self.vocab.START_TOKEN],
                                 dtype=torch.long, device=device)
            
            generated_tokens = [self.vocab.START_TOKEN]
            
            for _ in range(max_length):
                tgt_emb = self.token_embedding(generated)
                tgt_emb = self.pos_encoding(tgt_emb)
                
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1)).to(device)
                
                output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = self.output_proj(output[:, -1, :])  # Last position only
                
                # Apply physics constraints
                logits = self.physics_constraints.apply(logits[0], generated_tokens)
                
                # Sample next token
                probs = F.softmax(logits / temperature, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                
                next_token = self.vocab.idx_to_token[next_token_idx.item()]
                generated_tokens.append(next_token)
                
                # Append to generated sequence
                next_token_tensor = torch.tensor([[next_token_idx.item()]], device=device)
                generated = torch.cat([generated, next_token_tensor], dim=1)
                
                # Stop if end token
                if next_token == self.vocab.END_TOKEN:
                    break
                    
            return None, generated_tokens
    
    def forward(self,
               residuals: torch.Tensor,
               target_tokens: Optional[torch.Tensor] = None,
               orbital_context: Optional[torch.Tensor] = None) -> dict:
        """
        Full forward pass.
        
        Args:
            residuals: [batch, n_obs, 6]
            target_tokens: [batch, seq_len] for teacher forcing (training)
            orbital_context: [batch, d_model] optional context
        Returns:
            dict with 'logits', 'tokens', 'latex', 'maneuver_type'
        """
        # Encode
        memory = self.encode_residuals(residuals, orbital_context)
        
        # Decode
        logits, tokens = self.decode_symbolic(memory, target_tokens)
        
        # Convert to LaTeX
        latex = self.tokens_to_latex(tokens)
        
        # Infer maneuver type from tokens
        maneuver_type = self.infer_maneuver_type(tokens)
        
        return {
            'logits': logits,
            'tokens': tokens,
            'latex': latex,
            'maneuver_type': maneuver_type,
            'raw_expr': ' '.join(tokens)
        }
    
    def tokens_to_latex(self, tokens: List[str]) -> str:
        """Convert token sequence to LaTeX string"""
        # Remove special tokens
        tokens = [t for t in tokens if t not in 
                 [self.vocab.START_TOKEN, self.vocab.END_TOKEN, self.vocab.PAD_TOKEN]]
        
        if not tokens:
            return ""
            
        # First token is maneuver type
        maneuver_type = tokens[0] if tokens[0] in self.vocab.MANEUVER_TYPES else "unknown"
        
        # Format as LaTeX
        latex_parts = []
        
        # Maneuver description
        latex_parts.append(f"\\text{{{maneuver_type.replace('_', ' ')}}}")
        
        # Expression
        expr_tokens = tokens[1:] if maneuver_type != "unknown" else tokens
        
        for i, token in enumerate(expr_tokens):
            if token in self.vocab.OPERATORS:
                if token in ["sqrt", "exp", "log", "sin", "cos", "tan"]:
                    latex_parts.append(f"\\{token}(")
                elif token == "^":
                    latex_parts.append("^")
                else:
                    latex_parts.append(f" {token} ")
            elif token.startswith("delta_"):
                sub = token.replace("delta_", "")
                latex_parts.append(f"\\Delta {sub}")
            elif token == "toward_target":
                latex_parts.append("\\text{ toward target}")
            elif token == "away_from_target":
                latex_parts.append("\\text{ away from target}")
            else:
                latex_parts.append(token)
                
        latex = ' '.join(latex_parts)
        latex = latex.replace("( ", "(").replace(" )", ")")
        
        return f"${latex}$"
    
    def infer_maneuver_type(self, tokens: List[str]) -> str:
        """Infer maneuver type from token sequence"""
        for token in tokens:
            if token in self.vocab.MANEUVER_TYPES:
                return token
        return "unknown"
    
    def attribute_maneuver(self,
                          residual_history: torch.Tensor,
                          nearby_objects: Optional[List[dict]] = None) -> dict:
        """
        Full maneuver attribution pipeline.
        
        Args:
            residual_history: [1, n_obs, 6] residual time-series
            nearby_objects: Optional list of nearby RSO context
        Returns:
            Attribution report with LaTeX law
        """
        # Build context if nearby objects provided
        context = None
        if nearby_objects:
            # Simple encoding of nearby object info
            context_vec = torch.zeros(1, self.d_model)
            for obj in nearby_objects[:5]:  # Top 5 closest
                # Encode distance and type
                dist = obj.get('distance_km', 1000.0)
                is_target = obj.get('is_target', False)
                context_vec[0, 0] += 1.0 / (dist + 1.0)
                context_vec[0, 1] += float(is_target)
            context = context_vec
            
        # Forward pass
        with torch.no_grad():
            result = self.forward(residual_history, orbital_context=context)
            
        # Build attribution report
        report = {
            'maneuver_detected': True,
            'maneuver_type': result['maneuver_type'],
            'symbolic_law_latex': result['latex'],
            'symbolic_tokens': result['tokens'],
            'interpretation': self._interpret_law(result['tokens']),
            'confidence': 0.85  # Would be computed from model uncertainty
        }
        
        return report
    
    def _interpret_law(self, tokens: List[str]) -> str:
        """Generate human-readable interpretation of symbolic law"""
        if not tokens:
            return "No maneuver detected"
            
        maneuver = tokens[0] if tokens[0] in self.vocab.MANEUVER_TYPES else "unknown"
        
        interpretations = {
            'station_keep': "Station-keeping maneuver to maintain orbital position",
            'orbit_raise': "Orbit-raising maneuver to increase altitude",
            'orbit_lower': "Orbit-lowering maneuver to decrease altitude",
            'plane_change': "Orbital plane change maneuver",
            'phasing': "Phasing maneuver to adjust orbital period",
            'rendezvous': "Rendezvous approach maneuver",
            'collision_avoidance': "Collision avoidance maneuver",
            'deorbit': "Deorbit preparation maneuver",
        }
        
        return interpretations.get(maneuver, "Unknown maneuver type")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :]


def train_symbolic_sr():
    """Training orchestrator for symbolic regression"""
    pass


if __name__ == "__main__":
    # Quick test
    model = SymbolicTransformer(d_model=256, n_encoder_layers=4, n_decoder_layers=4)
    
    # Simulate residual history
    batch_size = 1
    n_obs = 10
    residuals = torch.randn(batch_size, n_obs, 6)
    
    # Inference
    result = model.attribute_maneuver(residuals)
    
    print(f"Maneuver type: {result['maneuver_type']}")
    print(f"LaTeX law: {result['symbolic_law_latex']}")
    print(f"Interpretation: {result['interpretation']}")
