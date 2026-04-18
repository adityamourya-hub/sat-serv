"""
NavIC Anomaly Detector — LSTM-Attention model for real-time GNSS spoofing/jamming detection.

Processes dual-frequency (L5 + S-band) pseudorange residual histories to classify:
  0 = Normal, 1 = Jamming, 2 = Spoofing

Architecture:
  [Dual-Freq Residuals × 12 features] → BiLSTM(64) → MultiHeadAttention(4 heads)
  → Classification Head (3 classes) + Confidence Head (1 scalar)

Training data: Synthetic from NavIC constellation simulator with injected anomalies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


# --- Feature Engineering ---

class NavICFeatureExtractor:
    """
    Extracts 12-dimensional feature vector from raw NavIC observables per epoch.
    
    Features:
        [0-1]   L5/S pseudorange residuals (m)
        [2-3]   L5/S carrier-phase residuals (cycles)
        [4]     Dual-frequency ionospheric delay difference (m)
        [5]     C/N0 ratio L5 (dB-Hz)
        [6]     C/N0 ratio S-band (dB-Hz)
        [7]     Geometric DOP
        [8]     Clock drift rate (ns/s)
        [9]     Multipath estimator (code-minus-carrier, m)
        [10]    Satellite elevation angle (rad)
        [11]    Cross-correlation metric (inter-SV consistency)
    """
    N_FEATURES = 12

    @staticmethod
    def extract(observables: Dict) -> np.ndarray:
        return np.array([
            observables.get('pr_residual_l5', 0.0),
            observables.get('pr_residual_s', 0.0),
            observables.get('cp_residual_l5', 0.0),
            observables.get('cp_residual_s', 0.0),
            observables.get('iono_diff', 0.0),
            observables.get('cn0_l5', 45.0),
            observables.get('cn0_s', 45.0),
            observables.get('gdop', 2.0),
            observables.get('clock_drift_ns_s', 0.0),
            observables.get('multipath_m', 0.0),
            observables.get('elevation_rad', np.pi / 4),
            observables.get('cross_corr', 1.0),
        ], dtype=np.float32)


# --- Model ---

class NavICAnomalyDetector(nn.Module):
    """
    Bi-directional LSTM + Multi-Head Attention for NavIC anomaly detection.
    
    Input:  [batch, seq_len, 12]  (dual-frequency residual history)
    Output: (class_logits [batch, 3], confidence [batch, 1])
    """
    def __init__(self, input_size: int = 12, hidden_size: int = 64,
                 num_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        enc_dim = hidden_size * 2  # bidirectional
        self.attention = nn.MultiheadAttention(
            embed_dim=enc_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(enc_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # [normal, jamming, spoofing]
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(enc_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, 12] pseudorange residual history
        Returns:
            logits: [batch, 3] class logits (normal/jamming/spoofing)
            confidence: [batch, 1] prediction confidence
        """
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)                  # [B, T, 128]
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # [B, T, 128]
        pooled = attn_out.mean(dim=1)                # [B, 128]
        logits = self.classifier(pooled)             # [B, 3]
        confidence = self.confidence_head(pooled)    # [B, 1]
        return logits, confidence

    def predict(self, x: torch.Tensor) -> Dict:
        """Inference convenience method."""
        self.eval()
        with torch.no_grad():
            logits, conf = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            pred_class = probs.argmax(dim=-1)
            labels = ['normal', 'jamming', 'spoofing']
            return {
                'class': labels[pred_class.item()] if x.shape[0] == 1 else [labels[c] for c in pred_class.tolist()],
                'probabilities': probs.cpu().numpy(),
                'confidence': conf.squeeze(-1).cpu().numpy(),
            }


# --- Synthetic Training Data Generator ---

class NavICScenarioGenerator:
    """
    Generates synthetic NavIC pseudorange residual sequences with injected anomalies.
    """
    def __init__(self, seq_len: int = 60, seed: Optional[int] = None):
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def generate_normal(self) -> np.ndarray:
        """Clean dual-frequency residuals."""
        data = np.zeros((self.seq_len, NavICFeatureExtractor.N_FEATURES), dtype=np.float32)
        data[:, 0] = self.rng.normal(0, 0.5, self.seq_len)       # L5 PR residual
        data[:, 1] = self.rng.normal(0, 0.5, self.seq_len)       # S PR residual
        data[:, 2] = self.rng.normal(0, 0.01, self.seq_len)      # L5 CP residual
        data[:, 3] = self.rng.normal(0, 0.01, self.seq_len)      # S CP residual
        data[:, 4] = self.rng.normal(0, 0.2, self.seq_len)       # Iono diff
        data[:, 5] = 45 + self.rng.normal(0, 1, self.seq_len)    # C/N0 L5
        data[:, 6] = 45 + self.rng.normal(0, 1, self.seq_len)    # C/N0 S
        data[:, 7] = 2.0 + self.rng.normal(0, 0.3, self.seq_len) # GDOP
        data[:, 8] = self.rng.normal(0, 0.01, self.seq_len)      # Clock drift
        data[:, 9] = self.rng.exponential(0.1, self.seq_len)     # Multipath
        data[:, 10] = self.rng.uniform(0.3, 1.4, self.seq_len)   # Elevation
        data[:, 11] = 1.0 + self.rng.normal(0, 0.02, self.seq_len) # Cross-corr
        return data

    def generate_jamming(self) -> np.ndarray:
        """Broadband jamming: C/N0 drops, residuals spike, cross-corr degrades."""
        data = self.generate_normal()
        onset = self.rng.integers(5, self.seq_len // 2)
        power = self.rng.uniform(5, 20)
        data[onset:, 0] += self.rng.normal(0, power, self.seq_len - onset)
        data[onset:, 1] += self.rng.normal(0, power, self.seq_len - onset)
        data[onset:, 5] -= power * 0.8    # C/N0 L5 drops
        data[onset:, 6] -= power * 0.8    # C/N0 S drops
        data[onset:, 11] -= 0.3 + self.rng.uniform(0, 0.4)  # Cross-corr degrades
        return data

    def generate_spoofing(self) -> np.ndarray:
        """
        Meaconing/spoofing: smooth residual drift, C/N0 stays normal,
        iono delay becomes inconsistent, cross-corr degrades subtly.
        """
        data = self.generate_normal()
        onset = self.rng.integers(5, self.seq_len // 2)
        drift_rate = self.rng.uniform(0.05, 0.5)
        t = np.arange(self.seq_len - onset, dtype=np.float32)
        drift = drift_rate * t
        data[onset:, 0] += drift            # L5 PR drifts
        data[onset:, 1] += drift * 0.7      # S PR drifts differently (freq-dependent)
        data[onset:, 4] += drift * 0.3      # Iono inconsistency grows
        data[onset:, 8] += drift_rate * 0.1 # Clock drift anomaly
        data[onset:, 11] -= 0.1 + self.rng.uniform(0, 0.15)  # Subtle cross-corr drop
        return data

    def generate_dataset(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate balanced training dataset."""
        per_class = n_samples // 3
        X, y = [], []
        for _ in range(per_class):
            X.append(self.generate_normal()); y.append(0)
        for _ in range(per_class):
            X.append(self.generate_jamming()); y.append(1)
        for _ in range(n_samples - 2 * per_class):
            X.append(self.generate_spoofing()); y.append(2)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        perm = self.rng.permutation(len(X))
        return X[perm], y[perm]


# --- Training Loop ---

def train_navic_detector(n_epochs: int = 50, batch_size: int = 128,
                         lr: float = 1e-3, n_samples: int = 30000,
                         device: str = 'cpu') -> NavICAnomalyDetector:
    """Train the NavIC anomaly detector on synthetic data."""
    gen = NavICScenarioGenerator(seq_len=60, seed=42)
    X, y = gen.generate_dataset(n_samples)

    # 80/20 split
    split = int(0.8 * len(X))
    X_train, y_train = torch.tensor(X[:split]), torch.tensor(y[:split])
    X_val, y_val = torch.tensor(X[split:]), torch.tensor(y[split:])

    model = NavICAnomalyDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx].to(device), y_train[idx].to(device)
            logits, conf = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val.to(device))
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == y_val.to(device)).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d} | Loss {total_loss:.4f} | Val Acc {val_acc:.4f} (best {best_val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    return model


if __name__ == "__main__":
    model = train_navic_detector(n_epochs=50, n_samples=30000)
    # Quick smoke test
    gen = NavICScenarioGenerator(seq_len=60, seed=99)
    test_spoof = torch.tensor(gen.generate_spoofing()).unsqueeze(0)
    result = model.predict(test_spoof)
    print(f"Spoofing test: {result}")
