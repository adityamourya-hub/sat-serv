#!/usr/bin/env python3
"""
Multi-Task GATv2 Training for 
Trains conjunction, maneuver, and threat-level heads on 10k AMOS-enhanced scenarios

: Physics-first multi-task learning with:
- Conjunction head: 3-class (Safe/Warning/Critical)
- Maneuver head: 3-class (None/Impulsive/Continuous)
- Threat-level head: 4-class (LOW/MEDIUM/HIGH/CRITICAL)
- AMOS 2025 features: spectral residuals, fractional drag, MGE threat score, photometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import json
import numpy as np
from pathlib import Path
from typing import List, Dict


class MultiTaskGATv2(nn.Module):
 """
 Multi-Task GATv2 for orbital dominance
 
 Architecture:
 - GATv2 encoder (64-dim input → 128 hidden)
 - Shared backbone → 3 task-specific heads
 - Conjunction: 3-class classification
 - Maneuver: 3-class classification
 - Threat-level: 4-class classification
 """
 def __init__(
 self,
 in_channels: int = 64,
 hidden_channels: int = 128,
 num_layers: int = 3,
 heads: int = 4,
 dropout: float = 0.1
):
 super().__init__()
 
 # GATv2 encoder layers
 self.convs = nn.ModuleList()
 # First layer: handle edge attributes (7-dim)
 self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=7))
 for _ in range(num_layers - 1):
 self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=7))
 
 # Shared backbone
 self.backbone = nn.Sequential(
 nn.Linear(hidden_channels * heads, hidden_channels),
 nn.ReLU(),
 nn.Dropout(dropout)
)
 
 # Task-specific heads
 # Conjunction head: 3-class (Safe/Warning/Critical)
 self.conjunction_head = nn.Sequential(
 nn.Linear(hidden_channels, 64),
 nn.ReLU(),
 nn.Dropout(dropout),
 nn.Linear(64, 3)
)
 
 # Maneuver head: 3-class (None/Impulsive/Continuous)
 self.maneuver_head = nn.Sequential(
 nn.Linear(hidden_channels, 64),
 nn.ReLU(),
 nn.Dropout(dropout),
 nn.Linear(64, 3)
)
 
 # Threat-level head: 4-class (LOW/MEDIUM/HIGH/CRITICAL)
 self.threat_head = nn.Sequential(
 nn.Linear(hidden_channels, 64),
 nn.ReLU(),
 nn.Dropout(dropout),
 nn.Linear(64, 4)
)
 
 def forward(self, data: Data):
 """
 Forward pass with multi-task outputs
 
 Returns:
 conj_logits: (B, 3) conjunction logits
 man_logits: (B, 3) maneuver logits
 threat_logits: (B, 4) threat-level logits
 """
 x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
 
 # GATv2 encoding
 for conv in self.convs:
 x = conv(x, edge_index, edge_attr)
 x = F.elu(x)
 
 # Global mean pooling (graph-level representation) with batch index
 x = global_mean_pool(x, batch) # (B, hidden)
 
 # Shared backbone
 x = self.backbone(x)
 
 # Task-specific heads
 conj_logits = self.conjunction_head(x)
 man_logits = self.maneuver_head(x)
 threat_logits = self.threat_head(x)
 
 return conj_logits, man_logits, threat_logits


def load_scenarios(data_path: Path) -> List[Dict]:
 """Load 10k scenarios from JSONL"""
 scenarios = []
 with data_path.open("r") as f:
 for line in f:
 scenarios.append(json.loads(line))
 return scenarios


def scenario_to_pyg_data(scenario: Dict) -> Data:
 """Convert scenario dict to PyG Data object"""
 x = torch.tensor(scenario["node_features"], dtype=torch.float)
 edge_index = torch.tensor(scenario["edge_index"], dtype=torch.long).t().contiguous()
 edge_attr = torch.tensor(scenario["edge_attr"], dtype=torch.float)
 
 # Labels need to be tensors for PyG batching
 y_conj = torch.tensor(scenario["labels"]["conjunction"], dtype=torch.float).unsqueeze(0) # (1, 3)
 y_man = torch.tensor([scenario["labels"]["maneuver"]], dtype=torch.long) # (1,)
 y_threat = torch.tensor([scenario["labels"]["threat_level"]], dtype=torch.long) # (1,)
 
 return Data(
 x=x,
 edge_index=edge_index,
 edge_attr=edge_attr,
 y_conj=y_conj,
 y_man=y_man,
 y_threat=y_threat
)


def train_epoch(model, loader, optimizer, device):
 """Train for one epoch"""
 model.train()
 total_loss = 0
 conj_correct = 0
 man_correct = 0
 threat_correct = 0
 total = 0
 
 for data in loader:
 data = data.to(device)
 optimizer.zero_grad()
 
 conj_logits, man_logits, threat_logits = model(data)
 
 # Multi-task loss
 conj_loss = F.cross_entropy(conj_logits, data.y_conj.argmax(dim=1))
 man_loss = F.cross_entropy(man_logits, data.y_man.squeeze())
 threat_loss = F.cross_entropy(threat_logits, data.y_threat.squeeze())
 
 # Weighted loss (prioritize conjunction)
 loss = 0.5 * conj_loss + 0.3 * man_loss + 0.2 * threat_loss
 
 loss.backward()
 optimizer.step()
 
 total_loss += loss.item()
 
 # Accuracy metrics
 conj_pred = conj_logits.argmax(dim=1)
 man_pred = man_logits.argmax(dim=1)
 threat_pred = threat_logits.argmax(dim=1)
 
 conj_correct += (conj_pred == data.y_conj.argmax(dim=1)).sum().item()
 man_correct += (man_pred == data.y_man.squeeze()).sum().item()
 threat_correct += (threat_pred == data.y_threat.squeeze()).sum().item()
 total += data.num_graphs
 
 avg_loss = total_loss / len(loader)
 conj_acc = conj_correct / total
 man_acc = man_correct / total
 threat_acc = threat_correct / total
 
 return avg_loss, conj_acc, man_acc, threat_acc


def evaluate(model, loader, device):
 """Evaluate model"""
 model.eval()
 conj_correct = 0
 man_correct = 0
 threat_correct = 0
 total = 0
 
 with torch.no_grad():
 for data in loader:
 data = data.to(device)
 conj_logits, man_logits, threat_logits = model(data)
 
 conj_pred = conj_logits.argmax(dim=1)
 man_pred = man_logits.argmax(dim=1)
 threat_pred = threat_logits.argmax(dim=1)
 
 conj_correct += (conj_pred == data.y_conj.argmax(dim=1)).sum().item()
 man_correct += (man_pred == data.y_man.squeeze()).sum().item()
 threat_correct += (threat_pred == data.y_threat.squeeze()).sum().item()
 total += data.num_graphs
 
 conj_acc = conj_correct / total
 man_acc = man_correct / total
 threat_acc = threat_correct / total
 
 return conj_acc, man_acc, threat_acc


def main():
 """Main training loop"""
 print("=== : Multi-Task GATv2 Training ===")
 
 # Configuration
 data_path = Path(__file__).parent / "data" / "ssa_graph_scenarios_10k.jsonl"
 batch_size = 32
 epochs = 50
 lr = 0.001
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 print(f"Device: {device}")
 print(f"Data path: {data_path}")
 print(f"Batch size: {batch_size}")
 print(f"Epochs: {epochs}")
 
 # Load scenarios
 print("\nLoading scenarios...")
 scenarios = load_scenarios(data_path)
 print(f"Loaded {len(scenarios)} scenarios")
 
 # Convert to PyG Data objects
 print("Converting to PyG Data objects...")
 pyg_data = [scenario_to_pyg_data(s) for s in scenarios]
 
 # Split train/val (80/20)
 train_size = int(0.8 * len(pyg_data))
 train_data = pyg_data[:train_size]
 val_data = pyg_data[train_size:]
 
 print(f"Train: {len(train_data)}, Val: {len(val_data)}")
 
 # Create data loaders
 train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
 val_loader = DataLoader(val_data, batch_size=batch_size)
 
 # Initialize model
 model = MultiTaskGATv2(
 in_channels=64,
 hidden_channels=128,
 num_layers=3,
 heads=4,
 dropout=0.1
).to(device)
 
 optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
 print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
 print("\n=== Training ===")
 
 best_val_acc = 0
 
 for epoch in range(epochs):
 train_loss, train_conj, train_man, train_threat = train_epoch(model, train_loader, optimizer, device)
 val_conj, val_man, val_threat = evaluate(model, val_loader, device)
 
 val_avg = (val_conj + val_man + val_threat) / 3
 
 if epoch % 10 == 0 or epoch == epochs - 1:
 print(f"Epoch {epoch+1}/{epochs}")
 print(f" Train Loss: {train_loss:.4f}")
 print(f" Train Acc: Conj={train_conj:.3f}, Man={train_man:.3f}, Threat={train_threat:.3f}")
 print(f" Val Acc: Conj={val_conj:.3f}, Man={val_man:.3f}, Threat={val_threat:.3f}")
 print(f" Val Avg: {val_avg:.3f}")
 
 if val_avg > best_val_acc:
 best_val_acc = val_avg
 torch.save(model.state_dict(), data_path.parent / "gatv2_multitask_best.pt")
 
 print(f"\n=== Training Complete ===")
 print(f"Best Val Accuracy: {best_val_acc:.3f}")
 print(f"Model saved to: {data_path.parent / 'gatv2_multitask_best.pt'}")
 print("\n=== Ready for deployment ===")
 print("Multi-task heads:")
 print(" - Conjunction: 3-class (Safe/Warning/Critical)")
 print(" - Maneuver: 3-class (None/Impulsive/Continuous)")
 print(" - Threat-level: 4-class (LOW/MEDIUM/HIGH/CRITICAL)")
 print("\nAMOS 2025 features integrated:")
 print(" - Spectral residuals (32-dim)")
 print(" - Fractional drag residual")
 print(" - MGE threat score")
 print(" - Photometric period")


if __name__ == "__main__":
 main()
