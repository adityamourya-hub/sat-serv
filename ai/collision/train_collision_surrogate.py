#!/usr/bin/env python3
"""
Finalized Phase 2 Multi-Task GNN Training Pipeline.
Integrated with:
1. JSONL Graph Scenarios Loading
2. 26-dim Node Features
3. Ionospheric Virtual Node Context
4. Evidential Dirichlet Loss + Maneuver Classification
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_path: str = "ai/simulator/data/scenarios_graph_v2.jsonl"
    max_nodes: int = 10
    feat_dim: int = 26
    hidden: int = 128
    heads: int = 4
    batch_size: int = 64
    epochs: int = 25
    lr: float = 1e-3

class CollisionGraphDataset(Dataset):
    def __init__(self, data_path: str, max_nodes: int, feat_dim: int):
        self.max_nodes = max_nodes
        self.feat_dim = feat_dim
        self.records = []
        
        path = Path(data_path)
        if not path.exists():
            print(f"ERROR: Data path {path} does not exist.")
            return

        with path.open("r") as f:
            for line in f:
                self.records.append(json.loads(line))
        print(f"Loaded {len(self.records)} scenarios.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        raw_nodes = rec["node_features"]
        num_raw = len(raw_nodes)
        
        # Zero-pad or truncate nodes
        nodes = np.zeros((self.max_nodes, self.feat_dim), dtype=np.float32)
        n_to_copy = min(num_raw, self.max_nodes)
        nodes[:n_to_copy] = np.array(raw_nodes[:n_to_copy], dtype=np.float32)
        
        # Adjacency
        adj = np.eye(self.max_nodes, dtype=np.float32)
        raw_adj = np.array(rec["adjacency"], dtype=np.float32)
        n_adj = min(len(raw_adj), self.max_nodes)
        adj[:n_adj, :n_adj] = raw_adj[:n_adj, :n_adj]
        
        # Ionospheric context
        iono = torch.tensor(rec["iono_context"], dtype=torch.float32)
        
        # Labels
        y_conj = torch.tensor(rec["labels"]["conjunction"], dtype=torch.float32)
        y_man = torch.tensor(rec["labels"]["maneuver"], dtype=torch.long)
        
        return torch.from_numpy(nodes), torch.from_numpy(adj), iono, y_conj, y_man

class MultiSensorFusionGNN(nn.Module):
    def __init__(self, node_dim=26, hidden=128, n_heads=4, vnode_dim=16):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden)
        self.vnode_embed = nn.Linear(vnode_dim, hidden)
        
        self.gat1 = nn.MultiheadAttention(hidden, num_heads=n_heads, batch_first=True)
        self.gat2 = nn.MultiheadAttention(hidden, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

        # Head 1: Evidential Conflict (Dirichlet)
        self.conjunction_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3), # [Safe, Warning, Critical]
            nn.Softplus()
        )

        # Head 2: Maneuver Detection
        self.maneuver_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3) # [None, Impulsive, Continuous]
        )

    def forward(self, nodes, adj, iono):
        # nodes: [B, N, 26], adj: [B, N, N], iono: [B, 16]
        h = self.node_embed(nodes)
        
        # --- GNN-Guard: Anti-Poisoning Layer ---
        # Calculate pairwise cosine similarity of embedded features
        # h_norm: [B, N, H]
        h_norm = torch.nn.functional.normalize(h, p=2, dim=-1)
        # sim_matrix: [B, N, N]
        sim_matrix = torch.bmm(h_norm, h_norm.transpose(1, 2))
        
        # Pruning Threshold theta = 0.85
        # Only allow messages between 'Verified Consistent' objects
        guard_mask = (sim_matrix > 0.85).float()
        adj_guarded = adj * guard_mask
        
        v = self.vnode_embed(iono).unsqueeze(1)
        h = torch.cat([h, v], dim=1) # [B, N+1, H]
        
        # Use guarded adjacency matrix as attention mask (N+1 x N+1)
        B, N, _ = nodes.shape
        full_mask = torch.ones(B, N+1, N+1).to(nodes.device)
        full_mask[:, :N, :N] = adj_guarded
        
        # In MultiheadAttention, attn_mask is (L, S) or (N*num_heads, L, S)
        # We use the mask to Zero out non-neighbors (with large negative values)
        # Convert 1/0 mask to -inf for zeros
        mask = (full_mask == 0).repeat_interleave(self.gat1.num_heads, dim=0)
        
        attn_out, _ = self.gat1(h, h, h, attn_mask=mask)
        h = self.norm(h + attn_out)
        
        v_ctx = h[:, -1, :]
        obj_nodes = h[:, :-1, :]
        obj_nodes = obj_nodes + v_ctx.unsqueeze(1)
        
        pooled = obj_nodes.mean(dim=1)
        alphas = self.conjunction_head(pooled) + 1.0 
        maneuver_logits = self.maneuver_head(pooled)
        
        return alphas, maneuver_logits

def evidential_loss(alphas, y, lambda_reg=0.2):
    S = torch.sum(alphas, dim=-1, keepdim=True)
    p = alphas / S
    
    # Class-weighted MSE for imbalance (5% Critical)
    # Weights: Safe=1.0, Warning=5.0, Critical=10.0
    weights = torch.tensor([1.0, 5.0, 10.0]).to(alphas.device)
    err = torch.mean(torch.sum(weights * (y - p)**2, dim=-1))
    
    var = torch.mean(torch.sum(p * (1 - p) / (S + 1), dim=-1))
    evidence_penalty = lambda_reg * torch.mean(torch.sum((1 - y) * (alphas - 1), dim=-1))
    
    return err + var + evidence_penalty

def train():
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = CollisionGraphDataset(cfg.data_path, cfg.max_nodes, cfg.feat_dim)
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    
    model = MultiSensorFusionGNN(node_dim=cfg.feat_dim, hidden=cfg.hidden, n_heads=cfg.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ce_loss = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    output_dir = Path("ai/training/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        for nodes, adj, iono, y_conj, y_man in train_loader:
            nodes, adj, iono = nodes.to(device), adj.to(device), iono.to(device)
            y_conj, y_man = y_conj.to(device), y_man.to(device)

            alphas, man_logits = model(nodes, adj, iono)

            loss_conj = evidential_loss(alphas, y_conj)
            loss_man = ce_loss(man_logits, y_man)

            # Simple homoscedastic weighting (could be uncertainty weighted)
            loss = loss_conj + 0.5 * loss_man

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{cfg.epochs} - Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "collision_multi_task_best.pt")
            print(f"  Saved best model with loss: {best_loss:.4f}")

        # Periodic checkpoint every 10 epochs (survives VM reboot)
        if epoch % 10 == 0 and epoch > 0:
            checkpoint_path = output_dir / f"collision_checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    # Final save
    torch.save(model.state_dict(), output_dir / "collision_multi_task_final.pt")

    # Export
    model.eval()

    # Save PT weights for Validator
    pt_path = output_dir / "collision_multi_task_v2.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"Saved PT weights to {pt_path}")

    dummy_nodes = torch.zeros(1, cfg.max_nodes, cfg.feat_dim).to(device)
    dummy_adj = torch.eye(cfg.max_nodes).unsqueeze(0).to(device)
    dummy_iono = torch.zeros(1, 16).to(device)
    
    out_path = Path("ai/training/models/collision_multi_task_v2.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model, (dummy_nodes, dummy_adj, dummy_iono), str(out_path),
        input_names=["nodes", "adj", "iono"],
        output_names=["risk_alphas", "maneuver_logits"],
        dynamic_axes={"nodes": {0: "batch"}, "adj": {0: "batch"}, "iono": {0: "batch"}}
    )
    print(f"Saved model to {out_path}")

if __name__ == "__main__":
    train()
