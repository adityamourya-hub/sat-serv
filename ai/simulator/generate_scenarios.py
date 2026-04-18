#!/usr/bin/env python3
"""
Generate 10k Graph-Native space situational scenarios with labels.
Each scenario represents a neighborhood graph of objects for GATv2 training.

Schema (26-dimensional nodes):
- Pos(3), Vel(3), Cov(6), Obs(3), Orbit(4), Mask(4), Temp(3)
- Ionospheric Global Context (16-dim)
"""

import json
import math
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

def bounded(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))

def generate_spherical_harmonics(n=16):
    """Simulate Ionospheric TEC context as 16-dim Spherical Harmonic embedding."""
    return np.random.normal(0, 0.1, n).tolist()

def generate_graph_scenario(scenario_id: str, rng: np.random.Generator):
    # Neighborhood size: 3 to 10 nodes
    num_nodes = rng.integers(3, 11)
    
    # Global Ionospheric Context
    iono_context = generate_spherical_harmonics(16)
    
    nodes = []
    # Primary conjunction pair (Node 0 and 1)
    for i in range(num_nodes):
        # Base kinematic state (Position in km, Velocity in km/s)
        # Random LEO sphere placement
        alt = rng.uniform(200, 2000)
        r = 6371 + alt
        q = rng.uniform(-1, 1, 3)
        pos = (q / np.linalg.norm(q)) * r
        
        v_mag = math.sqrt(398600.4418 / r) # Circular velocity
        v_dir = np.cross(pos, [0, 0, 1])
        v_dir = (v_dir / np.linalg.norm(v_dir))
        vel = v_dir * v_mag + rng.normal(0, 0.1, 3) # Add slight perturbation
        
        # 26-dim Feature Vector Construction
        node = np.zeros(26, dtype=np.float32)
        node[0:3] = pos / 1000.0 # Normalize km -> 1000km units
        node[3:6] = vel / 10.0   # Normalize km/s -> 10km/s units
        
        # 6-dim Covariance diagonal
        node[6:12] = np.abs(rng.normal(1e-4, 1e-5, 6))
        
        # Observables (RCS, Mag, Age)
        node[12] = rng.uniform(0.1, 10.0) # RCS m^2
        node[13] = rng.uniform(10, 20)    # Visual Mag
        node[14] = 0.0                    # Age (initial)
        
        # Orbit Regime (LEO, MEO, GEO, HEO) - One-hot
        if alt < 2000: node[15] = 1.0 # LEO
        elif alt < 35000: node[16] = 1.0 # MEO
        else: node[17] = 1.0 # GEO/HEO
        
        # Sensor Mask [Radar, Optical, SAR, NavIC]
        node[19:23] = (rng.random(4) > 0.5).astype(float)
        
        # Temporal [Last Obs Age, Confidence, Anomaly]
        node[23] = rng.uniform(0, 3600)
        node[24] = rng.uniform(0.7, 1.0)
        node[25] = rng.uniform(0, 0.1)
        
        nodes.append(node.tolist())
        
    # Adjacency matrix (Full connection for simplicity in small graphs)
    adj = np.ones((num_nodes, num_nodes), dtype=float).tolist()
    
    # Multi-task labels
    # 1. Conjunction: Binary [Safe, Warning, Critical]
    risk_prob = rng.random()
    if risk_prob > 0.95: conjunction_label = [0, 0, 1] # Critical
    elif risk_prob > 0.80: conjunction_label = [0, 1, 0] # Warning
    else: conjunction_label = [1, 0, 0] # Safe
    
    # 2. Maneuver: [None, Impulsive, Continuous]
    man_rand = rng.random()
    if man_rand > 0.90: maneuver_label = 1 # Impulsive
    elif man_rand > 0.85: maneuver_label = 2 # Continuous
    else: maneuver_label = 0
    
    return {
        "scenario_id": scenario_id,
        "iono_context": iono_context,
        "node_features": nodes,
        "adjacency": adj,
        "labels": {
            "conjunction": conjunction_label,
            "maneuver": maneuver_label,
            "correlation": scenario_id # Identity for contrastive matching
        }
    }

def main():
    count = 10000
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(42)
    output_path = out_dir / "scenarios_graph_v2.jsonl"
    
    print(f"Generating {count} graph scenarios...")
    with output_path.open("w") as f:
        for i in range(count):
            if i % 1000 == 0: print(f"  > Progressive: {i}/{count}")
            scen = generate_graph_scenario(f"PH2-{i+1:05d}", rng)
            f.write(json.dumps(scen) + "\n")
            
    print(f"Completed! Wrote {count} scenarios to {output_path}")

if __name__ == "__main__":
    main()
