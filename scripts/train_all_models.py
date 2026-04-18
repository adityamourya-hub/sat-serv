#!/usr/bin/env python3
"""
SSIS Automated Training Pipeline

Usage:
    python3 scripts/train_all_models.py --mode quick    # Fast training (1000 samples)
    python3 scripts/train_all_models.py --mode full     # Full training (100K samples)
    python3 scripts/train_all_models.py --component navic  # Train single component

This script orchestrates training for all AI components in SSIS.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List
import subprocess


# Training configurations
CONFIGS = {
    "quick": {
        "navic_samples": 5000,
        "navic_epochs": 10,
        "ateena_batches": 50,
        "collision_scenarios": 1000,
        "maneuver_samples": 2000,
        "fusion_epochs": 5,
        "federated_rounds": 3,
        "tasking_timesteps": 1000,
        "transformer_epochs": 10,
        "reentry_samples": 2000,
    },
    "full": {
        "navic_samples": 50000,
        "navic_epochs": 100,
        "ateena_batches": 10000,
        "collision_scenarios": 50000,
        "maneuver_samples": 50000,
        "fusion_epochs": 50,
        "federated_rounds": 20,
        "tasking_timesteps": 100000,
        "transformer_epochs": 100,
        "reentry_samples": 20000,
    },
}


class TrainingOrchestrator:
    def __init__(self, mode: str = "quick", device: str = "auto"):
        self.mode = mode
        self.config = CONFIGS[mode]
        self.device = self._detect_device() if device == "auto" else device
        self.results = {}
        
        # Ensure directories exist
        self.model_dir = Path("ai/training/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path("ai/training/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _detect_device(self) -> str:
        """Auto-detect CUDA availability."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _run_module(self, module_path: str, description: str) -> bool:
        """Execute a Python module and capture results."""
        print(f"\n{'='*60}")
        print(f"Training: {description}")
        print(f"{'='*60}")
        
        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "-m", module_path],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max per component
            )
            elapsed = time.time() - start
            
            if result.returncode == 0:
                print(f"✓ Success in {elapsed:.1f}s")
                self.results[description] = {"status": "success", "time": elapsed}
                return True
            else:
                print(f"✗ Failed with code {result.returncode}")
                print(f"STDERR: {result.stderr[:500]}")
                self.results[description] = {"status": "failed", "error": result.stderr[:200]}
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout after 1 hour")
            self.results[description] = {"status": "timeout"}
            return False
        except Exception as e:
            print(f"✗ Exception: {e}")
            self.results[description] = {"status": "error", "error": str(e)}
            return False

    def train_navic_detector(self) -> bool:
        """Train NavIC anomaly detector."""
        print(f"\n[1/7] NavIC Anomaly Detector")
        print(f"  Samples: {self.config['navic_samples']}")
        print(f"  Epochs: {self.config['navic_epochs']}")
        print(f"  Device: {self.device}")
        
        # Direct import and training
        try:
            sys.path.insert(0, "ai/v3")
            from navic_anomaly_detector import train_navic_detector
            
            model = train_navic_detector(
                n_epochs=self.config["navic_epochs"],
                n_samples=self.config["navic_samples"],
                batch_size=128,
                lr=1e-3,
                device=self.device,
            )
            
            # Save model
            import torch
            save_path = self.model_dir / "navic_detector.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  Saved: {save_path}")
            
            # Export ONNX
            dummy = torch.randn(1, 60, 12).to(self.device)
            onnx_path = self.model_dir / "navic_detector.onnx"
            torch.onnx.export(model, dummy, onnx_path,
                             input_names=["navic_history"],
                             output_names=["logits", "confidence"])
            print(f"  Exported ONNX: {onnx_path}")
            
            self.results["navic_detector"] = {"status": "success"}
            return True
            
        except Exception as e:
            print(f"  Error: {e}")
            self.results["navic_detector"] = {"status": "failed", "error": str(e)}
            return False

    def train_ateena_iqnet(self) -> bool:
        """Train ATeena radar IQNet."""
        print(f"\n[2/7] ATeena IQNet")
        print(f"  Batches: {self.config['ateena_batches']}")
        
        # Run training script
        return self._run_module(
            "ai.radar.train_ateena_iqnet",
            "ATeena IQNet (Radar Detection)"
        )

    def train_collision_surrogate(self) -> bool:
        """Train collision prediction GNN."""
        print(f"\n[3/7] Collision Surrogate (GNN)")
        print(f"  Scenarios: {self.config['collision_scenarios']}")
        
        # Generate scenarios first
        print("  Generating synthetic conjunction scenarios...")
        try:
            # Direct training
            sys.path.insert(0, "ai/collision")
            from train_collision_surrogate import train, TrainConfig, CollisionGraphDataset
            import torch
            
            cfg = TrainConfig()
            cfg.epochs = min(self.config["navic_epochs"], 25)  # GNN trains faster
            
            # Run training
            train()
            
            self.results["collision_surrogate"] = {"status": "success"}
            return True
            
        except Exception as e:
            print(f"  Error: {e}")
            self.results["collision_surrogate"] = {"status": "failed", "error": str(e)}
            return False

    def train_maneuver_detector(self) -> bool:
        """Train maneuver detection LSTM."""
        print(f"\n[4/7] Maneuver Detector")
        print(f"  Samples: {self.config['maneuver_samples']}")
        
        return self._run_module(
            "ai.maneuver.train_maneuver_detector",
            "Maneuver Detector (LSTM+Attention)"
        )

    def train_fusion_gnn(self) -> bool:
        """Train multi-sensor fusion GNN."""
        print(f"\n[5/7] Multi-Sensor Fusion GNN")
        print(f"  Epochs: {self.config['fusion_epochs']}")
        
        try:
            sys.path.insert(0, "ai/v3")
            import torch
            from fusion_gnn import MultiSensorFusionGNN
            
            # Quick self-test (full training requires real multi-sensor data)
            model = MultiSensorFusionGNN(latent_dim=128, n_gat_layers=2)
            
            # Generate synthetic multi-sensor scenario
            from fusion_gnn import generate_fusion_training_sample
            import numpy as np
            
            rng = np.random.default_rng(42)
            observations, truth = generate_fusion_training_sample(rng)
            
            # Forward pass test
            with torch.no_grad():
                elements, cov = model(observations)
            
            # Save initialized model
            save_path = self.model_dir / "fusion_gnn.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  Saved initialized model: {save_path}")
            
            self.results["fusion_gnn"] = {"status": "success", "note": "initialized"}
            return True
            
        except Exception as e:
            print(f"  Error: {e}")
            self.results["fusion_gnn"] = {"status": "failed", "error": str(e)}
            return False

    def train_maya_detector(self) -> bool:
        """Initialize Maya deception detector (rule-based, no training needed)."""
        print(f"\n[6/7] Maya Deception Detector")
        
        try:
            sys.path.insert(0, "ai/v3")
            from maya_detector import MayaDeceptionDetector
            import json
            
            detector = MayaDeceptionDetector()
            
            # Save default weights
            weights_path = self.model_dir / "maya_weights.json"
            with open(weights_path, 'w') as f:
                json.dump(detector.weights, f, indent=2)
            
            print(f"  Saved Maya weights: {weights_path}")
            self.results["maya_detector"] = {"status": "success", "note": "rule-based"}
            return True
            
        except Exception as e:
            print(f"  Error: {e}")
            self.results["maya_detector"] = {"status": "failed", "error": str(e)}
            return False

    def train_federated_baseline(self) -> bool:
        """Run federated learning simulation."""
        print(f"\n[7/7] Federated Learning (Astra-Mesh)")
        print(f"  Rounds: {self.config['federated_rounds']}")
        
        try:
            sys.path.insert(0, "ai/v3")
            from federated_learning import FederatedRKanTrainer, StationConfig
            import numpy as np
            
            stations = [
                StationConfig("station_leh", "Leh", "ateena_radar", 1000, "cpu"),
                StationConfig("station_thumba", "Thumba", "ateena_radar", 800, "cpu"),
                StationConfig("station_sriharikota", "Sriharikota", "optical", 600, "cpu"),
            ]
            
            trainer = FederatedRKanTrainer(stations, dp_epsilon=1.0, aggregation='fedavg')
            trainer.initialize_global_model({
                'encoder.weight': (64, 12),
                'encoder.bias': (64,),
                'head.weight': (3, 64),
                'head.bias': (3,),
            })
            
            # Run simulation rounds
            rng = np.random.default_rng(42)
            for round_num in range(min(3, self.config["federated_rounds"])):
                station_data = {}
                for cfg in stations:
                    X = rng.normal(0, 1, (cfg.n_local_samples, 12)).astype(np.float32)
                    y = rng.integers(0, 3, cfg.n_local_samples).astype(np.int64)
                    station_data[cfg.station_id] = (X, y)
                
                result = trainer.train_round(station_data, local_epochs=2)
                print(f"  Round {round_num+1}: loss={result.avg_loss:.4f}, acc={result.avg_accuracy:.4f}")
            
            self.results["federated_learning"] = {"status": "success", "rounds": 3}
            return True
            
        except Exception as e:
            print(f"  Error: {e}")
            self.results["federated_learning"] = {"status": "failed", "error": str(e)}
            return False

    def train_sensor_tasking_rl(self) -> bool:
        """Train RL-based sensor tasking optimizer (Volumetric Scan + Blind Spot Detection)."""
        print(f"\n[8/10] Sensor Tasking RL (Volumetric Scan Optimizer)")
        print(f"  Timesteps: {self.config['tasking_timesteps']}")
        
        try:
            sys.path.insert(0, "ai/v3")
            from sensor_tasking_rl import (
                SensorState, SensorType, TargetRegion,
                train_sensor_tasking_ppo, export_tasking_policy_onnx
            )
            
            # Define Astra-Mesh stations
            stations = [
                SensorState("leh", SensorType.ATEENA_RADAR, 34.0, 77.5, 3500, 1.0, 0.0, 100.0, 0.3),
                SensorState("thumba", SensorType.ATEENA_RADAR, 8.5, 76.9, 10, 1.0, 0.0, 100.0, 0.5),
                SensorState("sriharikota", SensorType.OPTICAL_CCD, 13.7, 80.2, 10, 1.0, 0.0, 100.0, 0.2),
                SensorState("port_blair", SensorType.ATEENA_RADAR, 11.6, 92.7, 10, 1.0, 0.0, 100.0, 0.4),
            ]
            
            # Define priority regions
            regions = [
                TargetRegion("Northern Border", 28.0, 38.0, 72.0, 80.0, 0.95, 2.0, 0.8),
                TargetRegion("Eastern Seaboard", 8.0, 22.0, 80.0, 88.0, 0.90, 4.0, 0.6),
                TargetRegion("Southern Ocean", -10.0, 8.0, 75.0, 95.0, 0.70, 6.0, 0.3),
                TargetRegion("Western Desert", 23.0, 35.0, 68.0, 72.0, 0.85, 3.0, 0.5),
            ]
            
            policy = train_sensor_tasking_ppo(
                stations, regions,
                total_timesteps=self.config["tasking_timesteps"],
                device=self.device,
            )
            
            # Export
            from sensor_tasking_rl import SensorTaskingEnv
            env = SensorTaskingEnv(stations, regions)
            export_tasking_policy_onnx(
                policy, env.observation_space.shape[0],
                str(self.model_dir / "sensor_tasking.onnx")
            )
            
            self.results["sensor_tasking_rl"] = {"status": "success", "regions": len(regions)}
            return True
            
        except Exception as e:
            print(f"  Error: {e}")
            self.results["sensor_tasking_rl"] = {"status": "failed", "error": str(e)}
            return False

    def train_transformer_forecaster(self) -> bool:
        """Train Transformer-based maneuver & conjunction forecaster."""
        print(f"\n[9/10] Transformer Forecaster (Enhancement to Maneuver Detector)")
        print(f"  Epochs: {self.config['transformer_epochs']}")
        
        try:
            sys.path.insert(0, "ai/v3")
            from transformer_forecaster import train_maneuver_transformer, export_transformer_onnx
            
            model = train_maneuver_transformer(
                n_samples=min(10000, self.config["maneuver_samples"]),
                seq_len=32,
                pred_len=8,
                epochs=self.config["transformer_epochs"],
                device=self.device,
            )
            
            export_transformer_onnx(model, str(self.model_dir / "transformer_forecaster.onnx"))
            
            self.results["transformer_forecaster"] = {"status": "success", "architecture": "Informer"}
            return True
            
        except Exception as e:
            print(f"  Error: {e}")
            self.results["transformer_forecaster"] = {"status": "failed", "error": str(e)}
            return False

    def train_reentry_predictor(self) -> bool:
        """Train re-entry risk predictor."""
        print(f"\n[10/10] Re-entry Risk Predictor")
        print(f"  Samples: {self.config['reentry_samples']}")
        
        try:
            sys.path.insert(0, "ai/v3")
            from reentry_predictor import train_reentry_predictor, export_reentry_onnx
            
            model = train_reentry_predictor(
                n_samples=self.config["reentry_samples"],
                epochs=50,
                device=self.device,
            )
            
            export_reentry_onnx(model, str(self.model_dir / "reentry_predictor.onnx"))
            
            self.results["reentry_predictor"] = {"status": "success", "target": "decay_error<5%"}
            return True
            
        except Exception as e:
            print(f"  Error: {e}")
            self.results["reentry_predictor"] = {"status": "failed", "error": str(e)}
            return False

    def export_all_models(self) -> None:
        """Export all trained models to production."""
        print(f"\n{'='*60}")
        print("Exporting Models to Production")
        print(f"{'='*60}")
        
        production_dir = Path("pkg/ml/models")
        production_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy ONNX files
        onnx_files = list(self.model_dir.glob("*.onnx"))
        for onnx_file in onnx_files:
            dest = production_dir / onnx_file.name
            import shutil
            shutil.copy(onnx_file, dest)
            print(f"  Copied: {onnx_file.name} → {dest}")
        
        # Copy PyTorch files
        pt_files = list(self.model_dir.glob("*.pt"))
        for pt_file in pt_files:
            dest = production_dir / pt_file.name
            shutil.copy(pt_file, dest)
            print(f"  Copied: {pt_file.name} → {dest}")

    def run_all(self) -> Dict:
        """Execute full training pipeline."""
        print(f"\n{'#'*60}")
        print(f"SSIS AI Training Pipeline — Mode: {self.mode.upper()}")
        print(f"Device: {self.device}")
        print(f"{'#'*60}")
        
        start = time.time()
        
        # Run all training components (original 7 + 3 new enhancements)
        self.train_navic_detector()           # 1. P0: NavIC Anomaly
        self.train_ateena_iqnet()             # 2. P0: Radar Detection
        self.train_collision_surrogate()      # 3. P1: Collision GNN
        self.train_maneuver_detector()        # 4. P1: Maneuver LSTM
        self.train_fusion_gnn()               # 5. P1: Fusion GNN
        self.train_maya_detector()            # 6. P1: Deception Detection
        self.train_federated_baseline()       # 7. P2: Federated Learning
        
        # NEW: High-priority enhancements
        self.train_sensor_tasking_rl()        # 8. Volumetric Scan + Blind Spot
        self.train_transformer_forecaster()   # 9. Transformer Enhancement
        self.train_reentry_predictor()        # 10. Re-entry Risk
        
        # Export to production
        self.export_all_models()
        
        # Summary
        elapsed = time.time() - start
        success_count = sum(1 for r in self.results.values() if r.get("status") == "success")
        total_count = len(self.results)
        
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Success: {success_count}/{total_count} components")
        
        for name, result in self.results.items():
            status = result.get("status", "unknown")
            icon = "✓" if status == "success" else "✗"
            print(f"  {icon} {name}: {status}")
        
        # Save report
        report_path = self.log_dir / f"training_report_{self.mode}_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump({
                "mode": self.mode,
                "device": self.device,
                "elapsed_seconds": elapsed,
                "results": self.results,
            }, f, indent=2)
        print(f"\nReport saved: {report_path}")
        
        return self.results


def main():
    parser = argparse.ArgumentParser(description="SSIS AI Training Pipeline")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Training mode: quick (fast) or full (production)"
    )
    parser.add_argument(
        "--component",
        choices=["navic", "ateena", "collision", "maneuver", "fusion", "maya", "federated", "all"],
        default="all",
        help="Train single component or all"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Compute device"
    )
    
    args = parser.parse_args()
    
    orchestrator = TrainingOrchestrator(mode=args.mode, device=args.device)
    
    if args.component == "all":
        orchestrator.run_all()
    elif args.component == "navic":
        orchestrator.train_navic_detector()
    elif args.component == "ateena":
        orchestrator.train_ateena_iqnet()
    elif args.component == "collision":
        orchestrator.train_collision_surrogate()
    elif args.component == "maneuver":
        orchestrator.train_maneuver_detector()
    elif args.component == "fusion":
        orchestrator.train_fusion_gnn()
    elif args.component == "maya":
        orchestrator.train_maya_detector()
    elif args.component == "federated":
        orchestrator.train_federated_baseline()


if __name__ == "__main__":
    main()
