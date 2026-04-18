import numpy as np
import onnxruntime as ort
import sys
import os
import time
from typing import List, Dict, Tuple

# Add simulator to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulator'))
from ateena_sim import ATeenaSimulator, ATEENA_DEFAULT_PARAMS, generate_random_scenario

class ATeenaStreamingValidator:
    """
    High-fidelity streaming validator for ATeena-IQNet.
    Executes inference on-the-fly to validate against 100k CPIs without using 6TB of disk.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.sim = ATeenaSimulator(seed=42)
        
        self.metrics = {
            'total_cpis': 0,
            'detections': 0,
            'ground_truth_targets': 0,
            'false_alarms': 0,
            'range_error_km': [],
            'velocity_error_kms': [],
            'azimuth_error_deg': [],
            'uncertainty_u': [],
            'snr_levels': []
        }

    def run_simulation(self, n_cpis: int = 1000, batch_size: int = 4):
        print(f"Starting High-Fidelity Simulation: {n_cpis} CPIs")
        print(f"Model: {self.model_path}")
        
        start_time = time.time()
        
        for i in range(0, n_cpis, batch_size):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {i}/{n_cpis} CPIs (Elapsed: {elapsed:.1f}s)")
            
            # 1. Generate Batch of Scenarios
            iq_batch = []
            batch_targets = []
            
            for _ in range(batch_size):
                targets, noise_fig = generate_random_scenario(min_targets=1, max_targets=10)
                iq = self.sim.simulate_cpi(targets, noise_figure_db=noise_fig)
                
                # Normalization: Applying 1e8 scale (Real/Imag)
                iq_dual = (np.stack([iq.real, iq.imag], axis=-1) * 1e8).astype(np.float32)
                iq_batch.append(iq_dual)
                batch_targets.append(targets)
                
                self.metrics['ground_truth_targets'] += len(targets)
                
            iq_input = np.stack(iq_batch) # [B, P, R, E, 2]
            
            # 2. Inference
            outputs = self.session.run(None, {"iq_samples": iq_input})
            detections = outputs[0] # [B, 50, 6]
            alphas = outputs[1]     # [B, 50]
            
            # 3. Analyze Results
            self._process_batch(detections, alphas, batch_targets)
            self.metrics['total_cpis'] += batch_size

        self._print_report(time.time() - start_time)

    def _process_batch(self, detections, alphas, batch_targets):
        for b in range(detections.shape[0]):
            pred_targets = detections[b] # [50, 6]
            pred_alphas = alphas[b]      # [50]
            true_targets = batch_targets[b]
            
            # DEBUG: Log first detection raw data
            if b == 0:
                print(f"DEBUG Batch 0: Max Conf: {np.max(pred_targets[:, 5]):.4f}, Mean Alpha: {np.mean(pred_alphas):.4f}")
            
            # Map detections using confidence threshold
            # Uncertainty u = 1 / S. Here we just use a confidence threshold from the 6th parameter
            active_preds = []
            for j in range(50):
                # Using a lower threshold for initial baseline verification
                if pred_targets[j, 5] > 0.1:
                    active_preds.append(pred_targets[j])
                    self.metrics['uncertainty_u'].append(1.0 / (pred_alphas[j] + 1.0))
            
            self.metrics['detections'] += len(active_preds)
            
            # Simple greedy matching for localization error
            for true_t in true_targets:
                # Find closest prediction
                min_dist = float('inf')
                best_match = None
                
                for pred_t in active_preds:
                    # R, V, Az, El, RCS, Conf
                    dist = np.abs(true_t['range_km'] - pred_t[0])
                    if dist < min_dist:
                        min_dist = dist
                        best_match = pred_t
                
                if best_match is not None and min_dist < 50: # 50km gate
                    self.metrics['range_error_km'].append(min_dist)
                    self.metrics['velocity_error_kms'].append(np.abs(true_t['velocity_kms'] - best_match[1]))
                    self.metrics['azimuth_error_deg'].append(np.abs(true_t['azimuth_deg'] - best_match[2]))
                else:
                    # Missed detection
                    pass

    def _print_report(self, duration: float):
        n_true = self.metrics['ground_truth_targets']
        n_pred = self.metrics['detections']
        pd = len(self.metrics['range_error_km']) / n_true if n_true > 0 else 1.0
        pfa = (n_pred - len(self.metrics['range_error_km'])) / self.metrics['total_cpis']
        
        print("\n" + "="*40)
        print("SSIS PHASE 2.1 SOVEREIGN RADAR REPORT")
        print("="*40)
        print(f"Total CPIs Simulated:  {self.metrics['total_cpis']}")
        print(f"Simulation Duration:   {duration:.1f}s")
        print(f"Avg Latency/CPI:       {duration*1000/self.metrics['total_cpis']:.1f}ms")
        print("-" * 40)
        print(f"Probability of Detection (P_d): {pd:.4f}")
        print(f"False Alarms per CPI (P_fa):    {pfa:.4f}")
        print("-" * 40)
        print(f"RMS Range Error:       {np.sqrt(np.mean(np.square(self.metrics['range_error_km']))):.3f} km")
        print(f"RMS Velocity Error:    {np.sqrt(np.mean(np.square(self.metrics['velocity_error_kms']))):.3f} km/s")
        print(f"Avg Epistemic Uncertainty (u): {np.mean(self.metrics['uncertainty_u']):.4f}")
        print("="*40)

if __name__ == "__main__":
    model_path = "ai/training/models/ateena_iqnet_production.onnx"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
        
    validator = ATeenaStreamingValidator(model_path)
    
    # Run sanity check (1000 CPIs) or full sim (100,000 CPIs)
    n_cpis = 1000
    if len(sys.argv) > 1:
        n_cpis = int(sys.argv[1])
        
    validator.run_simulation(n_cpis=n_cpis, batch_size=4)
