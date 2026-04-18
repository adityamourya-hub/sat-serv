# SSIS Model Training & ONNX Export Plan

**Objective**: Train all 5 Axiom-Zero models on MIT SPLID and CARA datasets, export trained weights to ONNX for Go inference.

**Timeline**: 2-3 weeks (parallel training possible on GPU cluster)

---

## Phase 1: Data Pipeline Setup (Days 1-3)

### 1.1 MIT SPLID Dataset Integration

**Dataset**: MIT ARCLab Satellite Pattern-of-Life Identification Dataset (SPLID 2024)
- Source: https://splid-data.mit.edu/
- 1,000+ labeled satellites
- 8 PoL classes (sun-sync orbiters, GEO station-keepers, etc.)
- Multi-sensor data (optical, radar, ISL)

**Pipeline Tasks**:

```python
# ai/training/data/splid_loader.py
class SPLIDLoader:
    """
    Loads and preprocesses MIT SPLID dataset for PatchTST-JEPA training.
    
    Output format:
    - observations: [N, 72, 6] - 72 timesteps of [pos(3), vel(3)]
    - labels: [N] - PoL class (0-7)
    - has_maneuver: [N] - binary anomaly flag
    """
    
    def __init__(self, data_dir: str, seq_length: int = 72):
        self.data_dir = data_dir
        self.seq_length = seq_length
        
    def load_trajectory(self, sat_id: str) -> np.ndarray:
        """Load TLE/JSON trajectory for satellite"""
        # Parse TLE to states
        # Return: [n_timesteps, 6] positions/velocities in km, km/s
        
    def create_sequences(self, trajectory: np.ndarray) -> List[np.ndarray]:
        """Sliding window sequences for self-supervised training"""
        # stride=6 for overlap
        # Return: list of [72, 6] sequences
        
    def get_pol_label(self, sat_id: str) -> int:
        """Map satellite to PoL class"""
        # Mapping:
        # 0: sun_sync_orbiters
        # 1: geo_station_keepers
        # 2: walker_constellation
        # 3: communication_satellites
        # 4: navigation_satellites
        # 5: weather_satellites
        # 6: earth_observation
        # 7: unknown / unlabeled
```

**Expected Data Volume**:
- ~10,000 sequences × 72 timesteps each
- ~720,000 state vectors for pre-training
- Split: 80% train / 10% val / 10% test

### 1.2 CARA Conjunction Dataset Integration

**Dataset**: NASA CARA Historical Conjunction Events
- ~1,000,000 conjunction events (synthetic + real)
- Fields: primary_state, secondary_state, covariances, TCA, miss_distance, Pc
- Target Pc range: 10⁻⁹ to 10⁻³ (highly imbalanced)

**Pipeline Tasks**:

```python
# ai/training/data/cara_loader.py
class CARALoader:
    """
    Loads CARA conjunction data for neural surrogate training.
    
    Output format:
    - primary: [N, 6] - [pos(3), vel(3)] in km, km/s
    - secondary: [N, 6]
    - cov_primary: [N, 6, 6]
    - cov_secondary: [N, 6, 6]
    - reference_pc: [N] - Alfano MC reference
    - confidence_flag: [N] - 1 for high-confidence cases
    """
    
    def load_conjunction(self, event_id: str) -> dict:
        """Load single conjunction event"""
        
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle Pc distribution (log-uniform sampling)"""
        # Most real conjunctions have Pc ~ 10⁻⁶ to 10⁻⁴
        # Need stratified sampling across Pc bins
        
    def augment_covariance(self, cov: np.ndarray) -> np.ndarray:
        """Add synthetic uncertainty for robustness"""
```

### 1.3 Orekit Synthetic Orbit Generation

**For**: GHNN training (Hamiltonian dynamics)

```python
# ai/training/data/orbital_simulator.py
class OrekitSimulator:
    """
    Generates synthetic orbital trajectories with physics.
    Uses Orekit (Java) via pyjnius or pre-computed datasets.
    """
    
    def generate_trajectory(self, 
                          a: float,  # semi-major axis
                          e: float,  # eccentricity
                          i: float,  # inclination
                          duration_days: float = 7.0,
                          dt_minutes: float = 1.0) -> np.ndarray:
        """
        Generate high-fidelity orbital trajectory.
        Includes: J2, J3, atmospheric drag, solar radiation pressure
        """
        
    def generate_debris_cloud(self, n_objects: int = 1000,
                             collision_point: np.ndarray = None) -> np.ndarray:
        """Generate synthetic debris field for volumetric training"""
```

---

## Phase 2: Model Training (Days 4-14)

### 2.1 PatchTST-JEPA Training (Days 4-7)

**File**: `ai/training/train_pol.py`

```python
"""
PatchTST-JEPA Training Pipeline

Phase 1: Self-supervised pre-training (50 epochs)
- Mask ratio: 0.6 (60% of patches masked)
- Optimizer: AdamW, lr=1e-4, weight_decay=0.05
- Batch size: 32
- GPU memory: ~4GB per batch

Phase 2: Supervised fine-tuning (50 epochs)
- Freeze encoder (optional), train classifier
- Class weights for imbalanced PoL classes
- Early stopping on validation F1
"""

import torch
from torch.utils.data import DataLoader
from ai.v3.pol_patchtst_jepa import PatchTSTJEPA

# Hyperparameters
PRETRAIN_EPOCHS = 50
FINETUNE_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MASK_RATIO = 0.6
PATCH_SIZE = 12
D_MODEL = 256

# Training loop
def pretrain_epoch(model, dataloader, optimizer):
    total_loss = 0
    for batch in dataloader:
        observations = batch['observations'].cuda()
        
        # JEPA pre-training step
        loss, metrics = model.pretrain_step(observations, mask_ratio=MASK_RATIO)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def finetune_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        observations = batch['observations'].cuda()
        labels = batch['labels'].cuda()
        has_maneuver = batch['has_maneuver'].cuda()
        
        outputs = model(observations)
        
        # Multi-task loss
        pol_loss = F.cross_entropy(outputs['pol_logits'], labels)
        anomaly_loss = F.binary_cross_entropy(outputs['anomaly_score'], has_maneuver)
        
        loss = pol_loss + 0.5 * anomaly_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Expected Results:
# - Pre-training loss: ~0.15 (MSE on reconstructed patches)
# - Fine-tuning accuracy: ~92% PoL classification
# - Anomaly detection F1: ~0.88
# - Training time: ~6 hours on RTX 4090
```

### 2.2 GHNN Training (Days 6-9)

**File**: `ai/training/train_hnn.py`

```python
"""
Generalized Hamiltonian Neural Network Training

Key: Physics-informed loss ensures energy conservation

Training objectives:
1. Trajectory reconstruction (MSE on positions/velocities)
2. Energy conservation (penalty on H drift)
3. Symplectic structure (optional, via constraints)

Expected energy drift: < 10^-15 over 168 hours
"""

from ai.v3.hamiltonian_dynamics import GeneralizedHNN

# Hyperparameters
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
DT = 60.0  # 1-minute integration steps
N_STEPS = 100  # Trajectory length
PHYSICS_WEIGHT = 1.0

def train_epoch(model, dataloader, optimizer):
    total_recon_loss = 0
    total_physics_loss = 0
    
    for batch in dataloader:
        initial_states = batch[:, 0, :].cuda()  # [B, 6]
        target_trajectories = batch.cuda()  # [B, T, 6]
        
        # Predict trajectory
        pred_trajectory = model.integrate(initial_states, DT, N_STEPS)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(pred_trajectory, target_trajectories)
        
        # Physics loss: energy drift
        energy_drift = model.energy_drift(pred_trajectory)
        physics_loss = energy_drift.mean()
        
        # Combined loss
        loss = recon_loss + PHYSICS_WEIGHT * physics_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_recon_loss += recon_loss.item()
        total_physics_loss += physics_loss.item()
    
    return total_recon_loss / len(dataloader), total_physics_loss / len(dataloader)

# Validation: Check energy conservation on held-out orbits
# Target: < 10^-15 drift / 168h for circular orbits
# < 10^-12 drift / 168h for highly eccentric orbits
```

### 2.3 Conjunction Surrogate Training (Days 7-10)

**File**: `ai/training/train_conjunction.py`

```python
"""
Neural Surrogate for Pc Training

Key: Learn both prediction and uncertainty

Loss: Negative log-likelihood with learned variance
L = 0.5 * (precision * (pred - target)^2 - log(precision))

This is equivalent to maximizing likelihood under Gaussian uncertainty.
"""

from ai.v3.conjunction_surrogate import BayesianConjunctionSurrogate

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MC_SAMPLES = 10  # For MC dropout uncertainty

def train_epoch(model, dataloader, optimizer):
    total_nll = 0
    total_mae = 0
    
    for batch in dataloader:
        # Forward with MC dropout
        result = model.forward_mc(
            batch['primary'].cuda(),
            batch['secondary'].cuda(),
            batch['cov_p'].cuda(),
            batch['cov_s'].cuda()
        )
        
        target = batch['reference_pc'].cuda()
        pred = result['pc']
        std = result['pc_std']
        
        # Negative log-likelihood
        precision = 1.0 / (std ** 2 + 1e-6)
        nll = 0.5 * (precision * (pred - target) ** 2 - torch.log(precision))
        loss = nll.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_nll += loss.item()
        total_mae += (pred - target).abs().mean().item()
    
    return total_nll / len(dataloader), total_mae / len(dataloader)

# Validation metrics:
# - Mean absolute error on Pc: < 10^-7
# - Coverage (90% CI includes true Pc): > 85%
# - Fallback rate (uncertainty > threshold): ~5-10%
```

### 2.4 Symbolic Transformer Training (Days 9-12)

**File**: `ai/training/train_symbolic.py`

```python
"""
Symbolic Transformer Training (2-phase)

Phase 1: Pre-train on Feynman physics equations
- Dataset: 100 equations from Feynman lectures
- Learn symbolic reasoning patterns
- 75 epochs

Phase 2: Fine-tune on orbital maneuver laws
- Dataset: 10,000 labeled maneuver descriptions
- LaTeX output: "station-keep +0.12 m/s radial"
- 75 epochs
"""

from ai.v3.symbolic_transformer_sr import SymbolicTransformer

# Phase 1: Pre-training
PRETRAIN_EPOCHS = 75
# Phase 2: Fine-tuning
FINETUNE_EPOCHS = 75

def pretrain_epoch(model, dataloader, optimizer):
    """Train on physics equations (Feynman)"""
    total_loss = 0
    
    for batch in dataloader:
        residuals = batch['residuals'].cuda()  # [B, 10, 6]
        target_tokens = batch['tokens'].cuda()  # [B, seq_len]
        
        # Teacher forcing
        logits, _ = model.decode_symbolic(
            model.encode_residuals(residuals),
            target_tokens
        )
        
        loss = F.cross_entropy(
            logits.view(-1, model.vocab.vocab_size),
            target_tokens.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def finetune_epoch(model, dataloader, optimizer):
    """Fine-tune on orbital maneuvers with physics constraints"""
    total_ce_loss = 0
    total_physics_loss = 0
    
    for batch in dataloader:
        residuals = batch['residuals'].cuda()
        target_tokens = batch['tokens'].cuda()
        
        # Forward
        memory = model.encode_residuals(residuals)
        logits, generated = model.decode_symbolic(memory, target_tokens)
        
        # CE loss
        ce_loss = F.cross_entropy(
            logits.view(-1, model.vocab.vocab_size),
            target_tokens.view(-1)
        )
        
        # Physics constraint: unit consistency
        physics_loss = 0
        for seq in generated:
            units = model.physics_constraints.compute_units(seq)
            if units is None:
                physics_loss += 1.0  # Penalize invalid
        
        loss = ce_loss + 0.1 * physics_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_ce_loss += ce_loss.item()
        total_physics_loss += physics_loss
    
    return total_ce_loss / len(dataloader)

# Expected:
# - Token accuracy: ~88%
# - Valid physics expressions: ~95%
# - Human-readable: 100% (by design)
```

### 2.5 Volumetric INR Training (Days 11-14)

**File**: `ai/training/train_volumetric.py`

```python
"""
Hash-Encoded INR Training

Dataset: Synthetic debris clouds from breakup models
- 1,000 cloud simulations
- 100-10,000 fragments each
- Density field: [0, 1] probability of debris

Training: Reconstruction + PINN physics loss
"""

from ai.v3.volumetric_hash_inr import PhysicsInformedVolumetricNN, HashEncodedField

EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
PINN_WEIGHT = 0.1

def train_epoch(model, dataloader, optimizer):
    total_recon = 0
    total_physics = 0
    
    for batch in dataloader:
        coords = batch['coords'].cuda()  # [B, 3]
        target = batch['density'].cuda()  # [B, 4] [density, prob_x, prob_y, prob_z]
        
        # Forward
        output = model(coords)
        
        # Reconstruction
        recon_loss = F.mse_loss(output, target)
        
        # PINN physics loss
        physics_loss = model.physics_loss(coords)
        
        loss = recon_loss + PINN_WEIGHT * physics_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_recon += recon_loss.item()
        total_physics += physics_loss.item()
    
    return total_recon / len(dataloader), total_physics / len(dataloader)

# Expected:
# - Reconstruction MSE: < 0.01
# - Physics loss: < 0.001
# - Query speed: <10ms per cell
```

---

## Phase 3: ONNX Export (Days 15-17)

### 3.1 Export Pipeline

**File**: `ai/training/export_trained.py`

```python
"""
Export trained models to ONNX for Go inference.

All exports use:
- ONNX opset 13 (stable, widely supported)
- Constant folding for optimization
- No dynamic shapes (fixed batch sizes)
"""

import torch
import torch.onnx

def export_model(model, dummy_input, path, input_names, output_names):
    """Generic export function"""
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=13,
        do_constant_folding=True,
        export_params=True  # Include trained weights
    )
    
    # Verify with onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    
    print(f"✅ Exported: {path}")
    print(f"   Inputs: {[i.name for i in sess.get_inputs()]}")
    print(f"   Outputs: {[o.name for o in sess.get_outputs()]}")
    
    return path

# Export each model
def export_all(artifacts_dir="artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. PatchTST-JEPA (inference mode)
    print("\n[1/5] Exporting PatchTST-JEPA...")
    pol_model = load_trained_model("pol_patchtst", "checkpoints/pol_best.pt")
    export_model(
        pol_model,
        torch.randn(1, 72, 6),
        f"{artifacts_dir}/pol_patchtst.onnx",
        ["observations"],
        ["pol_logits", "anomaly_score", "embedding"]
    )
    
    # 2. GHNN (Hamiltonian only)
    print("\n[2/5] Exporting GHNN...")
    hnn_model = load_trained_model("hnn_orbit", "checkpoints/hnn_best.pt")
    export_model(
        hnn_model.hamiltonian_net,
        torch.randn(1, 6),
        f"{artifacts_dir}/hnn_orbit.onnx",
        ["state"],
        ["hamiltonian"]
    )
    
    # 3. Symbolic Transformer (encoder only)
    print("\n[3/5] Exporting Symbolic SR...")
    sym_model = load_trained_model("symbolic_sr", "checkpoints/symbolic_best.pt")
    export_model(
        create_encoder_wrapper(sym_model),
        (torch.randn(1, 10, 6), torch.randn(1, 512)),
        f"{artifacts_dir}/symbolic_sr.onnx",
        ["residuals", "context"],
        ["encoded_memory"]
    )
    
    # 4. Conjunction Surrogate
    print("\n[4/5] Exporting Conjunction Surrogate...")
    conj_model = load_trained_model("conjunction_surrogate", "checkpoints/conjunction_best.pt")
    export_model(
        conj_model,
        (
            torch.randn(1, 6),
            torch.randn(1, 6),
            torch.eye(6).unsqueeze(0),
            torch.eye(6).unsqueeze(0)
        ),
        f"{artifacts_dir}/conjunction_surrogate.onnx",
        ["primary", "secondary", "cov_primary", "cov_secondary"],
        ["pc", "pc_std", "confidence", "fallback"]
    )
    
    # 5. Volumetric INR
    print("\n[5/5] Exporting Volumetric INR...")
    vol_model = load_trained_model("volumetric_inr", "checkpoints/volumetric_best.pt")
    export_model(
        vol_model.hash_field,
        torch.rand(1, 3),
        f"{artifacts_dir}/volumetric_inr.onnx",
        ["coords"],
        ["density", "prob_x", "prob_y", "prob_z"]
    )
    
    print("\n✅ All models exported successfully!")
    print(f"📁 Artifacts directory: {artifacts_dir}/")
```

### 3.2 Expected Artifact Sizes

| Model | Size (FP32) | Size (FP16) | Size (INT8) |
|-------|-------------|-------------|-------------|
| pol_patchtst | ~12 MB | ~6 MB | ~3 MB |
| hnn_orbit | ~2 MB | ~1 MB | ~500 KB |
| symbolic_sr | ~15 MB | ~8 MB | ~4 MB |
| conjunction_surrogate | ~8 MB | ~4 MB | ~2 MB |
| volumetric_inr | ~20 MB | ~10 MB | ~5 MB |
| **Total** | **~57 MB** | **~29 MB** | **~14.5 MB** |

---

## Phase 4: Validation (Days 18-21)

### 4.1 Test Suite

**File**: `ai/training/validate_exported.py`

```python
"""
Validation of exported ONNX models.
"""

import onnxruntime as ort
import numpy as np

def test_pol_inference():
    """Test PoL model"""
    sess = ort.InferenceSession("artifacts/pol_patchtst.onnx")
    
    # Test case: Sun-synchronous orbit pattern
    observations = np.random.randn(1, 72, 6).astype(np.float32)
    
    outputs = sess.run(None, {"observations": observations})
    pol_logits, anomaly_score, embedding = outputs
    
    assert pol_logits.shape == (1, 8), "PoL output shape mismatch"
    assert 0 <= anomaly_score[0] <= 1, "Anomaly score out of range"
    assert embedding.shape == (1, 256), "Embedding shape mismatch"
    
    print("✅ PoL model validated")

def test_hnn_energy_conservation():
    """Test GHNN energy conservation"""
    sess = ort.InferenceSession("artifacts/hnn_orbit.onnx")
    
    # Circular orbit state
    state = np.array([[7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]], dtype=np.float32)
    
    hamiltonian = sess.run(None, {"state": state})[0]
    
    # Energy should be negative (bound orbit)
    assert hamiltonian[0] < 0, "Energy positive for bound orbit"
    
    print("✅ HNN energy validated")

def test_conjunction_speed():
    """Test conjunction latency"""
    sess = ort.InferenceSession("artifacts/conjunction_surrogate.onnx")
    
    primary = np.random.randn(1, 6).astype(np.float32)
    secondary = np.random.randn(1, 6).astype(np.float32)
    cov_p = np.eye(6, dtype=np.float32).reshape(1, 6, 6)
    cov_s = np.eye(6, dtype=np.float32).reshape(1, 6, 6)
    
    import time
    start = time.time()
    for _ in range(100):
        sess.run(None, {
            "primary": primary,
            "secondary": secondary,
            "cov_primary": cov_p,
            "cov_secondary": cov_s
        })
    elapsed = (time.time() - start) / 100 * 1000  # ms
    
    assert elapsed < 10, f"Latency {elapsed}ms exceeds 10ms target"
    
    print(f"✅ Conjunction latency: {elapsed:.2f}ms")
```

### 4.2 Go Integration Test

```go
// pkg/ml/onnx/axiom_bridge_test.go
func TestAxiomBridgeIntegration(t *testing.T) {
    bridge, err := NewAxiomBridge("artifacts")
    if err != nil {
        t.Fatalf("Failed to create bridge: %v", err)
    }
    defer bridge.Close()
    
    // Test PoL detection
    observations := make([][][6]float32, 1)
    observations[0] = make([][6]float32, 72)
    
    result, err := bridge.DetectPoL(observations)
    if err != nil {
        t.Fatalf("PoL detection failed: %v", err)
    }
    
    if result.Confidence < 0 || result.Confidence > 1 {
        t.Errorf("Invalid confidence: %f", result.Confidence)
    }
    
    t.Logf("PoL Result: class=%s, confidence=%.3f", result.Class, result.Confidence)
}
```

---

## Execution Schedule

| Day | Task | GPU Hours | Dependencies |
|-----|------|-----------|--------------|
| 1-3 | Data pipeline setup | 0 | Dataset access |
| 4-7 | PatchTST-JEPA training | ~24 | SPLID data |
| 6-9 | GHNN training | ~32 | Orekit sims |
| 7-10 | Conjunction training | ~16 | CARA data |
| 9-12 | Symbolic training | ~20 | Feynman + orbital |
| 11-14 | Volumetric training | ~28 | Debris sims |
| 15-17 | ONNX export | ~4 | All trained models |
| 18-21 | Validation | ~8 | ONNX files |

**Total GPU Time**: ~132 hours (~5.5 days on single GPU)
**Wall Clock Time**: ~3 weeks (parallel training on 2-3 GPUs)

---

## Checkpoint Strategy

**Save checkpoints every 10 epochs**:
```
checkpoints/
├── pol_patchtst/
│   ├── epoch_10.pt
│   ├── epoch_20.pt
│   └── best.pt (highest val F1)
├── hnn_orbit/
├── symbolic_sr/
├── conjunction_surrogate/
└── volumetric_inr/
```

**Resume training** from any checkpoint if interrupted.

---

## Success Criteria

| Model | Metric | Target |
|-------|--------|--------|
| PoL | Val F1 | > 0.90 |
| PoL | Anomaly Recall | > 0.85 |
| GHNN | Energy drift / 168h | < 10^-12 |
| Conjunction | MAE on Pc | < 10^-7 |
| Conjunction | Latency | < 10ms |
| Symbolic | Token accuracy | > 85% |
| Volumetric | Query time | < 10ms |
| **All** | ONNX export | ✅ Pass |
| **All** | Go inference | ✅ Pass |

---

*Plan Version: 1.0*
*Target Completion: 21 days from data availability*
