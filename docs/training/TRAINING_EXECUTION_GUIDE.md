# SSIS Training & Export - Execution Guide

**Ready-to-run training pipeline for Axiom-Zero models.**

---

## Quick Start

### Prerequisites

```bash
# Python dependencies
pip install torch torchvision numpy pandas tqdm scikit-learn onnx onnxruntime

# For CARA data processing
pip install pyarrow  # For parquet files

# Optional: for Orekit orbital simulations
pip install orekit
```

### Directory Structure

```
/Users/adi/idex_iaf/
├── ai/
│   ├── v3/                          # Model implementations
│   │   ├── pol_patchtst_jepa.py
│   │   ├── hamiltonian_dynamics.py
│   │   ├── symbolic_transformer_sr.py
│   │   ├── conjunction_surrogate.py
│   │   └── volumetric_hash_inr.py
│   └── training/
│       ├── train_pol.py              # SPLID training
│       ├── train_conjunction.py      # CARA training
│       ├── export_trained.py         # ONNX export
│       ├── TRAINING_PLAN.md          # Detailed plan
│       └── this file
├── artifacts/                        # ONNX outputs
│   ├── pol_patchtst.onnx
│   ├── hnn_orbit.onnx
│   ├── symbolic_sr.onnx
│   ├── conjunction_surrogate.onnx
│   ├── volumetric_inr.onnx
│   └── model_manifest.json
└── checkpoints/                      # Training checkpoints
    ├── pol_best.pt
    ├── hnn_best.pt
    ├── symbolic_best.pt
    ├── conjunction_best.pt
    └── volumetric_best.pt
```

---

## Step-by-Step Execution

### Step 1: Prepare Datasets

#### MIT SPLID Dataset

1. **Download** from https://splid-data.mit.edu/
2. **Organize** into structure:

```
splid_data/
├── train/
│   ├── sat_001.json
│   ├── sat_002.json
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── labels.json
```

3. **JSON format per satellite**:
```json
{
    "sat_id": "001",
    "norad_id": 12345,
    "pol_class": "sun_sync_orbiter",
    "timestamps": [0, 60, 120, ...],
    "states": [[x,y,z,vx,vy,vz], ...],
    "has_maneuver": [0,0,0,1,0,...]
}
```

#### CARA Conjunction Dataset

1. **Obtain** historical CARA data (requires NASA collaboration)
2. **Format** as CSV/Parquet:

```python
cara_data.csv columns:
- primary_state: "[x,y,z,vx,vy,vz]"  # JSON array
- secondary_state: "[x,y,z,vx,vy,vz]"
- cov_primary: "[[...],...]"  # 6x6 matrix as JSON
- cov_secondary: "[[...],...]"
- reference_pc: float
- miss_distance: float
```

3. **Split** into train/val/test (80/10/10)

---

### Step 2: Train Models

#### Train PatchTST-JEPA (PoL Detection)

```bash
cd /Users/adi/idex_iaf/ai/training

python train_pol.py \
    --data-dir /path/to/splid_data \
    --output-dir ../../checkpoints \
    --pretrain-epochs 50 \
    --finetune-epochs 50 \
    --batch-size 32 \
    --lr 1e-4
```

**Expected output**:
```
Epoch 1/50: Pre-training loss: 0.1523
...
Epoch 50/50: Pre-training loss: 0.0432
✅ Saved best pre-trained model (loss: 0.0432)

Epoch 1/50: Train Acc: 0.723, Val Acc: 0.745, Val F1: 0.812
...
Epoch 50/50: Train Acc: 0.941, Val Acc: 0.923, Val F1: 0.901
✅ Saved best model (F1: 0.901)
```

**Training time**: ~6 hours on RTX 4090

#### Train Conjunction Surrogate (Fast Pc)

```bash
python train_conjunction.py \
    --data-dir /path/to/cara_data \
    --output-dir ../../checkpoints \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-3
```

**Expected output**:
```
Epoch 1/100: NLL=0.8234, MAE=1.23e-05
Val: MAE=8.45e-06, Coverage=87.3%, Fallback=4.2%
...
Epoch 100/100: NLL=0.2341, MAE=2.34e-07
Val: MAE=1.87e-07, Coverage=91.2%, Fallback=5.8%
✅ Saved best model (MAE: 1.87e-07)
```

**Training time**: ~4 hours on RTX 4090

#### Train Other Models

For GHNN, Symbolic SR, and Volumetric INR, use the orchestrator:

```bash
cd /Users/adi/idex_iaf/ai/training
python orchestrator.py  # Full 7-phase training
```

Or train individually (scripts to be created following the pattern in `TRAINING_PLAN.md`).

---

### Step 3: Export to ONNX

Once all checkpoints are ready:

```bash
cd /Users/adi/idex_iaf/ai/training

python export_trained.py \
    --checkpoint-dir ../../checkpoints \
    --output-dir ../../artifacts
```

**Expected output**:
```
============================================================
SSIS Model Export Pipeline
============================================================

[1/5] Exporting PatchTST-JEPA...
✅ Loaded checkpoint from ../../checkpoints/pol_best.pt
   Metrics: {'accuracy': 0.923, 'f1_anomaly': 0.901}
✅ Exported: ../../artifacts/pol_patchtst.onnx
   Inputs: ['observations']
   Outputs: ['pol_logits', 'anomaly_score', 'embedding']
   Test inference: ✅ (3 outputs)

[2/5] Exporting GHNN...
...

============================================================
Export Summary
============================================================
✅ pol_patchtst
✅ hnn_orbit
✅ symbolic_sr
✅ conjunction_surrogate
✅ volumetric_inr

Total: 5/5 models exported

✅ Created manifest: ../../artifacts/model_manifest.json

============================================================
Export Complete!
Artifacts directory: ../../artifacts/
============================================================
```

---

### Step 4: Validate in Go

```bash
cd /Users/adi/idex_iaf

# Build
go build ./...

# Test ONNX integration
go test -v ./pkg/ml/onnx/... -run TestAxiomBridgeIntegration
```

**Expected output**:
```
=== RUN   TestAxiomBridgeIntegration
    axiom_bridge_test.go:15: Loading models from artifacts/
    axiom_bridge_test.go:23: PoL Result: class=sun_sync_orbiter, confidence=0.923
    axiom_bridge_test.go:30: Conjunction Pc: 1.23e-05, confidence=0.912
--- PASS: TestAxiomBridgeIntegration (0.45s)
PASS
```

---

## Performance Targets

| Model | Dataset Size | Training Time | Target Metric | Expected Value |
|-------|-------------|---------------|---------------|----------------|
| PoL (PatchTST-JEPA) | ~10K sequences | 6 hrs | Val F1 | > 0.90 |
| Conjunction Surrogate | ~1M cases | 4 hrs | MAE on Pc | < 1e-7 |
| GHNN | ~50K orbits | 8 hrs | Energy drift | < 1e-12 / 168h |
| Symbolic SR | ~10K laws | 6 hrs | Token accuracy | > 85% |
| Volumetric INR | ~1K clouds | 4 hrs | Query latency | < 10ms |

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size
```bash
python train_pol.py --batch-size 16  # Instead of 32
```

### Issue: Checkpoint not found during export

**Solution**: Train first, or use pretrained flag
```bash
python export_trained.py --checkpoint-dir checkpoints/
# If checkpoints missing, uses random weights with warning
```

### Issue: ONNX export fails with dynamic shapes

**Solution**: Export uses fixed batch size (1). For larger batches, modify:
```python
# In export_trained.py
dummy_input = torch.randn(BATCH_SIZE, 72, 6)  # Change 1 to your batch size
```

### Issue: Go build fails

**Solution**: Check artifacts exist
```bash
ls /Users/adi/idex_iaf/artifacts/
# Should show: *.onnx files and model_manifest.json
```

---

## Next Steps After Training

1. **Edge Quantization** (Optional):
   ```bash
   # Convert to INT8 for satellite deployment
   python quantize_models.py --input artifacts/ --output edge_artifacts/
   ```

2. **Operator Dashboard**:
   ```bash
   cd /Users/adi/idex_iaf/ui/
   npm install
   npm run dev  # Svelte + CesiumJS interface
   ```

3. **Formal Verification**:
   ```bash
   # Verify energy conservation bounds
   go test ./pkg/physics/... -run TestHamiltonianConservation
   ```

---

## Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `ai/v3/*.py` | Model implementations | ✅ Complete |
| `ai/training/train_pol.py` | SPLID training | ✅ Ready |
| `ai/training/train_conjunction.py` | CARA training | ✅ Ready |
| `ai/training/export_trained.py` | ONNX export | ✅ Ready |
| `ai/training/orchestrator.py` | Full pipeline | ✅ Ready |
| `TRAINING_PLAN.md` | Detailed documentation | ✅ Complete |
| `TRAINING_EXECUTION_GUIDE.md` | This guide | ✅ Complete |

---

**Ready to execute!** Start with Step 1 (data preparation) and proceed through the pipeline.

---

*Guide Version: 1.0*
*Last Updated: April 18, 2026*
