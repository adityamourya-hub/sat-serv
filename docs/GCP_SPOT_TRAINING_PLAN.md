# GCP Spot GPU Training Plan — SSIS/IDEX AI Models

## Overview

Train all 10 AI/ML components using your `training-data/` datasets on GCP Spot GPU instances, achieving ~70% cost savings vs on-demand.

---

## 1. Dataset Inventory & Model Mapping

| Dataset Folder | Size | Format | Target Model(s) | GPU Needed |
|---|---|---|---|---|
| **A/** TLEs/Ephemeris | ~14 MB | TXT (TLE), CSV | Maneuver Detector (LSTM), Transformer Forecaster, Neural ODE, Cataloguing | T4 |
| **A/** `satcat-data.csv` | 6.3 MB | CSV | Pattern-of-Life, Cataloguing BVH | CPU only |
| **A/** `Space-weather-data-from-1962.csv` | 2.8 MB | CSV | Re-entry Predictor (atmospheric density inputs) | T4 |
| **A/** `earth-orientation-data.csv` | 2.3 MB | CSV | Propagator (EOP corrections) | CPU only |
| **B/** Conjunction/Collision | ~22 GB+ | CSV.gz, TAR.gz | Collision Surrogate GNN, Foster Pc surrogate | T4/V100 |
| **B/** `train_data.zip` (ESA CDMs) | 84 MB | CSV (zipped) | Collision Surrogate GNN | T4 |
| **C/** NavIC/GNSS spoofing | ~40 KB | JSON metadata + ZIP | NavIC Anomaly Detector (BiLSTM) | T4 |
| **D/** Re-entry history | 376 KB | CSV | Re-entry Risk Predictor (FFN+Transformer) | T4 |

---

## 2. Training Phases (Sequential by Priority)

### Phase 1 — Quick Wins (CPU + single T4, ~2 hrs, ~$0.50)

| # | Model | Script | Dataset | Est. Time |
|---|---|---|---|---|
| 1 | **Maneuver Detector** (LSTM+Attention) | `ai/maneuver/train_maneuver_detector.py` | A/ TLE files → synthetic residuals | 15 min |
| 2 | **Re-entry Predictor** (FFN) | `ai/v3/reentry_predictor.py` | D/ `Reentry_History_Spreadsheet.csv` + A/ `Space-weather-data-from-1962.csv` | 10 min |
| 3 | **Maya Deception Detector** | `ai/v3/maya_detector.py` | Rule-based (no GPU) | 1 min |
| 4 | **NavIC Anomaly Detector** (BiLSTM) | `ai/v3/navic_anomaly_detector.py` | C/ GNSS metadata + synthetic augmentation | 20 min |
| 5 | **Transformer Forecaster** | `ai/v3/transformer_forecaster.py` | A/ TLE time-series | 30 min |
| 6 | **Federated Learning** (Astra-Mesh) | `ai/v3/federated_learning.py` | Synthetic per-station | 15 min |

### Phase 2 — Heavy Training (T4/V100 spot, ~6 hrs, ~$2-5)

| # | Model | Script | Dataset | Est. Time |
|---|---|---|---|---|
| 7 | **Collision Surrogate GNN** | `ai/collision/train_collision_surrogate_spot.py` | B/ `train_data.zip` + `IVV_Releasable_Dataset_*.csv.gz` | 2-4 hrs |
| 8 | **Multi-Sensor Fusion GNN** | `ai/v3/fusion_gnn.py` | A/ multi-source TLEs + B/ CDMs | 1-2 hrs |
| 9 | **Sensor Tasking RL** (PPO) | `ai/v3/sensor_tasking_rl.py` | Synthetic env + A/ catalog | 1-2 hrs |

### Phase 3 — Full-Scale (V100/A100, optional)

| # | Model | Script | Dataset | Est. Time |
|---|---|---|---|---|
| 10 | **Collision GNN** full IVV dataset | `ai/collision/train_collision_surrogate_spot.py` | B/ `AerospaceIVVDataset_20251009a.tar.gz` (22 GB) | 6-12 hrs |

---

## 3. GCP Infrastructure Setup

### 3.1 One-Time Setup

```bash
# Set project
export GCP_PROJECT="your-project-id"
export GCS_BUCKET="ssis-training-$(date +%s)"
export REGION="us-central1"
export ZONE="us-central1-a"

# Create GCS bucket for checkpoints + data
gcloud storage buckets create gs://$GCS_BUCKET --location=$REGION

# Upload training data to GCS (do this from your local machine)
gsutil -m cp -r training-data/ gs://$GCS_BUCKET/training-data/

# Create service account for training
gcloud iam service-accounts create ssis-trainer \
  --display-name="SSIS Training SA"

gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --member="serviceAccount:ssis-trainer@$GCP_PROJECT.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud iam service-accounts keys create ~/ssis-trainer-key.json \
  --iam-account=ssis-trainer@$GCP_PROJECT.iam.gserviceaccount.com
```

### 3.2 Create Spot VM (Phase 1 — T4)

```bash
gcloud compute instances create ssis-train-t4 \
  --zone=$ZONE \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --boot-disk-size=100GB \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=True" \
  --scopes=storage-full
```

**Estimated cost**: T4 spot = ~$0.11/hr, n1-standard-8 spot = ~$0.07/hr → **~$0.18/hr total**

### 3.3 Create Spot VM (Phase 2 — V100, for large collision dataset)

```bash
gcloud compute instances create ssis-train-v100 \
  --zone=$ZONE \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --boot-disk-size=200GB \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=True" \
  --scopes=storage-full
```

**Estimated cost**: V100 spot = ~$0.74/hr → **~$0.81/hr total**

---

## 4. VM Startup & Data Prep Script

```bash
#!/bin/bash
# Run on the spot VM after SSH

# Clone repo
git clone https://github.com/your-org/idex.git /ssis
cd /ssis

# Install Python deps
pip3 install torch torchvision numpy scipy scikit-learn \
  pandas onnx onnxruntime google-cloud-storage

# Install project-specific deps
pip3 install -r ai/v3/requirements.txt
pip3 install -r ai/collision/requirements.txt
pip3 install -r ai/maneuver/requirements.txt

# Download training data from GCS
export GCS_BUCKET="ssis-training-XXXXX"  # your bucket
gsutil -m cp -r gs://$GCS_BUCKET/training-data/ ./training-data/

# Decompress datasets
cd training-data/B-Conjuction-Collision-dataset/
unzip train_data.zip -d train_data/
gunzip IVV_Releasable_Dataset_Spherical_DefaultHBR.csv.gz
gunzip IVV_Releasable_Dataset_SFSH_DiscreteHBR.csv.gz
cd /ssis

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## 5. Data Preprocessing Pipeline

Create the following script to preprocess your CSV/JSON datasets into training-ready formats:

```bash
# From /ssis on the VM:
python3 scripts/preprocess_training_data.py
```

### Key preprocessing steps per dataset:

**A/ TLEs → Maneuver & Propagation training data:**
- Parse TLE text files → extract orbital elements time series
- Compute residuals between consecutive TLEs → maneuver labels
- Generate sliding windows (seq_len=32) for LSTM input
- Output: `ai/maneuver/data/tle_residuals.jsonl`

**B/ CDMs → Collision GNN training data:**
- Parse ESA `train_data.zip` CSV → conjunction scenarios
- Extract miss distance, covariance, TCA, relative velocity
- Build graph structures (node features: 26-dim per object)
- Output: `ai/simulator/data/scenarios_graph_v2.jsonl`

**C/ GNSS metadata → NavIC features:**
- Parse JSON metadata files → extract spoofing/jamming labels
- Augment with synthetic dual-frequency residual histories
- Output: `ai/v3/data/navic_training.npz`

**D/ Re-entry CSV → Re-entry predictor features:**
- Parse `Reentry_History_Spreadsheet.csv` → extract orbital decay profiles
- Join with `Space-weather-data-from-1962.csv` for atmospheric conditions
- Output: `ai/v3/data/reentry_training.npz`

---

## 6. Execution Commands (Run on Spot VM)

### Phase 1 — Quick models (T4 spot)

```bash
export GCS_CHECKPOINT_BUCKET="ssis-training-XXXXX"

# 1. Maneuver Detector
python3 ai/maneuver/train_maneuver_detector.py

# 2. Re-entry Predictor
python3 -c "
import sys; sys.path.insert(0, 'ai/v3')
from reentry_predictor import train_reentry_predictor, export_reentry_onnx
model = train_reentry_predictor(n_samples=5000, epochs=50, device='cuda')
export_reentry_onnx(model, 'ai/training/models/reentry_predictor.onnx')
"

# 3. NavIC Anomaly Detector
python3 -c "
import sys; sys.path.insert(0, 'ai/v3')
from navic_anomaly_detector import train_navic_detector
import torch
model = train_navic_detector(n_epochs=20, n_samples=10000, device='cuda')
torch.save(model.state_dict(), 'ai/training/models/navic_detector.pt')
"

# 4. Transformer Forecaster
python3 -c "
import sys; sys.path.insert(0, 'ai/v3')
from transformer_forecaster import train_maneuver_transformer, export_transformer_onnx
model = train_maneuver_transformer(n_samples=10000, epochs=30, device='cuda')
export_transformer_onnx(model, 'ai/training/models/transformer_forecaster.onnx')
"

# 5. Sensor Tasking RL
python3 -c "
import sys; sys.path.insert(0, 'ai/v3')
from sensor_tasking_rl import *
stations = [
    SensorState('leh', SensorType.ATEENA_RADAR, 34.0, 77.5, 3500, 1.0, 0.0, 100.0, 0.3),
    SensorState('thumba', SensorType.ATEENA_RADAR, 8.5, 76.9, 10, 1.0, 0.0, 100.0, 0.5),
    SensorState('sriharikota', SensorType.OPTICAL_CCD, 13.7, 80.2, 10, 1.0, 0.0, 100.0, 0.2),
]
regions = [
    TargetRegion('Northern', 28.0, 38.0, 72.0, 80.0, 0.95, 2.0, 0.8),
    TargetRegion('Eastern', 8.0, 22.0, 80.0, 88.0, 0.90, 4.0, 0.6),
]
policy = train_sensor_tasking_ppo(stations, regions, total_timesteps=50000, device='cuda')
"
```

### Phase 2 — Collision GNN (T4/V100 spot, with checkpointing)

```bash
# Using the spot-aware trainer with auto-resume
python3 scripts/gcp_spot_vm_trainer.py \
  --component collision \
  --epochs 100 \
  --checkpoint-every 5 \
  --gcs-bucket $GCS_CHECKPOINT_BUCKET \
  --output-dir ai/training/models

# OR use the dedicated spot script
python3 ai/collision/train_collision_surrogate_spot.py \
  --epochs 100 \
  --checkpoint-every 5 \
  --auto-resume
```

### Full pipeline (all models)

```bash
python3 scripts/train_all_models.py --mode full --device cuda
```

---

## 7. Post-Training: Export & Deploy

```bash
# Copy ONNX models to production directory
mkdir -p pkg/ml/models
cp ai/training/models/*.onnx pkg/ml/models/
cp ai/training/models/*.pt pkg/ml/models/

# Upload to GCS for team access
gsutil -m cp -r ai/training/models/ gs://$GCS_BUCKET/trained-models/

# Download to local machine
gcloud compute scp ssis-train-t4:/ssis/ai/training/models/* ./ai/training/models/ --zone=$ZONE
```

---

## 8. Cost Estimate

| Phase | GPU | Hours | Spot $/hr | Total |
|---|---|---|---|---|
| Phase 1 (Quick) | T4 | 2 | $0.18 | **$0.36** |
| Phase 2 (Collision) | T4 | 6 | $0.18 | **$1.08** |
| Phase 2 (Collision) | V100 | 4 | $0.81 | **$3.24** |
| Phase 3 (Full IVV) | V100 | 12 | $0.81 | **$9.72** |
| **Total (Phases 1+2)** | | | | **~$1.50 - $4.50** |

**Tip**: Always stop the VM when not training: `gcloud compute instances stop ssis-train-t4 --zone=$ZONE`

---

## 9. Preemption Recovery

The existing `scripts/gcp_spot_vm_trainer.py` handles SIGTERM gracefully:

1. **SIGTERM received** → emergency checkpoint saved to GCS
2. **VM recreated** → `--auto-resume` loads latest checkpoint from GCS
3. **Training continues** from exact epoch

For Kubernetes-based training, use `k8s/gke-spot-training-job.yaml` (already configured with `backoffLimit: 3`).

---

## 10. Recommended Order of Execution

```
Day 1 (T4 spot, ~$0.50):
  ├── Preprocess all datasets (30 min)
  ├── Train Maneuver Detector with A/ TLEs (15 min)
  ├── Train NavIC Detector with C/ data (20 min)  
  ├── Train Re-entry Predictor with D/ + A/ data (10 min)
  ├── Train Transformer Forecaster with A/ TLEs (30 min)
  └── Train Sensor Tasking RL (30 min)

Day 2 (T4 or V100 spot, ~$2-5):
  ├── Train Collision GNN with B/train_data.zip (2-4 hrs)
  ├── Train Fusion GNN with A/ + B/ combined (1-2 hrs)
  └── Export all ONNX models to pkg/ml/models/

Day 3 (optional, V100 spot, ~$10):
  └── Retrain Collision GNN on full 22GB IVV dataset
```
