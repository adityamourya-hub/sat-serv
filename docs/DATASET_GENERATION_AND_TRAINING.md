# SSIS Dataset Generation & Model Training Guide

Physics-first, bottom-up training pipeline for the Sovereign Space Situational Awareness system. All training uses **synthetic + public open data only** on **GCP Spot GPU VMs** (L4/A100 recommended).

---

## GCS Bucket Layout

**Bucket**: `gs://ssis-training-1776624951`

```
gs://ssis-training-1776624951/
├── datasets/                          # Raw + preprocessed training data
│   ├── A-TLES-Ephemeris/              # CelesTrak TLEs, SATCAT, space weather
│   ├── B-Conjunction-Collision-dataset/# ESA CDMs, TraCSS ephemeris
│   ├── C-NAVIC-anamoly/               # FGI-JSDR I/Q, Jammertest 2024
│   └── D-Rentry-data/                 # Aerospace Corp reentry database
├── checkpoints/                       # Per-phase training checkpoints
│   ├── p0_atmosphere/
│   ├── p1_orbital/
│   ├── p2_hypersonic/
│   ├── collision/
│   ├── navic/
│   ├── maneuver/
│   └── reentry/
└── weights/                           # Final trained model weights (.onnx, .pt)
    ├── collision/
    ├── navic/
    ├── maneuver/
    └── reentry/
```

---

## Repository Structure (Training-Relevant)

```
idex/
├── ai/
│   ├── neural_ode/
│   │   ├── train.py                   # Yoshida-4 symplectic trainer (PyTorch)
│   │   ├── ghnn.py                    # Graph Hamiltonian Neural Network
│   │   ├── port_hamiltonian_pytorch.py# Port-Hamiltonian System (H, R, g)
│   │   └── hybrid_propagator.py       # Neural + Kepler hybrid propagator
│   ├── v3/
│   │   ├── train_phs_jax.py           # Canonical JAX training backend
│   │   ├── navic_anomaly_detector.py  # NavIC spoofing/jamming detector
│   │   ├── reentry_predictor.py       # Re-entry prediction model
│   │   ├── dphno.py                   # Deep Physics-informed HNO
│   │   ├── e3nn_equivariant_hnn.py    # SE(3)-equivariant Hamiltonian NN
│   │   ├── nrlmsise_wrapper.py        # NRLMSISE-00 atmosphere oracle
│   │   ├── hgv_phno.py               # Hypersonic glide vehicle PHNO
│   │   ├── conformal_uq.py           # Conformal uncertainty quantification
│   │   ├── onnx_export.py            # ONNX INT8 quantization + export
│   │   ├── sensor_tasking_rl.py       # RL sensor tasking policy
│   │   └── koopman_propagator.py      # Koopman operator propagation
│   ├── collision/
│   │   ├── train_collision_surrogate.py      # Collision GNN trainer
│   │   └── train_collision_surrogate_spot.py # Spot-optimized collision trainer
│   ├── maneuver/
│   │   └── train_maneuver_detector.py # Maneuver detection from TLE residuals
│   ├── radar/
│   │   └── train_ateena_iqnet.py      # ATeena radar IQ-Net trainer
│   ├── simulator/
│   │   └── generate_scenarios.py      # 64-dim graph-native scenario generator
│   └── analytics/
│       └── mge_models.py             # MGE template library for satellite ID
├── q/core/
│   ├── hno_pytorch.py                 # Fourier Neural Operator (atmosphere)
│   ├── jax_backend.py                 # JAX Forest-Ruth symplectic backend
│   └── bdg_hamiltonian.py             # BDG Hamiltonian dynamics
├── sda-train/
│   ├── train_init.py                  # Training orchestrator (ZMQ/Plasma)
│   ├── scripts/
│   │   ├── gen_catalog.py             # Synthetic catalog generation
│   │   ├── gen_conjunction.py         # Synthetic conjunction data
│   │   ├── gen_reentry.py             # Synthetic reentry trajectories
│   │   ├── gen_reentry_storms.py      # Reentry during geomagnetic storms
│   │   └── gen_volumetric.py          # Volumetric surveillance data
│   └── requirements.txt              # Training env requirements
├── scripts/
│   ├── train_all_sequential.sh        # Full sequential training (GCP Spot)
│   ├── gcp_spot_trainer_full.py       # Python Spot trainer w/ GCS integration
│   ├── train_neural_atmosphere.py     # P0 atmosphere FNO trainer
│   ├── download_public_datasets.py    # Download CelesTrak, ESA, FGI, etc.
│   ├── preprocess_training_data.py    # CSV/JSON → model-ready formats
│   ├── deploy_gcp_spot_vm.sh          # VM creation + startup
│   ├── gcp_spot_startup.sh            # Auto-startup script for Spot VMs
│   ├── validate_atmosphere.py         # P0 validation: <0.5% density error
│   ├── verify_symplectic_volume.py    # Symplectic integrator verification
│   └── validate_onnx_models.py        # ONNX model validation
└── training-data/                     # Local dataset mirror (gitignored)
    ├── A-TLES-Ephemeris-for-Collision-manauver-conjunction-cataloguing-data/
    ├── B-Conjuction-Collision-dataset/
    ├── C-NAVIC-anamoly-and-GNSS-spoofing-jamming-dataset/
    └── D-Rentry-data-dataset-optional/
```

---

## One-Time GCP Setup (~15 minutes)

> **Why venv?** The GCP deep learning VM images ship with a system Python that `gsutil`, `nvidia-smi`, and the CUDA toolkit depend on. Installing training packages directly into it can silently break these tools. A virtualenv isolates your training deps while keeping the system Python intact. It also makes the environment reproducible across VM restarts/preemptions.

SSH into your Spot VM:

```bash
# 1. System dependencies
sudo apt update && sudo apt install -y git python3-pip python3-venv hdf5-tools

# 2. Create isolated Python venv (REQUIRED — do NOT pip install into system Python)
python3 -m venv ~/ssis-env
source ~/ssis-env/bin/activate

# 3. ML + Physics stack (CUDA 12.1)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas h5py scipy scikit-learn tqdm tensorboard matplotlib
pip install torch-geometric pymsis poliastro google-cloud-storage onnx wandb

# 4. Clone repo
git clone https://github.com/adityamourya-hub/idex.git && cd idex
mkdir -p checkpoints weights logs

# 5. Upload local datasets to GCS (run once from local machine, outside the VM)
# gsutil -m cp -r ../training-data/* gs://ssis-training-1776624951/datasets/
```

**IMPORTANT — activate venv on every new SSH session**:
```bash
source ~/ssis-env/bin/activate && cd ~/idex
```

**Preemption Rule**: Every 5 epochs (or on SIGTERM), checkpoints auto-sync to GCS. On preemption, restart the VM and re-run — training resumes from the latest checkpoint.

---

## Phase P0: Neural Atmosphere Operator (Weeks 1–2)

**Goal**: Atmospheric density model with <0.5% error vs NRLMSISE-00.

### Dataset Generation

Generate 10M synthetic samples using NRLMSISE-00 as the oracle:

```bash
source ~/ssis-env/bin/activate && cd ~/idex

# Generate dataset only (saves to HDF5, no training)
python3 scripts/train_neural_atmosphere.py --generate-only \
  --num-samples 10000000 \
  --output-path ai/data/p0_atmosphere/train_data.h5
```

**Parameters sampled**: altitude (30–500 km), F10.7 (70–300), Ap (4–100), lat/lon, day-of-year.

**Validation data**: Download CHAMP/GOCE accelerometer densities from [TU Delft Thermosphere](http://thermosphere.tudelft.nl/) or NASA Earthdata.

### Training

```bash
source ~/ssis-env/bin/activate && cd ~/idex

# Train from pre-generated HDF5 dataset
python3 scripts/train_neural_atmosphere.py \
  --data-path ai/data/p0_atmosphere/train_data.h5 \
  --epochs 100 --batch-size 8192 --lr 1e-3 --device cuda

# Or generate + train in one shot (for smaller datasets)
python3 scripts/train_neural_atmosphere.py \
  --num-samples 100000 --epochs 100 --device cuda

# Upload weights to GCS
gsutil cp weights/neural_atmosphere_final.pth gs://ssis-training-1776624951/weights/p0_atmosphere/
gsutil cp weights/neural_atmosphere.onnx gs://ssis-training-1776624951/weights/p0_atmosphere/
```

**Training script**: `scripts/train_neural_atmosphere.py` — trains a Fourier Neural Operator (FNO) with physics loss:
```
L = ||log(ρ_nn) - log(ρ_msis)||² + λ_phys ||∂P/∂h + ρg||²
```

**FNO architecture**: `q/core/hno_pytorch.py`

**Quality gate**: Density error < 0.5% on CHAMP/GOCE validation set.
**Estimated time**: 4–7 days on L4 Spot (~$0.4–0.8/hr).

### Validation

```bash
source ~/ssis-env/bin/activate && cd ~/idex

python3 scripts/validate_atmosphere.py \
  --model weights/atmosphere_fno.pt \
  --validation-data ai/data/p0_atmosphere/validation/
```

---

## Phase P1: Orbital Foundation — SE(3)-GNN + GHNN (Weeks 3–4)

**Goal**: Physics-preserving orbital propagator. Energy drift < 0.0001% over long rollouts.

### Dataset Generation

**Step 1 — Download public TLEs**:

```bash
source ~/ssis-env/bin/activate && cd ~/idex

python3 scripts/download_public_datasets.py --source celestrak \
  --categories stations active starlink gps-ops

# Preprocess TLEs into training format
python3 scripts/preprocess_training_data.py --dataset A
```

This downloads from CelesTrak (no registration) and Space-Track.org (free registration).

**Step 2 — Generate synthetic orbital trajectories** (5M samples):

```bash
source ~/ssis-env/bin/activate && cd ~/idex

python3 ai/simulator/generate_scenarios.py \
  --n_scenarios 5000000 \
  --output ai/data/p1_orbital/scenarios_graph_v2.jsonl
```

The scenario generator produces 64-dim graph-native samples with:
- Position/velocity in canonical units (DU/TU, µ=1.0)
- SE(3)-equivariant features (spectral residuals, fractional drag, MGE threat score)
- Edge features for GATv2 attention (relative pos/vel/distance)
- 4-class orbit labels (LEO/MEO/GEO/HEO)

### Training

**Primary path** (PyTorch — Yoshida-4 symplectic):

```bash
source ~/ssis-env/bin/activate && cd ~/idex

python3 ai/neural_ode/train.py \
  --data ai/data/p1_orbital/scenarios_graph_v2.jsonl \
  --epochs 80 --output-dir models/ghnn \
  --lambda-sparse 0.01 --noise-floor 1e-5 \
  --validate-integrator
```

**Canonical path** (JAX — Forest-Ruth symplectic):

```bash
source ~/ssis-env/bin/activate && cd ~/idex

python3 ai/v3/train_phs_jax.py \
  --data ai/data/p1_orbital/ \
  --epochs 80
```

**Key models trained**:
| Model | Script | Architecture |
|-------|--------|--------------|
| Port-Hamiltonian System | `ai/neural_ode/train.py` | H-net + R-net + g-net (Yoshida-4) |
| SE(3)-Equivariant HNN | `ai/v3/e3nn_equivariant_hnn.py` | E(3)NN irreps for attitude-independent features |
| GHNN Vector Field | `ai/neural_ode/ghnn.py` | Graph Hamiltonian for multi-body interactions |

**Quality gate**: Run integrator validation:

```bash
source ~/ssis-env/bin/activate && cd ~/idex

python3 ai/neural_ode/train.py --data ai/data/p1_orbital/... --validate-integrator
# Target: avg energy drift < 1e-8, max drift < 1e-7 over 10k steps

python3 scripts/verify_symplectic_volume.py --model models/ghnn/axiom_hnode_v1.onnx
```

**Estimated time**: 5–8 days on L4/A100 Spot.

---

## Phase P2: Hypersonic + Neural SINDy (Weeks 5–6)

**Goal**: Discover unknown forces/maneuvers for ballistic/HGV threats. Position error < 50m after 60s prediction.

### Dataset Generation

```bash
source ~/ssis-env/bin/activate && cd ~/idex

# Generate 3M hypersonic trajectories (3-DoF/6-DoF with zero-day maneuvers)
python3 sda-train/scripts/gen_reentry_storms.py \
  --n_samples 3000000 \
  --include_zero_day_maneuvers \
  --output ai/data/p2_hypersonic/

# Also generate reentry scenarios with geomagnetic storm conditions
python3 sda-train/scripts/gen_reentry.py --n_samples 500000
```

### Training

```bash
source ~/ssis-env/bin/activate && cd ~/idex

# HGV Physics-informed HNO
python3 ai/v3/hgv_phno.py \
  --data ai/data/p2_hypersonic/ \
  --epochs 60 --device cuda

# Neural SINDy for maneuver discovery (Go-side integration)
# The SINDy discoverer in pkg/physics/sindy.go operates on residuals:
#   r(t) = ẍ_observed - ẍ_keplerian
# Training the neural SINDy latent encoder:
python3 ai/v3/eso_neural_sindy.py \
  --data ai/data/p2_hypersonic/ \
  --epochs 60 --device cuda
```

**SINDy sparsity** is enforced via `compute_sparse_loss` in `ai/neural_ode/train.py`:
```
L_sparse = λ · mean(max(0, ||z_t||₁ - ε_noise))
```
This prevents micro-thrust hallucination — the network cannot use tiny latent adjustments to fit sensor noise.

**Quality gate**: Position error < 50m after 60s prediction horizon.
**Estimated time**: 4–6 days.

---

## Phase P3: Task Heads on Frozen Foundation (Weeks 7–9)

**Goal**: Freeze P0–P2 foundation weights → train individual task heads.

### Option A — Full Sequential (Recommended for Spot VMs)

Uses the existing all-in-one script with GCS checkpointing:

```bash
# Upload datasets to GCS first (if not already done — run from local machine)
# gsutil -m cp -r ../training-data/* gs://ssis-training-1776624951/datasets/

# On Spot VM — activate venv, then run sequential training
source ~/ssis-env/bin/activate && cd ~/idex
bash scripts/train_all_sequential.sh
```

The script handles:
- NVIDIA driver installation
- Python venv setup
- Dataset download from `gs://ssis-training-1776624951/datasets/`
- Skip guards (if weights already exist in GCS)
- Checkpoint upload every 5 epochs
- Disk cleanup between phases

### Option B — Per-Component with Python Trainer

```bash
source ~/ssis-env/bin/activate && cd ~/idex

python3 scripts/gcp_spot_trainer_full.py \
  --gcs-bucket ssis-training-1776624951 \
  --component all \
  --epochs 100 \
  --checkpoint-every 5
```

Or train individual components:

```bash
source ~/ssis-env/bin/activate && cd ~/idex

# NavIC Anomaly Detection (~10 min)
python3 scripts/gcp_spot_trainer_full.py \
  --gcs-bucket ssis-training-1776624951 --component navic --epochs 50

# Maneuver Detection (~15 min)
python3 scripts/gcp_spot_trainer_full.py \
  --gcs-bucket ssis-training-1776624951 --component maneuver --epochs 12

# Re-entry Prediction (~10 min)
python3 scripts/gcp_spot_trainer_full.py \
  --gcs-bucket ssis-training-1776624951 --component reentry --epochs 100

# Collision GNN (~3-4 hours, largest workload)
python3 scripts/gcp_spot_trainer_full.py \
  --gcs-bucket ssis-training-1776624951 --component collision --epochs 25
```

### Task Head Details

| Head | Training Script | Dataset | Input | Output |
|------|----------------|---------|-------|--------|
| **NavIC Anomaly** | `ai/v3/navic_anomaly_detector.py` | C-NAVIC (FGI-JSDR, Jammertest) | I/Q samples | Spoof/Jam/Normal |
| **Maneuver Detect** | `ai/maneuver/train_maneuver_detector.py` | A-TLES (TLE residuals) | Δa, Δe, Δi sequences | Maneuver probability |
| **Re-entry Predict** | `ai/v3/reentry_predictor.py` | D-Reentry (Aerospace Corp) | Orbital elements + atmos | Time/location of reentry |
| **Collision GNN** | `ai/collision/train_collision_surrogate.py` | B-Collision (ESA CDMs) | Graph(nodes, edges, iono) | Risk + maneuver class |
| **Radar ATeena** | `ai/radar/train_ateena_iqnet.py` | Synthetic radar I/Q | Raw I/Q tensor | Target classification |

### Data Preprocessing

Before training task heads, preprocess raw datasets into model-ready formats:

```bash
source ~/ssis-env/bin/activate && cd ~/idex

# Preprocess all datasets
python3 scripts/preprocess_training_data.py --dataset all

# Or individually:
python3 scripts/preprocess_training_data.py --dataset A  # TLEs → maneuver residuals
python3 scripts/preprocess_training_data.py --dataset B  # CDMs → graph scenarios
python3 scripts/preprocess_training_data.py --dataset C  # I/Q → NavIC features
python3 scripts/preprocess_training_data.py --dataset D  # Reentry → sequences
```

**Estimated time**: 2–3 days per head (parallel on larger instances).

---

## Phase P4: RL Policies (Weeks 10–12)

**Goal**: Sensor tasking, Maya deception detection, wolfpack coordination.

```bash
source ~/ssis-env/bin/activate && cd ~/idex

# Sensor Tasking RL
python3 ai/v3/sensor_tasking_rl.py \
  --env ssis_sensor_tasking --epochs 200000 --device cuda

# MBPO Agent (model-based policy optimization)
python3 ai/v3/mbpo_hardened.py \
  --epochs 100000 --device cuda

# Maya Deception Detector
python3 ai/v3/maya_detector.py \
  --data ai/data/p4_deception/ --epochs 50 --device cuda
```

**Estimated time**: 5–7 days.

---

## Phase P5: Uncertainty Quantification (Week 13)

**Goal**: Conformal prediction + evidential outputs for every model.

```bash
source ~/ssis-env/bin/activate && cd ~/idex

python3 ai/v3/conformal_uq.py \
  --load-checkpoint models/ghnn/axiom_hnode_v1.onnx \
  --calibration-data ai/data/p1_orbital/validation/ \
  --epochs 30
```

Produces calibrated uncertainty intervals (D1–D10 defensive features) for the Sovereign Handshake VSC certificate pipeline.

**Estimated time**: 2–3 days.

---

## Phase P6: ONNX Export & Deployment (Week 14)

**Goal**: INT8 quantization → ONNX → Go inference runtime.

```bash
source ~/ssis-env/bin/activate && cd ~/idex

# Export all models to ONNX with INT8 quantization
python3 ai/v3/onnx_export.py \
  --model-path models/ghnn/axiom_hnode_v1.pt \
  --output pkg/ml/models/axiom_hnode_v1.onnx \
  --quantize int8

# Validate ONNX models match PyTorch outputs
python3 scripts/validate_onnx_models.py --model-dir pkg/ml/models/

# Copy to Go inference path
cp pkg/ml/models/*.onnx pkg/onnx/models/
```

The Go ONNX runtime wrapper (`pkg/onnx/session.go`) provides <2ms inference for the 100Hz reflex loop.

---

## Spot VM Quick Reference

### Create VM

```bash
# L4 GPU (cost-effective, sufficient for P0-P3)
gcloud compute instances create ssis-train-$(date +%m%d) \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=300GB --boot-disk-type=pd-ssd \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --max-run-duration=8h --scopes=cloud-platform
```

### Resume After Preemption

```bash
# Restart stopped VM
gcloud compute instances start ssis-train-MMDD --zone=us-central1-a

# SSH in
gcloud compute ssh ssis-train-MMDD --zone=us-central1-a

# Re-activate venv (persists across reboots, just needs re-sourcing)
source ~/ssis-env/bin/activate && cd ~/idex

# Re-run — auto-resumes from GCS checkpoint, skips completed phases
bash scripts/train_all_sequential.sh
```

### Check Training Status

```bash
# List completed weights
gsutil ls -lh gs://ssis-training-1776624951/weights/

# List checkpoints
gsutil ls gs://ssis-training-1776624951/checkpoints/

# Check completion markers
gsutil ls gs://ssis-training-1776624951/checkpoints/.*.complete
```

### Cost Estimates

| Phase | GPU | Duration | Approx Cost |
|-------|-----|----------|-------------|
| P0 Atmosphere FNO | L4 | 4–7 days | $40–70 |
| P1 Orbital Foundation | L4/A100 | 5–8 days | $50–160 |
| P2 Hypersonic SINDy | L4 | 4–6 days | $40–60 |
| P3 Task Heads (all) | L4 | 4–6 hours | $2–5 |
| P4 RL Policies | L4 | 5–7 days | $50–70 |
| P5 UQ Calibration | L4 | 2–3 days | $20–30 |
| P6 ONNX Export | CPU | < 1 hour | ~$0 |
| **Total** | | **~7–9 weeks** | **$200–400** |

---

## Dataset Sources (All Public / Synthetic)

| Dataset | Source | Registration | Size | Used By |
|---------|--------|-------------|------|---------|
| CelesTrak TLEs | [celestrak.org](https://www.celestrak.org/NORAD/elements/) | None | ~5 MB | P1, P3 Maneuver |
| Space-Track Bulk TLEs | [space-track.org](https://www.space-track.org/) | Free account | ~2 GB | P1, P3 Maneuver |
| ESA Collision CDMs | [Zenodo 4463683](https://zenodo.org/records/4463683) | None | 221 MB | P3 Collision |
| TraCSS Conjunction | [space.commerce.gov](https://space.commerce.gov/dataset-for-conjunction-assessment-verification/) | Google Form | 21 GB | P3 Collision |
| FGI-JSDR GNSS | [fairdata.fi](https://etsin.fairdata.fi/dataset/367379a8-7d78-4b08-91f0-8027ce7a621b) | None | ~1 GB | P3 NavIC |
| Jammertest 2024 | [Zenodo 15911359](https://zenodo.org/records/15911359) | None | ~500 MB | P3 NavIC |
| Aerospace Reentry DB | [aerospace.org/reentries](https://aerospace.org/reentries) | None | ~10 MB | P3 Reentry |
| CHAMP/GOCE Densities | [thermosphere.tudelft.nl](http://thermosphere.tudelft.nl/) | None | ~200 MB | P0 Validation |
| NRLMSISE-00 (pymsis) | Synthetic via `pymsis` | None | Generated | P0 Training |
| Orbital trajectories | Synthetic via `poliastro` | None | Generated | P1, P2 Training |

---

## Quality Gates Summary

| Phase | Metric | Target | Validation Script |
|-------|--------|--------|-------------------|
| P0 | Density error vs NRLMSISE-00 | < 0.5% | `scripts/validate_atmosphere.py` |
| P1 | Energy conservation drift | < 0.0001% | `ai/neural_ode/train.py --validate-integrator` |
| P1 | Symplectic volume preservation | M^T J M = J | `scripts/verify_symplectic_volume.py` |
| P2 | Position error @ 60s horizon | < 50 m | HGV propagation benchmark |
| P3 | Per-head accuracy | See task-specific | Component test scripts |
| P5 | Conformal coverage | 95% CI calibrated | `ai/v3/conformal_uq.py` |
| P6 | ONNX vs PyTorch parity | < 1e-5 | `scripts/validate_onnx_models.py` |
