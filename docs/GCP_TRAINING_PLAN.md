# GCP Spot L4 Training Plan — 3 Deal-Breaker Models
## ADITI 4.0 IAF Challenge | Budget: $120–150 from $300 credits

---

## Codebase Confirmation

I've audited your entire codebase. Here's exactly what exists for each model:

| Model | Architecture | Training Script | Data Source | ONNX Export | Go Consumer |
|---|---|---|---|---|---|
| **ATeena-IQNet** | 5-stage pipeline: ComplexConv1d → NeuralBeamformer → LearnedDoppler → SE-ResNet34 → EvidentialHead | `ai/radar/train_ateena_iqnet.py` + `ai/v3/sensor_tasking_rl.py` | `ai/radar/simulator/ateena_sim.py` (streaming synthetic IQ) | Built-in ONNX export in train script | `pkg/engine/onnx_pool.go` (ATeena-Sensing session) |
| **Rapid Ranker** | XGBoost regressor (7 features → threat score) | `ai/rapid/train_ranker_xgboost.py` | Fully synthetic (inline `synthesize()`) | `skl2onnx` → `rapid_ranker.onnx` | `pkg/ml/inference.go:PredictRapidScore()` expects 4 outputs: score, confidence, low, high |
| **Axiom-HNODE** | Port-Hamiltonian System (4×128 Tanh MLP + Cholesky dissipation) | `ai/neural_ode/train.py` + `ai/neural_ode/port_hamiltonian_pytorch.py` | `ai/v3/training/synthetic_generator.py` (JAX trajectory gen) OR public TLE→JSONL | `ai/neural_ode/ghnn.py:export_ghnn_to_onnx()` | `pkg/physics/onnx_runtime.go:InferHnodeCore()` expects `hnode_core.onnx` |

**Already trained & validated:** Collision GNN, NavIC detector, Maneuver LSTM, Atmosphere FNO, Conformal UQ.

---

## Cost Estimate

| Phase | GPU Hours | Cost @ $0.70/hr (L4 spot) | Notes |
|---|---|---|---|
| VM setup + env | 0.5 hr | $0.35 | One-time |
| ATeena-IQNet (RL sensor tasking) | 24 hr | $16.80 | 50k timesteps PPO |
| ATeena-IQNet (radar pipeline) | 36 hr | $25.20 | 100 epochs, streaming IQ |
| Rapid Ranker (XGBoost) | 0.5 hr | $0.35 | CPU-only, XGBoost is fast |
| Rapid Ranker (neural upgrade) | 8 hr | $5.60 | If needed for 4-output ONNX |
| Axiom-HNODE (data gen) | 4 hr | $2.80 | 100k synthetic trajectories |
| Axiom-HNODE (training) | 48 hr | $33.60 | 100 epochs PHS |
| Buffer (preemptions, retries) | 16 hr | $11.20 | ~2 preemptions expected |
| **Total** | **~137 hr** | **~$96** | Well under $150 cap |

---

## Phase 1: Spin Up Spot VM + Install Environment (Day 1, ~1 hour)

### 1.1 Create GCS Bucket for Checkpoints

```bash
# Run locally (your Mac)
export PROJECT_ID=$(gcloud config get-value project)
export BUCKET="ssis-training-1776624951"
export ZONE="us-central1-a"   # cheapest L4 zone

gcloud storage buckets create gs://${BUCKET} \
    --location=us-central1 \
    --uniform-bucket-level-access

echo "Bucket: ${BUCKET}" > /tmp/ssis_bucket.txt
```

### 1.2 Create Spot VM with L4 GPU

```bash
gcloud compute instances create ssis-trainer \
    --zone=${ZONE} \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --metadata="install-nvidia-driver=True" \
    --scopes=storage-rw \
    --labels=project=ssis,purpose=training

# Cost: ~$0.70/hr (L4 spot in us-central1)
```

### 1.3 SSH and Install Dependencies

```bash
gcloud compute ssh ssis-trainer --zone=${ZONE}

# On the VM:
sudo nvidia-smi   # Verify L4 GPU

# Clone repo
git clone https://github.com/adityamourya-hub/idex.git /idex
cd /idex

#Virtual env -

source ~/ssis-env/bin/activate && cd ~/idex

# Install Python deps
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install onnx onnxruntime-gpu xgboost scikit-learn skl2onnx lightgbm
pip3 install gymnasium numpy h5py google-cloud-storage
pip3 install jax[cuda12] jaxlib   # For trajectory generator only

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, Device: NVIDIA L4
```
#(ssis-env) root@ssis-all-0423-1753:~/idex# python3 -c "import torch; print(f'CUDA: {torch.cuda.#is_available()}, Device: {torch.cuda.get_device_name(0)}')"
#CUDA: True, Device: Tesla V100-SXM2-16GB
#(ssis-env) root@ssis-all-0423-1753:~/idex#

### 1.4 Upload Existing Weights (from your Mac)

```bash
# Run locally — push existing trained weights to VM
gcloud compute scp --recurse \
    pkg/ml/models/ ssis-trainer:/idex/pkg/ml/models/ \
    --zone=${ZONE}

gcloud compute scp --recurse \
    weights/ ssis-trainer:/idex/weights/ \
    --zone=${ZONE}
```

---

## Phase 2: Train ATeena-IQNet (Days 1–3) — Highest Demo Value

ATeena has TWO components: the radar signal processing pipeline AND the RL sensor tasking policy. We train both.

### 2A: Sensor Tasking RL (PPO) — 12–24 hours

This is the **volumetric scan + blind spot** demo piece. Uses the existing `ai/v3/sensor_tasking_rl.py`.

#### 2A.1 Training Command

```bash
cd /idex

# Create the training script wrapper for spot-safe execution
cat > /tmp/train_sensor_tasking.py << 'SCRIPT'
import sys, os, signal, time, json
sys.path.insert(0, "ai/v3")
os.makedirs("weights/sensor_tasking", exist_ok=True)

# Install gymnasium if not present
import subprocess
try:
    import gymnasium
except ImportError:
    print("Installing gymnasium...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium"])

import torch
import numpy as np
from sensor_tasking_rl import (
    SensorState, SensorType, TargetRegion,
    SensorTaskingEnv, SensorTaskingPolicy,
    train_sensor_tasking_ppo, export_tasking_policy_onnx
)

# Extended IAF station network (8 stations for realistic coverage)
stations = [
    SensorState("leh", SensorType.ATEENA_RADAR, 34.0, 77.5, 3500, 1.0, 0.0, 100.0, 0.3),
    SensorState("thumba", SensorType.ATEENA_RADAR, 8.5, 76.9, 10, 1.0, 0.0, 100.0, 0.5),
    SensorState("sriharikota", SensorType.OPTICAL_CCD, 13.7, 80.2, 10, 1.0, 0.0, 100.0, 0.2),
    SensorState("port_blair", SensorType.RISAT_SAR, 11.6, 92.7, 10, 1.0, 0.0, 100.0, 0.4),
    SensorState("mount_abu", SensorType.ATEENA_RADAR, 24.6, 72.7, 1220, 1.0, 0.0, 100.0, 0.3),
    SensorState("itanagar", SensorType.OPTICAL_CCD, 27.1, 93.6, 350, 1.0, 0.0, 100.0, 0.4),
    SensorState("kochi", SensorType.NAVIC_GNSS, 9.9, 76.3, 10, 1.0, 0.0, 100.0, 0.2),
    SensorState("jodhpur", SensorType.ATEENA_RADAR, 26.3, 73.0, 231, 1.0, 0.0, 100.0, 0.3),
]

# Strategic priority regions
regions = [
    TargetRegion("Northern_Border", 28.0, 38.0, 72.0, 80.0, 0.95, 2.0, 0.8),
    TargetRegion("Eastern_Seaboard", 8.0, 22.0, 80.0, 88.0, 0.90, 4.0, 0.6),
    TargetRegion("Southern_Ocean", -10.0, 8.0, 75.0, 95.0, 0.70, 6.0, 0.3),
    TargetRegion("Western_Desert", 23.0, 35.0, 68.0, 72.0, 0.85, 3.0, 0.5),
    TargetRegion("Siliguri_Corridor", 25.5, 27.2, 87.8, 89.5, 0.99, 1.0, 0.9),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# Curriculum: phase 1 small → phase 2 full
for phase, timesteps in [(1, 10000), (2, 50000)]:
    print(f"\n{'='*60}")
    print(f"Phase {phase}: {timesteps} timesteps")
    print(f"{'='*60}")
    
    policy = train_sensor_tasking_ppo(
        stations, regions,
        total_timesteps=timesteps,
        device=device,
        checkpoint_path="weights/sensor_tasking/checkpoint.pt",
        checkpoint_interval=1000,
    )
    
    # Checkpoint
    env = SensorTaskingEnv(stations, regions)
    state_dim = env.observation_space.shape[0]
    
    torch.save(policy.state_dict(), f"weights/sensor_tasking/policy_phase{phase}.pt")
    export_tasking_policy_onnx(
        policy, state_dim,
        f"weights/sensor_tasking/sensor_tasking_phase{phase}.onnx"
    )

# Final export to production path
import shutil
shutil.copy(
    "weights/sensor_tasking/sensor_tasking_phase2.onnx",
    "pkg/ml/models/sensor_tasking.onnx"
)
print("\n✅ Sensor Tasking RL training complete!")
print(f"Production ONNX: pkg/ml/models/sensor_tasking.onnx")
SCRIPT

mkdir -p weights/sensor_tasking
python3 /tmp/train_sensor_tasking.py 2>&1 | tee weights/sensor_tasking/train.log
```

#### 2A.2 Checkpoint to GCS

```bash
gsutil -m cp -r weights/sensor_tasking/ gs://ssis-training-1776624951/weights/sensor_tasking/
gsutil cp pkg/ml/models/sensor_tasking.onnx gs://ssis-training-1776624951/weights/production/
```

### 2B: ATeena Radar Pipeline (SE-ResNet + Evidential) — 24–36 hours

This is the full radar signal processing pipeline. Uses streaming synthetic IQ data from `ateena_sim.py`.

#### 2B.1 Training Command

```bash
cd /idex

cat > /tmp/train_ateena_radar.py << 'SCRIPT'
import sys, os, signal, time
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)
sys.path.insert(0, "ai/radar")
sys.path.insert(0, "ai/radar/simulator")
os.makedirs("weights/ateena_radar", exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from ateena_iqnet import ATeenaIQNet
from radar_augmentation import RadarAugmentation
from ateena_sim import ATeenaSimulator, generate_random_scenario
from train_ateena_iqnet import islr_loss, evidential_radar_loss

class StreamingRadarDataset(Dataset):
    def __init__(self, augmentation=None):
        self.sim = ATeenaSimulator({'fc_hz':3e9,'wavelength_m':0.1,'bandwidth_hz':100e6,'pulse_width_us':100,'pri_ms':1.0,'n_pulses':64,'n_range_bins':256,'n_elements':16,'element_spacing_m':0.05,'transmit_power_w':100e3,'transmit_gain_db':35,'receive_gain_db':35,'noise_figure_db':3.0,'system_temp_k':290})
        self.augmentation = augmentation
    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        targets, noise_fig = generate_random_scenario(min_targets=1, max_targets=3)
        iq_np = self.sim.simulate_cpi(targets, noise_figure_db=noise_fig)
        iq = torch.from_numpy(np.stack([iq_np.real, iq_np.imag], axis=-1)).float() * 1e4
        labels = torch.zeros(50, 6)
        for i, t in enumerate(targets[:50]):
            labels[i] = torch.tensor([t['range_km'], t['velocity_kms'], t.get('azimuth_deg',0), t.get('elevation_deg',45), t.get('rcs_m2',1), 1.0])
        if self.augmentation:
            iq = self.augmentation(iq)
        return iq, labels

device = torch.device("cuda")
print(f"Device: {device}", flush=True)

class Config:
    n_elements = 16
    n_pulses = 64
    n_doppler_bins = 32

model = ATeenaIQNet(Config()).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# scaler = GradScaler()  # AMP disabled due to NaN instability
aug = None  # Augmentation disabled to isolate NaN source

# Checkpoint path and best_loss defined before SIGTERM handler
ckpt_path = "weights/ateena_radar/latest_checkpoint.pt"
best_loss = float('inf')
_cur_epoch, _cur_step, _cur_loss = 0, 0, float('inf')

# SIGTERM handler for spot preemption — saves immediately
preempted = False
def on_sigterm(sig, frame):
    global preempted
    preempted = True
    print("\n⚠️ SIGTERM received! Saving emergency checkpoint...", flush=True)
    torch.save({
        'epoch': _cur_epoch, 'step': _cur_step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': _cur_loss, 'best_loss': best_loss,
    }, ckpt_path)
    os.system(f"gsutil -q cp {ckpt_path} gs://ssis-training-1776624951/checkpoints/ateena_radar/emergency.pt")
    print("✓ Emergency checkpoint saved.", flush=True)
signal.signal(signal.SIGTERM, on_sigterm)

# num_workers>0 requires augmentation to not use CUDA (fork issues)
# Pass augmentation=None to dataset; augmentation applied after .to(device) in loop
dataset = StreamingRadarDataset(augmentation=aug)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2,
                    pin_memory=True, prefetch_factor=2, persistent_workers=True)

# Resume from checkpoint if exists
start_epoch = 0
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    # scaler_state not used when AMP disabled
    start_epoch = ckpt['epoch'] + 1
    best_loss = ckpt.get('best_loss', float('inf'))
    print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}", flush=True)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

if __name__ == '__main__':
    for epoch in range(start_epoch, 100):
        if preempted:
            break
        _cur_epoch = epoch
        
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()
        
        for i, (iq_batch, labels) in enumerate(loader):
            if preempted:
                break
            _cur_step = i
            
            iq_batch = iq_batch.to(device, non_blocking=True)  # [B, P, R, E, 2]
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            detections, alphas, features = model(iq_batch)
            
            # NaN detection
            if torch.isnan(detections).any() or torch.isnan(alphas).any():
                print(f"  ⚠️ NaN in model outputs! Skipping step.", flush=True)
                continue
            
            l_reg = F.smooth_l1_loss(detections, labels)
            l_evd = evidential_radar_loss(alphas, (labels[:, :, 5] > 0.5).float())
            
            if torch.isnan(l_reg) or torch.isnan(l_evd):
                print(f"  ⚠️ NaN in loss! l_reg={l_reg}, l_evd={l_evd}", flush=True)
                continue
            
            loss = l_reg + l_evd
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # More aggressive clipping
            optimizer.step()
            
            loss_val = loss.item()
            _cur_loss = loss_val
            total_loss += loss_val
            n_batches += 1
            
            if i % 10 == 0:
                elapsed = time.time() - t0
                steps_per_sec = (i + 1) / max(elapsed, 1e-6)
                print(f"  E{epoch+1} step {i}/250 loss={loss_val:.4f} ({steps_per_sec:.1f} steps/s)", flush=True)
            
            # Step-level checkpoint every 50 steps
            if i > 0 and i % 50 == 0:
                torch.save({
                    'epoch': epoch, 'step': i,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    # 'scaler_state': scaler.state_dict(),
                    'loss': loss_val, 'best_loss': best_loss,
                }, ckpt_path)
                print(f"  💾 Step ckpt @ epoch {epoch+1} step {i}", flush=True)
            
            if i >= 250:
                break
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}/100 | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | {epoch_time:.0f}s", flush=True)
        
        # Epoch checkpoint every epoch (overwrite latest)
        ckpt_data = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
        }
        torch.save(ckpt_data, ckpt_path)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "weights/ateena_radar/best_model.pt")
            print(f"  ⭐ New best: {best_loss:.4f}", flush=True)
        
        # Sync to GCS every 5 epochs
        if (epoch + 1) % 5 == 0:
            os.system(f"gsutil -q cp {ckpt_path} gs://ssis-training-1776624951/checkpoints/ateena_radar/")
            print(f"  ☁️  GCS sync done (best={best_loss:.4f})", flush=True)
    
    # Final ONNX export
    model.eval()
    dummy_iq = torch.randn(1, 64, 256, 16, 2).to(device)
    onnx_path = "pkg/ml/models/ateena_iqnet.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model, dummy_iq, onnx_path,
        input_names=["iq_samples"],
        output_names=["detections", "alphas", "features"],
        dynamic_axes={"iq_samples": {0: "batch"}, "detections": {0: "batch"}},
        opset_version=17,
    )
    print(f"\n✅ ATeena-IQNet exported: {onnx_path}", flush=True)
SCRIPT

mkdir -p weights/ateena_radar
python3 /tmp/train_ateena_radar.py 2>&1 | tee weights/ateena_radar/train.log
```

---

## Phase 3: Train Rapid Ranker + Axiom-HNODE (Days 3–5)

### 3A: Rapid Ranker — XGBoost + Neural Wrapper (~2 hours)

The Go engine (`pkg/ml/inference.go:PredictRapidScore`) expects **4 ONNX outputs**: `score, confidence, uncertainty_low, uncertainty_high`. The existing XGBoost script only produces 1 output. We need a neural wrapper.

#### 3A.1 Train XGBoost Base + Neural Evidential Wrapper

```bash
cd /idex

cat > /tmp/train_rapid_ranker.py << 'SCRIPT'
import sys, os, json
sys.path.insert(0, "ai/rapid")
os.makedirs("weights/rapid_ranker", exist_ok=True)
os.makedirs("pkg/ml/models", exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ── Step 1: Train XGBoost base (fast, CPU-only) ──
print("="*60)
print("Step 1: XGBoost base ranker")
print("="*60)

def synthesize(n=50000, seed=42):
    rng = np.random.default_rng(seed)
    mach = rng.uniform(0.5, 12.0, n)
    altitude = rng.uniform(5.0, 250.0, n)
    burnout = rng.uniform(0.0, 6.0, n)
    flight_angle = rng.uniform(0.05, 1.1, n)
    distance = rng.uniform(50.0, 2500.0, n)
    rcs = rng.uniform(0.05, 6.0, n)
    intent = rng.uniform(0.0, 1.0, n)  # Maneuver intent score
    launch_lat = rng.uniform(20.0, 45.0, n)
    launch_lon = rng.uniform(60.0, 100.0, n)

    class_boost = np.where((mach >= 5) & (altitude >= 20) & (altitude <= 100), 100,
                  np.where((burnout >= 1) & (altitude < 300), 80,
                  np.where(altitude >= 120, 40, 20)))
    dist_boost = np.where(distance <= 500, 50, np.where(distance <= 1500, 25, 0))
    mach_boost = np.where(mach >= 8, 25, np.where(mach >= 5, 15, 0))
    
    # Dynamic score matching Go consumer features
    dynamic = class_boost + dist_boost + mach_boost + \
              0.9 * mach + 0.02 * (2500 - distance) + 3.0 * burnout + \
              15.0 * intent + rng.normal(0, 4.5, n)
    dynamic = np.clip(dynamic, 0, 300)
    
    # 9 features matching PredictRapidScore in inference.go
    x = np.stack([mach, altitude, burnout, flight_angle,
                  launch_lat, launch_lon, distance, rcs, intent], axis=1).astype(np.float32)
    y = dynamic.astype(np.float32)
    return x, y

x, y = synthesize()
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model_xgb = XGBRegressor(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
    tree_method='hist',  # GPU-friendly
)
model_xgb.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
pred = model_xgb.predict(x_val)
mae = float(mean_absolute_error(y_val, pred))
print(f"XGBoost MAE: {mae:.4f}")

# ── Step 2: Neural evidential wrapper (4 outputs for Go) ──
print("\n" + "="*60)
print("Step 2: Neural evidential wrapper")
print("="*60)

class RapidRankerNet(nn.Module):
    """Neural ranker with 4 outputs matching Go inference expectations."""
    def __init__(self, input_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.score_head = nn.Linear(64, 1)        # threat score
        self.conf_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())  # confidence
        self.low_head = nn.Linear(64, 1)           # uncertainty_low
        self.high_head = nn.Linear(64, 1)          # uncertainty_high
    
    def forward(self, x):
        h = self.net(x)
        score = self.score_head(h)
        conf = self.conf_head(h)
        low = self.low_head(h)
        high = self.high_head(h)
        return score, conf, low, high

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = RapidRankerNet().to(device)
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

# Use XGBoost predictions as soft labels (knowledge distillation)
xgb_scores_train = model_xgb.predict(x_train)
x_t = torch.tensor(x_train, device=device)
y_t = torch.tensor(xgb_scores_train, device=device).unsqueeze(1)

for epoch in range(50):
    net.train()
    perm = torch.randperm(len(x_t))
    total_loss = 0.0
    for i in range(0, len(x_t), 256):
        idx = perm[i:i+256]
        xb, yb = x_t[idx], y_t[idx]
        
        score, conf, low, high = net(xb)
        
        # Score regression
        loss_score = nn.functional.mse_loss(score, yb)
        # Confidence should be high when error is low
        error = (score - yb).abs().detach()
        target_conf = torch.exp(-error / 50.0)
        loss_conf = nn.functional.mse_loss(conf, target_conf)
        # Uncertainty: low < score < high
        loss_interval = nn.functional.relu(low - score).mean() + nn.functional.relu(score - high).mean()
        # Width penalty
        loss_width = (high - low).abs().mean() * 0.01
        
        loss = loss_score + 0.5 * loss_conf + loss_interval + loss_width
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 | Loss: {total_loss/(len(x_t)//256):.4f}")

# ── Step 3: Export ONNX ──
print("\n" + "="*60)
print("Step 3: ONNX Export")
print("="*60)

net.eval()
dummy = torch.randn(1, 9).to(device)
onnx_path = "pkg/ml/models/rapid_ranker.onnx"
torch.onnx.export(
    net, dummy, onnx_path,
    input_names=["rapid_features"],
    output_names=["dynamic_score", "confidence", "uncertainty_low", "uncertainty_high"],
    dynamic_axes={"rapid_features": {0: "batch"}},
    opset_version=17,
)

# Validate
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path)
out = sess.run(None, {"rapid_features": np.random.randn(1, 9).astype(np.float32)})
print(f"ONNX outputs: {[o.shape for o in out]}")
print(f"Expected: [(1,1), (1,1), (1,1), (1,1)]")

torch.save(net.state_dict(), "weights/rapid_ranker/rapid_ranker_net.pt")

metrics = {
    "xgboost_mae": mae,
    "neural_epochs": 50,
    "input_features": 9,
    "output_schema": ["dynamic_score[B,1]", "confidence[B,1]", "uncertainty_low[B,1]", "uncertainty_high[B,1]"],
    "onnx_path": onnx_path,
}
with open("weights/rapid_ranker/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Rapid Ranker exported: {onnx_path}")
print(f"   XGBoost MAE: {mae:.4f}")
print(f"   4-output ONNX matching Go PredictRapidScore()")
SCRIPT

python3 /tmp/train_rapid_ranker.py 2>&1 | tee weights/rapid_ranker/train.log
gsutil -m cp -r weights/rapid_ranker/ gs://ssis-training-1776624951/weights/rapid_ranker/
gsutil cp pkg/ml/models/rapid_ranker.onnx gs://ssis-training-1776624951/weights/production/
```

### 3B: Axiom-HNODE (Port-Hamiltonian System) — 2–3 days

#### 3B.1 Generate Synthetic Trajectory Data

```bash
cd /idex

cat > /tmp/generate_hnode_data.py << 'SCRIPT'
"""Generate synthetic Kepler+drag trajectories for HNODE training."""
import numpy as np
import json, os
from pathlib import Path

os.makedirs("ai/neural_ode/data", exist_ok=True)

# Physical constants (canonical units)
RE_KM = 6378.137
MU_KM = 398600.4418
TU_SEC = np.sqrt(RE_KM**3 / MU_KM)  # ~806.81s

def kepler_rhs(state):
    """Two-body + J2 dynamics in canonical units."""
    q = state[:3]
    p = state[3:]
    r = np.linalg.norm(q)
    
    # Two-body gravity (canonical: mu=1)
    a_2body = -q / (r**3)
    
    # J2 perturbation (canonical units)
    J2 = 1.0826e-3
    z2 = q[2]**2 / r**2
    factor = 1.5 * J2 / (r**5)
    a_j2 = np.array([
        q[0] * factor * (5 * z2 - 1),
        q[1] * factor * (5 * z2 - 1),
        q[2] * factor * (5 * z2 - 3),
    ])
    
    return np.concatenate([p, a_2body + a_j2])

def rk4_step(state, dt):
    k1 = kepler_rhs(state)
    k2 = kepler_rhs(state + 0.5 * dt * k1)
    k3 = kepler_rhs(state + 0.5 * dt * k2)
    k4 = kepler_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def generate_trajectory(rng, alt_km_range=(300, 1200), n_steps=200, dt_tu=0.01):
    """Generate one trajectory in canonical units."""
    alt_km = rng.uniform(*alt_km_range)
    r_du = (RE_KM + alt_km) / RE_KM  # Canonical distance
    
    # Circular orbit + small eccentricity perturbation
    ecc = rng.uniform(0, 0.05)
    inc = rng.uniform(0, np.pi)
    omega = rng.uniform(0, 2*np.pi)
    
    # Initial state (simplified circular)
    v_circ = 1.0 / np.sqrt(r_du)  # Canonical circular velocity
    
    # Random orientation
    q0 = np.array([r_du * np.cos(omega), r_du * np.sin(omega) * np.cos(inc), 
                    r_du * np.sin(omega) * np.sin(inc)])
    # Perpendicular velocity
    v_dir = np.cross(q0, [0, 0, 1])
    if np.linalg.norm(v_dir) < 1e-10:
        v_dir = np.cross(q0, [0, 1, 0])
    v_dir = v_dir / np.linalg.norm(v_dir)
    p0 = v_circ * (1 + ecc * 0.5) * v_dir
    
    state = np.concatenate([q0, p0])
    states = [state.copy()]
    times = [0.0]
    
    for i in range(n_steps):
        state = rk4_step(state, dt_tu)
        states.append(state.copy())
        times.append((i + 1) * dt_tu)
    
    return np.array(states, dtype=np.float32), np.array(times, dtype=np.float32)

# Generate dataset
rng = np.random.default_rng(42)
n_trajectories = 100000
output_path = Path("ai/neural_ode/data/hnode_training.jsonl")

print(f"Generating {n_trajectories} trajectories...")

with open(output_path, 'w') as f:
    for i in range(n_trajectories):
        states, times = generate_trajectory(rng)
        record = {
            "sat_id": i,
            "states": states.tolist(),
            "time_points": times.tolist(),
        }
        f.write(json.dumps(record) + "\n")
        
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1}/{n_trajectories}")

print(f"✅ Dataset: {output_path} ({output_path.stat().st_size / 1024**2:.1f} MB)")
SCRIPT

python3 /tmp/generate_hnode_data.py 2>&1 | tee ai/neural_ode/data/generation.log
```

#### 3B.2 Train Axiom-HNODE

```bash
cd /idex

cat > /tmp/train_hnode.py << 'SCRIPT'
"""
Train Axiom-HNODE (Port-Hamiltonian System) with spot preemption safety.
"""
import sys, os, signal, time
sys.path.insert(0, "ai/neural_ode")
sys.path.insert(0, "ai/v3")
os.makedirs("weights/axiom_hnode", exist_ok=True)

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader

from port_hamiltonian_pytorch import PortHamiltonianSystem, create_port_hamiltonian_system
from train import CanonicalTrajectoryDataset, GHNNTrainer, collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create model (conservative-only for core propagator)
model = create_port_hamiltonian_system(
    state_dim=6,
    control_dim=3,
    latent_dim=64,
    use_latent_conditioning=False,  # Conservative-only (GHNN equivalent)
)

# Dataset
data_path = "ai/neural_ode/data/hnode_training.jsonl"
if not os.path.exists(data_path):
    print(f"ERROR: {data_path} not found. Run generate_hnode_data.py first.")
    sys.exit(1)

dataset = CanonicalTrajectoryDataset(data_path)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)

# Trainer with SINDy sparsity
trainer = GHNNTrainer(model, device, lambda_sparse=0.01, noise_floor=1e-5)

# Spot preemption handler
preempted = False
def on_sigterm(sig, frame):
    global preempted
    preempted = True
    print("\n⚠️ SIGTERM! Emergency save...")
signal.signal(signal.SIGTERM, on_sigterm)

# Resume from checkpoint
start_epoch = 0
ckpt_path = "weights/axiom_hnode/latest_checkpoint.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    trainer.optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")

best_loss = float('inf')
print(f"\nTraining from epoch {start_epoch} to 100")
print(f"Dataset: {len(dataset)} trajectories")

for epoch in range(start_epoch, 100):
    if preempted:
        break
    
    loss = trainer.train_epoch(loader)
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    # Checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': trainer.optimizer.state_dict(),
            'best_loss': best_loss,
        }, ckpt_path)
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), "weights/axiom_hnode/best_model.pt")
        
        # Sync to GCS
        os.system(f"gsutil -q cp {ckpt_path} gs://${{BUCKET}}/checkpoints/axiom_hnode/")
        print(f"  💾 Checkpoint (best={best_loss:.6f})")

# Save final
torch.save(model.state_dict(), "weights/axiom_hnode/final_model.pt")

# ONNX export
model.eval()
from ghnn import export_ghnn_to_onnx
onnx_path = "pkg/ml/models/hnode_core.onnx"
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

dummy_input = torch.randn(1, 6).to(device)
torch.onnx.export(
    model, dummy_input, onnx_path,
    input_names=['state'],
    output_names=['dynamics'],
    dynamic_axes={'state': {0: 'batch'}, 'dynamics': {0: 'batch'}},
    opset_version=17,
    do_constant_folding=True,
)
print(f"\n✅ Axiom-HNODE exported: {onnx_path}")
print(f"   Best loss: {best_loss:.6f}")

# Also copy to physics runtime path
import shutil
shutil.copy(onnx_path, "weights/production/hnode_core.onnx")
SCRIPT

python3 /tmp/train_hnode.py 2>&1 | tee weights/axiom_hnode/train.log
gsutil -m cp -r weights/axiom_hnode/ gs://${BUCKET}/weights/axiom_hnode/
gsutil cp pkg/ml/models/hnode_core.onnx gs://${BUCKET}/weights/production/
```

---

## Phase 4: Export ONNX + Update model_registry.yaml + Rebuild (Day 5–6)

### 4.1 Download Weights to Local Machine

```bash
# Run locally (your Mac)
export BUCKET=$(cat /tmp/ssis_bucket.txt)

gsutil cp gs://${BUCKET}/weights/production/sensor_tasking.onnx pkg/ml/models/
gsutil cp gs://${BUCKET}/weights/production/rapid_ranker.onnx pkg/ml/models/
gsutil cp gs://${BUCKET}/weights/production/hnode_core.onnx pkg/ml/models/

# Verify files
ls -la pkg/ml/models/*.onnx
```

### 4.2 Update model_registry.yaml

Add these entries to `configs/model_registry.yaml`:

```yaml
  # ATeena Sensor Tasking — TRAINED (PPO, 50k timesteps, 8 stations × 5 regions)
  - name: sensor_tasking
    version: "1.0.0"
    onnx_path: pkg/ml/models/sensor_tasking.onnx
    opset: 17
    input_schema:
      - name: sensor_state
        shape: [1, -1]
        dtype: float32
    output_schema:
      - name: station_probs
        shape: [1, 8]
        dtype: float32
      - name: continuous_action
        shape: [1, 3]
        dtype: float32
      - name: value
        shape: [1, 1]
        dtype: float32
    quantization: fp32
    tier: 2
    description: PPO-trained sensor tasking policy for volumetric scan + blind spot detection
    go_wrapper: SensorTaskingModel

  # Rapid Ranker — TRAINED (XGBoost distillation + neural evidential, 9 features)
  - name: rapid_ranker
    version: "4.0.0"
    onnx_path: pkg/ml/models/rapid_ranker.onnx
    opset: 17
    input_schema:
      - name: rapid_features
        shape: [1, 9]
        dtype: float32
    output_schema:
      - name: dynamic_score
        shape: [1, 1]
        dtype: float32
      - name: confidence
        shape: [1, 1]
        dtype: float32
      - name: uncertainty_low
        shape: [1, 1]
        dtype: float32
      - name: uncertainty_high
        shape: [1, 1]
        dtype: float32
    quantization: fp32
    tier: 1
    description: Neural threat ranker with XGBoost knowledge distillation (9 features, 4 outputs)
    go_wrapper: RapidRankerModel

  # Axiom-HNODE — TRAINED (Port-Hamiltonian System, 100k trajectories, 100 epochs)
  - name: axiom_hnode
    version: "2.0.0"
    onnx_path: pkg/ml/models/hnode_core.onnx
    opset: 17
    input_schema:
      - name: state
        shape: [1, 6]
        dtype: float32
    output_schema:
      - name: dynamics
        shape: [1, 6]
        dtype: float32
    quantization: fp32
    tier: 2
    description: Port-Hamiltonian Neural ODE for symplectic orbital propagation with energy conservation
    go_wrapper: AxiomHNODEModel
```

### 4.3 Rebuild Go Engine

```bash
# Locally
cd /path/to/idex
CGO_ENABLED=1 go build -tags onnx -o build/ssis_engine ./engine
```

---

## Phase 5: Validation + Evidence PDF (Day 6–7)

### 5.1 Run Full Production Verification

```bash
cd /path/to/idex

# Update tests/production_verification.py to include 3 new models
python3 tests/production_verification.py

# Regenerate VALIDATION_EVIDENCE.md with live results
python3 scripts/generate_validation_report.py
```

### 5.2 Run Full Demo

```bash
# Start engine
SSIS_SIGNING_KEY=$(openssl rand -hex 32) ./build/ssis_engine &
sleep 3

# Run demo script
bash scripts/demo.sh

# Stop engine
kill %1
```

### 5.3 Validate Quality Gates

| Model | Quality Gate | Pass Criteria |
|---|---|---|
| ATeena-IQNet (RL) | >80% region coverage in 24h sim | `coverage > 0.80` in evaluation rollout |
| ATeena-IQNet (Radar) | ONNX load + inference shape correct | `detections[B,50,6], alphas[B,50]` |
| Rapid Ranker | 4-output ONNX, MAE < 8.0 | `score[1,1], conf[1,1], low[1,1], high[1,1]` |
| Axiom-HNODE | Energy drift < 1e-6 over 24h | `validate_yoshida4_integrator()` |
| Axiom-HNODE | ONNX load + state→dynamics shape | `[1,6] → [1,6]` |

---

## Cost Control & Preemption Recovery Script

```bash
# Save as scripts/spot_recovery.sh on the VM
#!/bin/bash
set -e

BUCKET="${1:?Usage: spot_recovery.sh <gcs-bucket>}"

echo "🔄 Spot VM Recovery Script"
echo "Bucket: gs://${BUCKET}"

# Check which components have checkpoints
for comp in sensor_tasking ateena_radar rapid_ranker axiom_hnode; do
    count=$(gsutil ls gs://${BUCKET}/checkpoints/${comp}/ 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "  ✅ ${comp}: ${count} checkpoints found"
        # Download latest
        mkdir -p /tmp/checkpoints/${comp}
        gsutil cp $(gsutil ls -l gs://${BUCKET}/checkpoints/${comp}/ | sort -k2 | tail -1 | awk '{print $3}') /tmp/checkpoints/${comp}/
    else
        echo "  ⚠️ ${comp}: no checkpoints"
    fi
done

# Download existing production weights
mkdir -p pkg/ml/models
gsutil -m cp gs://${BUCKET}/weights/production/*.onnx pkg/ml/models/ 2>/dev/null || true

echo ""
echo "✅ Recovery complete. Resume training with the same commands."
```

### Auto-Shutdown Budget Guard

```bash
# Run this in a tmux session on the VM
cat > /tmp/budget_guard.sh << 'EOF'
#!/bin/bash
# Kill VM if it runs more than 200 hours total
MAX_HOURS=200
START=$(date +%s)
while true; do
    ELAPSED=$(( ($(date +%s) - $START) / 3600 ))
    if [ $ELAPSED -ge $MAX_HOURS ]; then
        echo "⚠️ Budget guard: ${MAX_HOURS}h limit reached. Shutting down."
        sudo shutdown -h now
    fi
    sleep 300  # Check every 5 minutes
done
EOF
chmod +x /tmp/budget_guard.sh
nohup /tmp/budget_guard.sh &
```

---

## Execution Timeline

| Day | Activity | Est. Cost |
|---|---|---|
| **Day 1** | VM setup + env install + start ATeena RL training | $5 |
| **Day 2** | ATeena RL completes → start ATeena Radar training | $20 |
| **Day 3** | ATeena Radar continues + train Rapid Ranker (2h) | $20 |
| **Day 4** | ATeena Radar completes → start Axiom-HNODE data gen + training | $25 |
| **Day 5** | Axiom-HNODE training continues | $25 |
| **Day 6** | HNODE completes → download weights, update registry, rebuild | $5 |
| **Day 7** | Full validation + demo run + evidence generation | $0 (local) |
| **Total** | | **~$100** |

---

## One-Click Demo After Training

After all weights are trained and placed in `pkg/ml/models/`, the existing `scripts/demo.sh` will exercise all 9 capabilities. The only addition needed is to verify the 3 new ONNX files load:

```bash
# Quick verification
python3 -c "
import onnxruntime as ort
for name in ['sensor_tasking', 'rapid_ranker', 'hnode_core']:
    sess = ort.InferenceSession(f'pkg/ml/models/{name}.onnx')
    print(f'✅ {name}: inputs={[i.name for i in sess.get_inputs()]}, outputs={[o.name for o in sess.get_outputs()]}')
"
```

---

*Ready to execute Phase 1. Confirm and I'll generate the exact gcloud commands for your project.*
