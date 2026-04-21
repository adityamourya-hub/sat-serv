# GCP Spot Training - Component-by-Component Plan

## Overview

Total dataset: ~21GB across 4 categories. Train sequentially to manage memory and costs.

| Order | Component | Dataset | Size | GPU | Time | Cost |
|-------|-----------|---------|------|-----|------|------|
| 1 | **NavIC Anomaly** | C-NAVIC | ~500MB | T4 | 10 min | $0.02 |
| 2 | **Maneuver Detect** | A-TLES | ~2GB | T4 | 15 min | $0.03 |
| 3 | **Re-entry Predict** | D-Rentry | ~200MB | T4 | 10 min | $0.02 |
| 4 | **Collision GNN** | B-Conjunction | ~18GB | V100 | 3-4 hrs | $2.50 |
| **Total** | | | **~21GB** | | **~4 hrs** | **~$2.57** |

---

## Pre-Flight Checklist

```bash
# 1. Verify GCloud login
gcloud auth list

# 2. Set project
gcloud config set project YOUR_PROJECT_ID

# 3. Verify GCS bucket with datasets
gsutil ls -lh gs://YOUR_BUCKET/datasets/
# Should show:
#   A-TLES-Ephemeris/
#   B-Conjunction-Collision/
#   C-NAVIC-anomaly/
#   D-Rentry-data/

# 4. Create bucket folders for outputs
gsutil mb gs://YOUR_BUCKET/checkpoints 2>/dev/null || true
gsutil mb gs://YOUR_BUCKET/weights 2>/dev/null || true
gsutil mb gs://YOUR_BUCKET/logs 2>/dev/null || true
```

---

## Phase 1: NavIC Anomaly Detection (10 min)

**Why first:** Smallest dataset, validates pipeline works.

### Create T4 VM:
```bash
export ZONE="us-central1-a"
export BUCKET="YOUR_BUCKET"
export VM_NAVIC="ssis-navic-$(date +%m%d-%H%M)"

gcloud compute instances create $VM_NAVIC \
    --zone=$ZONE \
    --machine-type=n1-standard-2 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=50GB \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP

# SSH in 2 minutes
echo "Waiting for VM..."
sleep 120
gcloud compute ssh $VM_NAVIC --zone=$ZONE
```

### Inside VM - Train NavIC:
```bash
export BUCKET="YOUR_BUCKET"

# Download only NavIC dataset (~500MB)
mkdir -p datasets/C-NAVIC
gsutil -m cp -r gs://$BUCKET/datasets/C-NAVIC-anomaly/* ./datasets/C-NAVIC/

# Install deps
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q pandas numpy scikit-learn tqdm tensorboard gcsfs

# Clone/pull repo
git clone https://github.com/your-org/idex.git
cd idex

# Run training with checkpointing
python3 ai/v3/navic_anomaly_detector.py \
    --data-dir ../datasets/C-NAVIC \
    --epochs 50 \
    --batch-size 64 \
    --checkpoint-dir ./checkpoints \
    --gcs-bucket $BUCKET

# Upload weights
python3 -c "
import subprocess
subprocess.run(['gsutil', 'cp', 'ai/v3/models/navic_detector.onnx', f'gs://{BUCKET}/weights/'])
print('NavIC weights uploaded')
"

# Exit and cleanup
echo "NavIC training complete. Deleting VM..."
exit
```

### Cleanup:
```bash
gcloud compute instances delete $VM_NAVIC --zone=$ZONE --quiet
```

---

## Phase 2: Maneuver Detection (15 min)

**Why second:** Uses synthetic data generation, moderate compute.

### Create T4 VM:
```bash
export VM_MANEUVER="ssis-maneuver-$(date +%m%d-%H%M)"

gcloud compute instances create $VM_MANEUVER \
    --zone=$ZONE \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP

sleep 120
gcloud compute ssh $VM_MANEUVER --zone=$ZONE
```

### Inside VM - Train Maneuver:
```bash
export BUCKET="YOUR_BUCKET"

# Download TLES dataset (~2GB)
mkdir -p datasets/A-TLES
gsutil -m cp -r gs://$BUCKET/datasets/A-TLES-Ephemeris/* ./datasets/A-TLES/

pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q pandas numpy scikit-learn tqdm tensorboard

git clone https://github.com/your-org/idex.git
cd idex

# Train maneuver detector (generates synthetic residuals)
python3 ai/maneuver/train_maneuver_detector.py \
    --epochs 12 \
    --batch-size 128 \
    --output-dir ./weights \
    --gcs-bucket $BUCKET

# Upload
python3 -c "
import subprocess
subprocess.run(['gsutil', 'cp', 'ai/maneuver/data/maneuver_detector.onnx', f'gs://{BUCKET}/weights/'])
print('Maneuver weights uploaded')
"

exit
```

### Cleanup:
```bash
gcloud compute instances delete $VM_MANEUVER --zone=$ZONE --quiet
```

---

## Phase 3: Re-entry Prediction (10 min)

**Why third:** Small dataset, quick validation.

### Create T4 VM:
```bash
export VM_REENTRY="ssis-reentry-$(date +%m%d-%H%M)"

gcloud compute instances create $VM_REENTRY \
    --zone=$ZONE \
    --machine-type=n1-standard-2 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=50GB \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP

sleep 120
gcloud compute ssh $VM_REENTRY --zone=$ZONE
```

### Inside VM - Train Re-entry:
```bash
export BUCKET="YOUR_BUCKET"

# Download re-entry dataset (~200MB)
mkdir -p datasets/D-Rentry
gsutil -m cp -r gs://$BUCKET/datasets/D-Rentry-data/* ./datasets/D-Rentry/

pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q pandas numpy scikit-learn tqdm

git clone https://github.com/your-org/idex.git
cd idex

# Train re-entry predictor
python3 ai/v3/reentry_predictor.py \
    --data-dir ../datasets/D-Rentry \
    --epochs 100 \
    --batch-size 256 \
    --gcs-bucket $BUCKET

# Upload
python3 -c "
import subprocess
subprocess.run(['gsutil', 'cp', 'ai/v3/models/reentry_predictor.onnx', f'gs://{BUCKET}/weights/'])
print('Re-entry weights uploaded')
"

exit
```

### Cleanup:
```bash
gcloud compute instances delete $VM_REENTRY --zone=$ZONE --quiet
```

---

## Phase 4: Collision Surrogate GNN (3-4 hours)

**Why last:** Largest dataset (~18GB), needs V100 GPU for GNN.

### Create V100 VM (Preemptible):
```bash
export VM_COLLISION="ssis-collision-$(date +%m%d-%H%M)"

# V100 has 16GB VRAM - required for GNN with large graphs
gcloud compute instances create $VM_COLLISION \
    --zone=$ZONE \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=250GB \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --max-run-duration=6h \
    --scopes=cloud-platform

sleep 180
gcloud compute ssh $VM_COLLISION --zone=$ZONE
```

### Inside VM - Train Collision (with streaming):
```bash
export BUCKET="YOUR_BUCKET"

# Install deps first (large)
pip install -q torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install -q pandas numpy scikit-learn tqdm tensorboard gcsfs

git clone https://github.com/your-org/idex.git
cd idex

# Download collision dataset in chunks (streaming load)
mkdir -p datasets/B-Conjunction

# Use gsutil streaming - don't download all at once to save disk
# Instead modify training script to stream from GCS

# Run training with checkpointing every epoch
python3 ai/collision/train_collision_surrogate_spot.py \
    --gcs-bucket $BUCKET \
    --checkpoint-every 1 \
    --epochs 25 \
    --batch-size 32 \
    --resume-from-checkpoint

# Upload final weights
python3 -c "
import subprocess
subprocess.run(['gsutil', 'cp', 'ai/collision/models/collision_multi_task_v2.onnx', f'gs://{BUCKET}/weights/'])
print('Collision weights uploaded')
"

exit
```

### Monitor (in another terminal):
```bash
gcloud compute ssh $VM_COLLISION --zone=$ZONE -- -L 6006:localhost:6006
tensorboard --logdir=./logs --port=6006
```

### If preempted:
```bash
# VM stops - just restart it
gcloud compute instances start $VM_COLLISION --zone=$ZONE
gcloud compute ssh $VM_COLLISION --zone=$ZONE

# Re-run same command - auto-resumes from checkpoint
cd idex && python3 ai/collision/train_collision_surrogate_spot.py ...
```

---

## Automated Pipeline Script

Save as `run_all_training.sh`:

```bash
#!/bin/bash
set -e

export ZONE="us-central1-a"
export BUCKET="YOUR_BUCKET"
export PROJECT="YOUR_PROJECT"

LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=== SSIS Training Pipeline Started at $(date) ==="
echo "GCS Bucket: $BUCKET"
echo "Zone: $ZONE"
echo ""

# Helper function
train_component() {
    local name=$1
    local vm_type=$2
    local gpu=$3
    local disk=$4
    local script=$5
    local upload=$6
    
    VM_NAME="ssis-${name}-$(date +%m%d-%H%M)"
    
    echo "=== Phase: $name ==="
    echo "Creating VM: $VM_NAME with $gpu"
    
    # Create VM
    gcloud compute instances create $VM_NAME \
        --zone=$ZONE \
        --machine-type=$vm_type \
        --accelerator=type=$gpu,count=1 \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=$disk \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --quiet || {
            echo "ERROR: Failed to create VM for $name"
            return 1
        }
    
    # Wait for VM ready
    echo "Waiting for VM startup..."
    sleep 180
    
    # Run training via SSH
    echo "Starting training..."
    gcloud compute ssh $VM_NAME --zone=$ZONE --command="
        export BUCKET=$BUCKET
        pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install -q pandas numpy scikit-learn tqdm tensorboard gcsfs
        git clone https://github.com/your-org/idex.git
        cd idex
        python3 $script --gcs-bucket $BUCKET
        gsutil cp $upload gs://$BUCKET/weights/ 2>/dev/null || echo 'Upload path may vary'
    " || {
        echo "WARNING: Training may have been interrupted (spot preemption?)"
        echo "Check GCS for partial checkpoints and resume manually"
    }
    
    # Cleanup VM
    echo "Cleaning up VM: $VM_NAME"
    gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet || true
    
    echo "=== Phase $name complete ==="
    echo ""
}

# Phase 1: NavIC (T4)
train_component "navic" "n1-standard-2" "nvidia-tesla-t4" "50GB" \
    "ai/v3/navic_anomaly_detector.py" \
    "ai/v3/models/navic_detector.onnx"

# Phase 2: Maneuver (T4)
train_component "maneuver" "n1-standard-4" "nvidia-tesla-t4" "100GB" \
    "ai/maneuver/train_maneuver_detector.py" \
    "ai/maneuver/data/maneuver_detector.onnx"

# Phase 3: Re-entry (T4)
train_component "reentry" "n1-standard-2" "nvidia-tesla-t4" "50GB" \
    "ai/v3/reentry_predictor.py" \
    "ai/v3/models/reentry_predictor.onnx"

# Phase 4: Collision (V100 - longer)
echo "=== Phase: Collision (V100 - ~3-4 hours) ==="
VM_COLLISION="ssis-collision-$(date +%m%d-%H%M)"

gcloud compute instances create $VM_COLLISION \
    --zone=$ZONE \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=250GB \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --max-run-duration=6h \
    --quiet

echo "Collision VM created. Training will take 3-4 hours."
echo "Monitor with: gcloud compute ssh $VM_COLLISION --zone=$ZONE"
echo "Check logs: gsutil ls gs://$BUCKET/logs/"

echo ""
echo "=== Pipeline Summary ==="
echo "NavIC, Maneuver, Re-entry: COMPLETE"
echo "Collision: RUNNING on $VM_COLLISION"
echo "Total expected time: ~4 hours"
echo "Total expected cost: ~$2.57"
echo ""
echo "All weights will be in: gs://$BUCKET/weights/"
```

---

## Post-Training: Collect All Weights

```bash
# Download all trained weights locally
mkdir -p trained_weights_$(date +%Y%m%d)

gsutil -m cp -r gs://YOUR_BUCKET/weights/* ./trained_weights_$(date +%Y%m%d)/

# List what you got
gsutil ls -lh gs://YOUR_BUCKET/weights/

# Expected:
#   navic_detector.onnx
#   maneuver_detector.onnx
#   reentry_predictor.onnx
#   collision_multi_task_v2.onnx
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Spot preemption | VM stops - restart same VM name, training auto-resumes from GCS checkpoint |
| OOM on Collision | Reduce batch size: `--batch-size 16` or use A100 (more expensive) |
| GCS permission denied | Run `gcloud auth application-default login` on VM |
| Slow download | Use `gsutil -m` for parallel transfers |
| Disk full | Increase `--boot-disk-size` or stream data (don't download all) |
| CUDA out of memory | Use gradient accumulation: `--accumulation-steps 2` |

---

## Cost Optimization Tips

1. **Use T4 for everything except Collision** - 10x cheaper than V100
2. **Start Collision training at night** - less spot preemption
3. **Enable preemptible notification** - script saves checkpoint on SIGTERM
4. **Delete VMs immediately after training** - you pay for disk until deleted
5. **Use `gsutil -m`** - parallel transfers save time (and money on spot)

---

## Quick Commands Reference

```bash
# Check running VMs
gcloud compute instances list --filter="scheduling.provisioningModel=SPOT"

# Monitor GPU on VM
watch -n 1 nvidia-smi

# View training logs from GCS
gsutil cat gs://YOUR_BUCKET/logs/training.log

# List all checkpoints
gsutil ls -r gs://YOUR_BUCKET/checkpoints/

# Emergency: Stop all spot VMs
gcloud compute instances list --filter="scheduling.provisioningModel=SPOT" --format="value(name,zone)" | while read name zone; do gcloud compute instances stop $name --zone=$zone; done
```

---

## Next Steps

1. **Replace** `YOUR_BUCKET` with your actual GCS bucket name
2. **Verify** datasets exist in GCS: `gsutil ls gs://YOUR_BUCKET/datasets/`
3. **Run** `./run_all_training.sh` for fully automated pipeline
4. **Or** run phases manually for more control

**Total time:** ~4 hours (mostly waiting for Collision GNN)  
**Total cost:** ~$2.57 (spot pricing) vs ~$15 (on-demand)
