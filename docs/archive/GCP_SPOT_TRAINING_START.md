# GCP Spot Training - Quick Start Guide

## Prerequisites

- GCloud CLI installed and authenticated: `gcloud auth login`
- GCS bucket with datasets (you confirmed this is done)
- GCP project with billing enabled

---

## Step 1: Verify GCS Dataset Structure

```bash
# Set your bucket name
export GCS_BUCKET="your-ssis-training-bucket"

# Verify datasets exist
gsutil ls -r gs://$GCS_BUCKET/datasets/

# Expected structure:
# gs://$GCS_BUCKET/datasets/
#   ├── A-TLES-Ephemeris/
#   ├── B-Conjunction-Collision/
#   ├── C-NAVIC-anomaly/
#   └── D-Rentry-data/
```

---

## Step 2: Create Spot VM with GPU

```bash
# Configuration
export ZONE="us-central1-a"
export VM_NAME="ssis-train-$(date +%Y%m%d-%H%M)"
export GCS_BUCKET="your-ssis-training-bucket"

# Create spot instance with T4 GPU (cost-effective)
gcloud compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --scopes=cloud-platform \
    --metadata="startup-script=#!/bin/bash
    # Install NVIDIA drivers
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
    nvidia-smi
    echo 'GPU ready' > /var/log/gpu-ready.log"

echo "VM created: $VM_NAME"
```

---

## Step 3: SSH and Setup Environment

```bash
# SSH into VM (wait 2-3 minutes for startup)
gcloud compute ssh $VM_NAME --zone=$ZONE

# Inside VM - verify GPU
nvidia-smi

# Clone repo (or use your existing)
git clone https://github.com/your-org/idex.git
cd idex

# Verify GCS access
gsutil ls gs://$GCS_BUCKET/datasets/

# Create checkpoint/weights directories
mkdir -p checkpoints weights logs
```

---

## Step 4: Start Training with Checkpointing

Save this as `start_training.sh` on the VM:

```bash
#!/bin/bash
set -e

export GCS_BUCKET="your-ssis-training-bucket"
export CHECKPOINT_DIR="./checkpoints"
export WEIGHTS_DIR="./weights"

# Check for existing checkpoint to resume
LATEST_CHECKPOINT=$(gsutil ls gs://$GCS_BUCKET/checkpoints/ 2>/dev/null | sort | tail -1 || echo "")

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
    gsutil cp -r "$LATEST_CHECKPOINT"* ./checkpoints/
    RESUME_FLAG="--resume ./checkpoints/$(basename $LATEST_CHECKPOINT)"
else
    echo "Starting fresh training"
    RESUME_FLAG=""
fi

# Install Python deps
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q pandas numpy scikit-learn matplotlib tqdm tensorboard gcsfs

# Start training with checkpoint callback
echo "Starting training at $(date)"
python scripts/gcp_spot_trainer_full.py \
    --gcs-bucket $GCS_BUCKET \
    --checkpoint-dir $CHECKPOINT_DIR \
    --weights-dir $WEIGHTS_DIR \
    --epochs 100 \
    $RESUME_FLAG \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo "Training completed at $(date)"
```

Run it:
```bash
chmod +x start_training.sh
./start_training.sh
```

---

## Step 5: Handle Spot Preemption (Auto)

The `gcp_spot_trainer_full.py` already includes:
- SIGTERM handler for graceful shutdown
- Checkpoint save every N epochs
- GCS sync after each epoch
- Resume capability on restart

If VM is preempted, just restart it and re-run `start_training.sh` - it will auto-resume.

---

## Step 6: Monitor Training

```bash
# In another terminal, SSH to VM
gcloud compute ssh $VM_NAME --zone=$ZONE -- -L 6006:localhost:6006

# View TensorBoard
tensorboard --logdir=./logs --port=6006

# Monitor GPU usage (another terminal)
watch -n 1 nvidia-smi

# Check GCS for checkpoints
gsutil ls -lh gs://$GCS_BUCKET/checkpoints/
```

---

## Step 7: Download Results

```bash
# After training completes, download weights locally
gsutil -m cp -r gs://$GCS_BUCKET/weights/* ./trained-weights/

# Or sync entire bucket
gsutil -m rsync -r gs://$GCS_BUCKET/ ./gcs-backup/
```

---

## Cost Optimization

| Workload | GPU | Cost/hr | Est. Time | Total |
|----------|-----|---------|-----------|-------|
| NavIC Anomaly | T4 | $0.11 | 10 min | ~$0.02 |
| Maneuver Detect | T4 | $0.11 | 15 min | ~$0.03 |
| Re-entry | T4 | $0.11 | 10 min | ~$0.02 |
| Collision GNN | V100 | $0.74 | 2-4 hrs | ~$2-3 |
| **All (quick)** | T4 | $0.11 | 1-2 hrs | ~$0.20 |

**Use V100 only for Collision Surrogate (large GNN). All others work fine on T4.**

---

## Quick Reference Commands

```bash
# Stop VM (preserve disk for resume)
gcloud compute instances stop $VM_NAME --zone=$ZONE

# Delete VM (weights/checkpoints in GCS)
gcloud compute instances delete $VM_NAME --zone=$ZONE

# View VM logs
gcloud compute instances get-serial-port-output $VM_NAME --zone=$ZONE

# List all spot instances
gcloud compute instances list --filter="scheduling.provisioningModel=SPOT"
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "GPU not found" | Wait 2-3 min after VM creation for driver install |
| "OOM" | Reduce batch size: `--batch-size 16` (default 32) |
| GCS access denied | Run `gcloud auth application-default login` |
| Preemption during training | VM stops; restart with same command, auto-resumes |
| Slow training | Check GPU util: `nvidia-smi dmon` should show >80% |

---

## Next Steps

1. **Replace** `your-ssis-training-bucket` with your actual bucket name
2. **Run** Step 1 to verify datasets
3. **Run** Step 2 to create VM
4. **SSH** and run `start_training.sh`
5. **Monitor** via TensorBoard

Want me to customize the training script for a specific component (Collision, Maneuver, NavIC, or Re-entry)?
