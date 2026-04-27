# GCP Spot Training - Single VM Sequential Plan

## Strategy: One VM, All Components

To avoid spot capacity issues, use **ONE** spot VM and train all 4 components sequentially.

### VM Configuration

**GPU Choice:** T4 for everything **except** Collision GNN needs V100.

**Option A:** Use V100 for entire pipeline (more expensive but works for all)
**Option B:** Use T4 for first 3, then upgrade to V100 (stop → change → restart)
**Option C:** Use T4 only, stream Collision data in small batches (slower but cheapest)

**Recommended: Option A** - Single V100 VM, train all 4 components sequentially.

---

## Cost Comparison

| Option | GPU | Time | Cost | Notes |
|--------|-----|------|------|-------|
| A (Recommended) | V100 | 4 hrs | ~$3.00 | One VM, no interruptions |
| B (Hybrid) | T4→V100 | 4 hrs | ~$2.60 | Stop/upgrade once |
| C (T4 only) | T4 | 6-8 hrs | ~$0.88 | Slower, smaller batches |

---

## Option A: Single V100 VM (Recommended)

### Step 1: Create One V100 Spot VM

```bash
export ZONE="us-central1-a"
export BUCKET="YOUR_BUCKET"
export VM_NAME="ssis-all-$(date +%m%d-%H%M)"

# Create single V100 VM - sufficient for all components
gcloud compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=300GB \
    --boot-disk-type=pd-ssd \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --max-run-duration=8h \
    --scopes=cloud-platform

echo "VM created: $VM_NAME"
echo "Waiting 3 min for GPU drivers..."
sleep 180

# SSH into VM
gcloud compute ssh $VM_NAME --zone=$ZONE
```

### Step 2: Inside VM - Sequential Training Script

Save as `~/train_all_sequential.sh`:

```bash
#!/bin/bash
set -e

export BUCKET="YOUR_BUCKET"
export HOME_DIR=$(pwd)
export CHECKPOINT_DIR="./checkpoints"
export WEIGHTS_DIR="./weights"

mkdir -p $CHECKPOINT_DIR $WEIGHTS_DIR logs

LOG_FILE="logs/training_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "================================================"
echo "SSIS Sequential Training Started: $(date)"
echo "Bucket: $BUCKET"
echo "================================================"
echo ""

# Setup Python environment
setup_env() {
    echo "[SETUP] Installing dependencies..."
    pip install -q torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -q torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html 2>/dev/null || true
    pip install -q pandas numpy scikit-learn tqdm tensorboard gcsfs matplotlib
    
    # Clone repo if not exists
    if [ ! -d "idex" ]; then
        git clone https://github.com/your-org/idex.git
    fi
    cd idex
}

# Upload checkpoint to GCS
upload_checkpoint() {
    local component=$1
    local checkpoint=$2
    if [ -f "$checkpoint" ]; then
        echo "[UPLOAD] $component checkpoint to GCS..."
        gsutil cp "$checkpoint" gs://$BUCKET/checkpoints/${component}/ 2>/dev/null || true
    fi
}

# Upload final weights
upload_weights() {
    local component=$1
    local weight_path=$2
    if [ -f "$weight_path" ]; then
        echo "[UPLOAD] $component weights to GCS..."
        gsutil cp "$weight_path" gs://$BUCKET/weights/ 2>/dev/null || true
    fi
}

# Cleanup function between components
cleanup_component() {
    local dataset_dir=$1
    echo "[CLEANUP] Removing $dataset_dir to free disk space..."
    rm -rf "$dataset_dir"
    df -h .
}

# ============================================================
# PHASE 1: NavIC Anomaly Detection
# ============================================================
phase1_navic() {
    echo ""
    echo "================================================"
    echo "PHASE 1: NavIC Anomaly Detection (10 min)"
    echo "================================================"
    
    # Download dataset
    echo "[DOWNLOAD] NavIC dataset (~500MB)..."
    mkdir -p ../datasets/C-NAVIC
    gsutil -m cp -r gs://$BUCKET/datasets/C-NAVIC/* ../datasets/C-NAVIC/ 2>/dev/null || \
        gsutil -m cp -r gs://$BUCKET/datasets/C-NAVIC-anomaly/* ../datasets/C-NAVIC/
    
    # Train
    echo "[TRAIN] Starting NavIC training..."
    python3 ai/v3/navic_anomaly_detector.py \
        --data-dir ../datasets/C-NAVIC \
        --epochs 50 \
        --batch-size 64 \
        --checkpoint-dir $CHECKPOINT_DIR \
        --output-dir $WEIGHTS_DIR 2>&1 | tee logs/navic.log
    
    # Upload
    upload_weights "navic" "$WEIGHTS_DIR/navic_detector.onnx"
    upload_weights "navic" "$WEIGHTS_DIR/navic_detector.pth"
    
    # Cleanup
    cleanup_component "../datasets/C-NAVIC"
    
    echo "[COMPLETE] NavIC training finished at $(date)"
}

# ============================================================
# PHASE 2: Maneuver Detection
# ============================================================
phase2_maneuver() {
    echo ""
    echo "================================================"
    echo "PHASE 2: Maneuver Detection (15 min)"
    echo "================================================"
    
    # Download dataset
    echo "[DOWNLOAD] TLES dataset (~2GB)..."
    mkdir -p ../datasets/A-TLES
    gsutil -m cp -r gs://$BUCKET/datasets/A-TLES-Ephemeris/* ../datasets/A-TLES/ 2>/dev/null || \
        gsutil -m cp -r gs://$BUCKET/datasets/A-TLES/* ../datasets/A-TLES/
    
    # Train (synthetic data generation from TLES)
    echo "[TRAIN] Starting Maneuver training..."
    python3 ai/maneuver/train_maneuver_detector.py \
        --epochs 12 \
        --batch-size 128 \
        --output-dir $WEIGHTS_DIR \
        --checkpoint-dir $CHECKPOINT_DIR 2>&1 | tee logs/maneuver.log
    
    # Upload
    upload_weights "maneuver" "$WEIGHTS_DIR/maneuver_detector.onnx"
    upload_weights "maneuver" "$WEIGHTS_DIR/maneuver_v2.onnx"
    
    # Cleanup
    cleanup_component "../datasets/A-TLES"
    
    echo "[COMPLETE] Maneuver training finished at $(date)"
}

# ============================================================
# PHASE 3: Re-entry Prediction
# ============================================================
phase3_reentry() {
    echo ""
    echo "================================================"
    echo "PHASE 3: Re-entry Prediction (10 min)"
    echo "================================================"
    
    # Download dataset
    echo "[DOWNLOAD] Re-entry dataset (~200MB)..."
    mkdir -p ../datasets/D-Rentry
    gsutil -m cp -r gs://$BUCKET/datasets/D-Rentry-data/* ../datasets/D-Rentry/ 2>/dev/null || \
        gsutil -m cp -r gs://$BUCKET/datasets/D-Rentry/* ../datasets/D-Rentry/
    
    # Train
    echo "[TRAIN] Starting Re-entry training..."
    python3 ai/v3/reentry_predictor.py \
        --data-dir ../datasets/D-Rentry \
        --epochs 100 \
        --batch-size 256 \
        --checkpoint-dir $CHECKPOINT_DIR \
        --output-dir $WEIGHTS_DIR 2>&1 | tee logs/reentry.log
    
    # Upload
    upload_weights "reentry" "$WEIGHTS_DIR/reentry_predictor.onnx"
    upload_weights "reentry" "$WEIGHTS_DIR/reentry_model.pth"
    
    # Cleanup
    cleanup_component "../datasets/D-Rentry"
    
    echo "[COMPLETE] Re-entry training finished at $(date)"
}

# ============================================================
# PHASE 4: Collision GNN (Main Workload)
# ============================================================
phase4_collision() {
    echo ""
    echo "================================================"
    echo "PHASE 4: Collision GNN (3-4 hours)"
    echo "================================================"
    
    # Check for existing checkpoint to resume
    local checkpoint_file=""
    if gsutil ls gs://$BUCKET/checkpoints/collision/*.pt 2>/dev/null; then
        echo "[RESUME] Found existing checkpoint, downloading..."
        mkdir -p $CHECKPOINT_DIR/collision
        gsutil cp $(gsutil ls gs://$BUCKET/checkpoints/collision/*.pt | sort | tail -1) \
            $CHECKPOINT_DIR/collision/
        checkpoint_file=$(ls $CHECKPOINT_DIR/collision/*.pt 2>/dev/null | head -1)
    fi
    
    # Train with streaming (don't download full 18GB)
    echo "[TRAIN] Starting Collision GNN training with GCS streaming..."
    
    # Use spot trainer which handles streaming
    python3 scripts/gcp_spot_trainer_full.py \
        --component collision \
        --gcs-bucket $BUCKET \
        --checkpoint-dir $CHECKPOINT_DIR \
        --epochs 25 \
        --checkpoint-every 1 \
        ${checkpoint_file:+--resume $checkpoint_file} 2>&1 | tee logs/collision.log
    
    # Upload final weights
    upload_weights "collision" "$WEIGHTS_DIR/collision_multi_task_v2.onnx"
    upload_weights "collision" "$WEIGHTS_DIR/collision_gnn_final.pt"
    
    # Keep checkpoint for potential resume
    upload_checkpoint "collision" "$CHECKPOINT_DIR/collision/*.pt"
    
    echo "[COMPLETE] Collision training finished at $(date)"
}

# ============================================================
# MAIN EXECUTION
# ============================================================

main() {
    # Setup
    setup_env
    
    # Check GPU
    echo "[INFO] GPU Info:"
    nvidia-smi
    
    # Run all phases sequentially
    phase1_navic
    phase2_maneuver
    phase3_reentry
    phase4_collision
    
    # Final summary
    echo ""
    echo "================================================"
    echo "ALL TRAINING COMPLETE: $(date)"
    echo "================================================"
    echo ""
    echo "Weights uploaded to: gs://$BUCKET/weights/"
    echo ""
    echo "Final weights:"
    gsutil ls -lh gs://$BUCKET/weights/
    echo ""
    echo "Logs saved to: logs/"
    echo "================================================"
}

# Run main
main

# Upload final log
gsutil cp $LOG_FILE gs://$BUCKET/logs/ 2>/dev/null || true

echo "Script complete. VM can now be deleted."
```

### Step 3: Run the Script

```bash
# On the VM after SSH:
cd ~
chmod +x train_all_sequential.sh

# Run it (detached so it survives SSH disconnect)
nohup ./train_all_sequential.sh > training_nohup.log 2>&1 &

# Monitor progress
tail -f training_nohup.log
tail -f idex/logs/navic.log
tail -f idex/logs/collision.log
```

### Step 4: Monitor from Local Machine

```bash
# In another terminal, forward TensorBoard
gcloud compute ssh $VM_NAME --zone=$ZONE -- -L 6006:localhost:6006

# On VM, start tensorboard
cd ~/idex && tensorboard --logdir=./logs --port=6006

# Open browser: http://localhost:6006
```

### Step 5: Handle Preemption

If VM stops (preemption), restart and resume:

```bash
# From local machine
gcloud compute instances start $VM_NAME --zone=$ZONE
gcloud compute ssh $VM_NAME --zone=$ZONE

# On VM - script auto-resumes from GCS checkpoint
# Just re-run the script, it will skip completed phases
./train_all_sequential.sh
```

---

## Option B: T4 First, Then Upgrade to V100

If you want to save cost on first 3 components:

```bash
# Step 1: Create T4 VM for Phases 1-3
gcloud compute instances create ssis-phases1-3 \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --boot-disk-size=100GB \
    --provisioning-model=SPOT

# Run Phases 1-3 only (edit script to comment out phase4)
# ... training ...

# After Phase 3 completes, stop T4 VM
gcloud compute instances stop ssis-phases1-3 --zone=us-central1-a

# Step 2: Create V100 for Phase 4 (Collision)
gcloud compute instances create ssis-phase4 \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --boot-disk-size=250GB \
    --provisioning-model=SPOT

# Run Phase 4 only
```

**Cost:** T4 (35 min) + V100 (4 hrs) = ~$0.06 + $2.50 = **~$2.56**

---

## Option C: T4 Only (Slower but Cheapest)

If V100 spot capacity unavailable, use T4 with smaller batches:

```bash
# Create T4 VM with larger disk
gcloud compute instances create ssis-t4-all \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --boot-disk-size=300GB \
    --provisioning-model=SPOT

# Modify collision training:
python3 ai/collision/train_collision_surrogate_spot.py \
    --gcs-bucket $BUCKET \
    --batch-size 16 \
    --accumulation-steps 2 \
    --epochs 25
```

**Cost:** ~$0.88 for 6-8 hours

---

## Quick Reference: Single VM Commands

```bash
# Create one VM (V100 recommended)
export VM_NAME="ssis-all-$(date +%m%d)"
export ZONE="us-central1-a"
export BUCKET="YOUR_BUCKET"

gcloud compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=300GB \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --max-run-duration=8h

# SSH
gcloud compute ssh $VM_NAME --zone=$ZONE

# Download and run script
curl -o train_all_sequential.sh https://raw.githubusercontent.com/your-org/idex/main/docs/train_all_sequential.sh
chmod +x train_all_sequential.sh
export BUCKET="YOUR_BUCKET"
./train_all_sequential.sh

# Monitor
tail -f training_nohup.log

# If pre-empted, restart
gcloud compute instances start $VM_NAME --zone=$ZONE
```

---

## Summary

| Approach | VMs | GPU | Time | Cost | Spot Risk |
|----------|-----|-----|------|------|-----------|
| **Single V100** | **1** | V100 | 4 hrs | ~$3.00 | **Low** |
| Hybrid (T4→V100) | 2 | Mixed | 4 hrs | ~$2.56 | Medium |
| T4 Only | 1 | T4 | 6-8 hrs | ~$0.88 | Low |

**Recommendation:** Use **Single V100** approach - one VM creation, guaranteed capacity, auto-resume on preemption.
