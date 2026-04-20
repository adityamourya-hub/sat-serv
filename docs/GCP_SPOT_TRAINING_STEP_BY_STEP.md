# GCP Spot Training - Step-by-Step Guide

Complete guide for training SSIS AI models on GCP spot instances with GCS integration, checkpoint/resume capability, and automatic dataset/weight management.

---

## Prerequisites

1. **GCP Project** with billing enabled
2. **gcloud CLI** installed and authenticated
3. **Local datasets** in `training-data/` folder
4. **Google Cloud Storage** API enabled

---

## Step 1: Prepare Your GCS Bucket

### 1.1 Create the bucket (one-time setup)

```bash
# Set your bucket name (must be globally unique)
export GCS_BUCKET="ssis-training-$(date +%s)"
export GCP_PROJECT="your-project-id"
export GCP_REGION="us-central1"

# Create bucket
gcloud storage buckets create gs://$GCS_BUCKET \
    --location=$GCP_REGION \
    --uniform-bucket-level-access

# Or use gsutil (legacy)
# gsutil mb -l $GCP_REGION gs://$GCS_BUCKET
```

### 1.2 Create folder structure

```bash
# Create virtual folders in GCS
echo "" | gsutil cp - gs://$GCS_BUCKET/datasets/
echo "" | gsutil cp - gs://$GCS_BUCKET/checkpoints/
echo "" | gsutil cp - gs://$GCS_BUCKET/weights/
```

### 1.3 Upload your datasets to GCS

```bash
# Navigate to your repo
cd /path/to/your/idex

# Upload all training data (this preserves the folder structure)
gsutil -m cp -r training-data/* gs://$GCS_BUCKET/datasets/

# Verify upload
gsutil ls -r gs://$GCS_BUCKET/datasets/ | head -20
```

Expected GCS structure after upload:
```
gs://your-bucket/
├── datasets/
│   ├── A-TLES-Ephemeris-for-Collision-manauver-conjunction-cataloguing-data/
│   │   ├── GP-TLE-FORMAT-DATA-from-www-celestrak-org.txt
│   │   ├── active-satelites.txt
│   │   ├── satcat-data.csv
│   │   └── ...
│   ├── B-Conjuction-Collision-dataset/
│   │   ├── train_data.zip
│   │   ├── IVV_Releasable_Dataset_*.csv.gz
│   │   └── ...
│   ├── C-NAVIC-anamoly-and-GNSS-spoofing-jamming-dataset/
│   └── D-Rentry-data-dataset-optional/
├── checkpoints/          # Will be populated during training
└── weights/              # Will be populated after training
```

### 1.4 Set up authentication

Create a service account for the VM:

```bash
# Create service account
gcloud iam service-accounts create ssis-trainer \
    --display-name="SSIS Training Service Account"

# Grant GCS access
gcloud projects add-iam-policy-binding $GCP_PROJECT \
    --member="serviceAccount:ssis-trainer@$GCP_PROJECT.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Create and download key (if needed for local testing)
gcloud iam service-accounts keys create ~/ssis-trainer-key.json \
    --iam-account=ssis-trainer@$GCP_PROJECT.iam.gserviceaccount.com

# Set as default credentials (optional)
export GOOGLE_APPLICATION_CREDENTIALS=~/ssis-trainer-key.json
```

---

## Step 2: Configure .gitignore

Your repo should already exclude large datasets:

```bash
# Check .gitignore includes training-data/
cat .gitignore | grep training-data
```

Expected output:
```
# Training datasets (large files, stored in GCS)
training-data/
!training-data/readme-for-all-dataset-download-use-case.txt
```

This ensures:
- Datasets stay out of GitHub
- Only the README is committed (for documentation)
- Training data is pulled from GCS at runtime

---

## Step 3: Create GCP Spot VM

### 3.1 Create the spot instance

```bash
# Set variables
export VM_NAME="ssis-train-t4"
export ZONE="us-central1-a"
export GCS_BUCKET="your-bucket-name"  # Replace with your bucket

# Create spot VM with GPU
gcloud compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --boot-disk-size=200GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --metadata="install-nvidia-driver=True" \
    --service-account=ssis-trainer@$GCP_PROJECT.iam.gserviceaccount.com \
    --scopes=storage-full \
    --tags=ssis-training

# Wait for VM to be ready
gcloud compute instances wait $VM_NAME --zone=$ZONE
```

### 3.2 SSH into the VM

```bash
gcloud compute ssh $VM_NAME --zone=$ZONE
```

---

## Step 4: Setup on the Spot VM

Once SSH'd into the VM, run these commands:

### 4.1 Clone your repository

```bash
# Clone the repo (training-data is excluded, will download from GCS)
git clone https://github.com/your-org/idex.git /ssis
cd /ssis
```

### 4.2 Install dependencies

```bash
# Install Python dependencies
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install numpy scipy scikit-learn pandas onnx onnxruntime
pip3 install google-cloud-storage

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 4.3 Verify GCS access

```bash
# Check VM has GCS access
gsutil ls gs://$GCS_BUCKET/datasets/

# Should list your datasets
```

---

## Step 5: Start Training

### 5.1 Quick test (10 epochs, single component)

```bash
cd /ssis

# Test NavIC training (fastest, uses synthetic data if needed)
python3 scripts/gcp_spot_trainer_full.py \
    --gcs-bucket $GCS_BUCKET \
    --component navic \
    --epochs 10 \
    --download-datasets \
    --quick-test
```

### 5.2 Full training (all components)

```bash
cd /ssis

# Start full training
python3 scripts/gcp_spot_trainer_full.py \
    --gcs-bucket $GCS_BUCKET \
    --component all \
    --epochs 100 \
    --checkpoint-every 5 \
    --download-datasets
```

### 5.3 Train specific component

```bash
# Collision GNN (heaviest, uses B/ dataset)
python3 scripts/gcp_spot_trainer_full.py \
    --gcs-bucket $GCS_BUCKET \
    --component collision \
    --epochs 100 \
    --checkpoint-every 5 \
    --download-datasets

# Maneuver detector (uses A/ TLE dataset)
python3 scripts/gcp_spot_trainer_full.py \
    --gcs-bucket $GCS_BUCKET \
    --component maneuver \
    --epochs 50 \
    --checkpoint-every 10 \
    --download-datasets

# Re-entry predictor (uses D/ dataset)
python3 scripts/gcp_spot_trainer_full.py \
    --gcs-bucket $GCS_BUCKET \
    --component reentry \
    --epochs 50 \
    --checkpoint-every 10 \
    --download-datasets
```

---

## Step 6: Handle Spot Instance Preemption

### 6.1 If VM is terminated

When GCP reclaims your spot instance, you'll see:
```
⚠️ SIGTERM received! Spot instance being reclaimed...
💾 Emergency checkpoint will be saved...
```

The trainer automatically:
1. Saves emergency checkpoint to local disk
2. Syncs checkpoint to GCS (`gs://bucket/checkpoints/{component}/`)
3. Exits gracefully within 30 seconds

### 6.2 Restart training after preemption

Simply recreate the VM and run the **same command**:

```bash
# 1. Recreate the VM (it may take a few tries to get a spot)
gcloud compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --provisioning-model=SPOT \
    --boot-disk-size=200GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --service-account=ssis-trainer@$GCP_PROJECT.iam.gserviceaccount.com \
    --scopes=storage-full

# 2. SSH and restart
gcloud compute ssh $VM_NAME --zone=$ZONE

# 3. Setup (clone repo again)
git clone https://github.com/your-org/idex.git /ssis
cd /ssis
pip3 install torch numpy scipy google-cloud-storage

# 4. Run SAME command as before - it will auto-resume
python3 scripts/gcp_spot_trainer_full.py \
    --gcs-bucket $GCS_BUCKET \
    --component collision \
    --epochs 100 \
    --checkpoint-every 5
    # Note: No --download-datasets needed, data is in GCS
```

The trainer will:
1. Download latest checkpoint from GCS
2. Resume from exact epoch where it stopped
3. Continue training until completion

### 6.3 Monitor checkpoints

From your local machine:

```bash
# List checkpoints
gsutil ls gs://$GCS_BUCKET/checkpoints/

# List checkpoints for specific component
gsutil ls gs://$GCS_BUCKET/checkpoints/collision/

# Download specific checkpoint (if needed)
gsutil cp gs://$GCS_BUCKET/checkpoints/collision/checkpoint_collision_20240115_143022_epoch_0035.pt ./
```

---

## Step 7: Retrieve Trained Weights

### 7.1 After training completes

Trained weights are automatically uploaded to:
```
gs://your-bucket/weights/{component}/
```

### 7.2 Download weights to your local machine

```bash
# Set bucket
export GCS_BUCKET="your-bucket-name"

# Download all weights
mkdir -p ./trained-weights
gsutil -m cp -r gs://$GCS_BUCKET/weights/* ./trained-weights/

# Or download specific component
gsutil -m cp gs://$GCS_BUCKET/weights/collision/* ./trained-weights/

# List what weights exist
gsutil ls -r gs://$GCS_BUCKET/weights/
```

### 7.3 Copy weights to production

```bash
# Copy to your repo's model directory
cp ./trained-weights/*/collision_multi_task_v2.onnx pkg/ml/models/
cp ./trained-weights/*/navic_detector.onnx pkg/ml/models/
cp ./trained-weights/*/reentry_predictor.onnx pkg/ml/models/
cp ./trained-weights/*/maneuver_v2.onnx pkg/ml/models/

# Commit the ONNX files (these are production artifacts)
git add pkg/ml/models/*.onnx
git commit -m "Add trained models from GCP spot training"
```

---

## Step 8: Cost Optimization Tips

### 8.1 Choose right GPU for each component

| Component | Recommended GPU | Spot Cost/hr | Est. Training Time |
|-----------|----------------|--------------|-------------------|
| NavIC | T4 | ~$0.11 | 10 min |
| Maneuver | T4 | ~$0.11 | 15 min |
| Re-entry | T4 | ~$0.11 | 10 min |
| Collision | V100 | ~$0.74 | 2-4 hours |
| All (quick) | T4 | ~$0.11 | 1-2 hours |

### 8.2 Create different VMs for different workloads

```bash
# Small VM for quick components (NavIC, Maneuver, Re-entry)
gcloud compute instances create ssis-train-small \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --provisioning-model=SPOT

# Large VM for collision training
gcloud compute instances create ssis-train-large \
    --zone=us-central1-a \
    --machine-type=n1-standard-16 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --provisioning-model=SPOT
```

### 8.3 Stop VMs when not training

```bash
# Stop VM (preserves disk, stops billing for GPU/CPU)
gcloud compute instances stop ssis-train-t4 --zone=$ZONE

# Start again when needed
gcloud compute instances start ssis-train-t4 --zone=$ZONE
```

### 8.4 Use preemptible for preprocessing

```bash
# Preprocess data on CPU-only preemptible (cheaper)
gcloud compute instances create ssis-preprocess \
    --zone=$ZONE \
    --machine-type=n1-standard-8 \
    --provisioning-model=SPOT \
    --no-accelerator

# Run preprocessing only
python3 scripts/preprocess_training_data.py --dataset all
```

---

## Step 9: Troubleshooting

### 9.1 "No datasets found in GCS"

```bash
# Check datasets are uploaded
gsutil ls gs://$GCS_BUCKET/datasets/

# Re-upload if missing
gsutil -m cp -r training-data/* gs://$GCS_BUCKET/datasets/
```

### 9.2 "GCS connection failed"

```bash
# On the VM, check service account
gcloud auth list

# Should show: ssis-trainer@your-project.iam.gserviceaccount.com

# If not, activate:
gcloud auth activate-service-account \
    ssis-trainer@$GCP_PROJECT.iam.gserviceaccount.com \
    --key-file=/path/to/key.json
```

### 9.3 "CUDA out of memory"

```bash
# Reduce batch size in training scripts
# Or use gradient accumulation
# Or use smaller GPU model for testing
```

### 9.4 VM keeps getting preempted

```bash
# Try different zone
gcloud compute instances create ssis-train-t4 \
    --zone=us-west1-a \
    ...

# Or use regular VM for critical training (more expensive)
gcloud compute instances create ssis-train-regular \
    --provisioning-model=STANDARD \
    ...
```

### 9.5 Check training logs

```bash
# On the VM, check if training is running
ps aux | grep python3

# Check disk space
df -h

# Check GCS sync is working
gsutil ls gs://$GCS_BUCKET/checkpoints/
```

---

## Quick Reference Commands

```bash
# ===== ONE-TIME SETUP =====
export GCS_BUCKET="ssis-training-$(date +%s)"
export GCP_PROJECT="your-project-id"
gcloud storage buckets create gs://$GCS_BUCKET --location=us-central1
gsutil -m cp -r training-data/* gs://$GCS_BUCKET/datasets/

# ===== CREATE VM =====
gcloud compute instances create ssis-train-t4 \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --provisioning-model=SPOT \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --scopes=storage-full

# ===== SSH & TRAIN =====
gcloud compute ssh ssis-train-t4 --zone=us-central1-a
# (inside VM)
git clone https://github.com/your-org/idex.git /ssis
cd /ssis
python3 scripts/gcp_spot_trainer_full.py \
    --gcs-bucket $GCS_BUCKET \
    --component all \
    --epochs 100 \
    --download-datasets

# ===== IF PREEMPTED =====
gcloud compute instances create ssis-train-t4 --zone=us-central1-a ...
gcloud compute ssh ssis-train-t4 --zone=us-central1-a
# (inside VM - same command auto-resumes)
python3 scripts/gcp_spot_trainer_full.py --gcs-bucket $GCS_BUCKET --component all --epochs 100

# ===== DOWNLOAD WEIGHTS =====
gsutil -m cp -r gs://$GCS_BUCKET/weights/* ./trained-weights/
cp ./trained-weights/*/*.onnx pkg/ml/models/

# ===== CLEANUP =====
gcloud compute instances delete ssis-train-t4 --zone=us-central1-a
```

---

## Summary

| What | Where |
|------|-------|
| Datasets | `gs://bucket/datasets/` (uploaded once) |
| Checkpoints | `gs://bucket/checkpoints/{component}/` (auto-created) |
| Weights | `gs://bucket/weights/{component}/` (auto-uploaded) |
| Local cache | `/ssis/training-data/` (downloaded on VM) |
| Resume | Automatic - just re-run same command |
| Cost | ~70% savings vs on-demand GPU |
