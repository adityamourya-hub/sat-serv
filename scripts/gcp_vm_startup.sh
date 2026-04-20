#!/bin/bash
# GCP Spot VM Startup Script
# This runs automatically when the VM starts
# Set as metadata: --metadata-from-file startup-script=scripts/gcp_vm_startup.sh

set -e

# Configuration
REPO_URL="https://github.com/your-org/idex.git"  # Change to your repo
GCS_BUCKET="${GCS_BUCKET:-}"  # Set via metadata if needed
INSTALL_DIR="/ssis"

# Log output
exec > >(tee -a /var/log/ssis-startup.log)
exec 2>&1

echo "=========================================="
echo "SSIS Training VM Startup"
echo "Date: $(date)"
echo "=========================================="

# 1. Install system dependencies
echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git python3-pip python3-venv unzip

# 2. Install NVIDIA drivers if not present
if ! command -v nvidia-smi &> /dev/null; then
    echo "[2/6] Installing NVIDIA drivers..."
    apt-get install -y -qq linux-headers-$(uname -r)
    apt-get install -y -qq nvidia-driver-470
    # Reboot may be needed, but Deep Learning images usually have drivers
fi

# 3. Clone repository
echo "[3/6] Cloning SSIS repository..."
if [ -d "$INSTALL_DIR" ]; then
    echo "  Directory exists, pulling latest..."
    cd "$INSTALL_DIR"
    git pull origin main || true
else
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# 4. Install Python dependencies
echo "[4/6] Installing Python packages..."
pip3 install -q --upgrade pip
pip3 install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -q numpy scipy scikit-learn pandas onnx onnxruntime
pip3 install -q google-cloud-storage

# 5. Verify GPU
echo "[5/6] Verifying GPU..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('WARNING: No GPU detected!')
" || echo "GPU check failed, continuing anyway..."

# 6. Check GCS access
echo "[6/6] Checking GCS access..."
if command -v gsutil &> /dev/null; then
    # Try to list bucket if provided
    if [ -n "$GCS_BUCKET" ]; then
        echo "  Checking access to gs://$GCS_BUCKET..."
        if gsutil ls "gs://$GCS_BUCKET/" > /dev/null 2>&1; then
            echo "  ✓ GCS access confirmed"
        else
            echo "  ⚠ Cannot access gs://$GCS_BUCKET/"
            echo "  Ensure VM has storage-full scope or service account"
        fi
    fi
else
    echo "  ⚠ gsutil not found (install google-cloud-sdk)"
fi

echo ""
echo "=========================================="
echo "Startup complete!"
echo "Date: $(date)"
echo "=========================================="
echo ""
echo "To start training, SSH and run:"
echo "  gcloud compute ssh $(hostname) --zone=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H Metadata-Flavor:Google | cut -d/ -f4)"
echo "  cd $INSTALL_DIR"
echo "  python3 scripts/gcp_spot_trainer_full.py --gcs-bucket YOUR_BUCKET --component all --epochs 100"
echo ""

# Optional: Auto-start training if GCS_BUCKET is set via metadata
# METADATA_BUCKET=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket -H Metadata-Flavor:Google 2>/dev/null || echo "")
# if [ -n "$METADATA_BUCKET" ]; then
#     echo "Auto-starting training with bucket: $METADATA_BUCKET"
#     cd "$INSTALL_DIR"
#     python3 scripts/gcp_spot_trainer_full.py --gcs-bucket "$METADATA_BUCKET" --component all --epochs 100 --download-datasets 2>&1 | tee -a /var/log/ssis-training.log &
# fi
