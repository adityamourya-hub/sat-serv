#!/bin/bash
set -e

export BUCKET="ssis-training-1776624951"
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

# Install NVIDIA drivers (if not present)
install_nvidia_driver() {
    echo "[DRIVER] Checking NVIDIA driver..."
    if ! nvidia-smi &>/dev/null; then
        echo "[DRIVER] Installing NVIDIA drivers..."
        apt-get update -qq
        apt-get install -y -qq linux-headers-$(uname -r)
        
        # Add NVIDIA package repositories
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
        dpkg -i cuda-keyring_1.0-1_all.deb
        apt-get update -qq
        
        # Install driver
        apt-get install -y -qq nvidia-driver-550
        
        # Load driver
        modprobe nvidia
        
        echo "[DRIVER] Waiting for driver to initialize..."
        for i in {1..30}; do
            if nvidia-smi &>/dev/null; then
                echo "[DRIVER] NVIDIA driver ready!"
                nvidia-smi
                return 0
            fi
            echo "[DRIVER] Waiting... ($i/30)"
            sleep 10
        done
        
        echo "[ERROR] NVIDIA driver failed to load. Rebooting VM may help."
        exit 1
    else
        echo "[DRIVER] NVIDIA driver already installed"
        nvidia-smi
    fi
}

# Setup Python virtual environment
setup_env() {
    echo "[SETUP] Creating Python virtual environment..."
    
    # Install python3-venv if not present
    apt-get update -qq && apt-get install -y -qq python3-venv python3-pip
    
    # Create and activate venv
    python3 -m venv ~/ssis-venv
    source ~/ssis-venv/bin/activate
    
    echo "[SETUP] Installing dependencies into venv..."
    pip install --upgrade pip -q
    pip install -q torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -q torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html 2>/dev/null || true
    pip install -q pandas numpy scikit-learn tqdm tensorboard gcsfs matplotlib google-cloud-storage
    
    # Clone repo if not exists
    if [ ! -d "idex" ]; then
        echo "[SETUP] Cloning repository..."
        git clone https://github.com/your-org/idex.git
    fi
    cd idex
    
    echo "[SETUP] Environment ready"
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
    
    # Ensure logs directory exists
    mkdir -p logs
    
    # Download dataset
    echo "[DOWNLOAD] NavIC dataset (~500MB)..."
    mkdir -p ../datasets/C-NAVIC
    gsutil -m cp -r gs://$BUCKET/datasets/C-NAVIC-anamoly-and-GNSS-spoofing-jamming-dataset/* ../datasets/C-NAVIC/ 2>/dev/null
    
    # Train
    echo "[TRAIN] Starting NavIC training..."
    python3 ai/v3/navic_anomaly_detector.py \
        --data-dir ../datasets/C-NAVIC \
        --epochs 50 \
        --batch-size 64 \
        --checkpoint-dir $CHECKPOINT_DIR \
        --output-dir $WEIGHTS_DIR 2>&1 | tee logs/navic.log || \
        echo "[WARN] NavIC training script not found or failed, skipping"
    
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
    
    # Ensure logs directory exists
    mkdir -p logs
    
    # Download dataset
    echo "[DOWNLOAD] TLES dataset (~2GB)..."
    mkdir -p ../datasets/A-TLES
    gsutil -m cp -r gs://$BUCKET/datasets/A-TLES-Ephemeris-for-Collision-manauver-conjunction-cataloguing-data/* ../datasets/A-TLES/ 2>/dev/null
    
    # Train (synthetic data generation from TLES)
    echo "[TRAIN] Starting Maneuver training..."
    python3 ai/maneuver/train_maneuver_detector.py \
        --epochs 12 \
        --batch-size 128 \
        --output-dir $WEIGHTS_DIR \
        --checkpoint-dir $CHECKPOINT_DIR 2>&1 | tee logs/maneuver.log || \
        echo "[WARN] Maneuver training script not found or failed, skipping"
    
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
    
    # Ensure logs directory exists
    mkdir -p logs
    
    # Download dataset
    echo "[DOWNLOAD] Re-entry dataset (~200MB)..."
    mkdir -p ../datasets/D-Rentry
    gsutil -m cp -r gs://$BUCKET/datasets/D-Rentry-data-dataset-optional/* ../datasets/D-Rentry/ 2>/dev/null
    
    # Train
    echo "[TRAIN] Starting Re-entry training..."
    python3 ai/v3/reentry_predictor.py \
        --data-dir ../datasets/D-Rentry \
        --epochs 100 \
        --batch-size 256 \
        --checkpoint-dir $CHECKPOINT_DIR \
        --output-dir $WEIGHTS_DIR 2>&1 | tee logs/reentry.log || \
        echo "[WARN] Re-entry training script not found or failed, skipping"
    
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
    
    # Ensure logs directory exists
    mkdir -p logs
    
    # Download collision dataset (~18GB)
    echo "[DOWNLOAD] Collision dataset (~18GB) - this will take time..."
    mkdir -p training-data/B-Conjuction-Collision-dataset
    gsutil -m cp -r gs://$BUCKET/datasets/B-Conjunction-Collision-dataset/* training-data/B-Conjuction-Collision-dataset/ 2>/dev/null || \
        gsutil -m cp -r gs://$BUCKET/datasets/B-Conjuction-Collision-dataset/* training-data/B-Conjuction-Collision-dataset/
    
    echo "[DOWNLOAD] Collision dataset downloaded"
    df -h .
    
    # Check for existing checkpoint to resume
    local checkpoint_file=""
    if gsutil ls gs://$BUCKET/checkpoints/collision/*.pt 2>/dev/null; then
        echo "[RESUME] Found existing checkpoint, downloading..."
        mkdir -p $CHECKPOINT_DIR/collision
        gsutil cp $(gsutil ls gs://$BUCKET/checkpoints/collision/*.pt | sort | tail -1) \
            $CHECKPOINT_DIR/collision/
        checkpoint_file=$(ls $CHECKPOINT_DIR/collision/*.pt 2>/dev/null | head -1)
    fi
    
    # Train
    echo "[TRAIN] Starting Collision GNN training..."
    
    # Direct call to collision training script
    python3 ai/collision/train_collision_surrogate.py \
        --data-path training-data/B-Conjuction-Collision-dataset/train_data \
        --epochs 25 \
        --batch-size 32 \
        --checkpoint-dir $CHECKPOINT_DIR \
        --output-dir $WEIGHTS_DIR 2>&1 | tee logs/collision.log || \
        echo "[WARN] Collision training script not found or failed, skipping"
    
    # Upload final weights
    upload_weights "collision" "$WEIGHTS_DIR/collision_multi_task_v2.onnx"
    upload_weights "collision" "$WEIGHTS_DIR/collision_gnn_final.pt"
    
    # Keep checkpoint for potential resume
    upload_checkpoint "collision" "$CHECKPOINT_DIR/collision/*.pt"
    
    # Cleanup collision dataset to free disk space
    cleanup_component "training-data"
    
    echo "[COMPLETE] Collision training finished at $(date)"
}

# ============================================================
# MAIN EXECUTION
# ============================================================

main() {
    # Install NVIDIA driver first
    install_nvidia_driver
    
    # Setup Python environment
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
