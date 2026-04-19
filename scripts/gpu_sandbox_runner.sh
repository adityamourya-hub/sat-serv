#!/bin/bash
# GPU Sandbox Runner for SSIS Training
# Copy this script + manifest to GPU machine and run

set -e

echo "=========================================="
echo "SSIS GPU Sandbox Training Runner"
echo "=========================================="

# Configuration
MANIFEST=${1:-"gpu_training_manifest.json"}
INPUT_DIR=${2:-"training-data"}
OUTPUT_DIR=${3:-"ai/training/data_gpu"}
COMPONENT=${4:-"all"}

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ No GPU detected, will use CPU (slower)"
fi

# Check available disk space
AVAILABLE=$(df . | awk 'NR==2 {print $4}')
AVAILABLE_GB=$((AVAILABLE / 1024 / 1024))
echo "✓ Available disk space: ${AVAILABLE_GB} GB"

# Step 1: Stream process data
echo ""
echo "[1/3] Streaming process from archives..."
python3 scripts/streaming_dataset_processor.py \
    --manifest "$MANIFEST" \
    --input_dir "$INPUT_DIR" \
    --stream_process \
    --component "$COMPONENT" \
    --output_dir "$OUTPUT_DIR"

# Step 2: Train models
echo ""
echo "[2/3] Training models on GPU..."
python3 scripts/train_with_public_data.py \
    --manifest "$OUTPUT_DIR/training_manifest.json" \
    --mode full  # Use full mode on GPU

# Step 3: Export
echo ""
echo "[3/3] Exporting to production..."
cp ai/training/models/*.onnx pkg/ml/models/ 2>/dev/null || true
cp ai/training/models/*.pt pkg/ml/models/ 2>/dev/null || true

echo ""
echo "=========================================="
echo "✅ GPU Training Complete!"
echo "=========================================="
echo "Models exported to: pkg/ml/models/"
echo ""
echo "Copy back to host:"
echo "  scp -r pkg/ml/models/*.onnx user@host:/path/"
