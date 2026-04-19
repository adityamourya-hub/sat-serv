#!/bin/bash
# Setup script for GKE Spot Instance Training
# Run this to prepare all training scripts for GKE spot instances

set -e

echo "========================================"
echo "SSIS GKE Spot Training Setup"
echo "========================================"

# Make scripts executable
echo "Setting executable permissions..."
chmod +x ai/utils/gke_checkpoint_manager.py 2>/dev/null || true
chmod +x ai/collision/train_collision_surrogate_spot.py 2>/dev/null || true
chmod +x ai/v3/navic_anomaly_detector_spot.py 2>/dev/null || true
echo "✓ Scripts ready"

# Check dependencies
echo ""
echo "Checking dependencies..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "⚠ PyTorch not installed"
python3 -c "from google.cloud import storage; print('✓ Google Cloud Storage')" 2>/dev/null || echo "⚠ google-cloud-storage not installed (pip install google-cloud-storage)"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Set GCS bucket: export GCS_CHECKPOINT_BUCKET=your-bucket"
echo "2. Run local test:"
echo "   python3 ai/collision/train_collision_surrogate_spot.py --epochs 5 --checkpoint-every 2"
echo ""
echo "3. Deploy to GKE:"
echo "   kubectl apply -f k8s/gke-spot-training-job.yaml"
echo ""
