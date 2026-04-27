#!/usr/bin/env bash
# =============================================================================
# SSIS Local Laptop Training — P0 through P6
# Intel CPU (no GPU). All phases use reduced sample counts suited for a laptop.
#
# Usage:
#   chmod +x scripts/train_local_laptop.sh
#   ./scripts/train_local_laptop.sh            # run all phases
#   ./scripts/train_local_laptop.sh --phase p0  # run one phase only
#   ./scripts/train_local_laptop.sh --phase p1
#
# Weights saved to: weights/
# Logs saved to:    logs/
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="ssis-train"
PYTHON="$(conda run -n $ENV_NAME which python 2>/dev/null || echo '')"
LOG_DIR="$REPO_ROOT/logs"
WEIGHTS_DIR="$REPO_ROOT/weights"
PHASE="${1:-all}"

# Parse --phase argument
if [[ "$1" == "--phase" && -n "${2:-}" ]]; then
  PHASE="$2"
fi

mkdir -p "$LOG_DIR" "$WEIGHTS_DIR"

# =============================================================================
# Step 0: Create / verify conda environment
# =============================================================================
setup_env() {
  echo ""
  echo "=========================================="
  echo " Setting up conda env: $ENV_NAME"
  echo "=========================================="

  if conda env list | grep -q "^$ENV_NAME "; then
    echo " [OK] Env '$ENV_NAME' already exists — skipping creation."
  else
    echo " [CREATE] Creating '$ENV_NAME' with Python 3.10..."
    conda create -y -n "$ENV_NAME" python=3.10
  fi

  echo " [PIP] Installing training dependencies..."
  conda run -n "$ENV_NAME" pip install --quiet --upgrade pip
  # CPU-only torch (avoids ~2GB CUDA download on a laptop with no GPU)
  conda run -n "$ENV_NAME" pip install --quiet \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu
  # Pin numpy<2: torch 2.2 is compiled against numpy 1.x ABI
  conda run -n "$ENV_NAME" pip install --quiet "numpy<2"
  conda run -n "$ENV_NAME" pip install --quiet \
    scipy>=1.11.0 \
    h5py>=3.10.0 \
    pandas>=2.1.0 \
    scikit-learn>=1.3.0 \
    matplotlib>=3.8.0 \
    tqdm>=4.66.0 \
    pyyaml>=6.0.1 \
    onnx>=1.15.0 \
    onnxruntime>=1.17.0 \
    astropy>=6.0.0

  echo " [OK] Environment ready."
  PYTHON="$(conda run -n $ENV_NAME which python)"
}

run_python() {
  conda run -n "$ENV_NAME" python "$@"
}

# =============================================================================
# Phase P0: Neural Atmosphere FNO
# Dataset: Synthetic NRLMSISE-00 samples (generated in-memory)
# Laptop scale: 50k samples, 30 epochs, batch 256
# Output: weights/neural_atmosphere_final.pth + weights/neural_atmosphere.onnx
# =============================================================================
run_p0() {
  echo ""
  echo "=========================================="
  echo " P0: Neural Atmosphere Operator"
  echo "=========================================="

  local H5="$REPO_ROOT/training-data/p0/synthetic/train_data.h5"
  mkdir -p "$(dirname "$H5")"

  if [[ ! -f "$H5" ]]; then
    echo " [GEN] Generating 50k atmosphere samples → $H5"
    run_python scripts/train_neural_atmosphere.py \
      --generate-only \
      --num-samples 50000 \
      --output-path "$H5" \
      2>&1 | tee "$LOG_DIR/p0_generate.log"
  else
    echo " [SKIP] Dataset already exists: $H5"
  fi

  # P0 FNO uses complex FFT (torch.fft.rfft + complex einsum) which MPS does not support.
  # Must run on cpu.
  echo " [TRAIN] Training FNO (30 epochs, batch=256, device=cpu)..."
  run_python scripts/train_neural_atmosphere.py \
    --data-path "$H5" \
    --epochs 30 \
    --batch-size 256 \
    --lr 1e-3 \
    --lambda-phys 0.1 \
    --modes 16 \
    --width 64 \
    --device cpu \
    --model-dir "$WEIGHTS_DIR/p0_atmosphere" \
    2>&1 | tee "$LOG_DIR/p0_train.log"

  echo " [DONE] P0 complete. Weights: $WEIGHTS_DIR/p0_atmosphere/"
}

# =============================================================================
# Phase P1: Preprocess TLE/satcat data + generate orbital scenario cache
# Dataset: TLE files already in training-data/A-* and training-data/p1/
# Output: ai/maneuver/data/tle_residuals.jsonl, ai/simulator/data/satcat_catalog.json
# No neural training in P1 on laptop (trajectory generation needs Go hpop)
# =============================================================================
run_p1() {
  echo ""
  echo "=========================================="
  echo " P1: TLE Preprocessing & Orbital Dataset"
  echo "=========================================="

  echo " [PREPROCESS] Dataset A (TLEs → maneuver residuals + satcat)..."
  run_python scripts/preprocess_training_data.py --dataset A \
    2>&1 | tee "$LOG_DIR/p1_preprocess_A.log"

  echo " [PREPROCESS] Dataset B (Conjunction CDMs)..."
  run_python scripts/preprocess_training_data.py --dataset B \
    2>&1 | tee "$LOG_DIR/p1_preprocess_B.log"

  echo " [PREPROCESS] Dataset C (GNSS spoofing/jamming)..."
  run_python scripts/preprocess_training_data.py --dataset C \
    2>&1 | tee "$LOG_DIR/p1_preprocess_C.log"

  echo " [PREPROCESS] Dataset D (Re-entry history)..."
  run_python scripts/preprocess_training_data.py --dataset D \
    2>&1 | tee "$LOG_DIR/p1_preprocess_D.log"

  echo " [INFO] P1 neural training (SE3-GNN / GHNN) requires Go hpop trajectory generator."
  echo "        Skipping trajectory generation on laptop — preprocessed data is ready."
  echo " [DONE] P1 complete. Check ai/maneuver/data/ and ai/simulator/data/"
}

# =============================================================================
# Phase P2: Re-entry predictor training
# Dataset: D-Rentry + synthetic gen_reentry.py (laptop scale: 10k samples)
# Output: weights/p2_reentry/
# =============================================================================
run_p2() {
  echo ""
  echo "=========================================="
  echo " P2: Re-entry Predictor"
  echo "=========================================="

  local OUT="$REPO_ROOT/training-data/p2/raw/reentry"
  mkdir -p "$OUT"

  echo " [GEN] Generating 10k synthetic re-entry trajectories → $OUT"
  run_python sda-train/scripts/gen_reentry.py \
    --n_samples 10000 \
    --output "$OUT" \
    2>&1 | tee "$LOG_DIR/p2_gen_reentry.log"

  if [[ -f ai/v3/reentry_predictor.py ]]; then
    echo " [TRAIN] Training re-entry predictor (20 epochs)..."
    run_python ai/v3/reentry_predictor.py \
      --data "$OUT" \
      --epochs 20 \
      --device mps \
      --output-dir "$WEIGHTS_DIR/p2_reentry" \
      2>&1 | tee "$LOG_DIR/p2_train.log" || echo " [WARN] reentry_predictor.py returned non-zero — check log."
  else
    echo " [SKIP] ai/v3/reentry_predictor.py not found."
  fi

  echo " [DONE] P2 complete."
}

# =============================================================================
# Phase P3: Task heads — maneuver detector + collision surrogate
# Dataset: ai/maneuver/data/ (from P1) + ai/collision/data/ (from B preprocessing)
# Laptop scale: 20 epochs
# =============================================================================
run_p3() {
  echo ""
  echo "=========================================="
  echo " P3: Task Heads (Maneuver + Collision)"
  echo "=========================================="

  if [[ -f ai/maneuver/train_maneuver_detector.py ]]; then
    echo " [TRAIN] Maneuver detector (20 epochs)..."
    run_python ai/maneuver/train_maneuver_detector.py \
      --data ai/maneuver/data/tle_residuals.jsonl \
      --epochs 20 \
      --device mps \
      --output-dir "$WEIGHTS_DIR/p3_maneuver" \
      2>&1 | tee "$LOG_DIR/p3_maneuver.log" || echo " [WARN] maneuver trainer returned non-zero."
  fi

  if [[ -f ai/collision/train_collision_surrogate.py ]]; then
    echo " [TRAIN] Collision surrogate GNN (20 epochs)..."
    run_python ai/collision/train_collision_surrogate.py \
      --epochs 20 \
      --device mps \
      --output-dir "$WEIGHTS_DIR/p3_collision" \
      2>&1 | tee "$LOG_DIR/p3_collision.log" || echo " [WARN] collision trainer returned non-zero."
  fi

  echo " [DONE] P3 complete."
}

# =============================================================================
# Phase P4: NavIC anomaly detector
# Dataset: GNSS spoofing data from C preprocessing
# =============================================================================
run_p4() {
  echo ""
  echo "=========================================="
  echo " P4: NavIC Anomaly Detector"
  echo "=========================================="

  if [[ -f ai/v3/navic_anomaly_detector.py ]]; then
    echo " [TRAIN] NavIC anomaly detector (20 epochs)..."
    run_python ai/v3/navic_anomaly_detector.py \
      --epochs 20 \
      --device mps \
      --output-dir "$WEIGHTS_DIR/p4_navic" \
      2>&1 | tee "$LOG_DIR/p4_navic.log" || echo " [WARN] navic trainer returned non-zero."
  else
    echo " [SKIP] ai/v3/navic_anomaly_detector.py not found."
  fi

  echo " [DONE] P4 complete."
}

# =============================================================================
# Phase P5: Conformal UQ calibration
# =============================================================================
run_p5() {
  echo ""
  echo "=========================================="
  echo " P5: Conformal Uncertainty Quantification"
  echo "=========================================="

  if [[ -f ai/v3/conformal_uq.py ]]; then
    echo " [CALIBRATE] Running conformal UQ..."
    run_python ai/v3/conformal_uq.py \
      --model-dir "$WEIGHTS_DIR" \
      --device mps \
      2>&1 | tee "$LOG_DIR/p5_uq.log" || echo " [WARN] conformal_uq.py returned non-zero."
  else
    echo " [SKIP] ai/v3/conformal_uq.py not found."
  fi

  echo " [DONE] P5 complete."
}

# =============================================================================
# Phase P6: ONNX export + INT8 quantization
# =============================================================================
run_p6() {
  echo ""
  echo "=========================================="
  echo " P6: ONNX Export & INT8 Quantization"
  echo "=========================================="

  if [[ -f ai/v3/onnx_export.py ]]; then
    echo " [EXPORT] Exporting models to ONNX + INT8 quantization..."
    run_python ai/v3/onnx_export.py \
      --weights-dir "$WEIGHTS_DIR" \
      --output-dir "$WEIGHTS_DIR/p6_onnx" \
      2>&1 | tee "$LOG_DIR/p6_onnx.log" || echo " [WARN] onnx_export.py returned non-zero."
  else
    echo " [SKIP] ai/v3/onnx_export.py not found."
  fi

  echo " [DONE] P6 complete."
}

# =============================================================================
# Summary
# =============================================================================
print_summary() {
  echo ""
  echo "=========================================="
  echo " TRAINING SUMMARY"
  echo "=========================================="
  echo " Weights directory: $WEIGHTS_DIR/"
  ls -lh "$WEIGHTS_DIR"/ 2>/dev/null || echo " (empty — check logs)"
  echo ""
  echo " Logs: $LOG_DIR/"
  ls -1 "$LOG_DIR"/*.log 2>/dev/null | while read -r f; do
    local last
    last=$(tail -1 "$f" 2>/dev/null || echo "")
    echo "  $(basename "$f"): $last"
  done
  echo ""
  echo " To validate P0 atmosphere model:"
  echo "   conda run -n $ENV_NAME python scripts/validate_atmosphere.py \\"
  echo "     --model $WEIGHTS_DIR/p0_atmosphere/neural_atmosphere_best.pth"
  echo ""
  echo " To export all ONNX models for Go runtime:"
  echo "   conda run -n $ENV_NAME python ai/v3/onnx_export.py \\"
  echo "     --weights-dir $WEIGHTS_DIR --output-dir $WEIGHTS_DIR/p6_onnx"
  echo "=========================================="
}

# =============================================================================
# Main dispatch
# =============================================================================
setup_env

case "$PHASE" in
  all)
    run_p0; run_p1; run_p2; run_p3; run_p4; run_p5; run_p6
    ;;
  p0) run_p0 ;;
  p1) run_p1 ;;
  p2) run_p2 ;;
  p3) run_p3 ;;
  p4) run_p4 ;;
  p5) run_p5 ;;
  p6) run_p6 ;;
  *)
    echo "Unknown phase: $PHASE. Use: all | p0 | p1 | p2 | p3 | p4 | p5 | p6"
    exit 1
    ;;
esac

print_summary
