#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/docs/validation_reports"
mkdir -p "$OUT_DIR"

# Gobra formal verification run for floating-point robustness obligations.
# DO-333 FM.5: IEEE-754 finite-value safety + termination checks.
gobra \
  --divergence \
  --float-mode=ieee \
  --module ssis \
  "$ROOT_DIR/pkg/math/safe_cholesky.go" \
  "$ROOT_DIR/pkg/verification/gobra_specs.gobra" \
  > "$OUT_DIR/gobra_fp_verification.log" 2>&1

echo "Gobra FP verification log: $OUT_DIR/gobra_fp_verification.log"
