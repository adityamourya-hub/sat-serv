# SSIS Operator Manual (IAF ADITI 4.0 PS24)

## 1. Purpose
This manual defines baseline operator procedures for running SSIS in air-gapped training and operations environments.

## 2. Preconditions
- Offline machine with x86_64 or arm64 CPU
- SSIS delivery package extracted locally
- Optional: private signing key in environment variable SSIS_SIGNING_KEY (hex-encoded Ed25519 private key)

## 3. Core Commands
### 3.1 Operational run
./bin/ssa_ops_runner -catalog-size 100000 -concurrency 50 -watch-hours 24 -out ./runtime/ops_run -assistant

### 3.2 Scale validation run
./bin/ssa_ops_runner -scale-test -catalog-size 100000 -scale-hours 168 -out ./runtime/scale_run

### 3.3 One-click demo
./runtime/one_click_demo.sh

## 4. Generated Artifacts
- runtime/ops_run/operational_metrics.json
- runtime/ops_run/signed_alert_example.json
- runtime/ops_run/assistant_sitrep.txt
- runtime/ops_run/ADITI4_OPERATIONAL_TEST_REPORT.md

## 5. Signature Verification Procedure
1. Open signed_alert_example.json.
2. Confirm signature_valid=true.
3. Confirm public_key_hex is populated.
4. Store artifact hash from SHA256SUMS with mission record.

## 6. Runtime Safety Rules
- Do not disable fallback guardrails in production-like exercises.
- Require explicit ONNX availability for adjudication runs.
- Record seed and command line for replay determinism.
- Preserve all signed artifacts for audit trail.

## 7. Incident Response
- If ONNX model load fails: rerun with strict model path checks and inspect model sidecar files.
- If signatures fail: revalidate SSIS_SIGNING_KEY and artifact integrity via SHA256SUMS.
- If latency spikes: reduce concurrency and isolate model-specific timing in operational_metrics.json.
