# Multi-Object Track Association (MOTA) - Feature

Improves catalog correlation for noisy/fused sensor data (TLE + radar) using a hybrid path:
- deterministic geometric association (always available)
- optional Transformer/DeepSORT-style model reranking
- confidence and guardrail blending to preserve safety

## Go Integration

- Core logic: `engine/pkg/catalog/catalog.go`
  - `Correlate(...)` (backward-compatible)
  - `CorrelateAdvanced(...)` (association score/confidence/attention)
  - fused observation support via `TrackObservation` radar + TLE fields

- ONNX adapters:
  - `engine/pkg/ml/onnx/track_association_stub.go`
  - `engine/pkg/ml/onnx/track_association_onnx.go`

## Python Training

python3 -m venv .venv
source .venv/bin/activate
pip install -r ai/tracker/requirements.txt
python ai/tracker/train_transformer_association.py

Artifacts:
- `ai/tracker/data/association_transformer.onnx`
- `ai/tracker/data/association_metrics.json`

## Notes

- This module is intentionally hybrid: model outputs are blended only when confidence and score-delta guardrails pass.
- Fused position uses inverse-variance weighting for radar vs TLE uncertainty.
- Fallback behavior ensures deterministic operation if model is unavailable.
