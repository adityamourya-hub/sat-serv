# GNN-Augmented Collision Risk Engine (Feature 2)

Implements the ADITI Feature 2 hybrid design from docmd/ai.md:
- Graph Attention (GATv2-style surrogate) over neighborhood graph
- Hybrid score with classical Alfano/Gaussian/QMC fallback guardrails
- Outputs: risk score, uncertainty bounds, explanation heatmap
- ONNX export path for Go runtime

## Train and Export ONNX

python3 -m venv .venv
source .venv/bin/activate
pip install -r ai/collision/requirements.txt
python ai/collision/train_collision_surrogate.py

Artifacts:
- ai/collision/data/collision_gat_surrogate.onnx
- ai/collision/data/metrics.json

## Go Integration

- Hybrid engine and API: engine/pkg/orbital/collision.go
- Default build stub model: engine/pkg/ml/onnx/collision_model_stub.go
- ONNX-tag build adapter: engine/pkg/ml/onnx/collision_model_onnx.go

## Runtime Strategy

1. Compute classical risk with `EvaluateCollisionRisk`.
2. Optionally run surrogate with `EvaluateCollisionRiskHybrid` using graph input.
3. If model is missing, low-confidence, or deviates beyond guardrail, fallback to classical stage.

This ensures deterministic safety while still enabling cognitive acceleration and explainability.
