# Intelligent Maneuver Detector (Feature 1)

This module implements the ADITI Feature 1 model stack requested in docmd/ai.md:
- Architecture: LSTM + Multi-Head Attention
- Outputs: maneuver probability, maneuver type (impulsive/continuous), delta-v estimate, confidence score
- Explainability: SHAP-style feature contribution proxy
- Export: ONNX for Go inference path

## Train and Export ONNX

1) Create environment and install deps:

python3 -m venv .venv
source .venv/bin/activate
pip install -r ai/maneuver/requirements.txt

2) Train and export:

python ai/maneuver/train_maneuver_detector.py

Artifacts:
- ai/maneuver/data/maneuver_detector.onnx
- ai/maneuver/data/metrics.json

## Generate Explainability Proxy

python ai/maneuver/shap_proxy.py

Artifact:
- ai/maneuver/data/sample_shap_proxy.json

## Go Integration

- Core hybrid detector: engine/pkg/orbital/maneuver.go
- ONNX adapter interface: engine/pkg/ml/onnx/maneuver_model_stub.go
- Optional ONNX build tag path: engine/pkg/ml/onnx/maneuver_model_onnx.go

## Runtime Notes

- Default build keeps deterministic CUSUM fallback active.
- For deployment with native ONNX runtime libraries, compile with:

go build -tags onnx ./...

- The ONNX adapter currently provides a thin contract and output mapping spec; bind tensor IO in deployment image where onnxruntime shared libs are available.
