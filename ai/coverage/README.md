# Predictive Blind-Spot Forecaster (Feature 6)

Extends `pkg/spatial/coverage.go` with a Temporal Convolutional Network + Prophet-style hybrid forecaster for 24-72 hour blind-spot ratio prediction.

## What is implemented

### Go runtime
- Hybrid forecasting API in `engine/pkg/spatial/coverage.go`
  - `BlindSpotSample`
  - `BlindSpotForecastModel`
  - `HybridBlindSpotForecaster`
  - `ForecastBlindSpots(history, horizonHours, forecaster)`
  - Prophet-like deterministic fallback with trend + daily seasonality

### ONNX adapters
- `engine/pkg/ml/onnx/blindspot_forecaster_stub.go`
- `engine/pkg/ml/onnx/blindspot_forecaster_onnx.go`

### Python training/scripts
- `ai/coverage/train_tcn_forecaster.py` (PyTorch TCN -> ONNX)
- `ai/coverage/train_prophet_baseline.py` (Prophet baseline, with fallback mode)
- `ai/coverage/requirements.txt`

## Quickstart

python3 -m venv .venv
source .venv/bin/activate
pip install -r ai/coverage/requirements.txt

### 1) Train TCN and export ONNX
python ai/coverage/train_tcn_forecaster.py

Artifacts:
- `ai/coverage/data/blindspot_tcn.onnx`
- `ai/coverage/data/tcn_metrics.json`

### 2) Fit Prophet baseline (72h)
python ai/coverage/train_prophet_baseline.py

Artifacts:
- `ai/coverage/data/prophet_forecast_72h.csv`
- `ai/coverage/data/prophet_metrics.json`

## Runtime usage (Go)

Use `ForecastBlindSpots` with 24-72 horizon:
- if model is available and confident -> hybrid blended forecast
- else -> deterministic Prophet-like fallback

This ensures operational continuity and safe forecasting under model outages.
