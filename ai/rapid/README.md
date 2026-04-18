# AI Rapid-Threat Priority Ranker (Feature 3)

Implements ADITI Feature 3 for `pkg/physics/rapid.go`:
- ML ranker: LightGBM/XGBoost-style dynamic threat scoring
- Optional RL policy: PPO action recommendation
- Reachability-gated recommendations for defensive actioning
- ONNX adapter path for Go runtime + LightGBM C API path

## Go Integration

- Hybrid threat engine: `engine/pkg/physics/rapid.go`
- ONNX stubs/adapters:
  - `engine/pkg/ml/onnx/rapid_ranker_stub.go`
  - `engine/pkg/ml/onnx/rapid_ranker_onnx.go`
- LightGBM C API stubs/adapters:
  - `engine/pkg/ml/lightgbm/rapid_ranker_stub.go`
  - `engine/pkg/ml/lightgbm/rapid_ranker_capi.go`

## Outputs in RapidAssessment

- `DynamicScore`
- `ScoreConfidence`
- `UncertaintyLow` / `UncertaintyHigh`
- `Recommendation` (action + reason)
- `FeatureHeatmap`

## Train XGBoost Ranker and Export ONNX

python3 -m venv .venv
source .venv/bin/activate
pip install -r ai/rapid/requirements.txt
python ai/rapid/train_ranker_xgboost.py

Artifacts:
- `ai/rapid/data/rapid_ranker.onnx`
- `ai/rapid/data/ranker_metrics.json`

## Train LightGBM Ranker for C API Runtime

python ai/rapid/train_ranker_lightgbm.py

Artifacts:
- `ai/rapid/data/rapid_ranker_lightgbm.txt`
- `ai/rapid/data/rapid_ranker_lightgbm_metrics.json`

### Build with LightGBM C API

go build -tags lightgbm ./...

Note: `rapid_ranker_capi.go` includes scaffolded cgo hooks for LightGBM C API.
Bind exact `LGBM_BoosterCreateFromModelfile` / prediction symbols in deployment image where the native SDK version is pinned.

## Optional PPO Policy Training

python ai/rapid/train_policy_ppo.py

Artifacts:
- `ai/rapid/data/ppo_policy.zip`
- `ai/rapid/data/policy_notes.json`

## Runtime Hybrid Strategy

1. Compute deterministic baseline priority (existing physics logic).
2. Run ML ranker if configured.
3. Apply confidence + deviation guardrails.
4. Blend baseline and ML score when accepted.
5. Generate recommendation via PPO policy if available, else deterministic rules.
6. Enforce reachability gate before escalation actions.

This keeps operational safety deterministic while adding cognitive ranking and action recommendation.
