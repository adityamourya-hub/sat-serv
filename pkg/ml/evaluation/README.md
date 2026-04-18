# ML Evaluation & Benchmark Suite (`pkg/ml/evaluation`)

This package provides a focused evaluation harness for hybrid SSA pipelines:

- Latency distributions (avg/p50/p95/p99/min/max)
- Accuracy and false-positive rate for binary tasks
- Precision/recall and confusion counts
- SHAP fidelity against classical baseline attributions

## Main APIs

- `BinaryMetrics(truth, pred)`
- `EvaluateManeuverAccuracy(orchestrator, cases)`
- `EvaluateCollisionAccuracy(orchestrator, cases, threshold)`
- `EvaluateRapidAccuracy(orchestrator, cases, maxDistance, highScoreThreshold)`
- `MeasureLatency(iterations, fn)`
- `SHAPFidelityAgainstBaseline(model, baseline, topK)`

## Benchmarks

Run with:

```bash
go test ./pkg/ml/evaluation -bench . -benchmem
```

Included benchmarks:
- Maneuver orchestration latency
- Collision orchestration latency
- Rapid-threat orchestration latency

## Purpose

This suite is intended for continuous model-vs-classical tracking in ADITI demo runs, and for release gates on:
- Max acceptable false positive rate
- Minimum classification accuracy
- Minimum SHAP fidelity against classical reasoning
- Max latency budget
