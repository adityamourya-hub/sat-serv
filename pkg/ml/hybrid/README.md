# Hybrid Orchestration (`pkg/ml/hybrid`)

This package centralizes `classical -> ML -> fallback` orchestration with explicit threshold guardrails.

## What it does

- Runs deterministic classical SSA logic first.
- Queries ML inference via `pkg/ml.InferenceEngine`.
- Accepts ML only when confidence thresholds and deviation guardrails pass.
- Falls back to classical outputs otherwise.
- Emits decision metadata (`Source`, `Reason`) for auditability.

## APIs

- `NewOrchestrator(engine)`
- `DetectManeuvers(objectID, residuals, deltaVs)`
- `EvaluateCollision(input, graph)`
- `RankRapid(tracks, maxDistanceKm, maxResults)`

## Thresholds

Default thresholds:
- Maneuver probability: `0.60`
- Maneuver confidence: `0.50`
- Collision confidence: `0.55`
- Rapid confidence: `0.55`
- Max collision deviation: `0.35`
- Max rapid deviation pct: `0.60`

These can be overridden via `orchestrator.Thresholds`.
