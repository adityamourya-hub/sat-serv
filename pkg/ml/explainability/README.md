# Explainability Toolkit (`pkg/ml/explainability`)

This package provides model-agnostic explainability outputs for all SSA ML models:

- SHAP-like feature attribution for threat ranking
- Sequence attention visualization for maneuver windows
- Uncertainty quantification summary from prediction intervals
- Collision node heatmap generation from graph attention

## APIs

- `ExplainRapid(track, inference)`
  - Returns top feature attributions + uncertainty summary.
- `ExplainManeuver(window, inf)`
  - Returns SHAP contributions, temporal attention points, uncertainty.
- `ExplainCollision(graph, inf)`
  - Returns node heatmap and uncertainty summary.
- `BuildAttentionFromWindow(window)`
  - Creates normalized attention weights for each timestep.
- `SummarizeUncertainty(mean, lower, upper, confidence)`
  - Computes interval width and relative uncertainty.

## Example

```go
rapidExp := explainability.ExplainRapid(track, rankInf)
manExp := explainability.ExplainManeuver(window, manInf)
colExp := explainability.ExplainCollision(graph, colInf)
```

All outputs are JSON-serializable and intended to be attached to signed SSA alerts.
