# Orbit Zoo Export and Policy Ingestion Interface

This specification executes Phase D from OPEN_SOURCE_COMSPOC_ALIGNMENT.md.

## Scope

- Export BOS SSA simulation/state traces to a Gym-compatible trajectory schema.
- Ingest trained policy outputs (action + confidence) into BOS Orbit Zoo decision paths.
- Keep strict fallback behavior when policy artifacts are missing or low-confidence.

## Export Schema (JSONL)

Each line is one time-step transition:

```json
{
  "episode_id": "scenario-001",
  "step": 42,
  "observation": [0.72, 0.31, 0.44, 0.80],
  "action_mask": [1, 1, 1],
  "reward": 0.12,
  "done": false,
  "info": {
    "threat_score": 0.72,
    "fuel_state": 0.31,
    "geometry_risk": 0.44,
    "urgency": 0.80
  }
}
```

## Policy Ingestion Contract

Input features (normalized):
- threat_score
- fuel_state
- geometry_risk
- urgency

Output contract:
- action: `maintain` | `maneuver` | `retask`
- confidence: 0..1
- policy_score: raw decision score (optional)

## BOS Integration Hook

- Use `pkg/ml/inference.PredictOrbitZooAction` as the policy bridge.
- If model unavailable or confidence below guardrail, use deterministic baseline policy.

## Sample Acceptance Criteria

1. Exported trajectories are parseable line-by-line in air-gapped environments.
2. Policy ingestion always returns a valid action.
3. Confidence guardrail and fallback path are auditable.
4. Strict mode never silently drops to an unlogged fallback.
