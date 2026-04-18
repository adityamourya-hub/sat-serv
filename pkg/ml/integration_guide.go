package ml

// Integration notes for ML backends:
// 1) Initialize via NewInferenceEngine(modelDir).
// 2) Use existing hybrid wrappers:
//    - orbital.NewHybridManeuverDetector
//    - orbital.NewHybridCollisionEngine
//    - physics.NewHybridRapidThreatEngine
// 3) Attach explainability payloads from pkg/ml/explainability to signed alerts.
// 4) Keep confidence thresholds conservative so deterministic fallbacks remain primary guardrails.
