package ml

// This file documents the ML architecture integration points
// PHASE 2 IMPLEMENTATIONS:
// - pkg/ml/inference.go: DummyInferenceEngine + ONNX wrapper interface
// - [Future] pkg/ml/onnx_ranker.go: ONNX-based rapid threat ranker
// - [Future] pkg/ml/onnx_maneuver.go: ONNX-based maneuver detector
// - [Future] pkg/ml/onnx_collision.go: ONNX-based collision risk model
//
// INTEGRATION POINTS (already in place):
// - pkg/orbital/maneuver.go: HybridManeuverDetector interface + DetectManeuversHybrid()
// - pkg/orbital/collision.go: HybridCollisionEngine interface + EvaluateCollisionRiskHybrid()
// - pkg/physics/rapid.go: HybridRapidThreatEngine interface + ProcessRapidTracksHybrid()
//
// USAGE PATTERN:
//   engine, _ := ml.NewInferenceEngine("./models")
//   maneuverInf, _ := engine.PredictManeuver(window)
//   collisionInf, _ := engine.PredictCollisionRisk(input, graph)
//   rapidInf, _ := engine.PredictRapidScore(track)
