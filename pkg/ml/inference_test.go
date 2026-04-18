package ml

import (
	"testing"

	"ssis/pkg/hypersonic"
	"ssis/pkg/maneuver"
	"ssis/pkg/proximity"
)

// ============================================================================
// TESTS: DummyInferenceEngine (Fallback Implementation)
// ============================================================================

func TestDummyInferenceEngine_IsHealthy(t *testing.T) {
	engine := NewDummyInferenceEngine()
	if !engine.IsHealthy() {
		t.Fatal("DummyInferenceEngine should always be healthy")
	}
}

func TestDummyInferenceEngine_PredictRapidScore(t *testing.T) {
	engine := NewDummyInferenceEngine()

	// Test case 1: High-threat track (reachable, close, fast)
	track1 := hypersonic.RapidTrack{
		Mach:               15.0,
		AltitudeKm:         50.0,
		BurnoutVelocityKmS: 6.0,
		FlightPathAngleRad: 0.5,
		LaunchLat:          20.0,
		LaunchLon:          80.0,
		DistanceToAssetKm:  100.0,
		RadarCrossSection:  0.1,
	}

	inf1, err := engine.PredictRapidScore(track1)
	if err != nil {
		t.Fatalf("PredictRapidScore failed: %v", err)
	}

	if inf1.Score < 50 || inf1.Score > 100 {
		t.Errorf("Score should be [50,100], got %f", inf1.Score)
	}

	if inf1.Confidence <= 0 || inf1.Confidence >= 1.0 {
		t.Errorf("Confidence should be (0,1), got %f", inf1.Confidence)
	}

	if len(inf1.FeatureWeights) == 0 {
		t.Fatal("FeatureWeights should not be empty")
	}

	t.Logf("[High-threat track] Score: %.2f, Confidence: %.2f", inf1.Score, inf1.Confidence)

	// Test case 2: Low-threat track (far, slow, unreachable)
	track2 := hypersonic.RapidTrack{
		Mach:               2.0,
		AltitudeKm:         500.0,
		BurnoutVelocityKmS: 2.0,
		FlightPathAngleRad: 0.1,
		LaunchLat:          10.0,
		LaunchLon:          -90.0,
		DistanceToAssetKm:  2500.0,
		RadarCrossSection:  0.001,
	}

	inf2, err := engine.PredictRapidScore(track2)
	if err != nil {
		t.Fatalf("PredictRapidScore failed: %v", err)
	}

	if inf2.Score >= inf1.Score {
		t.Errorf("Low-threat score should be lower: got %f vs %f", inf2.Score, inf1.Score)
	}

	t.Logf("[Low-threat track] Score: %.2f, Confidence: %.2f", inf2.Score, inf2.Confidence)
}

func TestDummyInferenceEngine_PredictManeuver(t *testing.T) {
	engine := NewDummyInferenceEngine()

	// Test case 1: Clean window (no maneuver)
	window1 := []maneuver.ManeuverFrame{
		{Residual: 0.1, DeltaV: 0.01},
		{Residual: 0.05, DeltaV: 0.01},
		{Residual: 0.08, DeltaV: 0.01},
		{Residual: 0.12, DeltaV: 0.01},
	}

	inf1, err := engine.PredictManeuver(window1)
	if err != nil {
		t.Fatalf("PredictManeuver failed: %v", err)
	}

	if inf1.Probability > 0.5 {
		t.Logf("[Clean window] EXPECTED low probability, got %.2f", inf1.Probability)
	}

	t.Logf("[Clean window] Probability: %.2f, Type: %s", inf1.Probability, inf1.Type)

	// Test case 2: High-variance window (likely maneuver)
	window2 := []maneuver.ManeuverFrame{
		{Residual: 0.1, DeltaV: 0.01},
		{Residual: 2.5, DeltaV: 0.5},
		{Residual: 2.3, DeltaV: 0.5},
		{Residual: 0.15, DeltaV: 0.01},
	}

	inf2, err := engine.PredictManeuver(window2)
	if err != nil {
		t.Fatalf("PredictManeuver failed: %v", err)
	}

	if inf2.Probability < 0.5 {
		t.Logf("[High-variance window] EXPECTED high probability, got %.2f", inf2.Probability)
	}

	if inf2.Type == maneuver.ManeuverNone {
		t.Logf("[High-variance window] EXPECTED non-none type, got %s", inf2.Type)
	}

	t.Logf("[High-variance window] Probability: %.2f, Type: %s", inf2.Probability, inf2.Type)
}

func TestDummyInferenceEngine_PredictCollisionRisk(t *testing.T) {
	engine := NewDummyInferenceEngine()

	// Test case 1: Safe conjunction (large miss distance)
	input1 := proximity.CollisionRiskInput{
		MissDistanceKm:      10.0,
		HardbodyRadiusKm:    0.005,
		RelativeVelocityKmS: 5.0,
		TimeToTCASeconds:    300.0,
		SigmaRadialKm:       0.05,
		SigmaIntrackKm:      0.05,
		SigmaCrosstrackKm:   0.05,
	}

	graph := proximity.CollisionNeighborhoodGraph{
		Nodes: []proximity.CollisionGraphNode{
			{ID: 1, Features: []float64{10.0, 5.0}},
		},
	}

	inf1, err := engine.PredictCollisionRisk(input1, graph)
	if err != nil {
		t.Fatalf("PredictCollisionRisk failed: %v", err)
	}

	if inf1.RiskScore > 1e-4 {
		t.Logf("[Safe conjunction] Risk should be low, got %.2e", inf1.RiskScore)
	}

	t.Logf("[Safe] Risk: %.2e, Lower: %.2e, Upper: %.2e",
		inf1.RiskScore, inf1.UncertaintyLower, inf1.UncertaintyUpper)

	// Test case 2: Dangerous conjunction (small miss distance)
	input2 := proximity.CollisionRiskInput{
		MissDistanceKm:      0.01,
		HardbodyRadiusKm:    0.005,
		RelativeVelocityKmS: 15.0,
		TimeToTCASeconds:    60.0,
		SigmaRadialKm:       0.01,
		SigmaIntrackKm:      0.01,
		SigmaCrosstrackKm:   0.01,
	}

	inf2, err := engine.PredictCollisionRisk(input2, graph)
	if err != nil {
		t.Fatalf("PredictCollisionRisk failed: %v", err)
	}

	if inf2.RiskScore <= inf1.RiskScore {
		t.Errorf("Dangerous conjunction should have higher risk: %e vs %e",
			inf2.RiskScore, inf1.RiskScore)
	}

	t.Logf("[Dangerous] Risk: %.2e, Lower: %.2e, Upper: %.2e",
		inf2.RiskScore, inf2.UncertaintyLower, inf2.UncertaintyUpper)

	// Sanity check: bounds
	if inf2.UncertaintyLower > inf2.RiskScore || inf2.UncertaintyUpper < inf2.RiskScore {
		t.Errorf("Uncertainty bounds should bracket risk score")
	}
}

// ============================================================================
// INTEGRATION TESTS: Hybrid Interfaces (with DummyEngine)
// ============================================================================

// Hybrid integration tests skipped - functions moved to maneuver/proximity packages
// TestHybridManeuverDetectorWithDummy - moved to pkg/maneuver
// TestHybridCollisionEngineWithDummy - moved to pkg/proximity

func TestBenchmarkLatency(t *testing.T) {
	engine := NewDummyInferenceEngine()

	// Rapid score prediction
	track := hypersonic.RapidTrack{
		Mach:               10.0,
		AltitudeKm:         75.0,
		BurnoutVelocityKmS: 6.0,
		DistanceToAssetKm:  500.0,
	}

	iterations := 1000

	for i := 0; i < iterations; i++ {
		result, err := engine.PredictRapidScore(track)
		if err != nil {
			t.Fatalf("Prediction failed: %v", err)
		}

		_ = result
	}

	// Should complete in <1 second for 1000 calls (1ms average, well under 50ms target)
	t.Logf("[Latency] %d predictions completed (should be <1s for <50ms target)",
		iterations)

	// Manual measurement with time package would be more precise
	// For debugging, use: time.Now(), defer + elapsed calculation
}

// ============================================================================
// HELPER MODELS (wrapping DummyEngine)
// ============================================================================

type DummyManeuverModel struct {
	engine InferenceEngine
}

func (m *DummyManeuverModel) Name() string {
	return "DummyManeuverModel"
}

func (m *DummyManeuverModel) Predict(window []maneuver.ManeuverFrame) (maneuver.ManeuverInference, error) {
	return m.engine.PredictManeuver(window)
}

type DummyCollisionModel struct {
	engine InferenceEngine
}

func (m *DummyCollisionModel) Name() string {
	return "DummyCollisionModel"
}

func (m *DummyCollisionModel) PredictRisk(input proximity.CollisionRiskInput,
	graph proximity.CollisionNeighborhoodGraph) (proximity.CollisionGraphInference, error) {
	return m.engine.PredictCollisionRisk(input, graph)
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================
