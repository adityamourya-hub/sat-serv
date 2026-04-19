package proximity_test

import (
	"fmt"
	"testing"

	"ssis/pkg/ml/hybrid"
	"ssis/pkg/physics"
	"ssis/pkg/proximity"
)

type mockCollisionModel struct {
	risk float64
	low  float64
	high float64
	conf float64
	err  error
}

func (m mockCollisionModel) Name() string { return "mock_gatv2_surrogate" }

func (m mockCollisionModel) PredictCollisionRisk(input physics.CollisionRiskInput, graph physics.CollisionNeighborhoodGraph) (physics.CollisionGraphInference, error) {
	if m.err != nil {
		return physics.CollisionGraphInference{}, m.err
	}
	attn := map[int]float64{}
	for _, n := range graph.Nodes {
		attn[n.ObjectID] = 0.5
	}
	return physics.CollisionGraphInference{
		RiskScore:        m.risk,
		UncertaintyLower: m.low,
		UncertaintyUpper: m.high,
		Confidence:       m.conf,
		NodeAttention:    attn,
	}, nil
}

func (m mockCollisionModel) PredictRapidScore(track physics.RapidTrack) (physics.RapidRankInference, error) {
	return physics.RapidRankInference{}, nil
}

func (m mockCollisionModel) PredictManeuver(window []physics.ManeuverFrame) (physics.ManeuverInference, error) {
	return physics.ManeuverInference{}, nil
}

func (m mockCollisionModel) IsHealthy() bool { return true }

func TestEvaluateCollisionRiskStages(t *testing.T) {
	lowRisk := proximity.EvaluateCollisionRisk(physics.CollisionRiskInput{
		MissDistanceKm:      200,
		HardbodyRadiusKm:    0.01,
		SigmaRadialKm:       0.1,
		SigmaIntrackKm:      0.1,
		SigmaCrosstrackKm:   0.1,
		RelativeVelocityKmS: 8,
		TimeToTCASeconds:    600,
	})

	if lowRisk.Stage != "alfano_bound" {
		t.Fatalf("Expected low risk to short-circuit at alfano bound, got %s", lowRisk.Stage)
	}

	highRisk := proximity.EvaluateCollisionRisk(physics.CollisionRiskInput{
		MissDistanceKm:      0.01,
		HardbodyRadiusKm:    0.01,
		SigmaRadialKm:       0.005,
		SigmaIntrackKm:      0.005,
		SigmaCrosstrackKm:   0.005,
		RelativeVelocityKmS: 1.5,
		TimeToTCASeconds:    60,
	})

	if highRisk.Probability <= 0 {
		t.Fatalf("Expected non-zero collision probability")
	}
}

func TestEvaluateCollisionRiskDeterministic(t *testing.T) {
	input := physics.CollisionRiskInput{
		MissDistanceKm:      0.03,
		HardbodyRadiusKm:    0.01,
		SigmaRadialKm:       0.01,
		SigmaIntrackKm:      0.01,
		SigmaCrosstrackKm:   0.01,
		RelativeVelocityKmS: 2.0,
		TimeToTCASeconds:    200,
	}

	a := proximity.EvaluateCollisionRisk(input)
	b := proximity.EvaluateCollisionRisk(input)
	if a.Probability != b.Probability || a.Stage != b.Stage {
		t.Fatalf("Expected deterministic output, got (%v, %s) and (%v, %s)", a.Probability, a.Stage, b.Probability, b.Stage)
	}
}

func TestEvaluateCollisionRiskHybridFallback(t *testing.T) {
	input := physics.CollisionRiskInput{
		MissDistanceKm:      0.05,
		HardbodyRadiusKm:    0.01,
		SigmaRadialKm:       0.01,
		SigmaIntrackKm:      0.01,
		SigmaCrosstrackKm:   0.01,
		RelativeVelocityKmS: 2,
		TimeToTCASeconds:    120,
	}

	graph := physics.CollisionNeighborhoodGraph{
		Nodes: []physics.CollisionGraphNode{{ObjectID: 1, Features: []float64{0.05, 2.0}}},
	}

	orch := hybrid.NewOrchestrator(nil)
	res, _ := orch.EvaluateCollision(input, graph)
	if res.Probability <= 0 {
		t.Fatalf("Expected fallback probability > 0")
	}
	if res.UncertaintyUpper < res.UncertaintyLower {
		t.Fatalf("Expected valid uncertainty bounds")
	}
}

func TestEvaluateCollisionRiskHybridModelBlend(t *testing.T) {
	input := physics.CollisionRiskInput{
		MissDistanceKm:      0.03,
		HardbodyRadiusKm:    0.01,
		SigmaRadialKm:       0.008,
		SigmaIntrackKm:      0.009,
		SigmaCrosstrackKm:   0.01,
		RelativeVelocityKmS: 1.3,
		TimeToTCASeconds:    240,
	}

	graph := physics.CollisionNeighborhoodGraph{
		Nodes: []physics.CollisionGraphNode{
			{ObjectID: 10, Features: []float64{0.03, 1.3, 0.008}},
			{ObjectID: 11, Features: []float64{0.04, 1.0, 0.009}},
		},
		Edges: []physics.CollisionGraphEdge{{SourceID: 10, TargetID: 11, Weight: 0.7}},
	}

	model := mockCollisionModel{
		risk: 0.004,
		low:  0.002,
		high: 0.006,
		conf: 0.9,
	}

	orch := hybrid.NewOrchestrator(model)
	orch.Thresholds.CollisionConfidence = 0.5
	orch.Thresholds.MaxCollisionDeviation = 2.0

	res, dec := orch.EvaluateCollision(input, graph)
	if res.Stage == "" {
		t.Fatalf("Expected non-empty stage")
	}
	if dec.Source != "ml" {
		fmt.Printf("DEBUG: dec.Reason=%s\n", dec.Reason)
		t.Fatalf("Expected ML source, got %s", dec.Source)
	}
	if res.UncertaintyUpper < res.UncertaintyLower {
		t.Fatalf("Expected valid uncertainty bounds")
	}
	if len(res.Heatmap) != len(graph.Nodes) {
		t.Fatalf("Expected heatmap for all graph nodes")
	}
}
