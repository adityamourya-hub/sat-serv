package hybrid

import (
	"testing"

	"ssis/pkg/hypersonic"
	"ssis/pkg/ml"
	"ssis/pkg/proximity"
)

func TestDetectManeuvers_ClassicalThenFallbackOrML(t *testing.T) {
	orch := NewOrchestrator(ml.NewDummyInferenceEngine())

	residuals := make([]float64, 80)
	deltaVs := make([]float64, 80)
	for i := 0; i < 30; i++ {
		residuals[i] = 0.02
		deltaVs[i] = 0.005
	}
	for i := 30; i < 36; i++ {
		residuals[i] = 2.7
		deltaVs[i] = 0.11
	}
	for i := 36; i < 80; i++ {
		residuals[i] = 0.02
		deltaVs[i] = 0.005
	}

	events, decisions := orch.DetectManeuvers(77, residuals, deltaVs)
	if len(events) == 0 {
		t.Fatalf("expected maneuver events")
	}
	if len(events) != len(decisions) {
		t.Fatalf("events/decisions mismatch: %d vs %d", len(events), len(decisions))
	}

	for _, d := range decisions {
		if d.Source == "" || d.Reason == "" {
			t.Fatalf("decision metadata must be populated")
		}
	}
}

func TestEvaluateCollision_UsesMLOrFallback(t *testing.T) {
	orch := NewOrchestrator(ml.NewDummyInferenceEngine())
	input := proximity.CollisionRiskInput{
		MissDistanceKm:      0.2,
		HardbodyRadiusKm:    0.005,
		RelativeVelocityKmS: 8.0,
		TimeToTCASeconds:    300,
		SigmaRadialKm:       0.03,
		SigmaIntrackKm:      0.03,
		SigmaCrosstrackKm:   0.03,
	}
	graph := proximity.CollisionNeighborhoodGraph{
		Nodes: []proximity.CollisionGraphNode{{ID: 1}, {ID: 2}},
	}

	res, d := orch.EvaluateCollision(input, graph)
	if res.Probability < 0 || res.Probability > 1 {
		t.Fatalf("probability out of range: %f", res.Probability)
	}
	if d.Source == "" {
		t.Fatalf("source must be set")
	}
}

func TestRankRapid_ClassicalBaselineThenML(t *testing.T) {
	orch := NewOrchestrator(ml.NewDummyInferenceEngine())
	tracks := []hypersonic.RapidTrack{
		{ID: 1, Mach: 12, AltitudeKm: 55, BurnoutVelocityKmS: 6, DistanceToAssetKm: 200, RadarCrossSection: 0.1},
		{ID: 2, Mach: 3, AltitudeKm: 500, BurnoutVelocityKmS: 2, DistanceToAssetKm: 2500, RadarCrossSection: 1.2},
	}

	assessments, decisions := orch.RankRapid(tracks, 3000, 10)
	if len(assessments) == 0 {
		t.Fatalf("expected at least one assessment")
	}
	if len(decisions) != len(assessments) {
		t.Fatalf("decisions mismatch assessments")
	}
	for _, a := range assessments {
		if a.DynamicScore <= 0 {
			t.Fatalf("dynamic score should be > 0")
		}
	}
}
