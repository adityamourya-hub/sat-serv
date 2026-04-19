package evaluation

import (
	"testing"

	"ssis/pkg/ml"
	"ssis/pkg/ml/hybrid"
	"ssis/pkg/physics"
)

func TestBinaryMetrics(t *testing.T) {
	truth := []bool{true, false, true, false}
	pred := []bool{true, true, false, false}
	m, err := BinaryMetrics(truth, pred)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m.TP != 1 || m.FP != 1 || m.TN != 1 || m.FN != 1 {
		t.Fatalf("confusion counts mismatch: %+v", m)
	}
	if m.FalsePositiveRate <= 0 {
		t.Fatalf("expected positive FPR")
	}
}

func TestSHAPFidelityAgainstBaseline(t *testing.T) {
	baseline := []SHAPFeature{{Feature: "distance", Contribution: 0.7}, {Feature: "mach", Contribution: 0.3}}
	goodModel := []SHAPFeature{{Feature: "distance", Contribution: 0.6}, {Feature: "mach", Contribution: 0.4}}
	badModel := []SHAPFeature{{Feature: "altitude", Contribution: 1.0}}

	g := SHAPFidelityAgainstBaseline(goodModel, baseline, 5)
	b := SHAPFidelityAgainstBaseline(badModel, baseline, 5)
	if g.CombinedScore <= b.CombinedScore {
		t.Fatalf("expected good model fidelity > bad model fidelity: good=%.4f bad=%.4f", g.CombinedScore, b.CombinedScore)
	}
}

func TestEvaluateSuiteWithHybridOrchestrator(t *testing.T) {
	orch := hybrid.NewOrchestrator(ml.NewDummyInferenceEngine())

	maneuverCases := []ManeuverCase{
		{ObjectID: 1, Residuals: []float64{0.02, 0.02, 3.0, 3.0, 0.02}, DeltaVs: []float64{0.01, 0.01, 0.12, 0.12, 0.01}, HasManeuver: true},
		{ObjectID: 2, Residuals: []float64{0.01, 0.01, 0.01, 0.01, 0.01}, DeltaVs: []float64{0.001, 0.001, 0.001, 0.001, 0.001}, HasManeuver: false},
	}
	mMet, err := EvaluateManeuverAccuracy(orch, maneuverCases)
	if err != nil {
		t.Fatalf("maneuver evaluation failed: %v", err)
	}
	if mMet.Accuracy < 0 || mMet.Accuracy > 1 {
		t.Fatalf("invalid maneuver accuracy: %f", mMet.Accuracy)
	}

	collisionCases := []CollisionCase{
		{Input: physics.CollisionRiskInput{MissDistanceKm: 0.01, HardbodyRadiusKm: 0.005, RelativeVelocityKmS: 8, SigmaRadialKm: 0.01, SigmaIntrackKm: 0.01, SigmaCrosstrackKm: 0.01}, Graph: physics.CollisionNeighborhoodGraph{Nodes: []physics.CollisionGraphNode{{ObjectID: 1}}}, IsCollision: true},
		{Input: physics.CollisionRiskInput{MissDistanceKm: 10, HardbodyRadiusKm: 0.005, RelativeVelocityKmS: 5, SigmaRadialKm: 0.05, SigmaIntrackKm: 0.05, SigmaCrosstrackKm: 0.05}, Graph: physics.CollisionNeighborhoodGraph{Nodes: []physics.CollisionGraphNode{{ObjectID: 2}}}, IsCollision: false},
	}
	cMet, err := EvaluateCollisionAccuracy(orch, collisionCases, 1e-4)
	if err != nil {
		t.Fatalf("collision evaluation failed: %v", err)
	}
	if cMet.FalsePositiveRate < 0 || cMet.FalsePositiveRate > 1 {
		t.Fatalf("invalid collision FPR: %f", cMet.FalsePositiveRate)
	}

	rapidCases := []RapidCase{
		{Track: physics.RapidTrack{ID: 11, Mach: 12, AltitudeKm: 60, BurnoutVelocityKmS: 6, DistanceToAssetKm: 200, RadarCrossSection: 0.1}, IsHighThreat: true},
		{Track: physics.RapidTrack{ID: 12, Mach: 2, AltitudeKm: 500, BurnoutVelocityKmS: 2, DistanceToAssetKm: 2500, RadarCrossSection: 1.0}, IsHighThreat: false},
	}
	rMet, err := EvaluateRapidAccuracy(orch, rapidCases, 3000, 100)
	if err != nil {
		t.Fatalf("rapid evaluation failed: %v", err)
	}
	if rMet.Accuracy < 0 || rMet.Accuracy > 1 {
		t.Fatalf("invalid rapid accuracy: %f", rMet.Accuracy)
	}
}

func TestMeasureLatency(t *testing.T) {
	lat := MeasureLatency(10, func() {
		_ = 1 + 1
	})
	if lat.Iterations != 10 {
		t.Fatalf("expected 10 iterations, got %d", lat.Iterations)
	}
	if lat.P99Ms < lat.P50Ms {
		t.Fatalf("expected p99 >= p50")
	}
}
