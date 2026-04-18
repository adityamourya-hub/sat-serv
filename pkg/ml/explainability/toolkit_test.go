package explainability

import (
	"math"
	"testing"

	"ssis/pkg/hypersonic"
	"ssis/pkg/maneuver"
	"ssis/pkg/proximity"
)

func TestBuildAttentionFromWindowNormalizes(t *testing.T) {
	window := []maneuver.ManeuverFrame{
		{Residual: 0.1, DeltaV: 0.0},
		{Residual: 1.0, DeltaV: 0.2},
		{Residual: 0.3, DeltaV: 0.0},
	}

	att := BuildAttentionFromWindow(window)
	if len(att) != len(window) {
		t.Fatalf("expected %d points, got %d", len(window), len(att))
	}

	sum := 0.0
	for _, p := range att {
		sum += p.Weight
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Fatalf("attention must sum to 1, got %.12f", sum)
	}
	if att[1].Weight <= att[0].Weight {
		t.Fatalf("expected higher residual point to get larger attention")
	}
}

func TestSummarizeUncertainty(t *testing.T) {
	u := SummarizeUncertainty(0.4, 0.3, 0.5, 0.8)
	if u.Width <= 0 {
		t.Fatalf("expected positive width")
	}
	if u.RelativeWidth <= 0 {
		t.Fatalf("expected positive relative width")
	}
	if u.Confidence != 0.8 {
		t.Fatalf("expected confidence passthrough")
	}
}

func TestExplainRapid(t *testing.T) {
	track := hypersonic.RapidTrack{
		Mach:               12,
		AltitudeKm:         70,
		BurnoutVelocityKmS: 6,
		DistanceToAssetKm:  350,
		RadarCrossSection:  0.2,
	}
	inf := hypersonic.RapidRankInference{
		Score:      82,
		Confidence: 0.65,
	}

	ex := ExplainRapid(track, inf)
	if len(ex.TopFeatures) == 0 {
		t.Fatalf("expected at least one feature attribution")
	}
	if ex.Uncertainty.Width <= 0 {
		t.Fatalf("expected non-zero uncertainty width")
	}
}

func TestExplainManeuverAndCollision(t *testing.T) {
	window := []maneuver.ManeuverFrame{
		{Residual: 0.02, DeltaV: 0.001},
		{Residual: 0.80, DeltaV: 0.050},
		{Residual: 0.10, DeltaV: 0.002},
	}
	manInf := maneuver.ManeuverInference{
		Probability: 0.72,
		Confidence:  0.63,
	}
	man := ExplainManeuver(window, manInf)
	if len(man.Attention) != len(window) {
		t.Fatalf("attention length mismatch")
	}
	if len(man.SHAP) == 0 {
		t.Fatalf("expected SHAP fallback to be generated")
	}

	graph := proximity.CollisionNeighborhoodGraph{
		Nodes: []proximity.CollisionGraphNode{{ID: 10}, {ID: 11}},
	}
	colInf := proximity.CollisionGraphInference{
		RiskScore:        0.01,
		UncertaintyLower: 0.008,
		UncertaintyUpper: 0.02,
		Confidence:       0.7,
		NodeAttention: map[int]float64{
			10: 0.7,
			11: 0.3,
		},
	}
	col := ExplainCollision(graph, colInf)
	if len(col.Heatmap) != 2 {
		t.Fatalf("expected 2 heatmap cells")
	}
	if col.Heatmap[0].NodeID != 10 {
		t.Fatalf("expected highest attention first")
	}
}
