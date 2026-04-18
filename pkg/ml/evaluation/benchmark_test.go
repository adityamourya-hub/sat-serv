package evaluation

import (
	"testing"

	"ssis/pkg/hypersonic"
	"ssis/pkg/ml"
	"ssis/pkg/ml/hybrid"
	"ssis/pkg/proximity"
)

func BenchmarkDetectManeuversLatency(b *testing.B) {
	orch := hybrid.NewOrchestrator(ml.NewDummyInferenceEngine())
	residuals := make([]float64, 256)
	deltaVs := make([]float64, 256)
	for i := 80; i < 88; i++ {
		residuals[i] = 2.8
		deltaVs[i] = 0.11
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = orch.DetectManeuvers(1001, residuals, deltaVs)
	}
}

func BenchmarkEvaluateCollisionLatency(b *testing.B) {
	orch := hybrid.NewOrchestrator(ml.NewDummyInferenceEngine())
	input := proximity.CollisionRiskInput{
		MissDistanceKm:      0.08,
		HardbodyRadiusKm:    0.005,
		RelativeVelocityKmS: 9,
		TimeToTCASeconds:    180,
		SigmaRadialKm:       0.03,
		SigmaIntrackKm:      0.03,
		SigmaCrosstrackKm:   0.03,
	}
	graph := proximity.CollisionNeighborhoodGraph{
		Nodes: []proximity.CollisionGraphNode{{ID: 1}, {ID: 2}, {ID: 3}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = orch.EvaluateCollision(input, graph)
	}
}

func BenchmarkRankRapidLatency(b *testing.B) {
	orch := hybrid.NewOrchestrator(ml.NewDummyInferenceEngine())
	tracks := []hypersonic.RapidTrack{
		{ID: 1, Mach: 12, AltitudeKm: 55, BurnoutVelocityKmS: 6, DistanceToAssetKm: 200, RadarCrossSection: 0.1},
		{ID: 2, Mach: 7, AltitudeKm: 85, BurnoutVelocityKmS: 4, DistanceToAssetKm: 500, RadarCrossSection: 0.3},
		{ID: 3, Mach: 3, AltitudeKm: 450, BurnoutVelocityKmS: 2, DistanceToAssetKm: 2500, RadarCrossSection: 1.2},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = orch.RankRapid(tracks, 3000, 3)
	}
}
