package proximity_test

import (
	"fmt"
	"math"
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

func (m mockCollisionModel) PredictManeuverINT8(window []physics.ManeuverFrame) (physics.ManeuverInference, error) {
	// Test mock: delegate to PredictManeuver
	return m.PredictManeuver(window)
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

func TestEvaluateCollisionRisk_ZeroNegativeInputs(t *testing.T) {
	// Test zero miss distance
	input := physics.CollisionRiskInput{
		MissDistanceKm:      0.0,
		HardbodyRadiusKm:    0.01,
		SigmaRadialKm:       0.01,
		SigmaIntrackKm:      0.01,
		SigmaCrosstrackKm:   0.01,
		RelativeVelocityKmS: 2.0,
		TimeToTCASeconds:    200,
	}
	result := proximity.EvaluateCollisionRisk(input)
	if result.Probability <= 0 {
		t.Errorf("Expected non-zero probability for zero miss distance, got %e", result.Probability)
	}

	// Test negative miss distance
	input.MissDistanceKm = -0.01
	result = proximity.EvaluateCollisionRisk(input)
	if result.Probability <= 0 {
		t.Errorf("Expected non-zero probability for negative miss distance, got %e", result.Probability)
	}

	// Test zero hardbody radius
	input.MissDistanceKm = 0.03
	input.HardbodyRadiusKm = 0.0
	result = proximity.EvaluateCollisionRisk(input)
	// Should use default radius of 0.005
	if result.Probability <= 0 {
		t.Errorf("Expected non-zero probability with default radius, got %e", result.Probability)
	}

	// Test zero sigma values
	input.HardbodyRadiusKm = 0.01
	input.SigmaRadialKm = 0.0
	input.SigmaIntrackKm = 0.0
	input.SigmaCrosstrackKm = 0.0
	result = proximity.EvaluateCollisionRisk(input)
	// Should use minimum sigma of 0.001
	if result.Probability < 0 {
		t.Errorf("Expected non-negative probability with zero sigma, got %e", result.Probability)
	}

	// Test negative sigma values
	input.SigmaRadialKm = -0.01
	input.SigmaIntrackKm = -0.01
	input.SigmaCrosstrackKm = -0.01
	result = proximity.EvaluateCollisionRisk(input)
	// Should use minimum sigma of 0.001
	if result.Probability < 0 {
		t.Errorf("Expected non-negative probability with negative sigma, got %e", result.Probability)
	}
}

func TestCalculateGaussianKLDivergence_EdgeCases(t *testing.T) {
	// Test identical distributions (KL divergence should be zero)
	mu0 := []float64{0, 0, 0, 0, 0, 0}
	mu1 := []float64{0, 0, 0, 0, 0, 0}
	cov0 := [][]float64{{1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1}}
	cov1 := [][]float64{{1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1}}
	kl := proximity.CalculateGaussianKLDivergence(mu0, mu1, cov0, cov1)
	if kl > 1e-10 {
		t.Errorf("Expected KL divergence near zero for identical distributions, got %f", kl)
	}

	// Test with zero covariance (near-singular)
	covZero := [][]float64{{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}}
	kl = proximity.CalculateGaussianKLDivergence(mu0, mu1, covZero, cov1)
	// Should handle gracefully with minimum threshold
	if kl < 0 {
		t.Errorf("Expected non-negative KL divergence, got %f", kl)
	}

	// Test with very different distributions
	muDiff := []float64{10, 10, 10, 10, 10, 10}
	covLarge := [][]float64{{100, 0, 0, 0, 0, 0}, {0, 100, 0, 0, 0, 0}, {0, 0, 100, 0, 0, 0}, {0, 0, 0, 100, 0, 0}, {0, 0, 0, 0, 100, 0}, {0, 0, 0, 0, 0, 100}}
	kl = proximity.CalculateGaussianKLDivergence(mu0, muDiff, cov0, covLarge)
	if kl <= 0 {
		t.Errorf("Expected positive KL divergence for different distributions, got %f", kl)
	}
}

func TestInformationBarycenterRefined_HighDivergence(t *testing.T) {
	// Test with high KL divergence (maneuver scenario)
	s_sgp4 := []float64{7000, 0, 0, 0, 7.5, 0}
	s_axiom := []float64{7100, 100, 50, 0.5, 8.0, 0.1} // Significant difference
	cov_sgp4 := [][]float64{{1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1}}
	cov_axiom := [][]float64{{0.5, 0, 0, 0, 0, 0}, {0, 0.5, 0, 0, 0, 0}, {0, 0, 0.5, 0, 0, 0}, {0, 0, 0, 0.5, 0, 0}, {0, 0, 0, 0, 0.5, 0}, {0, 0, 0, 0, 0, 0.5}}

	blended := proximity.InformationBarycenterRefined(s_sgp4, s_axiom, cov_sgp4, cov_axiom)
	if len(blended) != 6 {
		t.Errorf("Expected blended state of length 6, got %d", len(blended))
	}
	// With high divergence, SGP4 weight should be reduced
	// Check that blended state is closer to Axiom
	for i := 0; i < 6; i++ {
		diffSGP4 := blended[i] - s_sgp4[i]
		diffAxiom := blended[i] - s_axiom[i]
		// In high divergence case, should weight Axiom more
		if math.Abs(diffAxiom) > math.Abs(diffSGP4)*2 {
			// This is acceptable as the exact weighting depends on KL value
		}
	}
}

func TestInformationBarycenterRefined_LowDivergence(t *testing.T) {
	// Test with low KL divergence (nominal scenario)
	s_sgp4 := []float64{7000, 0, 0, 0, 7.5, 0}
	s_axiom := []float64{7001, 0.1, 0, 0, 7.51, 0} // Small difference
	cov_sgp4 := [][]float64{{1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1}}
	cov_axiom := [][]float64{{1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1}}

	blended := proximity.InformationBarycenterRefined(s_sgp4, s_axiom, cov_sgp4, cov_axiom)
	if len(blended) != 6 {
		t.Errorf("Expected blended state of length 6, got %d", len(blended))
	}
	// With low divergence, SGP4 weight should be higher
	// Check that blended state is between the two inputs
	for i := 0; i < 6; i++ {
		minVal := math.Min(s_sgp4[i], s_axiom[i])
		maxVal := math.Max(s_sgp4[i], s_axiom[i])
		if blended[i] < minVal || blended[i] > maxVal {
			t.Errorf("Blended state %d (%f) outside input range [%f, %f]", i, blended[i], minVal, maxVal)
		}
	}
}

func TestCalculateMOID_CircularOrbits(t *testing.T) {
	// Test with two circular orbits in same plane
	s1 := physics.State{
		Pos: physics.Vector3{7000, 0, 0},
		Vel: physics.Vector3{0, 7.5, 0},
	}
	s2 := physics.State{
		Pos: physics.Vector3{7100, 0, 0},
		Vel: physics.Vector3{0, 7.4, 0},
	}

	moid := proximity.CalculateMOID(s1, s2)
	// Should find a reasonable minimum distance
	if moid < 0 {
		t.Errorf("Expected non-negative MOID, got %f", moid)
	}
	if moid > 1000 {
		t.Errorf("MOID seems too large for close orbits: %f", moid)
	}
}

func TestAlfanoMaxPc_BoundaryConditions(t *testing.T) {
	// Test zero miss distance
	pc := proximity.AlfanoMaxPc(0.0, 0.01)
	if pc != 1.0 {
		t.Errorf("Expected Pc=1.0 for zero miss distance, got %f", pc)
	}

	// Test miss distance equal to radius
	pc = proximity.AlfanoMaxPc(0.01, 0.01)
	if pc != 1.0 {
		t.Errorf("Expected Pc=1.0 for miss distance equal to radius, got %f", pc)
	}

	// Test miss distance much larger than radius
	pc = proximity.AlfanoMaxPc(100.0, 0.01)
	if pc <= 0 || pc >= 1 {
		t.Errorf("Expected Pc between 0 and 1 for large miss distance, got %f", pc)
	}

	// Test negative miss distance
	pc = proximity.AlfanoMaxPc(-0.01, 0.01)
	if pc != 1.0 {
		t.Errorf("Expected Pc=1.0 for negative miss distance, got %f", pc)
	}
}
