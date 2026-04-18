package maneuver

import (
	"math"
	"testing"

	"ssis/pkg/orbital"
)

type mockManeuverModel struct {
	prob float64
	conf float64
	typ  ManeuverType
	dv   float64
}

func (m mockManeuverModel) Name() string { return "mock_lstm_attention" }

func (m mockManeuverModel) Predict(window []ManeuverFrame) (ManeuverInference, error) {
	return ManeuverInference{
		Probability: m.prob,
		Confidence:  m.conf,
		Type:        m.typ,
		DeltaV:      m.dv,
		SHAP:        HeuristicSHAP(window),
	}, nil
}

func TestDetectManeuvers(t *testing.T) {
	detector := NewCUSUMDetector(0.005, 0.05)
	residuals := []float64{0.002, 0.004, 0.06, 0.07, 0.01}
	deltaVs := []float64{0.002, 0.003, 0.09, 0.01, 0.002}

	events := DetectManeuvers(43286, residuals, deltaVs, detector)
	if len(events) == 0 {
		t.Fatalf("Expected at least one maneuver event")
	}
	if events[0].Type != ManeuverImpulsive {
		t.Fatalf("Expected impulsive maneuver, got %s", events[0].Type)
	}
}

func TestDetectManeuversHybridUsesModel(t *testing.T) {
	model := mockManeuverModel{prob: 0.92, conf: 0.81, typ: ManeuverImpulsive, dv: 0.11}
	hybrid := NewHybridManeuverDetector(model)
	hybrid.WindowSize = 4
	hybrid.MinProbability = 0.5
	hybrid.MinConfidence = 0.5

	residuals := []float64{0.001, 0.002, 0.003, 0.004, 0.002}
	deltaVs := []float64{0.001, 0.001, 0.002, 0.002, 0.001}

	events := DetectManeuversHybrid(123, residuals, deltaVs, hybrid)
	if len(events) == 0 {
		t.Fatalf("Expected model-driven maneuver events")
	}
	if events[0].Type != ManeuverImpulsive {
		t.Fatalf("Expected model to classify impulsive maneuver, got %s", events[0].Type)
	}
	if events[0].Score <= 0 {
		t.Fatalf("Expected positive hybrid score")
	}
}

func TestHeuristicSHAP(t *testing.T) {
	window := []ManeuverFrame{{Residual: 0.04, DeltaV: 0.01}, {Residual: 0.09, DeltaV: 0.02}}
	shap := HeuristicSHAP(window)
	if len(shap) != 2 {
		t.Fatalf("Expected two feature contributions, got %d", len(shap))
	}
	if shap[0].Contribution <= 0 || shap[1].Contribution <= 0 {
		t.Fatalf("Expected positive contributions")
	}
}

func TestCharacterizeManeuverFromTracks(t *testing.T) {
	prior := []orbital.State{
		{X: 7000, Y: 0, Z: 0, VX: 0, VY: 7.50, VZ: 0.00},
		{X: 6998, Y: 20, Z: 0, VX: -0.02, VY: 7.49, VZ: 0.00},
	}
	post := []orbital.State{
		{X: 6998, Y: 20, Z: 0, VX: -0.02, VY: 7.56, VZ: 0.01},
		{X: 6995, Y: 38, Z: 0.2, VX: -0.04, VY: 7.56, VZ: 0.01},
	}
	tPrior := []float64{100, 110}
	tPost := []float64{111, 120}
	assets := []orbital.State{{X: 7020, Y: 0, Z: 0, VX: 0, VY: 7.48, VZ: 0.0}}

	c := CharacterizeManeuverFromTracks(prior, post, tPrior, tPost, assets)
	if c.ManeuverTimeSec <= 0 {
		t.Fatalf("expected positive maneuver time")
	}
	if math.Abs(c.DeltaVRSW[1]) < 1e-6 {
		t.Fatalf("expected non-zero along-track delta-v in RSW")
	}
	if c.BurnType == ManeuverNone {
		t.Fatalf("expected burn classification")
	}
	if c.Confidence <= 0 || c.Confidence > 1 {
		t.Fatalf("expected normalized confidence, got %.3f", c.Confidence)
	}
}

func TestDetectManeuversCovarianceAwareSuppressesNoiseDrift(t *testing.T) {
	residuals := []float64{0.03, 0.04, 0.035, 0.045, 0.04, 0.03}
	sigma := []float64{0.03, 0.035, 0.03, 0.04, 0.035, 0.03}
	deltaVs := []float64{0.001, 0.001, 0.001, 0.001, 0.001, 0.001}

	raw := DetectManeuvers(77, residuals, deltaVs, NewCUSUMDetector(0.005, 0.05))
	aware := DetectManeuversCovarianceAware(77, residuals, sigma, deltaVs, NewCUSUMDetector(0.005, 0.05))

	if len(raw) == 0 {
		t.Fatalf("expected raw detector to alarm under drift-like residuals")
	}
	if len(aware) != 0 {
		t.Fatalf("expected covariance-aware detector to suppress noise-driven alarms")
	}
}

func TestDetectManeuversCovarianceAwareDetectsTrueEvent(t *testing.T) {
	residuals := []float64{0.01, 0.01, 0.02, 0.60, 0.02, 0.01}
	sigma := []float64{0.03, 0.03, 0.03, 0.03, 0.03, 0.03}
	deltaVs := []float64{0.002, 0.002, 0.004, 0.09, 0.002, 0.002}

	events := DetectManeuversCovarianceAware(88, residuals, sigma, deltaVs, NewCUSUMDetector(0.005, 0.05))
	if len(events) == 0 {
		t.Fatalf("expected covariance-aware detector to catch strong maneuver event")
	}
	if events[0].Type == ManeuverNone {
		t.Fatalf("expected classified maneuver type")
	}
}
