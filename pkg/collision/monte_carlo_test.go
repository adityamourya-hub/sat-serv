package collision

import (
	"math"
	"strings"
	"testing"
)

type MockObject struct {
	State      []float64
	Covariance [][]float64
}

func (m *MockObject) GetState() []float64        { return m.State }
func (m *MockObject) GetCovariance() [][]float64 { return m.Covariance }
func (m *MockObject) Propagate(dt float64) []float64 {
	s := append([]float64(nil), m.State...)
	s[0] += s[3] * dt
	s[1] += s[4] * dt
	s[2] += s[5] * dt
	return s
}

func TestMonteCarloScreening(t *testing.T) {
	// Standard Gaussian conjunction case (Foster 1992 reference)
	// Miss distance 0.05 km, 1-sigma uncertainty 0.02 km
	p := &MockObject{
		State: []float64{0, 0, 0, 7.5, 0, 0},
		Covariance: [][]float64{
			{0.0004, 0, 0, 0, 0, 0},
			{0, 0.0004, 0, 0, 0, 0},
			{0, 0, 0.0004, 0, 0, 0},
			{0, 0, 0, 1e-6, 0, 0},
			{0, 0, 0, 0, 1e-6, 0},
			{0, 0, 0, 0, 0, 1e-6},
		},
	}
	s := &MockObject{
		State: []float64{0.05, 0, 0, -7.5, 0, 0},
		Covariance: [][]float64{
			{0.0004, 0, 0, 0, 0, 0},
			{0, 0.0004, 0, 0, 0, 0},
			{0, 0, 0.0004, 0, 0, 0},
			{0, 0, 0, 1e-6, 0, 0},
			{0, 0, 0, 0, 1e-6, 0},
			{0, 0, 0, 0, 0, 1e-6},
		},
	}

	opts := DefaultMCOptions()
	opts.NSamples = 100000 // Faster for unit test

	res, err := ExecuteMonteCarloScreening(p, s, 0, 0.005, opts)
	if err != nil {
		t.Fatalf("MC screening failed: %v", err)
	}

	if res.PcMean <= 0 {
		t.Errorf("Expected positive Pc for near-conjunction, got %f", res.PcMean)
	}

	t.Logf("MC Pc: %f +/- %f (95%% CI: [%f, %f])", res.PcMean, res.PcStd, res.PcCI95[0], res.PcCI95[1])
	t.Logf("Execution time: %d ms", res.ExecutionTimeMs)
}

func TestCDMSerialization(t *testing.T) {
	mc := &MonteCarloConjunctionResult{
		PcMean:          1.23e-4,
		TotalSamples:    1000000,
		ConvergenceDiag: 0.001,
	}
	cdm := NewCDM(mc)
	cdm.RelativeMetadata.MISS_DISTANCE = 0.54
	cdm.Object1Metadata.OBJECT_NAME = "ASSET_A"
	cdm.Object2Metadata.OBJECT_NAME = "DEBRIS_B"

	kvn := cdm.ToKVN()
	if !strings.Contains(kvn, "COLLISION_PROB = 1.2300e-04") {
		t.Errorf("KVN missing expected Pc value")
	}

	js, _ := cdm.ToJSON()
	if !strings.Contains(js, "\"pc\": 0.000123") {
		t.Errorf("JSON missing expected Pc value")
	}
}

func TestFactorCovNearSingularRegularized(t *testing.T) {
	cov := [][]float64{
		{1.0, 0.999999999999, 0.999999999998, 0, 0, 0},
		{0.999999999999, 0.999999999998, 0.999999999997, 0, 0, 0},
		{0.999999999998, 0.999999999997, 0.999999999996, 0, 0, 0},
		{0, 0, 0, 1e-8, 0, 0},
		{0, 0, 0, 0, 1e-8, 0},
		{0, 0, 0, 0, 0, 1e-8},
	}

	l, err := factorCov(cov)
	if err != nil {
		t.Fatalf("factorCov failed for near-singular covariance: %v", err)
	}
	if len(l) != 6 || len(l[0]) != 6 {
		t.Fatalf("unexpected factor dimensions")
	}
	for i := 0; i < 6; i++ {
		if l[i][i] <= 0 || math.IsNaN(l[i][i]) || math.IsInf(l[i][i], 0) {
			t.Fatalf("invalid diagonal entry l[%d][%d]=%v", i, i, l[i][i])
		}
	}
}
