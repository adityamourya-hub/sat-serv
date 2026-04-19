package orbital

import (
	"math"
	"testing"
	"time"

	"ssis/pkg/physics"
)

// TestAxiomPropagatorInitialization validates the Sovereign propagator can be created
func TestAxiomPropagatorInitialization(t *testing.T) {
	kernel := &MockDynamicsKernel{}
	initState := physics.State{
		Pos: [3]float64{6878.0, 0, 0},
		Vel: [3]float64{0, 7.5, 0},
	}
	params := physics.PhysicsParams{Mass: 1000.0, DragArea: 1.0, SRPArea: 1.0, Cd: 2.2, Cr: 1.3}

	prop := NewAxiomPropagator(kernel, initState, time.Now(), params)
	if prop == nil {
		t.Fatal("Failed to create AxiomPropagator")
	}

	if prop.MU != physics.EARTH_MU_IAU {
		t.Errorf("Expected MU %f, got %f", physics.EARTH_MU_IAU, prop.MU)
	}
}

// TestToJulianDate validates time conversion
func TestToJulianDate(t *testing.T) {
	// J2000.0 epoch: 2000-01-01 12:00:00 TT = JD 2451545.0
	j2000 := time.Date(2000, 1, 1, 12, 0, 0, 0, time.UTC)
	jd := toJulianDate(j2000)

	// Allow small tolerance for time scale differences
	expectedJD := 2451545.0
	if math.Abs(jd-expectedJD) > 0.01 {
		t.Errorf("J2000 Julian Date: expected %.2f, got %.2f", expectedJD, jd)
	}
}

// TestJulianDateEpoch validates Unix epoch conversion
func TestJulianDateEpoch(t *testing.T) {
	unixEpoch := time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC)
	jd := toJulianDate(unixEpoch)
	expectedJD := 2440587.5

	if math.Abs(jd-expectedJD) > 0.0001 {
		t.Errorf("Unix epoch JD: expected %.4f, got %.4f", expectedJD, jd)
	}
}

// MockDynamicsKernel implements DynamicsKernel for testing
type MockDynamicsKernel struct{}

func (m *MockDynamicsKernel) GetAcceleration(q, v [3]float64, jd float64) [3]float64 {
	// Simple Keplerian gravity at Earth surface (for testing)
	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	mu := physics.EARTH_MU_IAU
	factor := -mu / (r * r * r)
	return [3]float64{factor * q[0], factor * q[1], factor * q[2]}
}
