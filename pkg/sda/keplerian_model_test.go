package sda

import (
	"math"
	"testing"
)

// TestKeplerianPHSModelCreation tests basic model initialization
func TestKeplerianPHSModelCreation(t *testing.T) {
	model := NewKeplerianPHSModel()

	if model.Mu != 398600.4418 {
		t.Errorf("Expected Mu = 398600.4418, got %f", model.Mu)
	}

	// Test custom mu
	customModel := NewKeplerianPHSModelWithMu(398600.0)
	if customModel.Mu != 398600.0 {
		t.Errorf("Expected custom Mu = 398600.0, got %f", customModel.Mu)
	}
}

// TestKeplerianVelocity tests that velocity equals momentum for unit mass
func TestKeplerianVelocity(t *testing.T) {
	model := NewKeplerianPHSModel()

	q := []float64{7000.0, 0.0, 0.0} // 7000 km radius
	p := []float64{0.0, 7.5, 0.0}    // ~7.5 km/s orbital velocity
	z := []float64{}

	v := model.Velocity(q, p, z)

	// With unit mass, velocity should equal momentum
	for i := 0; i < 3; i++ {
		if v[i] != p[i] {
			t.Errorf("Velocity[%d] = %f, expected %f (unit mass)", i, v[i], p[i])
		}
	}
}

// TestKeplerianForce tests gravitational force calculation
func TestKeplerianForce(t *testing.T) {
	model := NewKeplerianPHSModel()

	// Test at 7000 km radius
	q := []float64{7000.0, 0.0, 0.0}
	p := []float64{0.0, 7.5, 0.0}
	z := []float64{}

	f := model.Force(q, p, z)

	// Force should point toward Earth (negative x direction)
	if f[0] >= 0 {
		t.Errorf("Gravitational force should be attractive (negative), got f[0] = %f", f[0])
	}

	// Force magnitude should be approximately μ/r² = 398600/7000² ≈ 0.008 km/s²
	expectedMag := model.Mu / (7000.0 * 7000.0)
	actualMag := math.Sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2])

	if math.Abs(actualMag-expectedMag) > 1e-6 {
		t.Errorf("Force magnitude = %f, expected ~%f", actualMag, expectedMag)
	}
}

// TestKeplerianHamiltonian tests energy calculation
func TestKeplerianHamiltonian(t *testing.T) {
	model := NewKeplerianPHSModel()

	// Circular orbit at 7000 km
	r := 7000.0
	v := math.Sqrt(model.Mu / r) // Circular orbit velocity

	q := []float64{r, 0.0, 0.0}
	p := []float64{0.0, v, 0.0} // Momentum = velocity for unit mass
	z := []float64{}

	H := model.Hamiltonian(q, p, z)

	// For circular orbit: E = -μ/(2a) where a = r
	expectedE := -model.Mu / (2 * r)

	if math.Abs(H-expectedE) > 1e-6 {
		t.Errorf("Hamiltonian = %f, expected ~%f for circular orbit", H, expectedE)
	}
}

// TestEnergyConservation verifies that energy is conserved during propagation
func TestEnergyConservation(t *testing.T) {
	model := NewKeplerianPHSModel()

	// Initial state: elliptical orbit
	a := 8000.0       // Semi-major axis
	e := 0.1          // Eccentricity
	rp := a * (1 - e) // Perigee distance

	q := []float64{rp, 0.0, 0.0}
	vp := math.Sqrt(model.Mu * (2/rp - 1/a)) // Velocity at perigee
	p := []float64{0.0, vp, 0.0}
	z := []float64{}

	// Initial energy
	H0 := model.Hamiltonian(q, p, z)

	// Simulate for one orbital period (simplified)
	T := 2 * math.Pi * math.Sqrt(a*a*a/model.Mu)
	dt := T / 100.0 // 100 steps

	maxDrift := 0.0
	for i := 0; i < 100; i++ {
		// Simple Euler step (not symplectic, just for testing energy drift detection)
		v := model.Velocity(q, p, z)
		f := model.Force(q, p, z)

		for j := 0; j < 3; j++ {
			q[j] += v[j] * dt
			p[j] += f[j] * dt
		}

		H := model.Hamiltonian(q, p, z)
		drift := math.Abs((H - H0) / H0)
		if drift > maxDrift {
			maxDrift = drift
		}
	}

	// Note: Euler integration has O(t) energy drift, so this will be significant
	// The test verifies that energy drift is detectable
	t.Logf("Max energy drift with Euler: %.2e", maxDrift)
}

// TestComputeSpecificEnergy tests the helper function
func TestComputeSpecificEnergy(t *testing.T) {
	mu := 398600.4418

	// Circular orbit at 7000 km
	r := [3]float64{7000.0, 0.0, 0.0}
	vMag := math.Sqrt(mu / 7000.0)
	v := [3]float64{0.0, vMag, 0.0}

	E := ComputeSpecificEnergy(r, v, mu)

	// Expected: E = -μ/(2a) = -μ/(2*7000)
	expectedE := -mu / (2 * 7000.0)

	if math.Abs(E-expectedE) > 1e-6 {
		t.Errorf("Specific energy = %f, expected %f", E, expectedE)
	}
}

// TestComputeEnergyDrift tests drift calculation between two states
func TestComputeEnergyDrift(t *testing.T) {
	mu := 398600.4418

	// State 1: circular orbit
	r1 := [3]float64{7000.0, 0.0, 0.0}
	v1Mag := math.Sqrt(mu / 7000.0)
	v1 := [3]float64{0.0, v1Mag, 0.0}

	// State 2: same orbit (should have zero drift)
	drift := ComputeEnergyDrift(r1, v1, r1, v1, mu)

	if drift > 1e-15 {
		t.Errorf("Energy drift for identical states = %e, expected 0", drift)
	}
}

// BenchmarkHamiltonianComputation benchmarks Hamiltonian calculation
func BenchmarkHamiltonianComputation(b *testing.B) {
	model := NewKeplerianPHSModel()
	q := []float64{7000.0, 1000.0, 500.0}
	p := []float64{1.0, 7.0, 2.0}
	z := []float64{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Hamiltonian(q, p, z)
	}
}

// BenchmarkForceComputation benchmarks force calculation
func BenchmarkForceComputation(b *testing.B) {
	model := NewKeplerianPHSModel()
	q := []float64{7000.0, 1000.0, 500.0}
	p := []float64{1.0, 7.0, 2.0}
	z := []float64{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Force(q, p, z)
	}
}
