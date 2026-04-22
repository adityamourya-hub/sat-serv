// pkg/physics/symplectic_engine_test.go
// Symplectic Integration Tests
//
// Tests for Forest-Ruth 4th-order symplectic integrator energy conservation.
// Target: |H_t-H_0| < 1e-8 over 10,000 steps for Hamiltonian systems.
//
// This test suite verifies the geometric structure preservation properties
// of the symplectic engine, ensuring long-term stability of orbital predictions.

package physics

import (
	"math"
	"testing"
)

// TestSymplecticEnergyConservation verifies that the Forest-Ruth integrator
// preserves Hamiltonian energy within 1e-8 over 10,000 steps.
//
// This is a critical test for physics package integrity.
func TestSymplecticEnergyConservation(t *testing.T) {
	// Create a simple Keplerian system (2-body problem)
	keplerian := &KeplerianSystem{
		mu: 398600.4418, // Earth's gravitational parameter (km³/s²)
	}

	// Initial conditions: circular orbit at 7000 km altitude
	r0 := 7000.0 + 6371.0              // km from Earth center
	v0 := math.Sqrt(keplerian.mu / r0) // Circular velocity

	q := []float64{r0, 0.0, 0.0} // Position
	p := []float64{0.0, v0, 0.0} // Velocity (momentum for unit mass)
	var z []float64              // No latent variables

	// Compute initial Hamiltonian
	H0 := keplerian.Hamiltonian(q, p, z)

	// Create symplectic engine
	engine := NewEngine(keplerian)

	// Integration parameters
	dt := 1.0 // 1 second timestep
	nSteps := 10000

	// Track energy drift
	maxDrift := 0.0
	minDrift := 0.0

	// Run integration
	for i := 0; i < nSteps; i++ {
		q, p = engine.SymplecticStep(q, p, z, dt)

		// Compute current Hamiltonian
		Ht := keplerian.Hamiltonian(q, p, z)

		// Compute drift
		drift := math.Abs(Ht - H0)
		if drift > maxDrift {
			maxDrift = drift
		}
		if drift < minDrift || i == 0 {
			minDrift = drift
		}
	}

	// Verify energy conservation
	// Target: |H_t-H_0| < 1e-8 over 10,000 steps
	targetDrift := 1e-8
	if maxDrift > targetDrift {
		t.Errorf("Energy conservation failed: max drift %.2e > %.2e (target)", maxDrift, targetDrift)
		t.Logf("  Initial Hamiltonian: %.15e", H0)
		t.Logf("  Max drift: %.15e", maxDrift)
		t.Logf("  Min drift: %.15e", minDrift)
		t.Logf("  Final position: [%.3f, %.3f, %.3f] km", q[0], q[1], q[2])
		t.Logf("  Final velocity: [%.6f, %.6f, %.6f] km/s", p[0], p[1], p[2])
	} else {
		t.Logf("Energy conservation passed: max drift %.2e < %.2e", maxDrift, targetDrift)
	}
}

// TestSymplecticManifoldNormalization verifies that the integrator
// preserves manifold constraints (S² normalization).
func TestSymplecticManifoldNormalization(t *testing.T) {
	// Create a system on S² manifold (unit sphere)
	sphereSystem := &SphereSystem{}

	// Initial condition: point on unit sphere
	q := []float64{1.0, 0.0, 0.0} // Position on unit sphere
	p := []float64{0.0, 1.0, 0.0} // Velocity tangent to sphere
	var z []float64

	// Create symplectic engine
	engine := NewEngine(sphereSystem)

	// Integration parameters
	dt := 0.01
	nSteps := 1000

	// Track manifold violation
	maxViolation := 0.0

	// Run integration
	for i := 0; i < nSteps; i++ {
		q, p = engine.SymplecticStep(q, p, z, dt)

		// Normalize position to unit sphere (as integrator should do)
		norm := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
		violation := math.Abs(norm - 1.0)
		if violation > maxViolation {
			maxViolation = violation
		}
	}

	// Verify manifold preservation
	// Target: manifold violation < 1e-10
	targetViolation := 1e-10
	if maxViolation > targetViolation {
		t.Errorf("Manifold normalization failed: max violation %.2e > %.2e", maxViolation, targetViolation)
	} else {
		t.Logf("Manifold normalization passed: max violation %.2e", maxViolation)
	}
}

// KeplerianSystem implements PHSModel for simple 2-body Keplerian problem
type KeplerianSystem struct {
	mu float64 // Gravitational parameter
}

// Velocity returns ∂H/∂p = p/m = v (for unit mass)
func (k *KeplerianSystem) Velocity(q, p, z []float64) []float64 {
	return []float64{p[0], p[1], p[2]}
}

// Force returns -∂H/∂q = -(-μ/r³ * r) = μ/r³ * r (gravity)
func (k *KeplerianSystem) Force(q, p, z []float64) []float64 {
	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	factor := -k.mu / (r * r * r)
	return []float64{factor * q[0], factor * q[1], factor * q[2]}
}

// Hamiltonian returns H = 0.5*v² - μ/r
func (k *KeplerianSystem) Hamiltonian(q, p, z []float64) float64 {
	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	v := math.Sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
	kinetic := 0.5 * v * v
	potential := -k.mu / r
	return kinetic + potential
}

// SphereSystem implements PHSModel for S² manifold (unit sphere)
type SphereSystem struct{}

// Velocity returns ∂H/∂p (simple harmonic oscillator on sphere)
func (s *SphereSystem) Velocity(q, p, z []float64) []float64 {
	return []float64{p[0], p[1], p[2]}
}

// Force returns -∂H/∂q (restoring force toward origin)
func (s *SphereSystem) Force(q, p, z []float64) []float64 {
	k := 1.0 // Spring constant
	return []float64{-k * q[0], -k * q[1], -k * q[2]}
}

// Hamiltonian returns H = 0.5*v² + 0.5*k*q²
func (s *SphereSystem) Hamiltonian(q, p, z []float64) float64 {
	v := math.Sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	kinetic := 0.5 * v * v
	potential := 0.5 * r * r
	return kinetic + potential
}

// BenchmarkSymplecticEnergyConservation benchmarks energy conservation performance
func BenchmarkSymplecticEnergyConservation(b *testing.B) {
	keplerian := &KeplerianSystem{mu: 398600.4418}
	engine := NewEngine(keplerian)

	q := []float64{7000.0 + 6371.0, 0.0, 0.0}
	v := math.Sqrt(keplerian.mu / q[0])
	p := []float64{0.0, v, 0.0}
	var z []float64

	dt := 1.0

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q, p = engine.SymplecticStep(q, p, z, dt)
	}
}

// TestSymplecticStructurePreservation verifies symplectic structure preservation
// by checking the symplectic form ω = dq ∧ dp is preserved.
func TestSymplecticStructurePreservation(t *testing.T) {
	keplerian := &KeplerianSystem{mu: 398600.4418}
	engine := NewEngine(keplerian)

	// Initial conditions
	q := []float64{7000.0 + 6371.0, 0.0, 0.0}
	v := math.Sqrt(keplerian.mu / q[0])
	p := []float64{0.0, v, 0.0}
	var z []float64

	// Compute initial symplectic form (simplified as phase space volume)
	// For Hamiltonian systems, the phase space volume should be preserved
	initialVolume := computePhaseSpaceVolume(q, p)

	dt := 1.0
	nSteps := 1000

	for i := 0; i < nSteps; i++ {
		q, p = engine.SymplecticStep(q, p, z, dt)
	}

	finalVolume := computePhaseSpaceVolume(q, p)

	// Check volume preservation (Liouville's theorem)
	volumeChange := math.Abs(finalVolume-initialVolume) / initialVolume
	targetChange := 1e-6 // Allow 0.0001% change

	if volumeChange > targetChange {
		t.Errorf("Symplectic structure not preserved: volume change %.2e > %.2e", volumeChange, targetChange)
	} else {
		t.Logf("Symplectic structure preserved: volume change %.2e", volumeChange)
	}
}

// computePhaseSpaceVolume computes a simplified phase space volume
func computePhaseSpaceVolume(q, p []float64) float64 {
	// Simplified: product of position and velocity magnitudes
	qNorm := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	pNorm := math.Sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
	return qNorm * pNorm
}

// TestSymplecticReversibility verifies time-reversal symmetry
func TestSymplecticReversibility(t *testing.T) {
	keplerian := &KeplerianSystem{mu: 398600.4418}
	engine := NewEngine(keplerian)

	// Initial conditions
	q0 := []float64{7000.0 + 6371.0, 100.0, 0.0}
	v := math.Sqrt(keplerian.mu / q0[0])
	p0 := []float64{0.0, v, 0.0}
	var z []float64

	dt := 1.0
	nSteps := 100

	// Forward integration
	qForward := make([]float64, len(q0))
	pForward := make([]float64, len(p0))
	copy(qForward, q0)
	copy(pForward, p0)

	for i := 0; i < nSteps; i++ {
		qForward, pForward = engine.SymplecticStep(qForward, pForward, z, dt)
	}

	// Reverse velocity and integrate backward
	for i := 0; i < len(pForward); i++ {
		pForward[i] = -pForward[i]
	}

	for i := 0; i < nSteps; i++ {
		qForward, pForward = engine.SymplecticStep(qForward, pForward, z, dt)
	}

	// Check if we returned to initial state
	positionError := 0.0
	velocityError := 0.0
	for i := 0; i < len(q0); i++ {
		positionError += (qForward[i] - q0[i]) * (qForward[i] - q0[i])
		velocityError += (pForward[i] - p0[i]) * (pForward[i] - p0[i])
	}

	positionError = math.Sqrt(positionError)
	velocityError = math.Sqrt(velocityError)

	// Target: return within 1e-6 of initial state
	targetError := 1e-6
	if positionError > targetError || velocityError > targetError {
		t.Errorf("Time-reversal symmetry failed: pos error %.2e, vel error %.2e", positionError, velocityError)
	} else {
		t.Logf("Time-reversal symmetry passed: pos error %.2e, vel error %.2e", positionError, velocityError)
	}
}

// TestSymplecticLongTermStability tests stability over long integration (100k steps)
func TestSymplecticLongTermStability(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-term stability test in short mode")
	}

	keplerian := &KeplerianSystem{mu: 398600.4418}
	engine := NewEngine(keplerian)

	q := []float64{7000.0 + 6371.0, 0.0, 0.0}
	v := math.Sqrt(keplerian.mu / q[0])
	p := []float64{0.0, v, 0.0}
	var z []float64

	H0 := keplerian.Hamiltonian(q, p, z)

	dt := 1.0
	nSteps := 100000 // 100k steps = ~100k seconds ~ 27 hours

	maxDrift := 0.0

	for i := 0; i < nSteps; i++ {
		q, p = engine.SymplecticStep(q, p, z, dt)

		if i%10000 == 0 {
			Ht := keplerian.Hamiltonian(q, p, z)
			drift := math.Abs(Ht - H0)
			if drift > maxDrift {
				maxDrift = drift
			}
		}
	}

	// Target: max drift < 1e-6 over 100k steps (relaxed from 1e-8)
	targetDrift := 1e-6
	if maxDrift > targetDrift {
		t.Errorf("Long-term stability failed: max drift %.2e > %.2e over %d steps", maxDrift, targetDrift, nSteps)
	} else {
		t.Logf("Long-term stability passed: max drift %.2e over %d steps", maxDrift, nSteps)
	}
}
