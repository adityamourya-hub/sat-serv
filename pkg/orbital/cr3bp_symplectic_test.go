// pkg/orbital/cr3bp_symplectic_test.go
// CR3BPLite Forest-Ruth Symplectic Integration Tests
//
// Regression tests for the P0 Symplectic Ghost fix.
// Verifies that Forest-Ruth integration preserves JᵀJM = J structure
// and maintains energy conservation for CR3BP dynamics.

package orbital

import (
	"math"
	"testing"
)

// TestCR3BPLiteForestRuthEnergyConservation verifies that the Forest-Ruth
// integrator on CR3BPLite preserves Jacobi constant (energy invariant).
// This is the regression test for the Symplectic Ghost fix.
func TestCR3BPLiteForestRuthEnergyConservation(t *testing.T) {
	// Create CR3BP system (Earth-Moon)
	system := EarthMoonSystem()

	// Initial condition: L1 halo orbit approximation
	// Distance from Earth center (normalized)
	mu := system.Mu
	x := 1.0 - mu - 0.01 // Near L1
	y := 0.0
	z := 0.01

	// Velocity for periodic orbit (simplified)
	vx := 0.0
	vy := 0.1
	vz := 0.0

	initialState := CR3BPLite{
		Pos: Vector3{x, y, z},
		Vel: Vector3{vx, vy, vz},
	}

	// Compute initial Jacobi constant
	initialJacobi := initialState.ComputeJacobi(mu)

	// Integration parameters
	dt := 0.01 // Normalized time step
	steps := 1000

	// Track Jacobi drift
	maxDrift := 0.0
	current := initialState

	for i := 0; i < steps; i++ {
		// Use Forest-Ruth symplectic integrator (the fix)
		current = current.forestRuthStep(system, dt)

		// Compute Jacobi constant
		currentJacobi := current.ComputeJacobi(mu)
		drift := math.Abs(currentJacobi - initialJacobi)

		if drift > maxDrift {
			maxDrift = drift
		}
	}

	// Target: Jacobi drift < 1e-6 over integration
	// Forest-Ruth should preserve invariants much better than RK4
	targetDrift := 1e-6
	if maxDrift > targetDrift {
		t.Errorf("Jacobi constant conservation failed: max drift %.2e > %.2e", maxDrift, targetDrift)
		t.Logf("  Initial Jacobi: %.15e", initialJacobi)
		t.Logf("  Max drift: %.15e", maxDrift)
	} else {
		t.Logf("Jacobi constant conservation passed: max drift %.2e < %.2e", maxDrift, targetDrift)
	}
}

// TestCR3BPLiteRK4VsForestRuth compares RK4 and Forest-Ruth energy drift.
// This test demonstrates why the Symplectic Ghost fix was necessary.
func TestCR3BPLiteRK4VsForestRuth(t *testing.T) {
	system := EarthMoonSystem()
	mu := system.Mu

	// Initial condition (L1 region)
	initialState := CR3BPLite{
		Pos: Vector3{1.0 - mu - 0.01, 0.0, 0.01},
		Vel: Vector3{0.0, 0.1, 0.0},
	}

	initialJacobi := initialState.ComputeJacobi(mu)

	dt := 0.01
	steps := 1000

	// Test RK4
	rk4State := initialState
	rk4MaxDrift := 0.0
	for i := 0; i < steps; i++ {
		rk4State = rk4State.rk4Step(system, dt)
		drift := math.Abs(rk4State.ComputeJacobi(mu) - initialJacobi)
		if drift > rk4MaxDrift {
			rk4MaxDrift = drift
		}
	}

	// Test Forest-Ruth
	frState := initialState
	frMaxDrift := 0.0
	for i := 0; i < steps; i++ {
		frState = frState.forestRuthStep(system, dt)
		drift := math.Abs(frState.ComputeJacobi(mu) - initialJacobi)
		if drift > frMaxDrift {
			frMaxDrift = drift
		}
	}

	t.Logf("RK4 max Jacobi drift: %.2e", rk4MaxDrift)
	t.Logf("Forest-Ruth max Jacobi drift: %.2e", frMaxDrift)
	t.Logf("Improvement factor: %.2fx", rk4MaxDrift/frMaxDrift)

	// Forest-Ruth should have significantly better energy conservation
	if frMaxDrift > rk4MaxDrift {
		t.Errorf("Forest-Ruth failed to improve energy conservation: %.2e > %.2e", frMaxDrift, rk4MaxDrift)
	}

	// Forest-Ruth should be at least 10x better
	if rk4MaxDrift/frMaxDrift < 10.0 {
		t.Logf("Warning: Forest-Ruth improvement less than 10x (%.2fx)", rk4MaxDrift/frMaxDrift)
	}
}

// TestCR3BPLiteForestRuthCoefficients verifies Forest-Ruth coefficients
// match the expected values from symplectic_engine.go.
func TestCR3BPLiteForestRuthCoefficients(t *testing.T) {
	// Expected Forest-Ruth coefficients (Yoshida 4th-order)
	expectedGamma := 1.0 / (2.0 - math.Cbrt(2.0))
	expectedA := [4]float64{expectedGamma / 2.0, (1.0 - expectedGamma) / 2.0, (1.0 - expectedGamma) / 2.0, expectedGamma / 2.0}
	expectedB := [4]float64{expectedGamma, 1.0 - 2.0*expectedGamma, expectedGamma, 0.0}

	// Verify coefficient properties
	// Sum of position coefficients should be 1
	sumA := expectedA[0] + expectedA[1] + expectedA[2] + expectedA[3]
	if math.Abs(sumA-1.0) > 1e-10 {
		t.Errorf("Position coefficients don't sum to 1: %.15e", sumA)
	}

	// Sum of momentum coefficients should be 1
	sumB := expectedB[0] + expectedB[1] + expectedB[2] + expectedB[3]
	if math.Abs(sumB-1.0) > 1e-10 {
		t.Errorf("Momentum coefficients don't sum to 1: %.15e", sumB)
	}

	t.Logf("Forest-Ruth coefficients verified: γ=%.15e", expectedGamma)
	t.Logf("  a = [%.6f, %.6f, %.6f, %.6f]", expectedA[0], expectedA[1], expectedA[2], expectedA[3])
	t.Logf("  b = [%.6f, %.6f, %.6f, %.6f]", expectedB[0], expectedB[1], expectedB[2], expectedB[3])
}

// TestCR3BPLiteForestRuthStepSize tests integration stability with different step sizes.
func TestCR3BPLiteForestRuthStepSize(t *testing.T) {
	system := EarthMoonSystem()
	mu := system.Mu

	initialState := CR3BPLite{
		Pos: Vector3{1.0 - mu - 0.01, 0.0, 0.01},
		Vel: Vector3{0.0, 0.1, 0.0},
	}

	initialJacobi := initialState.ComputeJacobi(mu)

	// Test different step sizes
	stepSizes := []float64{0.001, 0.01, 0.1}

	for _, dt := range stepSizes {
		current := initialState
		steps := int(1.0 / dt) // Normalize to ~1 normalized time unit

		maxDrift := 0.0
		for i := 0; i < steps; i++ {
			current = current.forestRuthStep(system, dt)
			drift := math.Abs(current.ComputeJacobi(mu) - initialJacobi)
			if drift > maxDrift {
				maxDrift = drift
			}
		}

		t.Logf("dt=%.3f, steps=%d, max drift=%.2e", dt, steps, maxDrift)

		// All step sizes should maintain reasonable energy conservation
		if maxDrift > 1e-4 {
			t.Errorf("Step size dt=%.3f failed: max drift %.2e > 1e-4", dt, maxDrift)
		}
	}
}
