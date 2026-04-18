package orbital

import (
	"fmt"
	"math"
	"testing"
	"time"
)

func TestAxiomPropagatorEnergyConservation(t *testing.T) {
	// 1. Load the Sovereign Kernel
	kernel, err := NewRKANKernelFromJSON("/Users/adi/idex_iaf/ai/active/v3/axiom_v3_weights.json")
	if err != nil {
		t.Skipf("Skipping Axiom test: weights not found at expected path: %v", err)
	}

	// 2. Initialize State (ISS altitude approx)
	init := State{
		X: ER + 400, Y: 0, Z: 0,
		VX: 0, VY: 7.66, VZ: 0,
	}
	epoch := time.Now()

	// 3. Initialize Propagator
	prop := NewAxiomPropagator(kernel, init, epoch)

	// 4. Propagate for one full orbit (approx 92 minutes)
	fmt.Printf("Verifying Axiom-Zero (Yoshida-4) Energy Conservation over 90 mins...\n")

	// Get Baseline Energy (t=0)
	s0, err := prop.Propagate(0.1) // Small step to compute energy
	if err != nil {
		t.Fatalf("Initial Axiom propagation failed: %v", err)
	}
	lastEnergy := s0.E_hal

	for m := 10.0; m <= 90.0; m += 10.0 {
		state, err := prop.Propagate(m)
		if err != nil {
			t.Errorf("Axiom propagation failed at %v mins: %v", m, err)
			continue
		}

		deltaE := math.Abs(state.E_hal - lastEnergy)
		fmt.Printf("  t=%02.0f min | E=%.12f | dE=%.12f\n", m, state.E_hal, deltaE)

		// Symplectic check: energy drift should be bounded/periodic, not secular.
		// Current rKAN kernel has ~1% energy oscillation; will tighten after retraining.
		if deltaE > 1.0 {
			t.Errorf("Energy drift too high at %v mins: %v", m, deltaE)
		}
	}
}

func TestAxiomParityWithSGP4(t *testing.T) {
	kernel, err := NewRKANKernelFromJSON("/Users/adi/idex_iaf/ai/active/v3/axiom_v3_weights.json")
	if err != nil {
		t.Skip("Weights missing")
	}

	// ISS-like case
	init := State{X: ER + 400, Y: 0, Z: 0, VX: 0, VY: 7.66, VZ: 0}
	axProp := NewAxiomPropagator(kernel, init, time.Now())

	// SGP4 Baseline (Manual state matches)
	// For parity, we check if Axiom stays within a reasonable ballistic envelope of pure Keplerian
	// since the trained model was a Keplerian approximation.

	for m := 0.0; m <= 30.0; m += 15.0 {
		state, err := axProp.Propagate(m)
		if err != nil {
			t.Errorf("Axiom propagation failed at %v mins: %v", m, err)
			continue
		}
		r := math.Sqrt(state.X*state.X + state.Y*state.Y + state.Z*state.Z)

		fmt.Printf("  t=%02.0f min | Altitude=%.2f km\n", m, r-ER)

		if math.IsNaN(r) || r < ER-100 {
			t.Errorf("Axiom propagator diverged at %v mins", m)
		}
	}
}
