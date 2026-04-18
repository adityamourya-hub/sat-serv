package orbital

import (
	"fmt"
	"math"
	"testing"
	"time"
)

func TestDifferentiablePropagator_Parity(t *testing.T) {
	// 1. Setup Propagator (Expect disabled if no model file)
	modelPath := "pkg/ml/models/dsgp4_hybrid.onnx"
	prop, err := NewDifferentiablePropagator(modelPath)
	if err != nil {
		t.Fatalf("Failed to create propagator: %v", err)
	}
	defer prop.Close()

	if !prop.enabled {
		t.Skip("Differentiable model missing; skipping parity test")
	}

	// 2. Setup Test TLE (IRNSS-1I)
	tle := &TLE{
		Inclination:  29.1724,
		RAAN:         50.9321,
		Eccentricity: 0.0019623,
		ArgOfPerigee: 207.6548,
		MeanAnomaly:  152.0954,
		MeanMotion:   1.00271501,
		BStar:        0.00000084,
		EpochYear:    2024,
		EpochDay:     99.30823340,
	}

	epoch := tleToTime(tle).Add(60 * time.Minute)
	solar := Input{F107: 150, F107A: 150, AP: 15}

	// 3. Propagate
	dState, err := prop.Propagate(tle, epoch, solar)
	if err != nil {
		t.Fatalf("Propagation failed: %v", err)
	}

	// 4. Basic sanity checks
	if dState.State.X == 0 && dState.State.Y == 0 {
		t.Errorf("Propagated state is zero")
	}
	
	r := math.Sqrt(dState.State.X*dState.State.X + dState.State.Y*dState.State.Y + dState.State.Z*dState.State.Z)
	if r < 6371 || r > 50000 {
		t.Errorf("Unreasonable orbital radius: %f km", r)
	}

	fmt.Printf("[Test] Propagated State: %+v\n", dState.State)
	if len(dState.Jacobian) == 42 {
		fmt.Printf("[Test] Jacobian extracted successfully (42 elements)\n")
	} else {
		t.Errorf("Expected 42-element Jacobian, got %d", len(dState.Jacobian))
	}
}

func TestDifferentiablePropagator_JacobianAccuracy(t *testing.T) {
	modelPath := "pkg/ml/models/dsgp4_hybrid.onnx"
	prop, _ := NewDifferentiablePropagator(modelPath)
	if !prop.enabled {
		t.Skip("Model missing")
	}

	tle := &TLE{Inclination: 29.17, RAAN: 50.93, Eccentricity: 0.0019, ArgOfPerigee: 207.65, MeanAnomaly: 152.09, MeanMotion: 1.0027, BStar: 1e-7, EpochYear: 2024, EpochDay: 99.3}
	epoch := tleToTime(tle).Add(120 * time.Minute)
	solar := Input{F107: 150, F107A: 150, AP: 15}

	base, _ := prop.Propagate(tle, epoch, solar)
	
	// Perturb RAAN by epsilon
	eps := 1e-5
	tlePerturbed := *tle
	tlePerturbed.RAAN += eps
	perturbed, _ := prop.Propagate(&tlePerturbed, epoch, solar)

	// Numerical derivative dX/dRAAN
	numGrad := (perturbed.State.X - base.State.X) / eps
	// Analytic derivative from Jacobian (RAAN is index 1)
	analyticGrad := base.Jacobian[0*7 + 1]

	diff := math.Abs(numGrad - analyticGrad)
	fmt.Printf("[Test] RAAN Gradient Check: Num=%f, Analytic=%f, Diff=%e\n", numGrad, analyticGrad, diff)
	
	if diff > 1e-2 && math.Abs(numGrad) > 1e-6 {
		t.Errorf("Jacobian accuracy failed for RAAN: %f vs %f", numGrad, analyticGrad)
	}
}

func BenchmarkDifferentiablePropagate(b *testing.B) {
	modelPath := "pkg/ml/models/dsgp4_hybrid.onnx"
	prop, _ := NewDifferentiablePropagator(modelPath)
	if !prop.enabled {
		b.Skip("Model missing")
	}

	tle := &TLE{Inclination: 28.5, MeanMotion: 15.5, EpochYear: 2024, EpochDay: 100}
	epoch := tleToTime(tle).Add(90 * time.Minute)
	solar := Input{F107: 150, F107A: 150, AP: 15}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = prop.Propagate(tle, epoch, solar)
	}
}
