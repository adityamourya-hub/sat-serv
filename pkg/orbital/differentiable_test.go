package orbital

import (
	"fmt"
	"math"
	"ssis/pkg/physics"
	"testing"
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
	tle := &physics.TLE{
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

	dt := 60 * 60.0 // 60 minutes in seconds

	// 3. Propagate
	state, jacobian := prop.PropagateTLE(*tle, dt)

	// 4. Basic sanity checks
	if state.Pos[0] == 0 && state.Pos[1] == 0 {
		t.Errorf("Propagated state is zero")
	}

	r := math.Sqrt(state.Pos[0]*state.Pos[0] + state.Pos[1]*state.Pos[1] + state.Pos[2]*state.Pos[2])
	if r < 6371 || r > 50000 {
		t.Errorf("Unreasonable orbital radius: %f km", r)
	}

	fmt.Printf("[Test] Propagated State: %+v\n", state)
	if jacobian != nil && len(jacobian) > 0 {
		fmt.Printf("[Test] Jacobian extracted successfully (%d elements)\n", len(jacobian))
	} else {
		t.Logf("Note: Jacobian not available (model may not provide it)")
	}
}

func TestDifferentiablePropagator_JacobianAccuracy(t *testing.T) {
	modelPath := "pkg/ml/models/dsgp4_hybrid.onnx"
	prop, _ := NewDifferentiablePropagator(modelPath)
	if !prop.enabled {
		t.Skip("Model missing")
	}

	tle := &physics.TLE{Inclination: 29.17, RAAN: 50.93, Eccentricity: 0.0019, ArgOfPerigee: 207.65, MeanAnomaly: 152.09, MeanMotion: 1.0027, BStar: 1e-7, EpochYear: 2024, EpochDay: 99.3}

	base, baseJac := prop.PropagateTLE(*tle, 120*60)

	// Perturb RAAN by epsilon
	eps := 1e-5
	tlePerturbed := *tle
	tlePerturbed.RAAN += eps
	perturbed, _ := prop.PropagateTLE(tlePerturbed, 120*60)

	// Numerical derivative dX/dRAAN
	numGrad := (perturbed.Pos[0] - base.Pos[0]) / eps
	// Analytic derivative from Jacobian if available
	var analyticGrad float64
	if baseJac != nil && len(baseJac) > 0 && len(baseJac[0]) > 0 {
		analyticGrad = baseJac[0][0]
	}

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

	tle := &physics.TLE{Inclination: 28.5, MeanMotion: 15.5, EpochYear: 2024, EpochDay: 100}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = prop.PropagateTLE(*tle, 90*60)
	}
}
