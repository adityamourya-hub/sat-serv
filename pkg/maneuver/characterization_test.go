package maneuver

import (
	"math"
	"testing"
	"time"
	"ssis/pkg/orbital"
)

func localNorm3(v [3]float64) float64 {
	return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
}

func TestManeuverCharacterizationImpulsive(t *testing.T) {
	// 1. Setup Initial State (Stable MEO - 8000km to avoid drag layer)
	initial := orbital.State{
		X: 8000.0, Y: 0, Z: 0,
		VX: 0, VY: 7.0, VZ: 0,
	}

	prop := func(s orbital.State, dt float64) orbital.State {
		const maxStep = 30.0
		v := orbital.StateToVector(s)
		n := int(math.Ceil(math.Abs(dt) / maxStep))
		if n == 0 { return s }
		step := dt / float64(n)
		jd := 2451545.0 // Test epoch
		for i := 0; i < n; i++ {
			v = orbital.PropagateCartesianState(v, step, jd)
		}
		return orbital.VectorToState(v)
	}

	// 2. Generate Prior Arc (5 points, 1 min spacing)
	prior := make([]orbital.State, 5)
	tPrior := make([]float64, 5)
	for i := 0; i < 5; i++ {
		tPrior[i] = float64(i * 60)
		prior[i] = prop(initial, tPrior[i])
	}

	// 3. Apply Impulsive Maneuver at t=300s
	maneuverTime := 300.0
	stateAtManeuver := prop(initial, maneuverTime)
	statePostManeuver := stateAtManeuver
	statePostManeuver.VY += 0.1

	// 4. Generate Post Arc (5 points, 1 min spacing after maneuver)
	post := make([]orbital.State, 5)
	tPost := make([]float64, 5)
	for i := 0; i < 5; i++ {
		tPost[i] = maneuverTime + float64((i+1)*60)
		post[i] = prop(statePostManeuver, float64((i+1)*60))
	}

	// 5. Invoke Characterization
	 indianAssets := []orbital.State{{X: 6040, Y: 3530, Z: 0, VX: -3.5, VY: 6.5, VZ: 0}}
	char := CharacterizeManeuverFromTracks(prior, post, tPrior, tPost, indianAssets)

	// 6. Verification
	dvMag := math.Sqrt(char.DeltaVECI[0]*char.DeltaVECI[0] + char.DeltaVECI[1]*char.DeltaVECI[1] + char.DeltaVECI[2]*char.DeltaVECI[2])
	
	if math.Abs(dvMag-0.1) > 0.05 {
		t.Errorf("Delta-V magnitude estimate outside tolerance: got %.4f, want 0.1", dvMag)
	}

	if char.BurnType != ManeuverImpulsive {
		t.Errorf("Expected BurnType impulsive, got %s", char.BurnType)
	}
}

func TestManeuverCharacterizationStealthy(t *testing.T) {
	// Stealthy maneuver: Small cross-track burn at 8000km MEO
	// Using 1200s gap (20 mins) to verify integration reversibility
	initial := orbital.State{X: 8000.0, Y: 0, Z: 0, VX: 0, VY: 7.0, VZ: 0}
	
	baseJd := float64(time.Now().Unix())/86400.0 + 2440587.5
	prop := func(s orbital.State, dt float64) orbital.State {
		const maxStep = 30.0
		v := orbital.StateToVector(s)
		n := int(math.Ceil(math.Abs(dt) / maxStep))
		if n == 0 { return s }
		step := dt / float64(n)
		for i := 0; i < n; i++ {
			v = orbital.PropagateCartesianState(v, step, baseJd)
		}
		return orbital.VectorToState(v)
	}

	prior := []orbital.State{initial}
	tPrior := []float64{0}
	
	midState := prop(initial, 600) 
	midState.VZ += 0.02 // Small burn at t=600
	postState := prop(midState, 600)
	
	post := []orbital.State{postState}
	tPost := []float64{1200}

	char := CharacterizeManeuverFromTracks(prior, post, tPrior, tPost, nil)
	
	t.Logf("Stealthy Test Results (1200s Arc):")
	t.Logf("  Delta-V: %.4f km/s", localNorm3(char.DeltaVECI))
	t.Logf("  Stealthy Detected: %v", char.StealthyManeuver)

	dvMag := localNorm3(char.DeltaVECI)
	if math.Abs(dvMag - 0.02) > 0.005 {
		t.Errorf("Stealthy Delta-V recovery failed with fixed propagator: got %.4f, want 0.02", dvMag)
	}
	if !char.StealthyManeuver {
		t.Errorf("Expected stealthy detector to flag small DV with significant arc gap")
	}
}
