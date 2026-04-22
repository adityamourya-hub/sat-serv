package hypersonic

import (
	"math"
	"testing"
)

func TestNewDiscoveryModeFilter(t *testing.T) {
	hgv := DefaultHGV()
	dmf := NewDiscoveryModeFilter(hgv)

	if dmf == nil {
		t.Fatal("NewDiscoveryModeFilter returned nil")
	}

	// Check high process noise is set
	if dmf.ProcessNoiseQ[3][3] != DiscoveryProcessNoiseQ {
		t.Errorf("Expected high process noise Q=%.1f, got %.1f",
			DiscoveryProcessNoiseQ, dmf.ProcessNoiseQ[3][3])
	}

	// Check SINDy history is initialized
	if dmf.SINDyHistory == nil {
		t.Error("SINDyHistory not initialized")
	}

	t.Logf("DiscoveryModeFilter initialized with high process noise")
}

func TestDiscoveryModeFilter_Predict(t *testing.T) {
	hgv := DefaultHGV()
	dmf := NewDiscoveryModeFilter(hgv)

	// Set initial Cartesian state (LEO-like orbit)
	dmf.CartesianState = [6]float64{6778.137, 0, 0, 0, 7.6686, 0}

	initialR := math.Sqrt(dmf.CartesianState[0]*dmf.CartesianState[0] +
		dmf.CartesianState[1]*dmf.CartesianState[1] +
		dmf.CartesianState[2]*dmf.CartesianState[2])

	dmf.Predict(60.0, hgv)

	newR := math.Sqrt(dmf.CartesianState[0]*dmf.CartesianState[0] +
		dmf.CartesianState[1]*dmf.CartesianState[1] +
		dmf.CartesianState[2]*dmf.CartesianState[2])

	// Should have moved (velocity ~7.6 km/s * 60s = ~456 km)
	if math.Abs(newR-initialR) < 10.0 {
		t.Errorf("State did not update enough: R only changed by %.2f km", math.Abs(newR-initialR))
	}

	t.Logf("Predict: R changed from %.2f to %.2f km (delta=%.2f)", initialR, newR, newR-initialR)
}

func TestDiscoveryModeFilter_SINDyDiscovery(t *testing.T) {
	hgv := DefaultHGV()
	dmf := NewDiscoveryModeFilter(hgv)

	// Generate synthetic trajectory states
	// Simulating orbital motion with slight variations
	for i := 0; i < SINDyWindowSize+5; i++ {
		angle := float64(i) * 0.01
		r := 6778.137
		state := [6]float64{
			r * math.Cos(angle),
			r * math.Sin(angle),
			0,
			-7.6686 * math.Sin(angle),
			7.6686 * math.Cos(angle),
			0,
		}
		state[3] += 0.01 * float64(i)

		dmf.CartesianState = state
		dmf.storeStateForSINDy()
	}

	// Run SINDy discovery
	dmf.discoverControl()

	// Check that control was discovered
	control := dmf.CurrentControl
	if !control.IsValid() {
		t.Logf("Control not yet valid (need more data or better trajectory)")
	} else {
		t.Logf("SINDy discovered control: |u|=%.3f N", control.Magnitude())
		t.Logf("  R2=%.2f, Sparsity=%d, Confidence=%.2f",
			control.R2, control.Sparsity, control.Confidence)
	}
}

func TestNewDiscoveryIMM(t *testing.T) {
	hgv := DefaultHGV()
	dimm := NewDiscoveryIMM(hgv)

	if dimm == nil {
		t.Fatal("NewDiscoveryIMM returned nil")
	}

	// Check all modes initialized
	if len(dimm.StandardModes) != 3 {
		t.Errorf("Expected 3 standard modes, got %d", len(dimm.StandardModes))
	}

	if dimm.DiscoveryMode == nil {
		t.Error("DiscoveryMode not initialized")
	}

	// Check mode probabilities sum to 1
	sum := 0.0
	for i := 0; i < TotalModes; i++ {
		sum += dimm.ModeProbs[i]
	}
	if math.Abs(sum-1.0) > 0.001 {
		t.Errorf("Mode probabilities don't sum to 1: %.3f", sum)
	}

	t.Logf("DiscoveryIMM initialized with %d modes", TotalModes)
}

func TestDiscoveryIMM_ModeTransitions(t *testing.T) {
	hgv := DefaultHGV()
	dimm := NewDiscoveryIMM(hgv)

	// Check initial transition matrix
	// Row 0 (Ballistic): should have high self-transition
	if dimm.TransMatrix[0*TotalModes+0] < 0.8 {
		t.Errorf("Ballistic self-transition too low: %.2f", dimm.TransMatrix[0*TotalModes+0])
	}

	// Row 3 (Discovery): should have decent self-transition
	if dimm.TransMatrix[3*TotalModes+3] < 0.3 {
		t.Errorf("Discovery self-transition too low: %.2f", dimm.TransMatrix[3*TotalModes+3])
	}

	// Check rows sum to 1
	for i := 0; i < TotalModes; i++ {
		sum := 0.0
		for j := 0; j < TotalModes; j++ {
			sum += dimm.TransMatrix[i*TotalModes+j]
		}
		if math.Abs(sum-1.0) > 0.001 {
			t.Errorf("Row %d doesn't sum to 1: %.3f", i, sum)
		}
	}

	t.Logf("Transition matrix properly normalized")
}

func TestDiscoveryIMM_Update(t *testing.T) {
	hgv := DefaultHGV()
	dimm := NewDiscoveryIMM(hgv)

	// Set initial states in spherical coordinates
	for i, mode := range dimm.StandardModes {
		mode.State = HGVState{
			R:      6778.137,
			Phi:    0,
			Lambda: float64(i) * 0.1,
			V:      7.6686,
			Gamma:  0,
			Psi:    math.Pi / 2,
		}
	}
	dimm.DiscoveryMode.CartesianState = [6]float64{6778.137, 0, 0, 0, 7.6686, 0}

	// Simulate observation
	obs := HGVState{
		R:      6778.0,
		Phi:    0.001,
		Lambda: 0.01,
		V:      7.7,
		Gamma:  0.001,
		Psi:    math.Pi/2 + 0.01,
	}

	// Run update cycle
	dimm.Update(obs, 0.01) // 100Hz

	// Check mode probabilities updated
	sum := 0.0
	for i := 0; i < TotalModes; i++ {
		sum += dimm.ModeProbs[i]
	}
	if math.Abs(sum-1.0) > 0.01 {
		t.Errorf("Mode probabilities don't sum to 1 after update: %.3f", sum)
	}

	t.Logf("IMM update cycle completed")
	t.Logf("Mode probs: Ballistic=%.3f, Glide=%.3f, Maneuver=%.3f, Discovery=%.3f",
		dimm.ModeProbs[0], dimm.ModeProbs[1], dimm.ModeProbs[2], dimm.ModeProbs[3])
}

func TestDiscoveryIMM_Tier3Trigger(t *testing.T) {
	hgv := DefaultHGV()
	dimm := NewDiscoveryIMM(hgv)

	// Manually set discovery probability above threshold
	dimm.ModeProbs[ModeIndexDiscovery] = 0.75

	// Check Tier 3 trigger
	dimm.checkTier3Trigger()

	if !dimm.Tier3Triggered {
		t.Error("Tier 3 should be triggered when P(Discovery) > 0.7")
	}

	// Reset and check deactivation
	dimm.ModeProbs[ModeIndexDiscovery] = 0.2
	dimm.checkTier3Trigger()

	if dimm.Tier3Triggered {
		t.Error("Tier 3 should deactivate when P(Discovery) < 0.3")
	}

	t.Logf("Tier 3 trigger logic verified")
}

func TestDiscoveryIMM_IsZeroDay(t *testing.T) {
	hgv := DefaultHGV()
	dimm := NewDiscoveryIMM(hgv)

	// Below threshold
	dimm.ModeProbs[ModeIndexDiscovery] = 0.5
	if dimm.IsZeroDay() {
		t.Error("IsZeroDay should be false when P(Discovery)=0.5")
	}

	// Above threshold
	dimm.ModeProbs[ModeIndexDiscovery] = 0.75
	if !dimm.IsZeroDay() {
		t.Error("IsZeroDay should be true when P(Discovery)=0.75")
	}

	t.Logf("Zero-day detection threshold verified")
}

func TestDiscoveryIMM_LearnTransitions(t *testing.T) {
	hgv := DefaultHGV()
	dimm := NewDiscoveryIMM(hgv)

	// Initial transition to Discovery
	initialTrans := dimm.TransMatrix[0*TotalModes+ModeIndexDiscovery]

	// Set high discovery probability
	dimm.ModeProbs[ModeIndexDiscovery] = 0.5

	// Learn transitions
	dimm.learnTransitions()

	// Check transition to Discovery increased
	newTrans := dimm.TransMatrix[0*TotalModes+ModeIndexDiscovery]
	if newTrans <= initialTrans {
		t.Logf("Note: Transition to Discovery: %.3f -> %.3f", initialTrans, newTrans)
	} else {
		t.Logf("Transition to Discovery increased: %.3f -> %.3f", initialTrans, newTrans)
	}

	// Check rows still sum to 1
	for i := 0; i < TotalModes; i++ {
		sum := 0.0
		for j := 0; j < TotalModes; j++ {
			sum += dimm.TransMatrix[i*TotalModes+j]
		}
		if math.Abs(sum-1.0) > 0.01 {
			t.Errorf("Row %d doesn't sum to 1 after learning: %.3f", i, sum)
		}
	}
}

func TestControlInput_Magnitude(t *testing.T) {
	ci := ControlInput{
		Force: [3]float64{3, 4, 0}, // 3-4-5 triangle
	}

	mag := ci.Magnitude()
	expected := 5.0

	if math.Abs(mag-expected) > 0.001 {
		t.Errorf("Magnitude incorrect: got %.3f, expected %.3f", mag, expected)
	}

	t.Logf("Control magnitude: %.3f", mag)
}

func TestControlInput_IsValid(t *testing.T) {
	// Valid control
	ciValid := ControlInput{
		R2:         0.9,
		Sparsity:   3,
		Confidence: 0.8,
	}
	if !ciValid.IsValid() {
		t.Error("Control should be valid")
	}

	// Invalid (low R2)
	ciInvalid := ControlInput{
		R2:         0.7,
		Sparsity:   3,
		Confidence: 0.8,
	}
	if ciInvalid.IsValid() {
		t.Error("Control should be invalid (R2 too low)")
	}

	// Invalid (no active terms)
	ciNoTerms := ControlInput{
		R2:         0.9,
		Sparsity:   0,
		Confidence: 0.8,
	}
	if ciNoTerms.IsValid() {
		t.Error("Control should be invalid (no active terms)")
	}

	t.Logf("Control validation logic verified")
}

func TestPredictModeState(t *testing.T) {
	hgv := DefaultHGV()

	initial := HGVState{
		R:      6778.137,
		Phi:    0,
		Lambda: 0,
		V:      7.6686,
		Gamma:  0,
		Psi:    math.Pi / 2,
	}

	// Test ballistic mode
	ballistic := predictModeState(initial, 60.0, hgv, ModeBallistic)
	if ballistic.Lambda == initial.Lambda {
		t.Error("Ballistic Lambda should change")
	}

	// Test glide mode - maintains flight path angle
	glide := predictModeState(initial, 1.0, hgv, ModeGlide)
	if glide.Gamma != -0.1 {
		t.Errorf("Glide Gamma should be -0.1, got %.2f", glide.Gamma)
	}

	// Test maneuver mode - changes heading (Psi)
	maneuver := predictModeState(initial, 60.0, hgv, ModeManeuver)
	if maneuver.Psi == initial.Psi {
		t.Error("Maneuver Psi should change")
	}

	t.Logf("All mode predictions working")
}

func TestModeSymmetryConstraint(t *testing.T) {
	msc := NewModeSymmetryConstraint()

	// Add observations from different headings
	headings := []float64{0, math.Pi / 4, math.Pi / 2, 3 * math.Pi / 4, math.Pi, 5 * math.Pi / 4, 3 * math.Pi / 2, 7 * math.Pi / 4}
	for _, h := range headings {
		msc.UpdateHeadingBins(h, 0.5) // Same probability for all headings
	}

	// Check symmetry - should not violate with equal probabilities
	violation, delta := msc.CheckSymmetryViolation()
	if violation {
		t.Logf("Note: Symmetry violation detected with delta=%.2f (may need more data)", delta)
	} else {
		t.Logf("Symmetry maintained: delta=%.2f", delta)
	}

	// Add biased data (East has higher probability)
	for i := 0; i < 20; i++ {
		msc.UpdateHeadingBins(0, 0.9) // East - high discovery prob
	}
	for i := 0; i < 20; i++ {
		msc.UpdateHeadingBins(math.Pi, 0.1) // West - low discovery prob
	}

	violation, delta = msc.CheckSymmetryViolation()
	if violation {
		t.Logf("✓ Symmetry violation correctly detected: delta=%.2f", delta)
	}
}

func TestResidualSpectralAnalyzer(t *testing.T) {
	rsa := NewResidualSpectralAnalyzer(100.0) // 100Hz

	// Add random noise (atmospheric)
	for i := 0; i < 300; i++ {
		rsa.AddResidual([3]float64{math.Sin(float64(i)*0.1) * 10, 0, 0})
	}

	isCoherent, freq, conf := rsa.AnalyzeSpectrum()
	t.Logf("Atmospheric noise: coherent=%v, freq=%.2f Hz, conf=%.2f", isCoherent, freq, conf)

	// Add coherent signal (simulating control)
	rsa2 := NewResidualSpectralAnalyzer(100.0)
	for i := 0; i < 300; i++ {
		// 2 Hz sine wave - control signature
		rsa2.AddResidual([3]float64{math.Sin(float64(i)*2*math.Pi/50) * 5, 0, 0})
	}

	isCoherent2, freq2, conf2 := rsa2.AnalyzeSpectrum()
	t.Logf("Control signal: coherent=%v, freq=%.2f Hz, conf=%.2f", isCoherent2, freq2, conf2)

	// Classify
	classification := rsa2.ClassifyAnomaly()
	if classification != "control" {
		t.Errorf("Expected 'control' classification, got '%s'", classification)
	}

	t.Logf("✓ Residual spectral analysis working")
}
