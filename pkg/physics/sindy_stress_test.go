package physics

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestSINDyStressTest1000Simulations runs 1,000 simulations with varying SNR
// Target: >95% coefficient recovery rate for Mars/Rahu labels
func TestSINDyStressTest1000Simulations(t *testing.T) {
	t.Fatal("NOT IMPLEMENTED: "Stress test: 1000 simulations — requires STLSQ performance optimization (reduce bootstrap iterations or cache factorizations)")
	const numSimulations = 1000
	const snrLevels = 5 // Test 5 different SNR levels

	marsSuccesses := make([]int, snrLevels)
	rahuSuccesses := make([]int, snrLevels)

	snrValues := []float64{100.0, 50.0, 20.0, 10.0, 5.0} // SNR in dB

	t.Log("========================================")
	t.Log("SINDY RECOVERY STRESS TEST - 1000 SIMULATIONS")
	t.Log("========================================")

	for snrIdx, snr := range snrValues {
		t.Logf("\nSNR Level: %.1f dB", snr)

		marsCorrect := 0
		rahuCorrect := 0

		for i := 0; i < numSimulations/snrLevels; i++ {
			// Test Mars Kinetic profile
			if testMarsRecovery(snr) {
				marsCorrect++
			}

			// Test Rahu Deceptive profile
			if testRahuRecovery(snr) {
				rahuCorrect++
			}
		}

		marsSuccesses[snrIdx] = marsCorrect
		rahuSuccesses[snrIdx] = rahuCorrect

		marsRate := float64(marsCorrect) / float64(numSimulations/snrLevels) * 100
		rahuRate := float64(rahuCorrect) / float64(numSimulations/snrLevels) * 100

		t.Logf("  Mars:  %d/%d (%.1f%%)", marsCorrect, numSimulations/snrLevels, marsRate)
		t.Logf("  Rahu:  %d/%d (%.1f%%)", rahuCorrect, numSimulations/snrLevels, rahuRate)
	}

	// Overall statistics
	totalMarsSuccesses := 0
	totalRahuSuccesses := 0
	for i := 0; i < snrLevels; i++ {
		totalMarsSuccesses += marsSuccesses[i]
		totalRahuSuccesses += rahuSuccesses[i]
	}

	marsRecoveryRate := float64(totalMarsSuccesses) / float64(numSimulations/2) * 100
	rahuRecoveryRate := float64(totalRahuSuccesses) / float64(numSimulations/2) * 100

	t.Log("\n========================================")
	t.Logf("OVERALL RECOVERY RATES:")
	t.Logf("  Mars Kinetic:     %.1f%% (target: >95%%)", marsRecoveryRate)
	t.Logf("  Rahu Deceptive:   %.1f%% (target: >95%%)", rahuRecoveryRate)
	t.Log("========================================")

	// Validate >95% threshold
	if marsRecoveryRate < 95.0 {
		t.Errorf("Mars recovery rate %.1f%% below 95%% threshold", marsRecoveryRate)
	} else {
		t.Logf("✅ Mars recovery PASSED")
	}

	if rahuRecoveryRate < 95.0 {
		t.Errorf("Rahu recovery rate %.1f%% below 95%% threshold", rahuRecoveryRate)
	} else {
		t.Logf("✅ Rahu recovery PASSED")
	}
}

// testMarsRecovery tests a single Mars simulation with given SNR
func testMarsRecovery(snr float64) bool {
	// Generate trajectory with noise
	trajectory := generateNoisyManeuver(TestMarsKinetic, 300, 1.0, snr)

	// Convert to SINDy format
	states := convertToSINDyStates(trajectory)

	// Run SINDy
	sindy := NewSINDy()
	sindy.PolynomialDegree = 2
	sindy.SparsityThreshold = 0.05

	result, err := sindy.DiscoverManeuver(states, 1.0)
	if err != nil {
		return false
	}

	// Check if thrust was detected
	if !result.ThrustDetected {
		return false
	}

	// Check if it's Mars-like (high magnitude, minimal oscillation)
	if result.ThrustMagnitude < 0.01 { // At least 0.01 m/s detected
		return false
	}

	// Check coefficient sparsity
	activeTerms := countActiveCoeffs(result.Coefficients, 0.001)
	if activeTerms > 5 { // Too many terms = overfitting
		return false
	}

	return true
}

// testRahuRecovery tests a single Rahu simulation with given SNR
func testRahuRecovery(snr float64) bool {
	// Generate trajectory with noise
	trajectory := generateNoisyManeuver(TestRahuDeceptive, 300, 1.0, snr)

	// Convert to SINDy format
	states := convertToSINDyStates(trajectory)

	// Run SINDy with trigonometric library
	sindy := NewSINDy()
	sindy.PolynomialDegree = 3
	sindy.SparsityThreshold = 0.02
	sindy.UseTrigonometric = true

	result, err := sindy.DiscoverManeuver(states, 1.0)
	if err != nil {
		return false
	}

	// Check if thrust was detected
	if !result.ThrustDetected {
		return false
	}

	// Check if it's Rahu-like (lower magnitude, oscillatory)
	if result.ThrustMagnitude > 0.05 { // Too high = not Rahu
		return false
	}

	return true
}

// generateNoisyManeuver creates a trajectory with specified SNR
func generateNoisyManeuver(mType TestManeuverProfile, durationSec float64, dt float64, snr float64) []SyntheticState {
	const MU = 398600.4418
	r := 6371.0 + 500.0
	vMag := math.Sqrt(MU / r)

	currPos := []float64{r, 0, 0}
	currVel := []float64{0, vMag, 0}

	steps := int(durationSec / dt)
	trajectory := make([]SyntheticState, steps)

	// Calculate noise amplitude based on SNR
	// Signal amplitude is roughly thrust magnitude
	signalAmp := 0.00005 // 0.05 m/s^2 in km/s^2
	noiseAmp := signalAmp / math.Pow(10, snr/20.0)

	for i := 0; i < steps; i++ {
		t := float64(i) * dt

		// Gravity
		dist3 := math.Pow(math.Sqrt(currPos[0]*currPos[0]+currPos[1]*currPos[1]+currPos[2]*currPos[2]), 3)
		gravAccel := []float64{
			-MU * currPos[0] / dist3,
			-MU * currPos[1] / dist3,
			-MU * currPos[2] / dist3,
		}

		// Maneuver thrust
		maneuverAccel := []float64{0, 0, 0}
		switch mType {
		case TestMarsKinetic:
			const thrustMag = 0.00005
			maneuverAccel[1] = thrustMag
		case TestRahuDeceptive:
			const freq = 0.1
			const amp = 0.00001
			maneuverAccel[0] = amp * math.Sin(2*math.Pi*freq*t)
		}

		// Add Gaussian noise
		noise := []float64{
			rand.NormFloat64() * noiseAmp,
			rand.NormFloat64() * noiseAmp,
			rand.NormFloat64() * noiseAmp,
		}

		totalAccel := []float64{
			gravAccel[0] + maneuverAccel[0] + noise[0],
			gravAccel[1] + maneuverAccel[1] + noise[1],
			gravAccel[2] + maneuverAccel[2] + noise[2],
		}

		trajectory[i] = SyntheticState{
			Time:  t,
			Pos:   append([]float64(nil), currPos...),
			Vel:   append([]float64(nil), currVel...),
			Accel: append([]float64(nil), totalAccel...),
		}

		// Euler integration
		for j := 0; j < 3; j++ {
			currPos[j] += currVel[j] * dt
			currVel[j] += totalAccel[j] * dt
		}
	}

	return trajectory
}

// countActiveCoeffs counts coefficients above threshold
func countActiveCoeffs(coeffs []float64, threshold float64) int {
	count := 0
	for _, v := range coeffs {
		if math.Abs(v) > threshold {
			count++
		}
	}
	return count
}

// TestSINDySlidingWindowStress validates Sliding Window SINDy at 100Hz simulation
func TestSINDySlidingWindowStress(t *testing.T) {
	t.Fatal("NOT IMPLEMENTED: "Sliding window: 30,000 updates — requires STLSQ performance optimization (reduce bootstrap iterations or cache factorizations)")
	// Create sliding window detector
	swsindy := NewSlidingSINDy(50, 5) // 50-state window, stride 5
	swsindy.Regressor.PolynomialDegree = 2
	swsindy.Regressor.SparsityThreshold = 0.05

	t.Log("Sliding Window SINDy Stress Test:")
	t.Log("  Simulating 300 seconds at 100Hz (30,000 updates)")

	// Phase 1: Saturn (0-100s)
	for i := 0; i < 10000; i++ {
		state := generateKeplerianState(float64(i)*0.01, 0.0)
		swsindy.Update(state, float64(i)*0.01)
	}

	// Phase 2: Mars (100-150s)
	for i := 10000; i < 15000; i++ {
		state := generateKeplerianState(float64(i)*0.01, 0.00005)
		swsindy.Update(state, float64(i)*0.01)
	}

	// Phase 3: Rahu (150-200s)
	for i := 15000; i < 20000; i++ {
		t := float64(i) * 0.01
		thrust := 0.00001 * math.Sin(2*math.Pi*0.1*t)
		state := generateKeplerianState(t, thrust)
		swsindy.Update(state, t)
	}

	// Phase 4: Back to Saturn (200-300s)
	for i := 20000; i < 30000; i++ {
		state := generateKeplerianState(float64(i)*0.01, 0.0)
		swsindy.Update(state, float64(i)*0.01)
	}

	// Analyze results
	transitions := swsindy.GetTransitionLog()
	fill, regTime, updates := swsindy.GetBufferStats()

	t.Logf("\nResults:")
	t.Logf("  Total updates: %d", updates)
	t.Logf("  Buffer fill: %d/%d", fill, swsindy.WindowSize)
	t.Logf("  Regression time: %v", regTime)
	t.Logf("  Detected transitions: %d", len(transitions))

	for i, tr := range transitions {
		t.Logf("  Transition %d: %s -> %s [%s] at %s",
			i+1, tr.OldIntent, tr.NewIntent, tr.Urgency,
			tr.Timestamp.Format("15:04:05.000"))
	}

	// Validate performance
	if regTime > 100*time.Millisecond {
		t.Errorf("Regression time %v too slow for real-time operation", regTime)
	}

	t.Log("✅ Sliding Window SINDy stress test complete")
}

// BenchmarkSINDyRegression100States measures SINDy with 100-state window
func BenchmarkSINDyRegression100States(b *testing.B) {
	trajectory := generateSyntheticManeuver(TestMarsKinetic, 100, 1.0)
	states := convertToSINDyStates(trajectory)

	sindy := NewSINDy()
	sindy.PolynomialDegree = 2
	sindy.SparsityThreshold = 0.05

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = sindy.DiscoverManeuver(states, 1.0)
	}
}
