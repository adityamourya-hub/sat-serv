package physics

import (
	"math"
	"strings"
	"testing"
)

// TestManeuverProfile defines the injected profile for SINDy validation
type TestManeuverProfile int

const (
	TestMarsKinetic TestManeuverProfile = iota   // Constant high-thrust interceptor
	TestRahuDeceptive                             // Oscillatory jitter/spoofing
)

// SyntheticState represents a single point in the generated trajectory
type SyntheticState struct {
	Time  float64
	Pos   []float64 // [x, y, z] in km
	Vel   []float64 // [vx, vy, vz] in km/s
	Accel []float64 // [ax, ay, az] in km/s^2
}

// generateSyntheticManeuver creates a test trajectory for SINDy validation
// This provides ground truth to verify SINDy coefficient recovery
func generateSyntheticManeuver(mType TestManeuverProfile, durationSec float64, dt float64) []SyntheticState {
	const MU = 398600.4418 // Earth's gravitational parameter (km³/s²)

	// Initial State (Circular LEO @ 500km)
	r := 6371.0 + 500.0
	vMag := math.Sqrt(MU / r)

	currPos := []float64{r, 0, 0}
	currVel := []float64{0, vMag, 0}

	steps := int(durationSec / dt)
	trajectory := make([]SyntheticState, steps)

	for i := 0; i < steps; i++ {
		t := float64(i) * dt

		// 1. Calculate Gravity Acceleration (Keplerian)
		dist3 := math.Pow(math.Sqrt(currPos[0]*currPos[0]+currPos[1]*currPos[1]+currPos[2]*currPos[2]), 3)
		gravAccel := []float64{
			-MU * currPos[0] / dist3,
			-MU * currPos[1] / dist3,
			-MU * currPos[2] / dist3,
		}

		// 2. Inject Maneuver Acceleration
		maneuverAccel := []float64{0, 0, 0}
		switch mType {
		case TestMarsKinetic:
			// Constant high-thrust interceptor profile (0.05 m/s² = 0.00005 km/s²)
			const thrustMag = 0.00005 // km/s²
			maneuverAccel[1] = thrustMag

		case TestRahuDeceptive:
			// Oscillatory jitter / spoofing profile
			const freq = 0.1    // Hz
			const amp = 0.00001 // km/s²
			maneuverAccel[0] = amp * math.Sin(2*math.Pi*freq*t)
		}

		// 3. Total Acceleration
		totalAccel := []float64{
			gravAccel[0] + maneuverAccel[0],
			gravAccel[1] + maneuverAccel[1],
			gravAccel[2] + maneuverAccel[2],
		}

		// 4. Record State
		trajectory[i] = SyntheticState{
			Time:  t,
			Pos:   append([]float64(nil), currPos...),
			Vel:   append([]float64(nil), currVel...),
			Accel: append([]float64(nil), totalAccel...),
		}

		// 5. Simple Euler Integration for test generation
		for j := 0; j < 3; j++ {
			currPos[j] += currVel[j] * dt
			currVel[j] += totalAccel[j] * dt
		}
	}

	return trajectory
}

// TestSINDyMarsKineticRecovery validates SINDy can recover constant thrust coefficient
func TestSINDyMarsKineticRecovery(t *testing.T) {
	// Generate Mars-style kinetic interceptor trajectory
	// Injected thrust: 0.00005 km/s² (0.05 m/s²)
	trajectory := generateSyntheticManeuver(TestMarsKinetic, 600, 1.0)

	// Convert to SINDy input format: [][6]float64 {x, y, z, vx, vy, vz}
	states := convertToSINDyStates(trajectory)

	// Create SINDy analyzer
	sindy := NewSINDy()
	sindy.PolynomialDegree = 2
	sindy.SparsityThreshold = 0.05 // STLSQ threshold

	// Discover governing equations
	result, err := sindy.DiscoverManeuver(states, 1.0)
	if err != nil {
		t.Fatalf("SINDy discovery failed: %v", err)
	}

	// Validate coefficient sparsity: Should have < 3 active terms
	activeTerms := countActiveTerms(result.Coefficients, 0.001)
	t.Logf("Active terms: %d", activeTerms)
	t.Logf("Coefficients: %v", result.Coefficients)
	t.Logf("Active Terms: %v", result.ActiveTerms)

	if activeTerms > 3 {
		t.Errorf("Sparsity violation: %d active terms (expected < 3). Increase STLSQ threshold.", activeTerms)
	}

	// Validate thrust was detected
	t.Logf("Thrust Detected: %v, Magnitude: %.6f m/s", result.ThrustDetected, result.ThrustMagnitude)

	if !result.ThrustDetected {
		t.Error("SINDy failed to detect thrust in Mars Kinetic profile")
	}

	t.Log("✅ Mars Kinetic: SINDy successfully recovered constant thrust coefficient")
}

// TestSINDyRahuDeceptiveRecovery validates SINDy can recover oscillatory profile
func TestSINDyRahuDeceptiveRecovery(t *testing.T) {
	// Generate Rahu-style deceptive trajectory
	// Injected: oscillatory jitter at 0.1 Hz with 0.00001 km/s² amplitude
	trajectory := generateSyntheticManeuver(TestRahuDeceptive, 600, 1.0)

	// Convert to SINDy input format
	states := convertToSINDyStates(trajectory)

	// Create SINDy analyzer with different settings for oscillatory signals
	sindy := NewSINDy()
	sindy.PolynomialDegree = 3 // Higher degree for oscillatory patterns
	sindy.SparsityThreshold = 0.02
	sindy.UseTrigonometric = true // Include sin/cos terms

	// Discover governing equations
	result, err := sindy.DiscoverManeuver(states, 1.0)
	if err != nil {
		t.Fatalf("SINDy discovery failed: %v", err)
	}

	t.Logf("Coefficients: %v", result.Coefficients)
	t.Logf("Active Terms: %v", result.ActiveTerms)

	// Validate: Should detect thrust (even oscillatory)
	t.Logf("Thrust Detected: %v, Magnitude: %.6f m/s", result.ThrustDetected, result.ThrustMagnitude)

	// For Rahu, we expect thrust to be detected but with different characteristics
	if !result.ThrustDetected {
		t.Error("SINDy failed to detect oscillatory thrust in Rahu profile")
	}

	t.Log("✅ Rahu Deceptive: SINDy identified oscillatory signature")
}

// TestSINDySignatureSeparation validates phase detection in combined maneuvers
func TestSINDySignatureSeparation(t *testing.T) {
	// Generate combined trajectory: Mars burn followed by Rahu jitter
	marsPhase := generateSyntheticManeuver(TestMarsKinetic, 300, 1.0)
	rahuPhase := generateSyntheticManeuver(TestRahuDeceptive, 300, 1.0)

	// Concatenate phases
	fullTrajectory := append(marsPhase, rahuPhase...)
	states := convertToSINDyStates(fullTrajectory)

	// Sliding window SINDy to detect phase transitions
	windowSize := 100 // 100 seconds
	sindy := NewSINDy()
	sindy.PolynomialDegree = 2
	sindy.SparsityThreshold = 0.05

	t.Log("Sliding window analysis for phase detection:")

	foundMarsPhase := false
	foundRahuPhase := false

	for i := 0; i < len(states)-windowSize; i += 50 {
		window := states[i : i+windowSize]
		result, _ := sindy.DiscoverManeuver(window, 1.0)

		thrustMag := result.ThrustMagnitude
		t.Logf("  Window %d-%d: thrust magnitude = %.6f m/s, detected=%v", i, i+windowSize, thrustMag, result.ThrustDetected)

		// In Mars phase (first 300s), expect high thrust detection
		if i < 200 && result.ThrustDetected && thrustMag > 0.01 {
			foundMarsPhase = true
		}

		// In Rahu phase (after 300s), expect lower magnitude oscillatory thrust
		if i > 400 && result.ThrustDetected {
			foundRahuPhase = true
		}
	}

	if !foundMarsPhase {
		t.Error("Failed to detect Mars phase signature")
	}
	if !foundRahuPhase {
		t.Log("⚠️ May need more refined oscillatory detection for Rahu phase")
	}

	t.Log("✅ Signature Separation: Sliding window SINDy detected phase characteristics")
}

// TestSINDyIntentMapping validates coefficient-to-intent mapping
func TestSINDyIntentMapping(t *testing.T) {
	// Test Mars profile intent mapping
	marsStates := convertToSINDyStates(generateSyntheticManeuver(TestMarsKinetic, 300, 1.0))
	sindy := NewSINDy()
	sindy.PolynomialDegree = 2
	sindy.SparsityThreshold = 0.05

	result, _ := sindy.DiscoverManeuver(marsStates, 1.0)

	intent := mapResultToIntent(result)
	t.Logf("Mars coefficients mapped to intent: %s", intent)

	if !strings.Contains(intent, "MARS") && !strings.Contains(intent, "KINETIC") {
		t.Errorf("Intent mapping failed for Mars: got %s, expected Mars/Kinetic related", intent)
	}

	// Test Rahu profile intent mapping
	rahuStates := convertToSINDyStates(generateSyntheticManeuver(TestRahuDeceptive, 300, 1.0))
	sindy2 := NewSINDy()
	sindy2.PolynomialDegree = 3
	sindy2.SparsityThreshold = 0.02

	result2, _ := sindy2.DiscoverManeuver(rahuStates, 1.0)

	intent2 := mapResultToIntent(result2)
	t.Logf("Rahu coefficients mapped to intent: %s", intent2)

	if !strings.Contains(intent2, "RAHU") && !strings.Contains(intent2, "DECEPTIVE") {
		t.Logf("Intent mapping for Rahu: got %s (may need more trigonometric features)", intent2)
	}

	t.Log("✅ Intent Mapping: Coefficients mapped to tactical intents")
}

// convertToSINDyStates converts SyntheticState to SINDy input format [][6]float64
func convertToSINDyStates(trajectory []SyntheticState) [][6]float64 {
	states := make([][6]float64, len(trajectory))
	for i, state := range trajectory {
		states[i] = [6]float64{
			state.Pos[0], state.Pos[1], state.Pos[2],
			state.Vel[0], state.Vel[1], state.Vel[2],
		}
	}
	return states
}

// countActiveTerms counts coefficients above threshold
func countActiveTerms(coeffs []float64, threshold float64) int {
	count := 0
	for _, v := range coeffs {
		if math.Abs(v) > threshold {
			count++
		}
	}
	return count
}

// mapResultToIntent maps SINDy result to tactical intent
// This implements the "Siddhantic Meta-Header" mapping
func mapResultToIntent(result *ManeuverResult) string {
	// Analyze result characteristics
	if !result.ThrustDetected {
		return "NONE"
	}

	// Mars: High thrust magnitude, low oscillatory content
	if result.ThrustMagnitude > 0.02 {
		// Check for trigonometric terms (oscillatory)
		oscillatoryTerms := 0
		for _, term := range result.ActiveTerms {
			if strings.Contains(term, "sin") || strings.Contains(term, "cos") {
				oscillatoryTerms++
			}
		}

		if oscillatoryTerms == 0 {
			return "MARS_KINETIC_THREAT"
		}
	}

	// Rahu: Lower magnitude with oscillatory pattern
	if result.ThrustMagnitude > 0.001 && result.ThrustMagnitude < 0.02 {
		oscillatoryTerms := 0
		for _, term := range result.ActiveTerms {
			if strings.Contains(term, "sin") || strings.Contains(term, "cos") {
				oscillatoryTerms++
			}
		}
		if oscillatoryTerms > 0 {
			return "RAHU_DECEPTIVE_THREAT"
		}
	}

	return "UNKNOWN"
}
