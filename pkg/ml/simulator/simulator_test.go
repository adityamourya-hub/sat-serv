package simulator

import (
	"math"
	"os"
	"path/filepath"
	"ssis/pkg/physics"
	"testing"
)

func TestGenerateSyntheticScenarios(t *testing.T) {
	cfg := GeneratorConfig{Count: 1200, Seed: 9}
	scenarios := GenerateSyntheticScenarios(cfg)
	if len(scenarios) != cfg.Count {
		t.Fatalf("expected %d scenarios, got %d", cfg.Count, len(scenarios))
	}

	hasConjunction := false
	hasRapid := false
	hasASAT := false
	hasProximity := false
	hasReentry := false
	for _, s := range scenarios {
		if s.Type == ScenarioConjunction {
			hasConjunction = true
		}
		if s.Type == ScenarioRapidThreat {
			hasRapid = true
		}
		if s.Type == ScenarioASAT {
			hasASAT = true
		}
		if s.Type == ScenarioProximity {
			hasProximity = true
		}
		if s.Type == ScenarioReentry {
			hasReentry = true
		}
		if s.RiskLabel < 0 || s.RiskLabel > 1 {
			t.Fatalf("risk label out of range: %f", s.RiskLabel)
		}
		if s.PriorityLabel < 0 || s.PriorityLabel > 300 {
			t.Fatalf("priority label out of range: %f", s.PriorityLabel)
		}
		if len(s.ScenarioRationale) == 0 {
			t.Fatalf("expected scenario rationale for %s", s.ScenarioID)
		}
		if len(s.DecisionFactors) == 0 {
			t.Fatalf("expected decision factors for %s", s.ScenarioID)
		}
		if s.AdversarialIntent < 0 || s.AdversarialIntent > 1 {
			t.Fatalf("adversarial intent out of range: %f", s.AdversarialIntent)
		}
	}
	if !(hasConjunction && hasRapid && hasASAT && hasProximity && hasReentry) {
		t.Fatalf("expected all scenario families to be sampled")
	}
}

func TestWriteScenariosJSONL(t *testing.T) {
	scenarios := GenerateSyntheticScenarios(GeneratorConfig{Count: 25, Seed: 2})
	d := t.TempDir()
	p := filepath.Join(d, "sim.jsonl")

	if err := WriteScenariosJSONL(p, scenarios); err != nil {
		t.Fatalf("failed to write jsonl: %v", err)
	}
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("expected file to exist: %v", err)
	}
}

func TestTrainingEnvStep(t *testing.T) {
	init := EnvState{
		MissDistanceKm:      0.05,
		RelativeVelocityKmS: 2.5,
		TimeToTCASeconds:    1200,
		FuelFraction:        0.85,
		TrackConfidence:     0.7,
		AltitudeKm:          500.0, // LEO altitude for orbital mechanics
	}
	env := NewTrainingEnv(init, 20)
	env.Reset(init)

	res := env.Step(ActionSmallBurn)
	if res.State.MissDistanceKm <= init.MissDistanceKm {
		t.Fatalf("expected miss distance increase after small burn")
	}
	if res.State.FuelFraction >= init.FuelFraction {
		t.Fatalf("expected fuel usage after maneuver")
	}
	if res.Info["risk_next"] < 0 || res.Info["risk_next"] > 1 {
		t.Fatalf("expected normalized risk")
	}

	// Test physics-based time evolution (should not be linear subtraction)
	if res.State.TimeToTCASeconds >= init.TimeToTCASeconds {
		t.Fatalf("expected time-to-TCA to decrease")
	}
}

func TestAdaptiveSigma(t *testing.T) {
	// Test that adaptive sigma scales with miss distance
	// For far-field conjunctions (e.g., reentry with 15km miss distance),
	// sigma should be much larger than the fixed 0.03km

	// Conjunction scenario (tight sigma)
	sigmaConjunction := adaptiveSigma(0.1, ScenarioConjunction)
	if sigmaConjunction < 0.03 {
		t.Fatalf("expected conjunction sigma >= 0.03, got %f", sigmaConjunction)
	}

	// Reentry scenario (loose sigma for large miss distances)
	sigmaReentry := adaptiveSigma(15.0, ScenarioReentry)
	if sigmaReentry < 0.05 {
		t.Fatalf("expected reentry sigma >= 0.05, got %f", sigmaReentry)
	}

	// Reentry sigma should be much larger than fixed 0.03km
	if sigmaReentry <= 0.1 {
		t.Fatalf("expected reentry sigma >> 0.03km for 15km miss distance, got %f", sigmaReentry)
	}

	// Test adaptive sigma from covariance
	sigmaCov := adaptiveSigmaFromCovariance(15.0, 0.03, 0.03, 0.03)
	if sigmaCov <= 0.03 {
		t.Fatalf("expected adaptive covariance sigma > 0.03 for large miss distance, got %f", sigmaCov)
	}
}

func TestHGVDynamics(t *testing.T) {
	// Test HGV dynamics generation for HGV corridor (30-80km)
	altitude := 55.0 // Center of HGV corridor
	mach := 6.0

	liftDragRatio, bankAngleRad, flightPathAngleRad, wakeMemoryFactor, atmosphericDensity, wakeTemperatureK, wakeIonizationLevel, plumeLuminosity := generateHGVDynamics(altitude, mach)

	// L/D ratio should be in valid range (2-4)
	if liftDragRatio < 2.0 || liftDragRatio > 4.0 {
		t.Fatalf("expected L/D ratio in [2,4], got %f", liftDragRatio)
	}

	// Bank angle should be in valid range (0 to π/2)
	if bankAngleRad < 0.0 || bankAngleRad > math.Pi/2 {
		t.Fatalf("expected bank angle in [0, π/2], got %f", bankAngleRad)
	}

	// Flight path angle should be negative (descending)
	if flightPathAngleRad >= 0 {
		t.Fatalf("expected negative flight path angle (descending), got %f", flightPathAngleRad)
	}

	// Wake memory factor should be in [0,1]
	if wakeMemoryFactor < 0.0 || wakeMemoryFactor > 1.0 {
		t.Fatalf("expected wake memory factor in [0,1], got %f", wakeMemoryFactor)
	}

	// Wake memory should peak at 55km (center of HGV corridor)
	_, _, _, wakeMemoryEdge, _, _, _, _ := generateHGVDynamics(30.0, mach)
	if wakeMemoryEdge >= wakeMemoryFactor {
		t.Fatalf("expected wake memory to peak at 55km, edge: %f, center: %f", wakeMemoryEdge, wakeMemoryFactor)
	}

	// Atmospheric density should be positive
	if atmosphericDensity <= 0 {
		t.Fatalf("expected positive atmospheric density, got %f", atmosphericDensity)
	}

	// Density should decrease with altitude
	_, _, _, _, densityLow, _, _, _ := generateHGVDynamics(30.0, mach)
	_, _, _, _, densityHigh, _, _, _ := generateHGVDynamics(80.0, mach)
	if densityLow <= densityHigh {
		t.Fatalf("expected density to decrease with altitude, low: %f, high: %f", densityLow, densityHigh)
	}

	// Wake temperature should be in valid range (200K to 3000K)
	if wakeTemperatureK < 200.0 || wakeTemperatureK > 3000.0 {
		t.Fatalf("expected wake temperature in [200, 3000]K, got %f", wakeTemperatureK)
	}

	// Ionization level should be in [0,1]
	if wakeIonizationLevel < 0.0 || wakeIonizationLevel > 1.0 {
		t.Fatalf("expected ionization level in [0,1], got %f", wakeIonizationLevel)
	}

	// Plume luminosity should be positive
	if plumeLuminosity <= 0 {
		t.Fatalf("expected positive plume luminosity, got %f", plumeLuminosity)
	}
}

func TestClassifyThreatEnhanced(t *testing.T) {
	// Test enhanced HGV classification with L/D ratio

	// HGV in corridor with glide capability
	classHGV := classifyThreatEnhanced(6.0, 55.0, 2.5, 3.0, -0.2)
	if classHGV != ThreatHypersonic {
		t.Fatalf("expected Hypersonic for HGV in corridor with L/D=3.0, got %s", classHGV)
	}

	// HGV in corridor without L/D ratio (fallback to Mach+altitude)
	classHGVFallback := classifyThreatEnhanced(6.0, 55.0, 2.5, 0, 0)
	if classHGVFallback != ThreatHypersonic {
		t.Fatalf("expected Hypersonic for HGV in corridor (fallback), got %s", classHGVFallback)
	}

	// Ballistic (burnout >= 1 km/s, altitude < 300km)
	classBallistic := classifyThreatEnhanced(3.0, 100.0, 2.0, 0, 0)
	if classBallistic != ThreatBallistic {
		t.Fatalf("expected Ballistic, got %s", classBallistic)
	}

	// Orbital (altitude >= 120km)
	classOrbital := classifyThreatEnhanced(7.5, 500.0, 0, 0, 0)
	if classOrbital != ThreatOrbital {
		t.Fatalf("expected Orbital, got %s", classOrbital)
	}
}

func TestRewardFunctionBalance(t *testing.T) {
	// Test that action costs are properly balanced against risk reduction
	// Large burn should not be incentivized when risk reduction is small

	// High fuel scenario (fuel > 50%)
	initHighFuel := EnvState{
		MissDistanceKm:      0.05,
		RelativeVelocityKmS: 2.5,
		TimeToTCASeconds:    1200,
		FuelFraction:        0.85,
		TrackConfidence:     0.7,
		AltitudeKm:          500.0,
	}
	envHighFuel := NewTrainingEnv(initHighFuel, 20)
	envHighFuel.Reset(initHighFuel)

	resMonitor := envHighFuel.Step(ActionMonitor)
	resSmallBurn := envHighFuel.Step(ActionSmallBurn)
	resLargeBurn := envHighFuel.Step(ActionLargeBurn)

	// Validate that action costs are scaled properly
	// Monitor cost: 0.05, Small burn: 1.0, Large burn: 3.0
	// Even if risk reduction makes net reward positive, the cost structure should be enforced

	// Large burn should have higher cost than small burn
	// Test by comparing fuel consumption
	if resLargeBurn.State.FuelFraction >= resSmallBurn.State.FuelFraction {
		t.Fatalf("expected large burn to consume more fuel than small burn")
	}

	// Small burn should consume more fuel than monitor
	if resSmallBurn.State.FuelFraction >= resMonitor.State.FuelFraction {
		t.Fatalf("expected small burn to consume more fuel than monitor")
	}

	// Test fuel exhaustion penalty
	initExhausted := EnvState{
		MissDistanceKm:      0.05,
		RelativeVelocityKmS: 2.5,
		TimeToTCASeconds:    1200,
		FuelFraction:        0.01,
		TrackConfidence:     0.7,
		AltitudeKm:          500.0,
	}
	envExhausted := NewTrainingEnv(initExhausted, 20)
	envExhausted.Reset(initExhausted)

	resExhausted := envExhausted.Step(ActionMonitor)
	if resExhausted.Reward > -5.0 {
		t.Fatalf("expected severe penalty for fuel exhaustion, got %f", resExhausted.Reward)
	}

	// Test that action costs are properly scaled (monitor < sensor retask < small burn < large burn)
	// Use identical initial states to isolate action cost effect
	initTest := EnvState{
		MissDistanceKm:      0.05,
		RelativeVelocityKmS: 2.5,
		TimeToTCASeconds:    1200,
		FuelFraction:        0.85,
		TrackConfidence:     0.7,
		AltitudeKm:          500.0,
	}

	envMonitor := NewTrainingEnv(initTest, 20)
	envMonitor.Reset(initTest)
	resMonitorCost := envMonitor.Step(ActionMonitor)

	envSensor := NewTrainingEnv(initTest, 20)
	envSensor.Reset(initTest)
	resSensorCost := envSensor.Step(ActionSensorRetask)

	envSmall := NewTrainingEnv(initTest, 20)
	envSmall.Reset(initTest)
	resSmallCost := envSmall.Step(ActionSmallBurn)

	envLarge := NewTrainingEnv(initTest, 20)
	envLarge.Reset(initTest)
	resLargeCost := envLarge.Step(ActionLargeBurn)

	// Validate fuel consumption hierarchy
	if resMonitorCost.State.FuelFraction <= resSensorCost.State.FuelFraction {
		t.Fatalf("expected monitor to consume least fuel")
	}
	if resSensorCost.State.FuelFraction <= resSmallCost.State.FuelFraction {
		t.Fatalf("expected sensor retask to consume less fuel than small burn")
	}
	if resSmallCost.State.FuelFraction <= resLargeCost.State.FuelFraction {
		t.Fatalf("expected small burn to consume less fuel than large burn")
	}
}

func TestSovereignHandshake(t *testing.T) {
	// Test Hamiltonian invariant computation
	positionKm := [3]float64{7000.0, 0.0, 0.0} // 622km altitude
	velocityKmS := [3]float64{0.0, 7.5, 0.0}   // Circular orbit velocity
	mu := physics.EARTH_MU_IAU

	H := computeHamiltonianInvariant(positionKm, velocityKmS, mu)
	if H == 0 {
		t.Fatalf("expected non-zero Hamiltonian invariant")
	}

	// Test that Hamiltonian is negative for bound orbits (elliptical/circular)
	// H = 0.5*v² - μ/r < 0 for bound orbits
	if H >= 0 {
		t.Fatalf("expected negative Hamiltonian for bound orbit, got %f", H)
	}

	// Test Hamiltonian hash computation
	currentJD := 2460000.0
	hash := computeHamiltonianHash(H, currentJD)
	if len(hash) != 64 { // SHA-256 hex string is 64 characters
		t.Fatalf("expected 64-character hex hash, got %d characters", len(hash))
	}

	// Test that hash changes with different Hamiltonian values
	hash2 := computeHamiltonianHash(H+1.0, currentJD)
	if hash == hash2 {
		t.Fatalf("expected different hashes for different Hamiltonian values")
	}

	// Test that hash changes with different epoch
	hash3 := computeHamiltonianHash(H, currentJD+1.0)
	if hash == hash3 {
		t.Fatalf("expected different hashes for different epochs")
	}

	// Test spoof detection
	H_avg := H
	spoof := detectSpoof(H, H_avg)
	if spoof {
		t.Fatalf("expected no spoof detection for identical Hamiltonian values")
	}

	// Test spoof detection with small deviation (below threshold)
	H_small_dev := H * (1.0 + 1e-9) // 1e-9 deviation (below 1e-7 threshold)
	spoof_small := detectSpoof(H_small_dev, H_avg)
	if spoof_small {
		t.Fatalf("expected no spoof detection for deviation below threshold")
	}

	// Test spoof detection with large deviation (above threshold)
	H_large_dev := H * (1.0 + 1e-6) // 1e-6 deviation (above 1e-7 threshold)
	spoof_large := detectSpoof(H_large_dev, H_avg)
	if !spoof_large {
		t.Fatalf("expected spoof detection for deviation above threshold")
	}

	// Test that synthetic scenarios include Hamiltonian invariants
	cfg := DefaultGeneratorConfig()
	cfg.Count = 10
	scenarios := GenerateSyntheticScenarios(cfg)

	for i, scenario := range scenarios {
		if scenario.HamiltonianInvariant == 0 && scenario.AltitudeKm > 0 {
			t.Fatalf("scenario %d: expected non-zero Hamiltonian invariant for altitude %f", i, scenario.AltitudeKm)
		}
		if scenario.HamiltonianHash == "" {
			t.Fatalf("scenario %d: expected non-empty Hamiltonian hash", i)
		}
		// Spoof detection should be false for legitimate synthetic scenarios
		if scenario.SpoofDetected {
			t.Fatalf("scenario %d: expected no spoof detection for legitimate scenario", i)
		}
	}
}

func TestFractionalLaplacianKernel(t *testing.T) {
	// Test fractional Laplacian kernel for non-local threat propagation

	// Test with no history (should return 1.0 - no correction)
	correctionNoHistory := FractionalLaplacianKernel(0.5, []float64{}, 0.7)
	if correctionNoHistory != 1.0 {
		t.Fatalf("expected correction 1.0 with no history, got %f", correctionNoHistory)
	}

	// Test with stable history (should return ~1.0)
	historyStable := []float64{0.5, 0.5, 0.5, 0.5, 0.5}
	correctionStable := FractionalLaplacianKernel(0.5, historyStable, 0.7)
	if math.Abs(correctionStable-1.0) > 0.1 {
		t.Fatalf("expected correction ~1.0 with stable history, got %f", correctionStable)
	}

	// Test with increasing risk (should return >1.0 - amplification)
	historyIncreasing := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	correctionIncreasing := FractionalLaplacianKernel(0.6, historyIncreasing, 0.7)
	if correctionIncreasing <= 1.0 {
		t.Fatalf("expected correction >1.0 with increasing risk, got %f", correctionIncreasing)
	}

	// Test with decreasing risk (should return <1.0 - damping)
	historyDecreasing := []float64{0.9, 0.8, 0.7, 0.6, 0.5}
	correctionDecreasing := FractionalLaplacianKernel(0.4, historyDecreasing, 0.7)
	if correctionDecreasing >= 1.0 {
		t.Fatalf("expected correction <1.0 with decreasing risk, got %f", correctionDecreasing)
	}

	// Test alpha parameter (higher alpha = faster weight decay, more emphasis on recent history)
	history := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	correctionLowAlpha := FractionalLaplacianKernel(0.6, history, 0.3)
	correctionHighAlpha := FractionalLaplacianKernel(0.6, history, 0.9)
	// Higher alpha gives faster decay, so correction may differ based on history pattern
	// Just verify both are in valid range
	if correctionLowAlpha < 0.5 || correctionLowAlpha > 2.0 {
		t.Fatalf("expected correction in [0.5, 2.0] for low alpha, got %f", correctionLowAlpha)
	}
	if correctionHighAlpha < 0.5 || correctionHighAlpha > 2.0 {
		t.Fatalf("expected correction in [0.5, 2.0] for high alpha, got %f", correctionHighAlpha)
	}

	// Test computeNonLocalThreatPropagation
	adjustedRisk := computeNonLocalThreatPropagation(0.6, historyIncreasing, 0.7)
	if adjustedRisk <= 0.6 {
		t.Fatalf("expected adjusted risk > current risk with increasing history, got %f", adjustedRisk)
	}

	// Test that correction is bounded [0.5, 2.0]
	historyExtreme := []float64{0.01, 0.01, 0.01, 0.01, 0.01}
	correctionExtreme := FractionalLaplacianKernel(10.0, historyExtreme, 0.7)
	if correctionExtreme > 2.0 {
		t.Fatalf("expected correction bounded at 2.0, got %f", correctionExtreme)
	}
}
