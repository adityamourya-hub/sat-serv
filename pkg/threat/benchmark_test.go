// Package threat provides real-world simulation benchmarks for threat detection
package threat

import (
	"fmt"
	"testing"
	"time"
)

// BenchmarkScenario defines a realistic threat scenario for testing
type BenchmarkScenario struct {
	Name              string
	ThreatType        string
	AltitudeKm        float64
	VelocityKms       float64
	TemperatureK      float64
	ExpectedDetection bool
	Description       string
}

var benchmarkScenarios = []BenchmarkScenario{
	{
		Name:              "HGV_SkipGlide_50km",
		ThreatType:        "HGV",
		AltitudeKm:        50.0,
		VelocityKms:       5.0,
		TemperatureK:      2200.0,
		ExpectedDetection: true,
		Description:       "Hypersonic glide vehicle at 50km in transitional flow regime",
	},
	{
		Name:              "HGV_SkipGlide_30km",
		ThreatType:        "HGV",
		AltitudeKm:        30.0,
		VelocityKms:       6.0,
		TemperatureK:      2500.0,
		ExpectedDetection: true,
		Description:       "HGV at lower boundary of transitional flow regime",
	},
	{
		Name:              "HGV_SkipGlide_80km",
		ThreatType:        "HGV",
		AltitudeKm:        80.0,
		VelocityKms:       4.0,
		TemperatureK:      2000.0,
		ExpectedDetection: true,
		Description:       "HGV at upper boundary of transitional flow regime",
	},
	{
		Name:              "Ballistic_Missile_Boost",
		ThreatType:        "BALLISTIC",
		AltitudeKm:        80.0,
		VelocityKms:       2.5,
		TemperatureK:      3500.0,
		ExpectedDetection: true,
		Description:       "Ballistic missile in boost phase with hot plume",
	},
	{
		Name:              "Ballistic_Missile_Midcourse",
		ThreatType:        "BALLISTIC",
		AltitudeKm:        200.0,
		VelocityKms:       6.0,
		TemperatureK:      400.0,
		ExpectedDetection: false,
		Description:       "Ballistic missile in midcourse (cold, high altitude)",
	},
	{
		Name:              "Subsonic_Aircraft",
		ThreatType:        "AIRCRAFT",
		AltitudeKm:        10.0,
		VelocityKms:       0.25,
		TemperatureK:      300.0,
		ExpectedDetection: false,
		Description:       "Subsonic aircraft (should not trigger)",
	},
	{
		Name:              "Space_Debris",
		ThreatType:        "DEBRIS",
		AltitudeKm:        500.0,
		VelocityKms:       7.5,
		TemperatureK:      200.0,
		ExpectedDetection: false,
		Description:       "Orbital debris (cold, high velocity)",
	},
}

// BenchmarkRealWorldScenarios tests threat detection against realistic scenarios
func BenchmarkRealWorldScenarios(b *testing.B) {
	engine := NewLaunchTripwireEngine()

	// Add test corridor
	corridor := &LaunchCorridor{
		Name:     "BenchmarkCorridor",
		MinLat:   20.0,
		MaxLat:   30.0,
		MinLon:   70.0,
		MaxLon:   85.0,
		MinAltKm: 0,
		MaxAltKm: 200,
	}
	engine.AddLaunchCorridor(1, corridor)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scenario := benchmarkScenarios[i%len(benchmarkScenarios)]
		obs := createObservationFromScenario(scenario)
		engine.CheckLaunchTripwire(obs)
	}
}

// BenchmarkHGVTrackingFractional compares fractional vs standard tracking
func BenchmarkHGVTrackingFractional(b *testing.B) {
	filter := NewIMMFilter(1)

	// Simulate HGV trajectory at 50km altitude
	obs := LaunchObservation{
		Timestamp:    time.Now(),
		Position:     [3]float64{6421.0, 0, 0}, // 50km altitude
		Velocity:     [3]float64{0, 5.0, 0},
		Temperature:  2200.0,
		Intensity:    0.8,
		SourceSensor: "IR_SENSOR_1",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		filter.UpdateIMMFilter(obs)
	}
}

// BenchmarkConstraintEnforcement measures overhead of DPHNO constraints
func BenchmarkConstraintEnforcement(b *testing.B) {
	pc := NewPhysicalConstraints()
	pos := [3]float64{7000.0, 0, 0}
	vel := [3]float64{0, 7.0, 0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pc.CheckEnergyConservation(pos, vel, 1000.0)
		pc.CheckMomentumConservation(pos, vel, [3]float64{0, 0, -0.0098}, 1000.0)
		pc.ApplyAugmentedLagrangian(pos, vel, false, 0, false, 0)
	}
}

// BenchmarkSymbolicCertificateGeneration measures certificate creation overhead
func BenchmarkSymbolicCertificateGeneration(b *testing.B) {
	state := [6]float64{7000.0, 0, 0, 0, 7.0, 0}
	pc := NewPhysicalConstraints()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewSymbolicCertificate(fmt.Sprintf("OBS-%d", i), "SENSOR", state, pc)
	}
}

// TestScenarioDetection validates detection accuracy across benchmark scenarios
func TestScenarioDetection(t *testing.T) {
	t.Fatal("NOT IMPLEMENTED: Scenario detection test needs review - detection logic may have changed")
	engine := NewLaunchTripwireEngine()

	// Add test corridor
	corridor := &LaunchCorridor{
		Name:     "TestCorridor",
		MinLat:   20.0,
		MaxLat:   30.0,
		MinLon:   70.0,
		MaxLon:   85.0,
		MinAltKm: 0,
		MaxAltKm: 200,
	}
	engine.AddLaunchCorridor(1, corridor)

	for _, scenario := range benchmarkScenarios {
		t.Run(scenario.Name, func(t *testing.T) {
			obs := createObservationFromScenario(scenario)
			detected, _, _ := engine.CheckLaunchTripwire(obs)

			if detected != scenario.ExpectedDetection {
				t.Errorf("Scenario %s: expected detection=%v, got %v. Description: %s",
					scenario.Name, scenario.ExpectedDetection, detected, scenario.Description)
			}
		})
	}
}

// TestHGVCorridorAccuracy validates fractional dynamics in HGV corridor
func TestHGVCorridorAccuracy(t *testing.T) {
	filter := NewIMMFilter(1)

	// Test altitudes across HGV corridor (30-80km)
	testAltitudes := []float64{25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0}

	for _, alt := range testAltitudes {
		t.Run(fmt.Sprintf("Altitude_%.0fkm", alt), func(t *testing.T) {
			obs := LaunchObservation{
				Timestamp:    time.Now(),
				Position:     [3]float64{6371.0 + alt, 0, 0},
				Velocity:     [3]float64{0, 5.0, 0},
				Temperature:  2200.0,
				Intensity:    0.8,
				SourceSensor: "IR_SENSOR_1",
			}

			// Build wake history
			for i := 0; i < 15; i++ {
				filter.UpdateIMMFilter(obs)
			}

			// Check if fractional dynamics are active in corridor
			inCorridor := isInHGVCorridor(alt)

			if inCorridor && len(filter.WakeHistory) < 2 {
				t.Errorf("Expected wake history to be populated in HGV corridor at %.0fkm", alt)
			}

			// Reset for next test
			filter = NewIMMFilter(1)
		})
	}
}

// TestConstraintEnforcementAccuracy validates DPHNO constraint enforcement
func TestConstraintEnforcementAccuracy(t *testing.T) {
	t.Fatal("NOT IMPLEMENTED: Constraint enforcement test needs review - energy/momentum constraints")
	pc := NewPhysicalConstraints()

	// Test valid orbital state (should have minimal violations)
	validPos := [3]float64{7000.0, 0, 0}
	validVel := [3]float64{0, 7.35, 0} // Circular orbit velocity

	energyViolated, energyRatio := pc.CheckEnergyConservation(validPos, validVel, 1000.0)
	momentumViolated, momentumRatio := pc.CheckMomentumConservation(validPos, validVel, [3]float64{0, 0, -0.0098}, 1000.0)

	if energyViolated {
		t.Errorf("Valid orbital state should not violate energy constraint, ratio=%e", energyRatio)
	}

	if momentumViolated {
		t.Errorf("Valid orbital state should not violate momentum constraint, ratio=%e", momentumRatio)
	}

	// Test invalid state (should have violations)
	invalidVel := [3]float64{0, 20.0, 0} // Unrealistically high velocity

	energyViolated, energyRatio = pc.CheckEnergyConservation(validPos, invalidVel, 1000.0)
	if !energyViolated {
		t.Error("Invalid state should violate energy constraint")
	}
}

// TestCertificateValidity validates symbolic certificate self-invalidation
func TestCertificateValidity(t *testing.T) {
	state := [6]float64{7000.0, 0, 0, 0, 7.0, 0}
	pc := NewPhysicalConstraints()

	cert := NewSymbolicCertificate("TEST-OBS", "SENSOR", state, pc)

	// Certificate should be valid initially
	if !cert.IsValid {
		t.Error("New certificate should be valid")
	}

	// Manually set high sigma to trigger invalidation
	cert.HamiltonianSigma = 1.0
	cert.HamiltonianAvg = 100.0
	cert.validate()

	if cert.IsValid {
		t.Error("Certificate with high sigma should self-invalidate")
	}
}

// createObservationFromScenario creates a LaunchObservation from a benchmark scenario
func createObservationFromScenario(scenario BenchmarkScenario) LaunchObservation {
	return LaunchObservation{
		Timestamp:    time.Now(),
		Position:     [3]float64{6371.0 + scenario.AltitudeKm, 0, 0},
		Velocity:     [3]float64{0, scenario.VelocityKms, 0},
		Temperature:  scenario.TemperatureK,
		Intensity:    0.8,
		SourceSensor: "IR_SENSOR_1",
	}
}

// BenchmarkEndToEndLatency measures full pipeline latency
func BenchmarkEndToEndLatency(b *testing.B) {
	engine := NewLaunchTripwireEngine()
	filter := NewIMMFilter(1)

	// Add corridor
	corridor := &LaunchCorridor{
		Name:     "LatencyCorridor",
		MinLat:   20.0,
		MaxLat:   30.0,
		MinLon:   70.0,
		MaxLon:   85.0,
		MinAltKm: 0,
		MaxAltKm: 200,
	}
	engine.AddLaunchCorridor(1, corridor)

	// HGV scenario
	obs := LaunchObservation{
		Timestamp:    time.Now(),
		Position:     [3]float64{6421.0, 0, 0},
		Velocity:     [3]float64{0, 5.0, 0},
		Temperature:  2200.0,
		Intensity:    0.8,
		SourceSensor: "IR_SENSOR_1",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Full pipeline: detection -> tracking -> constraints
		detected, _, _ := engine.CheckLaunchTripwire(obs)
		if detected {
			filter.UpdateIMMFilter(obs)
		}
	}
}

// BenchmarkConfigReload measures configuration reload overhead
func BenchmarkConfigReload(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = GetConfig()
	}
}

// TestPerformanceMetrics reports performance metrics for threat detection
func TestPerformanceMetrics(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance metrics in short mode")
	}

	engine := NewLaunchTripwireEngine()
	corridor := &LaunchCorridor{
		Name:     "PerfCorridor",
		MinLat:   20.0,
		MaxLat:   30.0,
		MinLon:   70.0,
		MaxLon:   85.0,
		MinAltKm: 0,
		MaxAltKm: 200,
	}
	engine.AddLaunchCorridor(1, corridor)

	// Measure detection latency
	start := time.Now()
	iterations := 10000
	for i := 0; i < iterations; i++ {
		obs := LaunchObservation{
			Timestamp:    time.Now(),
			Position:     [3]float64{6471.0, 0, 0},
			Velocity:     [3]float64{0, 2.5, 0},
			Temperature:  3500.0,
			Intensity:    0.8,
			SourceSensor: "IR_SENSOR_1",
		}
		engine.CheckLaunchTripwire(obs)
	}
	elapsed := time.Since(start)

	avgLatency := elapsed / time.Duration(iterations)
	t.Logf("Average detection latency: %v per observation", avgLatency)
	t.Logf("Throughput: %.0f observations/second", float64(iterations)/elapsed.Seconds())

	// Target: <100ms per observation (from memory)
	if avgLatency > 100*time.Millisecond {
		t.Errorf("Detection latency too high: %v (target: <100ms)", avgLatency)
	}
}
