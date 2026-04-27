// Package threat provides unit tests for missile and hypersonic detection
package threat

import (
	"testing"
	"time"
)

func TestLaunchTripwireEngine_CheckLaunchTripwire(t *testing.T) {
t.Fatal("NOT IMPLEMENTED: "Launch tripwire test needs review - detection logic may have changed")
	engine := NewLaunchTripwireEngine()

	// Add a test corridor
	corridor := &LaunchCorridor{
		Name:     "TestCorridor",
		MinLat:   20.0,
		MaxLat:   25.0,
		MinLon:   75.0,
		MaxLon:   80.0,
		MinAltKm: 0,
		MaxAltKm: 150,
	}
	engine.AddLaunchCorridor(1, corridor)

	// Test case 1: Valid launch detection
	validObs := LaunchObservation{
		Timestamp:    time.Now(),
		Position:     [3]float64{6471.0, 0, 0}, // 100km altitude
		Velocity:     [3]float64{0, 0, 2.0},    // 2 km/s vertical velocity
		Temperature:  3500.0,                   // Hot plume
		Intensity:    0.8,
		SourceSensor: "IR_SENSOR_1",
	}

	detected, corridorID, confidence := engine.CheckLaunchTripwire(validObs)
	if !detected {
		t.Errorf("Expected launch detection, got false")
	}
	if corridorID != 1 {
		t.Errorf("Expected corridor ID 1, got %d", corridorID)
	}
	if confidence <= 0 {
		t.Errorf("Expected positive confidence, got %f", confidence)
	}

	// Test case 2: Below velocity threshold
	lowVelObs := LaunchObservation{
		Timestamp:    time.Now(),
		Position:     [3]float64{6471.0, 0, 0},
		Velocity:     [3]float64{0, 0, 0.5}, // Below 1 km/s threshold
		Temperature:  3500.0,
		Intensity:    0.8,
		SourceSensor: "IR_SENSOR_1",
	}

	detected, _, _ = engine.CheckLaunchTripwire(lowVelObs)
	if detected {
		t.Errorf("Expected no detection for low velocity, got true")
	}

	// Test case 3: Below temperature threshold
	coldObs := LaunchObservation{
		Timestamp:    time.Now(),
		Position:     [3]float64{6471.0, 0, 0},
		Velocity:     [3]float64{0, 0, 2.0},
		Temperature:  2000.0, // Below 3000K threshold
		Intensity:    0.8,
		SourceSensor: "IR_SENSOR_1",
	}

	detected, _, _ = engine.CheckLaunchTripwire(coldObs)
	if detected {
		t.Errorf("Expected no detection for cold plume, got true")
	}
}

func TestIMMFilter_UpdateIMMFilter(t *testing.T) {
	filter := NewIMMFilter(1)

	// Test observation
	obs := LaunchObservation{
		Timestamp:    time.Now(),
		Position:     [3]float64{6471.0, 0, 0},
		Velocity:     [3]float64{0, 0, 2.0},
		Temperature:  3500.0,
		Intensity:    0.8,
		SourceSensor: "IR_SENSOR_1",
	}

	// Update filter multiple times to build wake history
	for i := 0; i < 15; i++ {
		filter.UpdateIMMFilter(obs)
	}

	// Check that wake history is populated
	if len(filter.WakeHistory) != WakeMemoryDepth {
		t.Errorf("Expected wake history depth %d, got %d", WakeMemoryDepth, len(filter.WakeHistory))
	}

	// Check that state vector is updated
	if len(filter.StateVector) != 6 {
		t.Errorf("Expected state vector length 6, got %d", len(filter.StateVector))
	}

	// Check that constraints are initialized
	if filter.Constraints == nil {
		t.Error("Expected constraints to be initialized")
	}
}

func TestCalculateFractionalLaplacian(t *testing.T) {
t.Fatal("NOT IMPLEMENTED: "Fractional Laplacian test needs review - implementation may have changed")
	// Create wake history
	history := make([]WakeMemoryState, 5)
	for i := range history {
		history[i] = WakeMemoryState{
			Position: [3]float64{float64(i), 0, 0},
			Velocity: [3]float64{float64(i) * 0.1, 0, 0},
			Time:     time.Now(),
			Altitude: 50.0,
		}
	}

	// Calculate fractional Laplacian
	effect := calculateFractionalLaplacian(history, 1.7, 0.95)

	// Check that result is valid (all components should be non-zero)
	for i := 0; i < 3; i++ {
		if effect[i] == 0 {
			t.Errorf("Expected non-zero fractional effect at component %d", i)
		}
	}
}

func TestIsInHGVCorridor(t *testing.T) {
	// Test within corridor
	if !isInHGVCorridor(50.0) {
		t.Error("Expected true for altitude 50km")
	}

	// Test below corridor
	if isInHGVCorridor(20.0) {
		t.Error("Expected false for altitude 20km")
	}

	// Test above corridor
	if isInHGVCorridor(100.0) {
		t.Error("Expected false for altitude 100km")
	}

	// Test at boundaries
	if !isInHGVCorridor(30.0) {
		t.Error("Expected true at lower boundary 30km")
	}
	if !isInHGVCorridor(80.0) {
		t.Error("Expected true at upper boundary 80km")
	}
}

func TestPhysicalConstraints_CheckEnergyConservation(t *testing.T) {
t.Fatal("NOT IMPLEMENTED: "Energy conservation test needs review - constraint logic may have changed")
	pc := NewPhysicalConstraints()

	// Test valid orbital state
	pos := [3]float64{7000.0, 0, 0}
	vel := [3]float64{0, 7.0, 0}

	violated, ratio := pc.CheckEnergyConservation(pos, vel, 1000.0)

	if violated {
		t.Error("Expected no energy violation for valid orbital state")
	}
	if ratio < 0 {
		t.Errorf("Expected non-negative ratio, got %f", ratio)
	}

	// Check that violation was recorded
	if len(pc.EnergyHistory) == 0 {
		t.Error("Expected energy violation to be recorded")
	}
}

func TestPhysicalConstraints_CheckMomentumConservation(t *testing.T) {
	pc := NewPhysicalConstraints()

	pos := [3]float64{7000.0, 0, 0}
	vel := [3]float64{0, 7.0, 0}
	acc := [3]float64{0, 0, -0.0098}

	_, ratio := pc.CheckMomentumConservation(pos, vel, acc, 1000.0)

	if ratio < 0 {
		t.Errorf("Expected non-negative ratio, got %f", ratio)
	}

	// Check that violation was recorded
	if len(pc.MomentumHistory) == 0 {
		t.Error("Expected momentum violation to be recorded")
	}
}

func TestPhysicalConstraints_ApplyAugmentedLagrangian(t *testing.T) {
	pc := NewPhysicalConstraints()

	pos := [3]float64{7000.0, 0, 0}
	vel := [3]float64{0, 7.0, 0}

	// Test with energy violation
	_, adjustedVel := pc.ApplyAugmentedLagrangian(pos, vel, true, 1e-5, false, 0)

	// Check that velocity was adjusted
	velChanged := false
	for i := 0; i < 3; i++ {
		if adjustedVel[i] != vel[i] {
			velChanged = true
			break
		}
	}

	if !velChanged {
		t.Error("Expected velocity to be adjusted for energy violation")
	}

	// Check that lambda increased
	if pc.EnergyLambda <= 1.0 {
		t.Error("Expected energy lambda to increase")
	}
}

func TestSymbolicCertificate(t *testing.T) {
	state := [6]float64{7000.0, 0, 0, 0, 7.0, 0}
	pc := NewPhysicalConstraints()

	cert := NewSymbolicCertificate("TEST-OBS-1", "IR_SENSOR", state, pc)

	if cert.CertificateID == "" {
		t.Error("Expected certificate ID to be generated")
	}

	if cert.HamiltonianHash == "" {
		t.Error("Expected Hamiltonian hash to be generated")
	}

	if !cert.IsValid {
		t.Error("Expected certificate to be valid initially")
	}

	// Test constraint update
	cert.UpdateConstraints(1e-8, 1e-8)
	if cert.ConsistencyGrade != "A" {
		t.Errorf("Expected grade A for small violations, got %s", cert.ConsistencyGrade)
	}

	// Test with large violations
	cert.UpdateConstraints(1e-4, 1e-4)
	if cert.ConsistencyGrade != "F" {
		t.Errorf("Expected grade F for large violations, got %s", cert.ConsistencyGrade)
	}

	// Test prediction verification
	actualState := [6]float64{7010.0, 0, 0, 0, 7.0, 0}
	valid, rmse := cert.VerifyPrediction(actualState)
	if !valid {
		t.Error("Expected prediction to be valid for small deviation")
	}
	if rmse < 0 {
		t.Errorf("Expected non-negative RMSE, got %f", rmse)
	}
}

func TestConfig_DefaultConfig(t *testing.T) {
	config := DefaultConfig()

	if config.MachThreshold != 3.0 {
		t.Errorf("Expected MachThreshold 3.0, got %f", config.MachThreshold)
	}

	if config.FractionalOrder != 1.7 {
		t.Errorf("Expected FractionalOrder 1.7, got %f", config.FractionalOrder)
	}

	if config.WakeMemoryDepth != 10 {
		t.Errorf("Expected WakeMemoryDepth 10, got %d", config.WakeMemoryDepth)
	}
}

func TestConfig_SaveLoad(t *testing.T) {
	config := DefaultConfig()
	config.MachThreshold = 5.0 // Modify a value

	// Save to temp file (in production, use actual temp file)
	err := SaveConfig(config, "/tmp/test_threat_config.json")
	if err != nil {
		t.Fatalf("Failed to save config: %v", err)
	}

	// Load config
	loadedConfig, err := LoadConfig("/tmp/test_threat_config.json")
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	if loadedConfig.MachThreshold != 5.0 {
		t.Errorf("Expected MachThreshold 5.0 after load, got %f", loadedConfig.MachThreshold)
	}
}

func TestThreatFSM_NewThreatFSM(t *testing.T) {
	fsm := NewThreatFSM(12345)

	if fsm.SatID != 12345 {
		t.Errorf("Expected SatID 12345, got %d", fsm.SatID)
	}

	if fsm.CurrentState != StatePassive {
		t.Errorf("Expected initial state Passive, got %v", fsm.CurrentState)
	}

	if fsm.DwellTimer != time.Duration(FSMDwellTimerMinutes)*time.Minute {
		t.Errorf("Expected dwell timer %d minutes, got %v", FSMDwellTimerMinutes, fsm.DwellTimer)
	}
}

func TestThreatFSM_Update(t *testing.T) {
	fsm := NewThreatFSM(12345)

	// Test transition from Passive to Monitoring
	inputs := FSMInput{
		DistanceKm: 50.0, // Within FSMDistanceKm threshold
	}

	changed, msg := fsm.Update(inputs)
	if !changed {
		t.Error("Expected state change to Monitoring")
	}
	if fsm.CurrentState != StateMonitoring {
		t.Errorf("Expected state Monitoring, got %v", fsm.CurrentState)
	}
	if msg == "" {
		t.Error("Expected transition message")
	}
}

func BenchmarkIMMFilter_Update(b *testing.B) {
	filter := NewIMMFilter(1)
	obs := LaunchObservation{
		Timestamp:    time.Now(),
		Position:     [3]float64{6471.0, 0, 0},
		Velocity:     [3]float64{0, 0, 2.0},
		Temperature:  3500.0,
		Intensity:    0.8,
		SourceSensor: "IR_SENSOR_1",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		filter.UpdateIMMFilter(obs)
	}
}

func BenchmarkFractionalLaplacian(b *testing.B) {
	history := make([]WakeMemoryState, 10)
	for i := range history {
		history[i] = WakeMemoryState{
			Position: [3]float64{float64(i), 0, 0},
			Velocity: [3]float64{float64(i) * 0.1, 0, 0},
			Time:     time.Now(),
			Altitude: 50.0,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		calculateFractionalLaplacian(history, 1.7, 0.95)
	}
}
