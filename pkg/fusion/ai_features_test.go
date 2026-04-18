package fusion

import (
	"math"
	"testing"
)

// --- Multi-Sensor Fusion Tests ---

func TestWeightedFuserBasic(t *testing.T) {
	fuser := NewWeightedFuser()

	obs := []SensorObservation{
		{
			SensorType: SensorRadar,
			EpochJD:    2460000.0,
			Features: map[string]float64{
				"a_km":             7000.0,
				"eccentricity":     0.001,
				"inclination_rad":  0.9,
			},
			Confidence: 0.95,
		},
		{
			SensorType: SensorOptical,
			EpochJD:    2460000.01,
			Features: map[string]float64{
				"a_km":             7010.0,
				"eccentricity":     0.0015,
				"inclination_rad":  0.89,
			},
			Confidence: 0.80,
		},
	}

	result, err := fuser.Fuse(obs)
	if err != nil {
		t.Fatalf("Fuse failed: %v", err)
	}

	if result.SensorCount != 2 {
		t.Errorf("expected 2 sensors, got %d", result.SensorCount)
	}
	if len(result.SensorTypes) != 2 {
		t.Errorf("expected 2 sensor types, got %d", len(result.SensorTypes))
	}
	if result.Confidence <= 0 || result.Confidence > 1 {
		t.Errorf("confidence out of range: %f", result.Confidence)
	}
	// Fused a_km should be between 7000 and 7010
	if result.SemiMajorAxis < 6999 || result.SemiMajorAxis > 7011 {
		t.Errorf("fused a_km out of range: %f", result.SemiMajorAxis)
	}
	t.Logf("Fused state: a=%.1f km, conf=%.3f, sensors=%v", result.SemiMajorAxis, result.Confidence, result.SensorTypes)
}

func TestWeightedFuserEmpty(t *testing.T) {
	fuser := NewWeightedFuser()
	_, err := fuser.Fuse(nil)
	if err == nil {
		t.Fatal("expected error for empty observations")
	}
}

func TestWeightedFuserAllSensors(t *testing.T) {
	fuser := NewWeightedFuser()

	obs := []SensorObservation{
		{SensorType: SensorRadar, EpochJD: 2460000.0,
			Features: map[string]float64{"a_km": 7000}, Confidence: 0.95},
		{SensorType: SensorSAR, EpochJD: 2460000.01,
			Features: map[string]float64{"a_km": 7005}, Confidence: 0.85},
		{SensorType: SensorOptical, EpochJD: 2460000.02,
			Features: map[string]float64{"a_km": 7002}, Confidence: 0.75},
		{SensorType: SensorNavIC, EpochJD: 2460000.03,
			Features: map[string]float64{"a_km": 6998}, Confidence: 0.60},
	}

	result, err := fuser.Fuse(obs)
	if err != nil {
		t.Fatalf("Fuse failed: %v", err)
	}
	if len(result.SensorTypes) != 4 {
		t.Errorf("expected 4 sensor types, got %d", len(result.SensorTypes))
	}
	// Full diversity should boost confidence
	if result.Confidence < 0.5 {
		t.Errorf("expected higher confidence with 4 sensors, got %.3f", result.Confidence)
	}
	t.Logf("4-sensor fusion: a=%.1f, conf=%.3f", result.SemiMajorAxis, result.Confidence)
}

// --- NavIC Anomaly Detector Tests ---

func TestNavICDetectorNormal(t *testing.T) {
	det := NewNavICDetector()
	history := make([]NavICFeatures, 10)
	for i := range history {
		history[i] = NavICFeatures{
			PRResidualL5: 0.1, PRResidualS: 0.1,
			CN0L5: 45, CN0S: 45,
			IonoDiff: 0.1, CrossCorr: 1.0,
			GDOP: 2.0, ClockDrift: 0.001,
		}
	}

	result, err := det.Detect(history)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if result.Type != "normal" {
		t.Errorf("expected normal, got %s", result.Type)
	}
	t.Logf("Normal detection: %+v", result)
}

func TestNavICDetectorJamming(t *testing.T) {
	det := NewNavICDetector()
	history := make([]NavICFeatures, 10)
	for i := range history {
		history[i] = NavICFeatures{
			PRResidualL5: 0.1, PRResidualS: 0.1,
			CN0L5: 45, CN0S: 45,
			IonoDiff: 0.1, CrossCorr: 1.0,
			GDOP: 2.0,
		}
	}
	// Inject jamming: sudden C/N0 drop + residual spikes
	for i := 5; i < 10; i++ {
		history[i].CN0L5 = 20 // 25 dB drop
		history[i].CN0S = 20
		history[i].PRResidualL5 = 15.0
		history[i].PRResidualS = 12.0
		history[i].CrossCorr = 0.3
	}

	result, err := det.Detect(history)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if result.Type != "jamming" {
		t.Errorf("expected jamming, got %s (score=%.2f)", result.Type, result.Score)
	}
	if result.Score <= 0.5 {
		t.Errorf("expected score > 0.5, got %.2f", result.Score)
	}
	t.Logf("Jamming detection: %+v", result)
}

func TestNavICDetectorSpoofing(t *testing.T) {
	det := NewNavICDetector()
	history := make([]NavICFeatures, 15)
	for i := range history {
		drift := 0.0
		if i > 5 {
			drift = float64(i-5) * 0.3 // Smooth drift
		}
		history[i] = NavICFeatures{
			PRResidualL5: 0.1 + drift,
			PRResidualS:  0.1 + drift*0.7,
			CN0L5:        45, CN0S: 45, // C/N0 stays normal (unlike jamming)
			IonoDiff:     0.1 + drift*0.3,
			CrossCorr:    0.82, // Subtle degradation
			ClockDrift:   drift * 0.05,
			GDOP:         2.0,
		}
	}

	result, err := det.Detect(history)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if result.Type != "spoofing" {
		t.Errorf("expected spoofing, got %s (score=%.2f)", result.Type, result.Score)
	}
	t.Logf("Spoofing detection: %+v", result)
}

// --- Maya Deception Detector Tests ---

func TestMayaDetectorNominal(t *testing.T) {
	det := NewMayaDetector()

	states := []ObjectState{
		{EpochJD: 2460000.0, X: 6800, Y: 0, Z: 0, VX: 0, VY: 7.5, VZ: 0},
		{EpochJD: 2460000.01, X: 6800, Y: 50, Z: 0, VX: 0, VY: 7.5, VZ: 0},
		{EpochJD: 2460000.02, X: 6800, Y: 100, Z: 0, VX: 0, VY: 7.5, VZ: 0},
	}

	result := det.Detect(99999, states, []float64{0.2, 0.25, 0.22}, "smallsat", 430,
		nil, nil)

	if result.MayaScore > 0.3 {
		t.Errorf("expected low maya score for nominal object, got %.2f", result.MayaScore)
	}
	if result.DeceptionType != DeceptionNone {
		t.Errorf("expected no deception, got %s", result.DeceptionType)
	}
	t.Logf("Nominal: %s", result.Summary)
}

func TestMayaDetectorSuspicious(t *testing.T) {
	det := NewMayaDetector()

	// Energy jump without maneuver
	states := []ObjectState{
		{EpochJD: 2460000.0, X: 6800, Y: 0, Z: 0, VX: 0, VY: 7.5, VZ: 0},
		{EpochJD: 2460000.01, X: 6800, Y: 50, Z: 0, VX: 0, VY: 7.5, VZ: 0},
		{EpochJD: 2460000.02, X: 6800, Y: 100, Z: 0, VX: 0, VY: 8.5, VZ: 0}, // Velocity jump!
		{EpochJD: 2460000.03, X: 6800, Y: 150, Z: 0, VX: 0, VY: 8.5, VZ: 0},
	}

	// RCS mismatch for cubesat
	rcs := []float64{0.03, 0.04, 15.0, 14.0, 16.0}

	// Rapid small maneuvers
	maneuvers := []ManeuverEvent{
		{EpochJD: 2460000.01, DeltaVMS: 0.2, DirectionECI: [3]float64{1, 0, 0}, DurationS: 60},
		{EpochJD: 2460000.012, DeltaVMS: 0.1, DirectionECI: [3]float64{-1, 0, 0}, DurationS: 30},
		{EpochJD: 2460000.014, DeltaVMS: 0.15, DirectionECI: [3]float64{1, 0, 0}, DurationS: 45},
		{EpochJD: 2460000.016, DeltaVMS: 0.1, DirectionECI: [3]float64{-1, 0, 0}, DurationS: 30},
	}

	// Catalog anomalies
	tracks := []TrackEvent{
		{EpochJD: 2459999.0, Tracked: false, NObservations: 0, ResidualRMS: 0},
		{EpochJD: 2460000.0, Tracked: true, NObservations: 5, ResidualRMS: 0.5},
		{EpochJD: 2460000.5, Tracked: true, NObservations: 3, ResidualRMS: 8.0},
		{EpochJD: 2460001.0, Tracked: false, NObservations: 0, ResidualRMS: 0},
		{EpochJD: 2460001.5, Tracked: true, NObservations: 2, ResidualRMS: 12.0},
	}

	result := det.Detect(99999, states, rcs, "cubesat", 430, maneuvers, tracks)

	if result.MayaScore < 0.4 {
		t.Errorf("expected high maya score for suspicious object, got %.2f", result.MayaScore)
	}
	if result.DeceptionType == DeceptionNone {
		t.Error("expected some deception type")
	}

	triggeredCount := 0
	for _, e := range result.Evidence {
		if e.Triggered {
			triggeredCount++
		}
	}
	if triggeredCount < 2 {
		t.Errorf("expected at least 2 triggered checks, got %d", triggeredCount)
	}
	t.Logf("Suspicious: maya=%.2f type=%s rec=%s", result.MayaScore, result.DeceptionType, result.Recommendation)
	for _, e := range result.Evidence {
		t.Logf("  [%v] %s: %s (severity=%.2f)", e.Triggered, e.CheckName, e.Description, e.Severity)
	}
}

func TestMayaDetectorDecoy(t *testing.T) {
	det := NewMayaDetector()

	states := []ObjectState{
		{EpochJD: 2460000.0, X: 6800, Y: 0, Z: 0, VX: 0, VY: 7.5, VZ: 0},
		{EpochJD: 2460000.01, X: 6800, Y: 50, Z: 0, VX: 0, VY: 7.5, VZ: 0},
		{EpochJD: 2460000.02, X: 6800, Y: 100, Z: 0, VX: 0, VY: 7.5, VZ: 0},
	}

	// Large RCS for cubesat (decoy balloon?)
	rcs := []float64{18.0, 20.0, 19.0}

	// Rapid small maneuvers (decoy deployment pattern)
	maneuvers := []ManeuverEvent{
		{EpochJD: 2460000.01, DeltaVMS: 0.3, DirectionECI: [3]float64{1, 0, 0}, DurationS: 20},
		{EpochJD: 2460000.0105, DeltaVMS: 0.2, DirectionECI: [3]float64{0, 1, 0}, DurationS: 15},
		{EpochJD: 2460000.011, DeltaVMS: 0.1, DirectionECI: [3]float64{0, 0, 1}, DurationS: 10},
		{EpochJD: 2460000.0115, DeltaVMS: 0.15, DirectionECI: [3]float64{-1, 0, 0}, DurationS: 12},
	}

	result := det.Detect(88888, states, rcs, "cubesat", 430, maneuvers, nil)

	if result.DeceptionType != DeceptionDecoy {
		t.Errorf("expected decoy detection, got %s", result.DeceptionType)
	}
	t.Logf("Decoy: maya=%.2f type=%s rec=%s", result.MayaScore, result.DeceptionType, result.Recommendation)
}

// --- Utility Tests ---

func TestSortedMedian(t *testing.T) {
	tests := []struct {
		data     []float64
		expected float64
	}{
		{[]float64{1, 2, 3}, 2.0},
		{[]float64{1, 2, 3, 4}, 2.5},
		{[]float64{5}, 5.0},
		{[]float64{}, 0.0},
	}
	for _, tt := range tests {
		got := sortedMedian(tt.data)
		if math.Abs(got-tt.expected) > 1e-9 {
			t.Errorf("sortedMedian(%v) = %f, want %f", tt.data, got, tt.expected)
		}
	}
}

func TestDotProduct(t *testing.T) {
	a := [3]float64{1, 2, 3}
	b := [3]float64{4, 5, 6}
	got := dotProduct(a, b)
	want := 32.0
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("dotProduct = %f, want %f", got, want)
	}
}

func TestSensorTypeString(t *testing.T) {
	if SensorRadar.String() != "ATeena" {
		t.Errorf("SensorRadar string = %s", SensorRadar.String())
	}
	if SensorNavIC.String() != "NavIC" {
		t.Errorf("SensorNavIC string = %s", SensorNavIC.String())
	}
}
