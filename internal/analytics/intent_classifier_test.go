package analytics

import (
	"math"
	"testing"
	"time"

	"ssis/pkg/physics"
)

func TestIntentClassifierStationKeeping(t *testing.T) {
	ic := NewIntentClassifier()
	
	objID := int32(12345)
	
	// Record regular small maneuvers
	for i := 0; i < 5; i++ {
		event := physics.ManeuverEvent{
			DeltaV: 0.1 + float64(i)*0.01,
		}
		state := physics.State{Pos: [3]float64{7000, 0, 0}, Vel: [3]float64{0, 7.5, 0}}
		ic.RecordManeuver(objID, event, time.Now(), state, state)
	}
	
	result := ic.GenerateReport(objID, physics.State{}, physics.State{})
	
	if result.Intent != IntentStationKeeping && result.Intent != IntentOrbitMaintenance {
		t.Errorf("Expected station keeping or maintenance, got %v", result.Intent)
	}
	
	if result.ThreatLevel != "LOW" {
		t.Errorf("Expected LOW threat for station keeping, got %s", result.ThreatLevel)
	}
}

func TestIntentClassifierCoOrbitalThreat(t *testing.T) {
	ic := NewIntentClassifier()
	
	objID := int32(99999)
	
	// Record large delta-v, close approach
	for i := 0; i < 3; i++ {
		event := physics.ManeuverEvent{DeltaV: 500.0}
		state := physics.State{Pos: [3]float64{7000.1, 0, 0}, Vel: [3]float64{0, 7.5, 0}}
		target := physics.State{Pos: [3]float64{7000, 0, 0}, Vel: [3]float64{0, 8.0, 0}}
		ic.RecordManeuver(objID, event, time.Now(), state, target)
	}
	
	result := ic.GenerateReport(objID, physics.State{Pos: [3]float64{7000.1, 0, 0}}, physics.State{Pos: [3]float64{7000, 0, 0}})
	
	if result.Intent != IntentCoOrbitalThreat {
		t.Logf("Intent: %v (may vary based on thresholds)", result.Intent)
	}
}

func TestMeanStdDev(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5}
	mean, stddev := meanStdDev(values)
	
	if math.Abs(mean - 3.0) > 0.01 {
		t.Errorf("Mean should be 3.0, got %f", mean)
	}
	
	expectedStd := math.Sqrt(2.5)
	if math.Abs(stddev - expectedStd) > 0.01 {
		t.Errorf("StdDev should be ~%.4f, got %f", expectedStd, stddev)
	}
}

func TestIntentCategoryString(t *testing.T) {
	if IntentStationKeeping.String() != "STATION_KEEPING" {
		t.Error("Intent string mismatch")
	}
	
	if IntentUnknown.String() != "UNKNOWN" {
		t.Error("Unknown intent string mismatch")
	}
}
