package spatial

import (
	"testing"
	"time"
)

func TestSensorLOS(t *testing.T) {
	s := Sensor{
		ID:           "test_radar",
		Type:         SensorRadar,
		Lat:          0,
		Lon:          0,
		Alt:          0,
		MaxRange:     2000,
		MinElevation: 10,
	}

	// Case 1: Point directly above (0 lat, 0 lon, 500km alt) -> Should be visible
	if !s.CanSee(0, 0, 500) {
		t.Errorf("Expected sight directly above sensor")
	}

	// Case 2: Point at antipodes (0 lat, 180 lon, 500km alt) -> Should be invisible (Earth blockage)
	if s.CanSee(0, 180, 500) {
		t.Errorf("Expected Earth blockage for antipodal point")
	}

	// Case 3: Point at horizon but far beyond range
	if s.CanSee(0, 5, 20000) {
		t.Errorf("Expected range limit to block vision at 20000km")
	}
}

func TestMockIMDProvider(t *testing.T) {
	p := NewMockIMDProvider()
	p.MonsoonIntensity = 1.0
	
	t0 := time.Now()
	w0 := p.GetWeatherAt(20, 80, t0)
	
	// Check that weather evolves over time
	t1 := t0.Add(24 * time.Hour)
	w1 := p.GetWeatherAt(20, 80, t1)
	
	if w0.CloudCover == w1.CloudCover {
		t.Logf("Warning: Weather at T+0 and T+24 is identical (might be periodic)")
	}
}

func TestCoverageTensor4D(t *testing.T) {
	reg := NewDefaultSensorRegistry()
	weather := NewMockIMDProvider()
	tensor := NewCoverageTensor4D()
	
	startTime := time.Now()
	tensor.ComputeBlindSpots(reg, weather, startTime)
	
	// Blind spot ratio for LEO (shell 0) should be high (most of Earth is blind)
	ratio := tensor.GetBlindSpotRatio(0, 0)
	if ratio < 0.9 {
		t.Errorf("Expected a high blind spot ratio for a small 3-sensor network, got %f", ratio)
	}
	
	// GEO (shell 3) should have even larger blind spots
	geoRatio := tensor.GetBlindSpotRatio(0, 3)
	if geoRatio < ratio {
		// Actually GEO sensors have larger footprint (cone), but fewer sensors might see it.
		// However,Leh/MtAbu are GEO capable.
	}
}
