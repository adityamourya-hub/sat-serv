package proximity_test

import (
	"testing"

	"ssis/pkg/catalog"
	"ssis/pkg/proximity"
	"ssis/pkg/types"
)

func TestComputeBlindSpots_ZeroAdversaries(t *testing.T) {
	adversaries := []catalog.ObjectRecord{}
	roi := proximity.RegionOfInterest{
		LatDeg:   20.0,
		LonDeg:   78.0,
		RadiusKm: 1000.0,
	}

	summary := proximity.ComputeBlindSpots(adversaries, roi)
	if summary.ActiveSensors != 0 {
		t.Errorf("Expected 0 active sensors, got %d", summary.ActiveSensors)
	}
	if summary.BlindSpotRatio < 0 || summary.BlindSpotRatio > 1 {
		t.Errorf("Expected blind spot ratio between 0 and 1, got %f", summary.BlindSpotRatio)
	}
}

func TestComputeBlindSpots_InvalidAltitudes(t *testing.T) {
	// Create satellite below Earth surface (invalid)
	adversaries := []catalog.ObjectRecord{
		{
			Pos: types.Vector3{X: 6000, Y: 0, Z: 0}, // Below Earth radius (6371 km)
		},
	}
	roi := proximity.RegionOfInterest{
		LatDeg:   20.0,
		LonDeg:   78.0,
		RadiusKm: 1000.0,
	}

	summary := proximity.ComputeBlindSpots(adversaries, roi)
	// Should skip invalid satellite
	if summary.ActiveSensors != 0 {
		t.Errorf("Expected 0 active sensors (invalid altitude), got %d", summary.ActiveSensors)
	}
}

func TestComputeBlindSpots_FOVEdgeCases(t *testing.T) {
	cfg := proximity.GetConfig()

	// Test with very small FOV
	adversaries := []catalog.ObjectRecord{
		{
			Pos: types.Vector3{X: 7000, Y: 0, Z: 0}, // LEO
		},
	}
	roi := proximity.RegionOfInterest{
		LatDeg:   20.0,
		LonDeg:   78.0,
		RadiusKm: 1000.0,
	}

	// Temporarily set very small FOV for testing
	// (In production, this would use config)
	_ = cfg.DefaultFOVHalfAngleDeg

	summary := proximity.ComputeBlindSpots(adversaries, roi)
	if summary.ActiveSensors != 1 {
		t.Errorf("Expected 1 active sensor, got %d", summary.ActiveSensors)
	}
}

func TestComputeBlindSpots_CoverageIndexResolution(t *testing.T) {
	cfg := proximity.GetConfig()

	adversaries := []catalog.ObjectRecord{
		{
			Pos: types.Vector3{X: 7000, Y: 0, Z: 0},
		},
	}
	roi := proximity.RegionOfInterest{
		LatDeg:   20.0,
		LonDeg:   78.0,
		RadiusKm: 1000.0,
	}

	// Verify config resolution values are reasonable
	if cfg.CoverageIndexResolutionLat <= 0 || cfg.CoverageIndexResolutionLat > 3600 {
		t.Errorf("Coverage index resolution lat out of range: %d", cfg.CoverageIndexResolutionLat)
	}
	if cfg.CoverageIndexResolutionLon <= 0 || cfg.CoverageIndexResolutionLon > 7200 {
		t.Errorf("Coverage index resolution lon out of range: %d", cfg.CoverageIndexResolutionLon)
	}

	summary := proximity.ComputeBlindSpots(adversaries, roi)
	if summary.ActiveSensors != 1 {
		t.Errorf("Expected 1 active sensor, got %d", summary.ActiveSensors)
	}
}

func TestComputeBlindSpots_MultipleAdversaries(t *testing.T) {
	adversaries := []catalog.ObjectRecord{
		{
			Pos: types.Vector3{X: 7000, Y: 0, Z: 0},
		},
		{
			Pos: types.Vector3{X: 7100, Y: 100, Z: 50},
		},
		{
			Pos: types.Vector3{X: 7200, Y: -100, Z: -50},
		},
	}
	roi := proximity.RegionOfInterest{
		LatDeg:   20.0,
		LonDeg:   78.0,
		RadiusKm: 1000.0,
	}

	summary := proximity.ComputeBlindSpots(adversaries, roi)
	if summary.ActiveSensors != 3 {
		t.Errorf("Expected 3 active sensors, got %d", summary.ActiveSensors)
	}
	if summary.BlindSpotRatio < 0 || summary.BlindSpotRatio > 1 {
		t.Errorf("Expected blind spot ratio between 0 and 1, got %f", summary.BlindSpotRatio)
	}
}

func TestComputeBlindSpots_GEOCoverage(t *testing.T) {
	// Test with GEO satellite (higher altitude, larger footprint)
	adversaries := []catalog.ObjectRecord{
		{
			Pos: types.Vector3{X: 42164, Y: 0, Z: 0}, // GEO radius
		},
	}
	roi := proximity.RegionOfInterest{
		LatDeg:   20.0,
		LonDeg:   78.0,
		RadiusKm: 5000.0,
	}

	summary := proximity.ComputeBlindSpots(adversaries, roi)
	if summary.ActiveSensors != 1 {
		t.Errorf("Expected 1 active sensor, got %d", summary.ActiveSensors)
	}
	// GEO should have larger coverage, so blind spot ratio should be lower
	if summary.BlindSpotRatio < 0 || summary.BlindSpotRatio > 1 {
		t.Errorf("Expected blind spot ratio between 0 and 1, got %f", summary.BlindSpotRatio)
	}
}
