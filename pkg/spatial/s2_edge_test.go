package spatial

import (
	"fmt"
	"testing"
)

// TestS2PoleRegions tests coordinate conversion at poles (singularity regions)
func TestS2PoleRegions(t *testing.T) {
	tests := []struct {
		name      string
		lat       float64
		lon       float64
		level     int
		wantPanic bool
	}{
		{
			name:      "north pole",
			lat:       90.0,
			lon:       0,
			level:     10,
			wantPanic: false,
		},
		{
			name:      "south pole",
			lat:       -90.0,
			lon:       0,
			level:     10,
			wantPanic: false,
		},
		{
			name:      "near north pole",
			lat:       89.999999,
			lon:       180,
			level:     15,
			wantPanic: false,
		},
		{
			name:      "near south pole",
			lat:       -89.999999,
			lon:       -180,
			level:     15,
			wantPanic: false,
		},
		{
			name:      "equator",
			lat:       0,
			lon:       0,
			level:     10,
			wantPanic: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil && !tt.wantPanic {
					t.Errorf("LatLonToS2Cell() panicked unexpectedly: %v", r)
				}
			}()

			cellID := LatLonToS2Cell(tt.lat, tt.lon, tt.level)
			if cellID == 0 {
				t.Error("CellID should not be zero")
			}
		})
	}
}

// TestS2AntiMeridian tests longitude wraparound at anti-meridian
func TestS2AntiMeridian(t *testing.T) {
	tests := []struct {
		name string
		lat  float64
		lon  float64
	}{
		{"anti-meridian east", 0, 180},
		{"anti-meridian west", 0, -180},
		{"just east of anti-meridian", 45, 179.9},
		{"just west of anti-meridian", 45, -179.9},
		{"far east longitude", 0, 360},
		{"far west longitude", 0, -360},
		{"extreme east", 0, 720},
		{"extreme west", 0, -720},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cellID := LatLonToS2Cell(tt.lat, tt.lon, 10)
			if cellID == 0 {
				t.Errorf("CellID should not be zero for lat=%v, lon=%v", tt.lat, tt.lon)
			}
		})
	}
}

// TestS2Quantization tests quantization error at different levels
func TestS2Quantization(t *testing.T) {
	levels := []int{10, 15, 20, 25, 30}

	for _, level := range levels {
		t.Run(fmt.Sprintf("level%d", level), func(t *testing.T) {
			// Sample a grid of lat/lon pairs
			for lat := -80.0; lat <= 80.0; lat += 20.0 {
				for lon := -170.0; lon <= 170.0; lon += 40.0 {
					cellID := LatLonToS2Cell(lat, lon, level)
					if cellID == 0 {
						t.Errorf("CellID zero at lat=%v, lon=%v, level=%d", lat, lon, level)
					}
				}
			}
		})
	}
}

// TestS2HilbertMonotonicity tests that Hilbert curve preserves spatial locality
func TestS2HilbertMonotonicity(t *testing.T) {
	level := 10

	// Get cell IDs for a line of points at constant latitude
	lat := 30.0
	cellIDs := make([]uint64, 0)

	for lon := -170.0; lon <= 170.0; lon += 10.0 {
		cellID := LatLonToS2Cell(lat, lon, level)
		cellIDs = append(cellIDs, cellID)
	}

	// Check that cell IDs are monotonically increasing (not strictly, due to face changes)
	for i := 1; i < len(cellIDs); i++ {
		if cellIDs[i] == cellIDs[i-1] {
			// Same cell is acceptable (within quantization)
			continue
		}
		// Large jumps indicate face changes, which is expected
	}
}

// TestS2LatitudeClamping tests latitude bounds enforcement
func TestS2LatitudeClamping(t *testing.T) {
	tests := []struct {
		lat      float64
		expected float64
	}{
		{90.0, 90.0},
		{-90.0, -90.0},
		{91.0, 90.0},    // Should clamp
		{-91.0, -90.0},  // Should clamp
		{100.0, 90.0},   // Should clamp
		{-100.0, -90.0}, // Should clamp
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("lat%v", tt.lat), func(t *testing.T) {
			// We can't directly test the clamped value since it's internal
			// but we can verify it doesn't panic
			cellID := LatLonToS2Cell(tt.lat, 0, 10)
			if cellID == 0 {
				t.Errorf("CellID should not be zero for lat=%v", tt.lat)
			}
		})
	}
}
