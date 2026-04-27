package physics

import (
	"math"
	"testing"
)

func TestDE440Ephemeris_Initialize(t *testing.T) {
	config := &DE440Config{
		SpkKernelPath:  "test_kernels/de440.bsp", // Won't exist in test
		ReferenceFrame: "J2000",
	}

	eph := NewDE440Ephemeris(config)
	err := eph.Initialize()

	// Should not error even if kernels don't exist (uses analytical fallback)
	if err != nil {
		t.Errorf("Initialize should not fail: %v", err)
	}

	if !eph.loaded {
		t.Error("Ephemeris should be marked loaded after Initialize")
	}
}

func TestDE440Ephemeris_GetMoonPosition(t *testing.T) {
	eph := GetGlobalEphemeris()

	// Test at a known epoch: 2024-01-01 00:00:00 UTC
	// JD 2460310.5
	jd := 2460310.5

	pos := eph.GetMoonPosition(jd)

	// Verify position is reasonable
	// Moon distance ~384,400 km ± 20,000 km
	dist := math.Sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2])

	if dist < 360000 || dist > 410000 {
		t.Errorf("Moon distance %f km out of expected range [360000, 410000]", dist)
	}

	// Moon should be within ~500,000 km of Earth
	maxCoord := math.Max(math.Abs(pos[0]), math.Max(math.Abs(pos[1]), math.Abs(pos[2])))
	if maxCoord > 500000 {
		t.Errorf("Moon coordinate %f exceeds 500000 km", maxCoord)
	}
}

func TestDE440Ephemeris_GetSunPosition(t *testing.T) {
	eph := GetGlobalEphemeris()

	jd := 2460310.5 // 2024-01-01

	pos := eph.GetSunPosition(jd)

	// Verify Sun-Earth distance ~1 AU (149.6 million km)
	dist := math.Sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2])
	AU := 149597870.7

	// Allow 5% tolerance
	if dist < 0.95*AU || dist > 1.05*AU {
		t.Errorf("Sun distance %f km not within 5%% of 1 AU (expected ~%f)", dist, AU)
	}
}

func TestDE440Ephemeris_CalculateThirdBodyAcceleration(t *testing.T) {
	eph := GetGlobalEphemeris()

	// Satellite in LEO at 400 km altitude
	satPos := [3]float64{6778.0, 0, 0} // ~400 km altitude
	jd := 2460310.5

	acc := eph.CalculateThirdBodyAcceleration(satPos, jd)

	// Third-body acceleration should be very small compared to Earth's gravity (~8.7 m/s² at LEO)
	accMag := math.Sqrt(acc[0]*acc[0] + acc[1]*acc[1] + acc[2]*acc[2])

	// Moon perturbation ~0.003 m/s² max, Sun ~0.006 m/s² max
	// Total should be < 0.01 m/s²
	if accMag > 0.01 {
		t.Errorf("Third-body acceleration %f m/s² too large (expected < 0.01)", accMag)
	}

	// Should be non-zero
	if accMag < 1e-10 {
		t.Error("Third-body acceleration should be non-zero")
	}
}

func TestDE440Ephemeris_IsTimeCovered(t *testing.T) {
	eph := NewDE440Ephemeris(nil)

	tests := []struct {
		jd       float64
		expected bool
	}{
		{2298874.5, true},  // 1550-01-01 (boundary)
		{2451545.0, true},  // 2000-01-01 (modern)
		{2916359.5, true},  // 2650-01-01 (boundary)
		{2298874.4, false}, // Before coverage
		{2916359.6, false}, // After coverage
	}

	for _, tt := range tests {
		result := eph.IsTimeCovered(tt.jd)
		if result != tt.expected {
			t.Errorf("JD %f: expected coverage=%v, got=%v", tt.jd, tt.expected, result)
		}
	}
}

func TestDE440Ephemeris_DE440Coverage(t *testing.T) {
	eph := NewDE440Ephemeris(nil)
	start, end := eph.DE440Coverage()

	// DE440 covers 1550-01-01 to 2650-01-01
	// JD 2298874.5 to 2916359.5
	expectedStart := 2298874.5
	expectedEnd := 2916359.5

	if math.Abs(start-expectedStart) > 0.1 {
		t.Errorf("Expected start JD %f, got %f", expectedStart, start)
	}

	if math.Abs(end-expectedEnd) > 0.1 {
		t.Errorf("Expected end JD %f, got %f", expectedEnd, end)
	}
}

func TestDE440Ephemeris_AccuracyReport(t *testing.T) {
	eph := NewDE440Ephemeris(nil)
	report := eph.AccuracyReport()

	// Should have entries for major bodies
	expectedBodies := []string{"moon", "sun", "mars", "venus", "mercury"}
	for _, body := range expectedBodies {
		if _, ok := report[body]; !ok {
			t.Errorf("Accuracy report missing entry for %s", body)
		}
	}

	// Moon should be most accurate (~1 km)
	if report["moon"] > 10 {
		t.Error("Moon accuracy should be < 10 km")
	}
}

func TestPreciseMoonPosition(t *testing.T) {
	eph := NewDE440Ephemeris(nil)

	// Compare with known almanac value for 2024-01-01
	// Using JPL Horizons as reference
	jd := 2460310.5

	pos := eph.preciseMoonPosition(jd)

	// Just check it's reasonable - detailed validation would require
	// comparing against JPL Horizons ephemeris
	dist := math.Sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2])

	if dist < 350000 || dist > 420000 {
		t.Errorf("Moon distance %f km out of range [350000, 420000]", dist)
	}
}

func TestPreciseSunPosition(t *testing.T) {
	t.Fatal("NOT IMPLEMENTED: "SPICE kernels not available - test needs external data")
	eph := NewDE440Ephemeris(nil)

	jd := 2460310.5 // 2024-01-01

	pos := eph.preciseSunPosition(jd)

	// Distance should be close to 1 AU
	dist := math.Sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2])
	AU := 149597870.7

	pctDiff := math.Abs(dist-AU) / AU * 100
	if pctDiff > 1 { // Within 1%
		t.Errorf("Sun distance differs from 1 AU by %.2f%%", pctDiff)
	}
}

func TestGetBodyPosition_UnknownBody(t *testing.T) {
	eph := NewDE440Ephemeris(nil)
	eph.Initialize()

	_, err := eph.GetBodyPosition("pluto", 2451545.0)
	if err == nil {
		t.Error("Expected error for unknown body")
	}
}

func TestGetGlobalEphemeris(t *testing.T) {
	// Should return same instance
	eph1 := GetGlobalEphemeris()
	eph2 := GetGlobalEphemeris()

	if eph1 != eph2 {
		t.Error("GetGlobalEphemeris should return singleton instance")
	}
}
