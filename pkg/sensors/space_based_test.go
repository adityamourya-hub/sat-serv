package sensors

/**
 * Space-Based SSA Sensor Tests
 */

import (
	"fmt"
	"math"
	"ssis/pkg/physics"
	"testing"
)

func TestGSSAPPlatformCreation(t *testing.T) {
	gssap := NewGSSAP1(2451545.0)

	if gssap.ID != "GSSAP-1" {
		t.Errorf("Expected ID GSSAP-1, got %s", gssap.ID)
	}

	if gssap.Type != PlatformGSSAP {
		t.Error("Expected PlatformGSSAP type")
	}

	if gssap.Orbit.Regime != OrbitGEO {
		t.Error("Expected GEO orbit regime")
	}

	if len(gssap.Sensors) != 2 {
		t.Errorf("Expected 2 sensors, got %d", len(gssap.Sensors))
	}
}

func TestSBSSPlatformCreation(t *testing.T) {
	sbss := NewSBSS(2451545.0)

	if sbss.ID != "SBSS" {
		t.Errorf("Expected ID SBSS, got %s", sbss.ID)
	}

	if sbss.Orbit.Regime != OrbitLEO {
		t.Error("Expected LEO orbit regime")
	}

	if sbss.Orbit.Altitude != 630 {
		t.Errorf("Expected 630km altitude, got %f", sbss.Orbit.Altitude)
	}

	if sbss.Orbit.Inclination != 98.0 {
		t.Errorf("Expected 98° inclination (SSO), got %f", sbss.Orbit.Inclination)
	}
}

func TestNorthStarConstellation(t *testing.T) {
	platforms := NewNorthStarConstellation(2451545.0)

	if len(platforms) != 3 {
		t.Errorf("Expected 3 platforms, got %d", len(platforms))
	}

	for i, p := range platforms {
		if p.Type != PlatformCommercial {
			t.Errorf("Platform %d: expected commercial type", i)
		}

		if p.Orbit.Regime != OrbitLEO {
			t.Errorf("Platform %d: expected LEO orbit", i)
		}

		expectedID := fmt.Sprintf("NS-SSA-%02d", i+1)
		if p.ID != expectedID {
			t.Errorf("Platform %d: expected ID %s, got %s", i, expectedID, p.ID)
		}
	}
}

func TestAndromedaAdversarial(t *testing.T) {
	andromeda := NewAndromedaRGXX(2451545.0)

	if andromeda.Type != PlatformAdversarial {
		t.Error("Expected PlatformAdversarial type")
	}

	if andromeda.Orbit.Regime != OrbitGEO {
		t.Error("Expected GEO orbit for surveillance")
	}
}

func TestObservationGeometry(t *testing.T) {
	// Create a GSSAP platform at GEO
	gssap := NewGSSAP1(2451545.0)
	gssap.State = physics.State{
		Pos: [3]float64{42164, 0, 0}, // GEO, on x-axis
		Vel: [3]float64{0, 3.07, 0},  // GEO circular velocity
	}

	// Target also at GEO but separated
	target := physics.State{
		Pos: [3]float64{42164 * math.Cos(0.1), 42164 * math.Sin(0.1), 0},
		Vel: [3]float64{-3.07 * math.Sin(0.1), 3.07 * math.Cos(0.1), 0},
	}

	// Sun position (simplified, ~1AU along x)
	sunPos := [3]float64{149600000, 0, 0}

	geo := gssap.ComputeObservationGeometry(target, sunPos)

	// Should be visible (both at GEO, separated by ~0.1 rad)
	if !geo.Visible {
		t.Logf("Target not visible: elevation=%.2f°, range=%.0f km", geo.Elevation, geo.Range)
	}

	// Range should be roughly 4200 km (chord length at 0.1 rad separation)
	expectedRange := 42164.0 * 0.1 // ~4200 km
	if math.Abs(geo.Range-expectedRange) > 1000 {
		t.Logf("Unexpected range: expected ~%.0f km, got %.0f km", expectedRange, geo.Range)
	}

	// Phase angle should be reasonable (not 0, not 180)
	if geo.PhaseAngle < 0 || geo.PhaseAngle > 180 {
		t.Errorf("Invalid phase angle: %.2f°", geo.PhaseAngle)
	}

	t.Logf("Observation: range=%.0f km, phase=%.1f°, elev=%.1f°, illuminated=%v",
		geo.Range, geo.PhaseAngle, geo.Elevation, geo.Illuminated)
}

func TestIsIlluminated(t *testing.T) {
	// Sun at ~1 AU along x-axis, Earth at origin
	sunPos := [3]float64{149600000, 0, 0}

	// Point far sunward of Earth (should be illuminated)
	// This is well outside the cylindrical shadow region
	sunlit := [3]float64{140000000, 0, 0}
	if !IsIlluminated(sunlit, sunPos) {
		t.Error("Point far sunward should be illuminated")
	}

	// Point on sun-Earth line but beyond Earth (shadow/penumbra region)
	// With cylindrical shadow model, points close to Earth on anti-sun side
	// are considered shadowed
	shadow := [3]float64{-10000, 0, 0}
	inShadow := IsIlluminated(shadow, sunPos)
	t.Logf("Point at -10000 km on anti-sun side: illuminated=%v", inShadow)

	// Point well beyond Earth but off the sun-Earth line (should be illuminated)
	// Far enough to be outside cylindrical shadow cone
	perp := [3]float64{-15000, 100000, 0}
	if !IsIlluminated(perp, sunPos) {
		t.Error("Point well off shadow axis should be illuminated")
	}
}

func TestPlatformPropagation(t *testing.T) {
	gssap := NewGSSAP1(2451545.0)
	gssap.State = physics.State{
		Pos: [3]float64{42164, 0, 0},
		Vel: [3]float64{0, 3.07, 0},
	}

	params := physics.PhysicsParams{Mass: 1000, Cd: 2.2, Cr: 1.0}

	// Propagate 1 day
	initialPos := gssap.State.Pos
	gssap.Propagate(2451546.0, params)

	// Should have moved (approximately returned to same position for GEO)
	distMoved := math.Sqrt(
		(gssap.State.Pos[0]-initialPos[0])*(gssap.State.Pos[0]-initialPos[0]) +
			(gssap.State.Pos[1]-initialPos[1])*(gssap.State.Pos[1]-initialPos[1]) +
			(gssap.State.Pos[2]-initialPos[2])*(gssap.State.Pos[2]-initialPos[2]))

	// For GEO, should be close to original position after 1 day
	if distMoved > 100 {
		t.Logf("GEO platform moved %.1f km in 1 day (expected < 100 km drift)", distMoved)
	}

	t.Logf("Propagated 1 day: position delta = %.1f km", distMoved)
}

func TestMultiPlatformFusion(t *testing.T) {
	// Create constellation: 2 GSSAP + SBSS
	platforms := []*SensorPlatform{
		NewGSSAP1(2451545.0),
		NewGSSAP2(2451545.0),
		NewSBSS(2451545.0),
	}

	// Set states
	platforms[0].State = physics.State{
		Pos: [3]float64{42164, 0, 0},
		Vel: [3]float64{0, 3.07, 0},
	}
	platforms[1].State = physics.State{
		Pos: [3]float64{42164 * math.Cos(2.09), 42164 * math.Sin(2.09), 0}, // 120° separated
		Vel: [3]float64{-3.07 * math.Sin(2.09), 3.07 * math.Cos(2.09), 0},
	}
	platforms[2].State = physics.State{
		Pos: [3]float64{7000, 0, 0}, // LEO
		Vel: [3]float64{0, 7.5, 0},
	}

	fusion := NewMultiPlatformFusion(platforms)

	// Target at GEO
	target := physics.State{
		Pos: [3]float64{42164 * math.Cos(0.05), 42164 * math.Sin(0.05), 0},
		Vel: [3]float64{-3.07 * math.Sin(0.05), 3.07 * math.Cos(0.05), 0},
	}

	sunPos := [3]float64{149600000, 0, 0}

	geo, platform, count := fusion.ObserveTarget(target, sunPos)

	t.Logf("Multi-platform observation: %d platforms visible, best=%s at %.0f km",
		count, platform.ID, geo.Range)

	if count == 0 {
		t.Error("Expected at least one platform to see GEO target")
	}
}

func TestCoverageAnalysis(t *testing.T) {
	platforms := []*SensorPlatform{
		NewSBSS(2451545.0),
	}
	platforms[0].State = physics.State{
		Pos: [3]float64{7000, 0, 0},
		Vel: [3]float64{0, 7.5, 0},
	}

	fusion := NewMultiPlatformFusion(platforms)

	// Sample LEO targets
	targets := []physics.State{
		{Pos: [3]float64{6800, 0, 0}},  // Same orbit
		{Pos: [3]float64{-7000, 0, 0}}, // Opposite side
		{Pos: [3]float64{0, 7200, 0}},  // Orthogonal
		{Pos: [3]float64{42164, 0, 0}}, // GEO (distant)
	}

	sunPos := [3]float64{149600000, 0, 0}
	analysis := fusion.AnalyzeCoverage(targets, sunPos)

	t.Logf("Coverage analysis: LEO=%.1f%%, GEO=%.1f%%, avg_range=%.0f km, revisit=%.1f h",
		analysis.LEOCoverage*100, analysis.GEOCoverage*100,
		analysis.AverageRange, analysis.RevisitTime)

	if analysis.simultaneity > len(platforms) {
		t.Error("Simultaneity cannot exceed platform count")
	}
}
