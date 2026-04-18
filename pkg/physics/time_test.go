package physics

import (
	"math"
	"testing"
)

/**
 * TestMoonPosition verifies the lunar ephemeris implementation.
 * Key checks:
 * - Z-component is non-zero (was hardcoded to 0)
 * - Distance varies with anomaly (eccentricity effect)
 * - Position is in correct ballpark (~380,000 km)
 */
func TestMoonPosition(t *testing.T) {
	// Test at a known epoch (J2000.0)
	jd := 2451545.0
	x, y, z := MoonPosition(jd)

	// Check that z is no longer hardcoded to zero
	if math.Abs(z) < 1000 {
		t.Errorf("Moon Z-component too small: %f km (should be ~10,000-40,000 km due to inclination)", z)
	}

	// Check distance is reasonable (350,000 - 410,000 km)
	dist := math.Sqrt(x*x + y*y + z*z)
	if dist < 350000 || dist > 410000 {
		t.Errorf("Moon distance unreasonable: %f km (expected 350,000-410,000 km)", dist)
	}

	t.Logf("Moon at J2000: x=%.1f, y=%.1f, z=%.1f, dist=%.1f km", x, y, z, dist)
}

/**
 * TestMoonPositionEccentricity verifies the distance varies over an orbit period.
 * The Moon's distance varies by ~50,000 km due to eccentricity.
 */
func TestMoonPositionEccentricity(t *testing.T) {
	jd := 2451545.0

	// Sample distances over ~27 days (orbital period)
	var minDist, maxDist float64 = 1e9, 0
	for day := 0; day < 30; day++ {
		x, y, z := MoonPosition(jd + float64(day))
		dist := math.Sqrt(x*x + y*y + z*z)
		if dist < minDist {
			minDist = dist
		}
		if dist > maxDist {
			maxDist = dist
		}
	}

	// Distance should vary by at least 30,000 km (eccentricity effect)
	variation := maxDist - minDist
	if variation < 30000 {
		t.Errorf("Moon distance variation too small: %.1f km (expected >30,000 km due to eccentricity)", variation)
	}

	t.Logf("Moon distance variation over 30 days: %.1f km (range: %.1f - %.1f km)",
		variation, minDist, maxDist)
}

/**
 * TestMoonPositionZComponent verifies the Moon has significant out-of-plane motion.
 * Inclination to ecliptic is ~5.14°, which produces z ~ ±35,000 km.
 */
func TestMoonPositionZComponent(t *testing.T) {
	jd := 2451545.0

	// Find max |z| over an orbit
	maxZ := 0.0
	for day := 0; day < 30; day++ {
		_, _, z := MoonPosition(jd + float64(day))
		if math.Abs(z) > maxZ {
			maxZ = math.Abs(z)
		}
	}

	// Max z should be at least 20,000 km (inclination effect)
	if maxZ < 20000 {
		t.Errorf("Moon max Z-component too small: %.1f km (expected >20,000 km due to ~5° inclination)", maxZ)
	}

	t.Logf("Moon max |Z| over 30 days: %.1f km", maxZ)
}
