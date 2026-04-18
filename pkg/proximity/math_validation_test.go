package proximity

import (
	"math"
	"testing"
	"time"

	"ssis/pkg/orbital"
)

func TestPrecisionOrbitalConversion(t *testing.T) {
	// 1. Setup: GSAT-10 (GEO) TLE
	tle1 := "1 38779U 12051B   24108.52361111  .00000123  00000-0  00000-0 0  9994"
	tle2 := "2 38779   0.0526 235.4123 0002568 123.4567 236.6543  1.00271258 41233"
	mu := 398600.4418

	// 2. Propagate 1 day (1440 minutes)
	start := time.Date(2024, 4, 17, 12, 0, 0, 0, time.UTC)
	future := start.Add(24 * time.Hour)
	
	p, v, err := orbital.PropagateSGP4(tle1, tle2, future)
	if err != nil {
		t.Fatalf("SGP4 propagation failed: %v", err)
	}

	stateOrig := orbital.State{
		X: p[0], Y: p[1], Z: p[2],
		VX: v[0], VY: v[1], VZ: v[2],
	}

	// 3. Round-trip: State -> Kepler -> State
	kepler := CartesianToKepler(stateOrig, mu)
	stateFinal := KeplerToECI(kepler, kepler.M, mu)

	// 4. Calculate error
	dx := stateOrig.X - stateFinal[0]
	dy := stateOrig.Y - stateFinal[1]
	dz := stateOrig.Z - stateFinal[2]
	errorMeters := math.Sqrt(dx*dx + dy*dy + dz*dz) * 1000.0

	t.Logf("State: %+v", stateOrig)
	t.Logf("Kepler: %+v", kepler)
	t.Logf("Round-trip Error: %.6f meters", errorMeters)

	if errorMeters > 1.0 {
		t.Errorf("Precision check failed! Round-trip error %.6f meters is > 1m", errorMeters)
	}
}

func TestLEOPrecision(t *testing.T) {
	// 1. Setup: ISS (LEO) TLE
	tle1 := "1 25544U 98067A   24108.83510521  .00016717  00000-0  30165-3 0  9990"
	tle2 := "2 25544  51.6409 181.7459 0004546  51.9332 308.1750 15.49841857449557"
	mu := 398600.4418

	start := time.Date(2024, 4, 17, 12, 0, 0, 0, time.UTC)
	future := start.Add(12 * time.Hour) // 12 hours
	
	p, v, err := orbital.PropagateSGP4(tle1, tle2, future)
	if err != nil {
		t.Fatalf("SGP4 propagation failed: %v", err)
	}

	stateOrig := orbital.State{
		X: p[0], Y: p[1], Z: p[2],
		VX: v[0], VY: v[1], VZ: v[2],
	}

	kepler := CartesianToKepler(stateOrig, mu)
	stateFinal := KeplerToECI(kepler, kepler.M, mu)

	dx := stateOrig.X - stateFinal[0]
	dy := stateOrig.Y - stateFinal[1]
	dz := stateOrig.Z - stateFinal[2]
	errorMeters := math.Sqrt(dx*dx + dy*dy + dz*dz) * 1000.0

	t.Logf("LEO Error: %.6f meters", errorMeters)
	if errorMeters > 1.0 {
		t.Errorf("LEO Precision check failed! Round-trip error %.6f meters is > 1m", errorMeters)
	}
}
