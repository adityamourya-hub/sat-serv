package orbital

import (
	"testing"
)

func TestParseTLE(t *testing.T) {
	tleData := `ISS (ZARYA)
1 25544U 98067A   24099.41847222  .00016717  00000+0  30142-3 0  9998
2 25544  51.6416  24.7481 0004817  74.1293  63.3831 15.49830560447388`

	tle, err := ParseTLE(tleData)
	if err != nil {
		t.Fatalf("ParseTLE failed on validated TLE: %v", err)
	}

	if tle.Name != "ISS (ZARYA)" {
		t.Errorf("Expected name ISS (ZARYA), got %s", tle.Name)
	}
	if tle.SatelliteNumber != 25544 {
		t.Errorf("Expected sat num 25544, got %d", tle.SatelliteNumber)
	}
	if tle.Inclination != 51.6416 {
		t.Errorf("Expected inclination 51.6416, got %f", tle.Inclination)
	}
	// Check BStar: 30142-3 -> 0.30142 * 10^-3 = 0.00030142
	expectedBStar := 0.00030142
	if (tle.BStar - expectedBStar) > 1e-10 {
		t.Errorf("Expected BStar %f, got %f", expectedBStar, tle.BStar)
	}
}
