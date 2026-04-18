package orbital

import (
	"testing"
)

func TestTLEHardening(t *testing.T) {
	t.Run("ColumnStrict_BStar", func(t *testing.T) {
		// Field: " 30142-3" (Exactly 8 chars)
		// 0: ' ' (positive)
		// 1-5: 30142 (0.30142)
		// 6: '-'
		// 7: '3'
		// Result: 0.00030142
		val := parseTLEFloat(" 30142-3")
		expected := 0.00030142
		if (val - expected) > 1e-12 {
			t.Errorf("Expected %f, got %f", expected, val)
		}
	})

	t.Run("ColumnStrict_NegativeMantissa", func(t *testing.T) {
		// Field: "-12345-5"
		val := parseTLEFloat("-12345-5")
		expected := -0.12345e-5
		if (val - expected) > 1e-12 {
			t.Errorf("Expected %f, got %f", expected, val)
		}
	})

	t.Run("ReciprocalTrimming_Padding", func(t *testing.T) {
		// Line 1 with trailing spaces (75 chars total)
		line1 := "1 25544U 98067A   24099.41847222  .00016717  00000+0  30142-3 0  9998    "
		var tle TLE
		err := parseLine1(line1, &tle)
		if err != nil {
			t.Fatalf("parseLine1 failed on padded line: %v", err)
		}
		if tle.SatelliteNumber != 25544 {
			t.Errorf("Expected 25544, got %d", tle.SatelliteNumber)
		}
	})

	t.Run("ReciprocalTrimming_ShortLine", func(t *testing.T) {
		// Line 1 too short
		line1 := "1 25544U 98067A"
		var tle TLE
		err := parseLine1(line1, &tle)
		if err == nil {
			t.Errorf("Expected error for short line, got nil")
		}
	})
}
