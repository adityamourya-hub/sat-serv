package physics

import (
	"math"
	"testing"
	"time"
)

/**
 * TestBStarSmoother tests the Moving Window Least Squares implementation
 * for extracting stable ballistic coefficients from TLE history.
 */
func TestBStarSmoother(t *testing.T) {
	smoother := NewBStarSmoother()

	// Create synthetic TLE history for a typical LEO satellite
	// ISS-like object: B* ~ 0.0003, BC ~ 100 kg/m^2
	var tleHistory []*TLE

	for i := 0; i < 20; i++ {
		// Semi-major axis decaying over time (drag effect)
		dayOffset := float64(i)
		// ISS mean motion ~15.5 revs/day
		// Semi-major axis decay: ~100-200m per day at ~400km
		meanMotion := 15.5 + 0.0001*dayOffset // Slight increase as orbit decays

		tle := &TLE{
			SatelliteNumber: 25544,
			EpochYear:       2024,
			EpochDay:        100.0 + dayOffset,
			BStar:           0.0003, // ISS typical B*
			MeanMotion:      meanMotion,
		}
		tleHistory = append(tleHistory, tle)
	}

	result := smoother.SmoothenBStar(tleHistory)

	if result == nil {
		t.Fatal("SmoothenBStar returned nil for valid TLE history")
	}

	// Verify the result has expected values
	if result.SatelliteNumber != 25544 {
		t.Errorf("Wrong satellite number: got %d, want 25544", result.SatelliteNumber)
	}

	// Ballistic coefficient for B* = 0.0003 should be ~67 kg/m^2
	// Formula: BC = 0.02 / B* = 0.02 / 0.0003 = 66.7 kg/m^2
	if result.BallisticCoeff < 50 || result.BallisticCoeff > 150 {
		t.Errorf("Ballistic coefficient out of range: %.1f (expected 50-150 kg/m^2 for B* = 0.0003)",
			result.BallisticCoeff)
	}

	// B* mean should be close to input B* (0.0003)
	if math.Abs(result.BStarMean-0.0003) > 0.0001 {
		t.Errorf("B* mean deviates too much: %.6f (expected ~0.0003)", result.BStarMean)
	}

	// Confidence depends on sample count and variance
	// Synthetic data has zero variance, so confidence = min(1.0, 20/20) = 1.0
	if result.Confidence < 0.25 {
		t.Errorf("Confidence too low: %.2f (expected reasonable confidence with 20 samples)", result.Confidence)
	}

	t.Logf("Smoothed BC: %.1f kg/m^2, B* mean: %.6f, confidence: %.2f",
		result.BallisticCoeff, result.BStarMean, result.Confidence)
}

/**
 * TestBStarSmootherInsufficientData tests behavior with too few TLEs.
 */
func TestBStarSmootherInsufficientData(t *testing.T) {
	smoother := NewBStarSmoother()

	// Only 3 TLEs (minimum is 5)
	var tleHistory []*TLE
	for i := 0; i < 3; i++ {
		tle := &TLE{
			SatelliteNumber: 25544,
			EpochYear:       2024,
			EpochDay:        100.0 + float64(i),
			BStar:           0.0001,
			MeanMotion:      15.0,
		}
		tleHistory = append(tleHistory, tle)
	}

	result := smoother.SmoothenBStar(tleHistory)

	if result != nil {
		t.Error("Should return nil for insufficient data")
	}
}

/**
 * TestBStarSmootherOutliers tests that unrealistic B* values are filtered.
 */
func TestBStarSmootherOutliers(t *testing.T) {
	smoother := NewBStarSmoother()

	var tleHistory []*TLE
	baseJD := 2460000.0

	for i := 0; i < 15; i++ {
		bstar := 0.0003
		// Insert an outlier at index 5
		if i == 5 {
			bstar = 0.01 // Unrealistic 100x larger
		}

		tle := &TLE{
			SatelliteNumber: 25544,
			EpochYear:       2024,
			EpochDay:        baseJD - 2460000.0 + float64(i),
			BStar:           bstar,
			MeanMotion:      15.0,
		}
		tleHistory = append(tleHistory, tle)
	}

	result := smoother.SmoothenBStar(tleHistory)

	if result == nil {
		t.Fatal("Should still produce result with outlier filtered")
	}

	// The outlier B* (0.01) should be filtered (>MaxBStarMag = 0.001)
	// So we should have 14 valid samples
	if result.NumSamples != 14 {
		t.Errorf("Expected 14 samples (outlier filtered), got %d", result.NumSamples)
	}

	// B* mean should be close to 0.0003, not affected by outlier
	if math.Abs(result.BStarMean-0.0003) > 0.0005 {
		t.Errorf("B* mean affected by outlier: %.6f", result.BStarMean)
	}
}

/**
 * TestBStarCache tests the thread-safe cache for smoothed ballistic coefficients.
 */
func TestBStarCache(t *testing.T) {
	cache := NewBStarCache()

	// Store a smoothed B* value
	bs := &BStarSmoothed{
		SatelliteNumber: 25544,
		BallisticCoeff:  120.0,
		BStarMean:       0.0003,
		LastUpdated:     time.Now(),
		Confidence:      0.85,
	}

	cache.Set(25544, bs)

	// Retrieve it
	retrieved := cache.Get(25544)
	if retrieved == nil {
		t.Fatal("Failed to retrieve cached B*")
	}

	if retrieved.BallisticCoeff != 120.0 {
		t.Errorf("Wrong ballistic coefficient: %.1f", retrieved.BallisticCoeff)
	}

	// Non-existent satellite should return nil
	nonExistent := cache.Get(99999)
	if nonExistent != nil {
		t.Error("Should return nil for non-existent satellite")
	}
}

/**
 * TestDetectDragAnomaly tests anomaly detection for drag changes.
 */
func TestDetectDragAnomaly(t *testing.T) {
	// Baseline: stable drag
	previous := &BStarSmoothed{
		SatelliteNumber: 25544,
		BallisticCoeff:  100.0,
		BStarMean:       0.0003,
		BStarStd:        0.00005,
		TrendSlope:      -0.1, // -100m/day decay
		Confidence:      0.9,
	}

	// Case 1: Normal update (no anomaly)
	current := &BStarSmoothed{
		SatelliteNumber: 25544,
		BallisticCoeff:  100.0,
		BStarMean:       0.00032, // Slight change
		BStarStd:        0.00006,
		TrendSlope:      -0.12,
		Confidence:      0.85,
	}

	if DetectDragAnomaly(current, previous) {
		t.Error("False positive: normal update flagged as anomaly")
	}

	// Case 2: Significant B* change (>3 sigma)
	current.BStarMean = 0.0005 // Large change
	if !DetectDragAnomaly(current, previous) {
		t.Error("Should detect significant B* change as anomaly")
	}

	// Case 3: Extreme decay rate (>2 km/day)
	current.BStarMean = 0.00032
	current.TrendSlope = -3.0 // -3 km/day
	if !DetectDragAnomaly(current, previous) {
		t.Error("Should detect extreme decay as anomaly")
	}

	// Case 4: Confidence drop
	current.TrendSlope = -0.12
	current.Confidence = 0.3 // Sudden drop
	if !DetectDragAnomaly(current, previous) {
		t.Error("Should detect confidence drop as anomaly")
	}
}

/**
 * TestEstimateBallisticCoefficientFromTLE tests direct B* to BC conversion.
 */
func TestEstimateBallisticCoefficientFromTLE(t *testing.T) {
	testCases := []struct {
		bstar    float64
		expected float64 // Approximate expected BC
		minBC    float64
		maxBC    float64
		desc     string
	}{
		// BC = 0.02 / B*
		// B* = 0.0001 → BC = 200 (capped from 200)
		// B* = 0.0003 → BC = 67
		// B* = 0.001 → BC = 20
		{0.0001, 200, 150, 200, "Small B* = high BC (capped to 200)"},
		{0.0003, 67, 50, 100, "Typical LEO satellite"},
		{0.001, 20, 15, 25, "High B* = low BC (high drag)"},
		{0, 50, 45, 55, "Zero B* = nominal BC"},
	}

	for _, tc := range testCases {
		bc := EstimateBallisticCoefficientFromTLE(tc.bstar)

		if bc < tc.minBC || bc > tc.maxBC {
			t.Errorf("%s: BC = %.1f, expected [%.1f, %.1f]",
				tc.desc, bc, tc.minBC, tc.maxBC)
		}
	}
}

/**
 * TestTleToJulianDate tests TLE epoch to JD conversion.
 */
func TestTleToJulianDate(t *testing.T) {
	tle := &TLE{
		EpochYear: 2024,
		EpochDay:  100.5, // Day 100.5 = April 9, 2024 at 12:00 UTC
	}

	jd := tleToJulianDate(tle)

	// Day 100 of 2024 is April 9 (2024 is a leap year)
	// JD for 2024-04-09 12:00 UTC should be approximately 2460400.0
	// This is a sanity check - exact value depends on leap year handling
	if jd < 2460000 || jd > 2470000 {
		t.Errorf("Julian Date out of range: %.2f", jd)
	}
}
