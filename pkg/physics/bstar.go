package physics

import (
	"math"
	"sort"
	"time"
)

/**
 * BStarSmoothed represents a filtered, time-averaged ballistic coefficient
 * derived from historical TLE data. This addresses the "corrupted B*" problem
 * where SGP4 fitting absorbs unmodeled perturbations.
 *
 * Reference: ADITI 4.0 High-Fidelity Drag Differentiation Requirements
 */
type BStarSmoothed struct {
	SatelliteNumber int
	BallisticCoeff  float64 // Physical BC = m/(Cd*A) in kg/m^2
	BStarMean       float64 // Smoothed B* from time-series
	BStarStd        float64 // Standard deviation for uncertainty quantification
	LastUpdated     time.Time
	NumSamples      int
	TrendSlope      float64 // da/dt trend for anomaly detection
	Confidence      float64 // 0-1 confidence in the BC estimate
}

/**
 * BStarSmoothedCache maintains per-object smoothed ballistic coefficients.
 * Thread-safe for concurrent access from multiple propagation workers.
 */
type BStarSmoothedCache struct {
	cache map[int]*BStarSmoothed
}

/**
 * NewBStarCache creates a new cache for smoothed ballistic coefficients.
 */
func NewBStarCache() *BStarSmoothedCache {
	return &BStarSmoothedCache{
		cache: make(map[int]*BStarSmoothed),
	}
}

/**
 * Get retrieves the smoothed B* for a satellite, or nil if not available.
 */
func (c *BStarSmoothedCache) Get(satNum int) *BStarSmoothed {
	if c == nil {
		return nil
	}
	return c.cache[satNum]
}

/**
 * Set updates the cache with a new smoothed B* value.
 */
func (c *BStarSmoothedCache) Set(satNum int, bs *BStarSmoothed) {
	if c == nil || c.cache == nil {
		return
	}
	c.cache[satNum] = bs
}

/**
 * TLEWithEpoch represents a TLE with parsed epoch for time-series analysis.
 */
type TLEWithEpoch struct {
	TLE       *TLE
	EpochJD   float64
	EpochTime time.Time
}

/**
 * bstarDataPoint is an internal struct for time-series analysis.
 */
type bstarDataPoint struct {
	jd    float64
	a     float64 // semi-major axis (km)
	bstar float64
}

/**
 * BStarSmoother implements Moving Window Least Squares (MWLS) for B* filtering.
 *
 * Algorithm: Weighted least squares fit to semi-major axis decay over 15-30 days
 * of TLE history to derive a stable, "physical" ballistic coefficient.
 */
type BStarSmoother struct {
	WindowDays       int     // Analysis window (default: 30 days)
	MinSamples       int     // Minimum TLEs required (default: 5)
	MaxBStarMag      float64 // Reject |B*| > this (default: 1e-3, unrealistic drag)
	ReferenceDensity float64 // rho_0 for B* to BC conversion (kg/m^3)
}

/**
 * NewBStarSmoother creates a default MWLS smoother with ADITI 4.0 standards.
 */
func NewBStarSmoother() *BStarSmoother {
	return &BStarSmoother{
		WindowDays:       30,
		MinSamples:       5,
		MaxBStarMag:      1e-3,
		ReferenceDensity: 2.461e-5, // Reference density at one Earth radius (kg/m^3)
	}
}

/**
 * SmoothenBStar performs MWLS on TLE history to extract a stable ballistic coefficient.
 *
 * @param tleHistory: Chronological TLEs for the same object (oldest first)
 * @return: Smoothed B* with confidence metric
 *
 * Physics: da/dt is proportional to drag. By fitting a(t), we separate true drag
 * from SGP4 fitting artifacts.
 */
func (s *BStarSmoother) SmoothenBStar(tleHistory []*TLE) *BStarSmoothed {
	if len(tleHistory) < s.MinSamples {
		return nil
	}

	// Prepare time-series data
	var data []bstarDataPoint
	for _, tle := range tleHistory {
		// Filter unrealistic B* values (fitting artifacts)
		if math.Abs(tle.BStar) > s.MaxBStarMag {
			continue
		}

		// Compute semi-major axis from mean motion (Kepler's 3rd law)
		// n = sqrt(mu/a^3) => a = (mu/n^2)^(1/3)
		// n is in revs/day, convert to rad/s
		nRevsPerDay := tle.MeanMotion
		nRadPerSec := nRevsPerDay * 2 * math.Pi / 86400.0
		a := math.Pow(EARTH_MU_IAU/(nRadPerSec*nRadPerSec), 1.0/3.0)

		jd := tleToJulianDate(tle)
		data = append(data, bstarDataPoint{jd: jd, a: a, bstar: tle.BStar})
	}

	if len(data) < s.MinSamples {
		return nil
	}

	// Sort by time
	sort.Slice(data, func(i, j int) bool {
		return data[i].jd < data[j].jd
	})

	// Check time span
	timeSpan := data[len(data)-1].jd - data[0].jd
	if timeSpan < float64(s.WindowDays) {
		// Not enough history, use simple averaging with lower confidence
		return s.simpleAverage(tleHistory[0].SatelliteNumber, data)
	}

	// Weighted Least Squares fit: a(t) = a0 + slope * (t - t0)
	// Higher weight for more recent data (exponential decay weighting)
	t0 := data[0].jd
	sumW, sumWt, sumWa, sumWat := 0.0, 0.0, 0.0, 0.0
	sumWtt := 0.0

	for i, dp := range data {
		// Exponential weighting: recent = 1.0, 30 days ago = 0.1
		dt := dp.jd - t0
		weight := math.Exp(-dt / float64(s.WindowDays) * 2.3) // half-life ~30 days

		// Also weight by position in array (more recent = higher weight)
		recencyWeight := float64(i+1) / float64(len(data))
		w := weight * recencyWeight

		t := dt
		sumW += w
		sumWt += w * t
		sumWa += w * dp.a
		sumWat += w * dp.a * t
		sumWtt += w * t * t
	}

	// Solve normal equations for linear fit: a = a0 + slope * t
	det := sumW*sumWtt - sumWt*sumWt
	if math.Abs(det) < 1e-12 {
		return nil
	}

	slope := (sumW*sumWat - sumWt*sumWa) / det

	// Compute residuals and B* statistics
	var bstarSum, bstarSqSum float64
	numValid := 0
	for _, dp := range data {
		bstarSum += dp.bstar
		bstarSqSum += dp.bstar * dp.bstar
		numValid++
	}

	bstarMean := bstarSum / float64(numValid)
	bstarVar := (bstarSqSum / float64(numValid)) - bstarMean*bstarMean
	if bstarVar < 0 {
		bstarVar = 0
	}
	bstarStd := math.Sqrt(bstarVar)

	// Convert smoothed B* to physical ballistic coefficient
	// Empirical formula: BC = 0.02 / |B*|
	var bc float64
	if math.Abs(bstarMean) > 1e-10 {
		bc = 0.02 / math.Abs(bstarMean)
		// Clamp to realistic range (5 to 400 kg/m^2)
		if bc < 5.0 {
			bc = 5.0
		}
		if bc > 400.0 {
			bc = 400.0
		}
	} else {
		bc = 50.0 // Nominal value for unknown drag characteristics
	}

	// Confidence based on sample count and variance
	confidence := math.Min(1.0, float64(numValid)/20.0) // 20 samples = full confidence
	if bstarStd > 0 {
		// Lower confidence for high variance
		confidence *= math.Exp(-bstarStd / math.Abs(bstarMean))
	}

	return &BStarSmoothed{
		SatelliteNumber: tleHistory[0].SatelliteNumber,
		BallisticCoeff:  bc,
		BStarMean:       bstarMean,
		BStarStd:        bstarStd,
		LastUpdated:     time.Now(),
		NumSamples:      numValid,
		TrendSlope:      slope, // da/dt in km/day
		Confidence:      confidence,
	}
}

/**
 * simpleAverage computes a basic average when insufficient history for MWLS.
 */
func (s *BStarSmoother) simpleAverage(satNum int, data []bstarDataPoint) *BStarSmoothed {
	var bstarSum, bstarSqSum float64
	for _, dp := range data {
		bstarSum += dp.bstar
		bstarSqSum += dp.bstar * dp.bstar
	}

	n := float64(len(data))
	bstarMean := bstarSum / n
	bstarVar := (bstarSqSum / n) - bstarMean*bstarMean
	if bstarVar < 0 {
		bstarVar = 0
	}

	// Convert to BC: BC = 0.02 / |B*|
	var bc float64
	if math.Abs(bstarMean) > 1e-10 {
		bc = 0.02 / math.Abs(bstarMean)
		if bc < 5.0 {
			bc = 5.0
		}
		if bc > 400.0 {
			bc = 400.0
		}
	} else {
		bc = 50.0
	}

	return &BStarSmoothed{
		SatelliteNumber: satNum,
		BallisticCoeff:  bc,
		BStarMean:       bstarMean,
		BStarStd:        math.Sqrt(bstarVar),
		LastUpdated:     time.Now(),
		NumSamples:      len(data),
		TrendSlope:      0,
		Confidence:      0.3, // Low confidence due to limited data
	}
}

/**
 * tleToJulianDate converts TLE epoch to Julian Date.
 */
func tleToJulianDate(tle *TLE) float64 {
	// TLE epoch: year (2 digits) and day of year (including fraction)
	// Convert to Julian Date
	year := tle.EpochYear
	doy := tle.EpochDay

	// Julian Day for Jan 1 of the epoch year
	var jdYear0 float64
	if year >= 2000 {
		jdYear0 = 2451545.0 + float64(year-2000)*365.25
	} else {
		jdYear0 = 2451545.0 - float64(2000-year)*365.25
	}

	return jdYear0 + doy - 1.0
}

/**
 * DetectDragAnomaly flags significant changes in drag profile that may indicate:
 * - Maneuver (disguised as drag change)
 * - Solar storm effects (increased atmospheric density)
 * - Object breakup or attitude change
 *
 * @param current: Current smoothed B*
 * @param previous: Previous smoothed B* (from earlier update)
 * @return: true if anomaly detected
 */
func DetectDragAnomaly(current, previous *BStarSmoothed) bool {
	if current == nil || previous == nil {
		return false
	}

	// Check for significant B* change (>3 sigma from previous)
	if previous.BStarStd > 0 {
		deltaSigma := math.Abs(current.BStarMean-previous.BStarMean) / previous.BStarStd
		if deltaSigma > 3.0 {
			return true
		}
	}

	// Check for unrealistic trend in semi-major axis
	// Typical LEO decay: ~100-1000 m/day depending on altitude
	if math.Abs(current.TrendSlope) > 2.0 { // > 2 km/day is very high
		return true
	}

	// Check for sudden confidence drop (inconsistent new data)
	if current.Confidence < 0.5 && previous.Confidence > 0.8 {
		return true
	}

	return false
}

/**
 * EstimateBallisticCoefficientFromTLE extracts BC from a single TLE.
 *
 * SGP4 B* is a normalized drag parameter. Empirically, for most satellites:
 * BC ≈ 0.02 / B*  (in kg/m^2)
 *
 * This comes from the relationship where:
 * - B* = 0.0002 → BC ≈ 100 kg/m^2 (typical satellite)
 * - B* = 0.001 → BC ≈ 20 kg/m^2 (high drag object)
 * - B* = 0.00005 → BC ≈ 400 kg/m^2 (low drag, capped)
 *
 * Reference: Vallado, "Fundamentals of Astrodynamics and Applications"
 *
 * @param bstar: TLE B* value
 * @return: Estimated ballistic coefficient in kg/m^2
 */
func EstimateBallisticCoefficientFromTLE(bstar float64) float64 {
	if math.Abs(bstar) < 1e-10 {
		return 50.0 // Nominal value for unknown
	}

	// Empirical conversion: BC = 0.02 / |B*|
	bc := 0.02 / math.Abs(bstar)

	// Clamp to realistic range (5 to 400 kg/m^2 for most space objects)
	if bc < 5.0 {
		return 5.0
	}
	if bc > 400.0 {
		return 400.0
	}
	return bc
}
