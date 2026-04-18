package physics

import (
	"math"
	"time"
)

/**
 * TimeScaler implements the transformations between relativistic and 
 * coordinate time scales used in high-fidelity SDA operations.
 * 1. UTC (Coordinated Universal Time): Standard clock time (subject to leap seconds).
 * 2. TAI (International Atomic Time): Contiguous atomic scale.
 * 3. TT (Terrestrial Time): Ideal coordinate time at the geoid.
 */

const (
	// JD_EPOCH is the Julian Date of the Unix Epoch (1970-01-01 00:00:00 UTC)
	JD_EPOCH = 2440587.5
	// SEC_PER_DAY is the number of seconds in a standard SI day
	SEC_PER_DAY = 86400.0
)

// LeapSecond represents a historical leap second insertion point.
type LeapSecond struct {
	JD    float64
	Count int
}

// LeapSecondTable is current as of 2024. No leap seconds have been added since Jan 2017.
var LeapSecondTable = []LeapSecond{
	{2441317.5, 10}, // 1972-01-01
	{2441499.5, 11}, // 1972-07-01
	{2441682.5, 12}, // 1973-01-01
	{2442047.5, 13}, // 1974-01-01
	{2442412.5, 14}, // 1975-01-01
	{2442777.5, 15}, // 1976-01-01
	{2443144.5, 16}, // 1977-01-01
	{2443509.5, 17}, // 1978-01-01
	{2443874.5, 18}, // 1979-01-01
	{2444239.5, 19}, // 1980-01-01
	{2444786.5, 20}, // 1981-07-01
	{2445151.5, 21}, // 1982-07-01
	{2445516.5, 22}, // 1983-07-01
	{2445882.5, 23}, // 1985-07-01
	{2447161.5, 24}, // 1988-01-01
	{2447892.5, 25}, // 1990-01-01
	{2448257.5, 26}, // 1991-01-01
	{2448804.5, 27}, // 1992-07-01
	{2449169.5, 28}, // 1993-07-01
	{2449534.5, 29}, // 1994-07-01
	{2449169.5, 30}, // 1996-01-01 (Correction: Feb 1996)
	{2450630.5, 31}, // 1997-07-01
	{2451179.5, 32}, // 1999-01-01
	{2453736.5, 33}, // 2006-01-01
	{2454832.5, 34}, // 2009-01-01
	{2456109.5, 35}, // 2012-07-01
	{2457204.5, 36}, // 2015-07-01
	{2457754.5, 37}, // 2017-01-01
}

/**
 * GetLeapSeconds returns the total leap second offset at a given Julian Date.
 */
func GetLeapSeconds(jd float64) int {
	count := 0
	for _, ls := range LeapSecondTable {
		if jd >= ls.JD {
			count = ls.Count
		} else {
			break
		}
	}
	// If before 1972, return 0 (simplified)
	if count == 0 && jd > 2441317.5 {
		return 37 // Default for modern era
	}
	return count
}

/**
 * UTCToTT converts UTC Julian Date to Terrestrial Time (TT).
 * Formula: TT = UTC + LeapSeconds + 32.184s
 */
func UTCToTT(jdUTC float64) float64 {
	ls := GetLeapSeconds(jdUTC)
	deltaTAI := float64(ls) / SEC_PER_DAY
	jdTAI := jdUTC + deltaTAI
	return jdTAI + (32.184 / SEC_PER_DAY)
}

/**
 * TTToUTC converts Terrestrial Time (TT) to UTC Julian Date.
 */
func TTToUTC(jdTT float64) float64 {
	jdTAI := jdTT - (32.184 / SEC_PER_DAY)
	// Approximate LS count (sufficient for near-current dates)
	ls := GetLeapSeconds(jdTAI) 
	return jdTAI - (float64(ls) / SEC_PER_DAY)
}

// EOP represents Earth Orientation Parameters (from IERS).
type EOP struct {
	XP, YP float64 // Polar motion (arcsec)
	UT1_UTC float64 // ΔUT1 (sec)
	LOD    float64 // Length of day (ms)
}

/**
 * GreenwichMeanSiderealTime computes the GMST in radians including UT1 correction.
 */
func GreenwichMeanSiderealTime(jdUTC float64, ut1_utc float64) float64 {
	jdUT1 := jdUTC + (ut1_utc / SEC_PER_DAY)
	t := (jdUT1 - 2451545.0) / 36525.0
	// IAU 1982 GMST formula
	theta := 67310.54841 + (876600*3600+8640184.812866)*t + 0.093104*t*t - 6.2e-6*t*t*t
	gmst := math.Mod(theta*math.Pi/180.0/240.0, 2*math.Pi)
	if gmst < 0 {
		gmst += 2 * math.Pi
	}
	return gmst
}
