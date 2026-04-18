package physics

import (
	"math"
	"sync"
	"time"
)

/**
 * SpaceWeatherData holds real-time solar and geomagnetic indices
 * required for high-fidelity atmospheric density modeling.
 *
 * Sources: NOAA SWPC (Space Weather Prediction Center)
 * - F10.7 cm Solar Flux: 10.7 cm radio emission (solar EUV proxy)
 * - Ap Index: Planetary geomagnetic activity index
 * - Kp Index: Planetary geomagnetic storm scale (0-9)
 *
 * ADITI 4.0 Requirement: 2-hour propagation accuracy demands
 * real-time space weather integration for dynamic drag calculation.
 */
type SpaceWeatherData struct {
	F107    float64 // Current 10.7 cm solar flux (sfu)
	F107Avg float64 // 81-day centered average of F10.7
	Ap      float64 // Current Ap index (nT)
	Kp      float64 // Current Kp index (0-9)
	Timestamp time.Time
}

/**
 * SpaceWeatherCache provides thread-safe access to current space weather
 * with automatic expiration and fallback values.
 */
type SpaceWeatherCache struct {
	mu        sync.RWMutex
	data      SpaceWeatherData
	valid     bool
	maxAge    time.Duration
	
	// Fallback values for when live data unavailable
	DefaultF107 float64
	DefaultAp   float64
}

/**
 * NewSpaceWeatherCache creates a cache with ADITI 4.0 standard defaults.
 *
 * Default F10.7 = 140 sfu (moderate solar activity)
 * Default Ap = 15 (quiet to unsettled conditions)
 */
func NewSpaceWeatherCache() *SpaceWeatherCache {
	return &SpaceWeatherCache{
		maxAge:    6 * time.Hour, // Data older than 6 hours considered stale
		DefaultF107: 140.0,
		DefaultAp:   15.0,
		valid:     false,
	}
}

/**
 * Update sets new space weather data with current timestamp.
 */
func (c *SpaceWeatherCache) Update(sw SpaceWeatherData) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	
	sw.Timestamp = time.Now()
	c.data = sw
	c.valid = true
}

/**
 * Get retrieves current space weather data, falling back to defaults if stale.
 */
func (c *SpaceWeatherCache) Get() SpaceWeatherData {
	if c == nil {
		return SpaceWeatherData{
			F107: 140.0,
			Ap:   15.0,
		}
	}
	
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if !c.valid || time.Since(c.data.Timestamp) > c.maxAge {
		// Return defaults with zero timestamp to indicate fallback
		return SpaceWeatherData{
			F107:    c.DefaultF107,
			F107Avg: c.DefaultF107,
			Ap:      c.DefaultAp,
			Kp:      c.DefaultAp / 3.0, // Rough conversion: Ap/3 ≈ Kp
		}
	}
	
	return c.data
}

/**
 * IsLive returns true if cached data is current (not using fallbacks).
 */
func (c *SpaceWeatherCache) IsLive() bool {
	if c == nil {
		return false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.valid && time.Since(c.data.Timestamp) <= c.maxAge
}

/**
 * NRLMSISE00Density computes atmospheric density using the NRLMSISE-00 model.
 *
 * This is a simplified implementation of the NRLMSISE-00 (Naval Research Laboratory
 * Mass Spectrometer and Incoherent Scatter Radar) model for altitudes 0-1000 km.
 *
 * For altitudes > 1000 km, returns exospheric density.
 * For production use, consider linking to the full NRLMSISE-00 C library.
 *
 * @param altKm: Altitude in kilometers
 * @param latDeg: Latitude in degrees (-90 to 90)
 * @param lonDeg: Longitude in degrees (-180 to 180)
 * @param jd: Julian Date for diurnal variations
 * @param sw: Space weather indices (F10.7, Ap)
 * @return: Atmospheric density in kg/m^3
 */
func NRLMSISE00Density(altKm, latDeg, lonDeg, jd float64, sw SpaceWeatherData) float64 {
	// Clamp inputs to valid ranges
	if altKm < 0 {
		altKm = 0
	}
	if altKm > 2000 {
		// Exosphere - very low density
		return 1e-15
	}
	
	// Solar flux effect on density (higher F10.7 = more EUV = expanded atmosphere)
	// NRLMSISE-00 uses both current F10.7 and 81-day average
	f107 := sw.F107
	f107a := sw.F107Avg
	if f107a < 1 {
		f107a = f107 // Fallback
	}
	
	// Geomagnetic activity effect (Ap index)
	ap := sw.Ap
	
	// Diurnal variation (day/night bulge)
	// Peak density at ~14:00 local solar time
	lst := localSolarTime(jd, lonDeg)
	diurnal := 1.0 + 0.15*math.Cos((lst-14.0)*math.Pi/12.0)
	
	// Latitude variation (lower density at poles)
	latRad := latDeg * math.Pi / 180.0
	latEffect := 1.0 - 0.1*math.Abs(math.Sin(latRad))
	
	// Solar activity scaling factor
	// Higher F10.7 expands atmosphere, increasing density at all altitudes
	solarFactor := 0.5 + 0.5*(f107+f107a)/280.0 // Normalize around F10.7=140
	
	// Geomagnetic storm effect (significant at high altitudes)
	geomagFactor := 1.0
	if altKm > 200 {
		// Ap effect increases with altitude
		geomagFactor = 1.0 + (ap/100.0)*math.Min(1.0, (altKm-200)/400.0)
	}
	
	// Base density model - piecewise exponential with solar/geomag scaling
	rho := baseDensity(altKm) * diurnal * latEffect * solarFactor * geomagFactor
	
	return rho
}

/**
 * JB2008Density computes atmospheric density using the Jacchia-Bowman 2008 model.
 *
 * JB2008 is more accurate than NRLMSISE-00 during solar maximum and for
 * high geomagnetic activity conditions (Kp > 5).
 *
 * This is a simplified implementation. For production, use the full JB2008 model.
 *
 * @param altKm: Altitude in kilometers
 * @param latDeg: Latitude in degrees
 * @param jd: Julian Date
 * @param sw: Space weather indices (F10.7, F10.7avg, Ap)
 * @return: Atmospheric density in kg/m^3
 */
func JB2008Density(altKm, latDeg, jd float64, sw SpaceWeatherData) float64 {
	if altKm < 0 {
		altKm = 0
	}
	if altKm > 2000 {
		return 1e-15
	}
	
	// JB2008 uses temperature as the primary driver
	// Exospheric temperature from solar EUV heating
	f107 := sw.F107
	f107a := sw.F107Avg
	if f107a < 1 {
		f107a = f107
	}
	
	// Base exospheric temperature (Jacchia formula)
	// Texo = 379.0 + 3.24*F10.7 + 1.3*(F10.7 - F10.7avg)
	texo := 379.0 + 3.24*f107 + 1.3*(f107-f107a)
	
	// Geomagnetic heating (significant effect above 200 km)
	ap := sw.Ap
	if altKm > 200 {
		dtGeomag := 27.0 * ap * math.Exp(-(altKm-200)/100.0)
		texo += dtGeomag
	}
	
	// Diurnal variation
	lst := localSolarTime(jd, 0) // Use local time effect
	diurnalTemp := 30.0 * math.Cos((lst-14.0)*math.Pi/12.0)
	texo += diurnalTemp
	
	// Density from temperature (simplified barometric formula)
	// rho = rho0 * exp(-alt/H) where H is scale height
	scaleHeight := texo / 30.0 // Approximate scale height in km
	
	// Base sea-level density adjusted for temperature
	rho0 := 1.225 * (290.0 / (texo - 150.0)) // Temperature-corrected
	
	rho := rho0 * math.Exp(-altKm/scaleHeight)
	
	return rho
}

/**
 * GetDensityDynamic provides unified atmospheric density with space weather.
 *
 * Selects between NRLMSISE-00 and JB2008 based on altitude and solar conditions.
 * - Alt < 500 km: NRLMSISE-00 (better for LEO)
 * - Solar max (F10.7 > 200) or high Ap (>50): JB2008
 * - Otherwise: NRLMSISE-00 (faster, well-validated)
 *
 * @param altKm: Altitude in km
 * @param latDeg: Latitude in degrees
 * @param lonDeg: Longitude in degrees
 * @param jd: Julian Date
 * @param sw: Space weather data (F10.7, Ap)
 * @return: Atmospheric density in kg/m^3
 */
func GetDensityDynamic(altKm, latDeg, lonDeg, jd float64, sw SpaceWeatherData) float64 {
	// Use JB2008 for high solar activity or geomagnetic storms
	useJB2008 := sw.F107 > 200.0 || sw.Ap > 50.0
	
	if useJB2008 {
		return JB2008Density(altKm, latDeg, jd, sw)
	}
	
	return NRLMSISE00Density(altKm, latDeg, lonDeg, jd, sw)
}

/**
 * localSolarTime computes the local solar hour (0-24) for a given longitude and JD.
 */
func localSolarTime(jd, lonDeg float64) float64 {
	// Julian Date starts at noon, so adjust
	dayFraction := math.Mod(jd+0.5, 1.0)
	
	// UTC hour
	utcHour := dayFraction * 24.0
	
	// Longitude correction: 15 degrees = 1 hour
	lonCorrection := lonDeg / 15.0
	
	lst := utcHour + lonCorrection
	
	// Normalize to 0-24
	lst = math.Mod(lst, 24.0)
	if lst < 0 {
		lst += 24.0
	}
	
	return lst
}

/**
 * baseDensity provides a static density model without space weather.
 * Used as the baseline for NRLMSISE-00 scaling.
 */
func baseDensity(altKm float64) float64 {
	// Piecewise exponential model
	switch {
	case altKm < 25:
		// Troposphere
		return 1.225 * math.Exp(-altKm/8.5)
	case altKm < 90:
		// Stratosphere and mesosphere
		hScale := 6.5 + (altKm-25)*0.02
		return 0.04 * math.Exp(-(altKm-25)/hScale)
	case altKm < 200:
		// Lower thermosphere
		return 3.0e-6 * math.Exp(-(altKm-90)/30.0)
	case altKm < 500:
		// Thermosphere
		return 5.0e-9 * math.Exp(-(altKm-200)/50.0)
	case altKm < 1000:
		// Upper thermosphere
		return 2.0e-12 * math.Exp(-(altKm-500)/100.0)
	default:
		// Exosphere
		return 1e-15 * math.Exp(-(altKm-1000)/500.0)
	}
}

/**
 * EstimateDragErrorBound computes the uncertainty in density-based drag
 * given space weather forecast uncertainty.
 *
 * For operational SSA, drag uncertainty is a major source of conjunction
 * assessment error. This function quantifies that for covariance inflation.
 *
 * @param altKm: Altitude in km
 * @param f107Uncertainty: Uncertainty in F10.7 forecast (typically ±20 sfu for 1-day forecast)
 * @param apUncertainty: Uncertainty in Ap forecast (typically ±10 nT)
 * @return: Relative uncertainty in drag acceleration (0.3 = 30%)
 */
func EstimateDragErrorBound(altKm, f107Uncertainty, apUncertainty float64) float64 {
	// Altitude determines sensitivity to space weather
	// LEO < 400 km: dominated by F10.7 (thermal expansion)
	// LEO > 400 km and GEO: also sensitive to geomagnetic storms
	
	f107Sensitivity := 0.0
	apSensitivity := 0.0
	
	if altKm < 200 {
		// Lower atmosphere less sensitive to space weather
		f107Sensitivity = 0.2
		apSensitivity = 0.1
	} else if altKm < 500 {
		// Mid-LEO: high sensitivity to F10.7
		f107Sensitivity = 0.5
		apSensitivity = 0.3
	} else if altKm < 1000 {
		// Upper LEO: sensitive to both
		f107Sensitivity = 0.4
		apSensitivity = 0.4
	} else {
		// GEO/high orbit: minimal drag, but what exists is geomagnetic
		f107Sensitivity = 0.1
		apSensitivity = 0.5
	}
	
	// Normalize uncertainties
	f107Rel := f107Uncertainty / 140.0 // Normalized to typical F10.7=140
	apRel := apUncertainty / 15.0      // Normalized to typical Ap=15
	
	// Combined uncertainty (RSS)
	totalRel := math.Sqrt(
		(f107Sensitivity*f107Rel)*(f107Sensitivity*f107Rel) +
		(apSensitivity*apRel)*(apSensitivity*apRel),
	)
	
	// Cap at reasonable bounds
	if totalRel > 1.0 {
		totalRel = 1.0
	}
	if totalRel < 0.1 {
		totalRel = 0.1
	}
	
	return totalRel
}
