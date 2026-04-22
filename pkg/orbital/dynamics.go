package orbital

import (
	"math"
	"ssis/pkg/physics"
)

/**
 * OrbitalAccelerationDynamic computes acceleration with per-object drag
 * and real-time space weather integration (High-Fidelity).
 *
 * This function extends the base OrbitalAcceleration with:
 * - Per-object ballistic coefficient from TLE B* smoothing
 * - Dynamic NRLMSISE-00/JB2008 atmospheric density with F10.7/Ap
 * - Sun and Moon third-body perturbations
 * - Earth rotation correction for drag
 *
 * @param x,y,z: Position in ECI km
 * @param vx,vy,vz: Velocity in ECI km/s
 * @param jd: Julian Date
 * @param params: Object parameters including BallisticCoeff
 * @param sw: Space weather indices (F10.7, Ap)
 * @return: Acceleration (ax, ay, az) in km/s^2
 */
func OrbitalAccelerationDynamic(
	x, y, z, vx, vy, vz float64,
	jd float64,
	params physics.PhysicsParams,
	sw physics.SpaceWeatherData,
) (ax, ay, az float64) {
	// 1. Point Mass + J2-J4 Zonal + J2,2 Tesseral Harmonics
	ax, ay, az = GravityAccelerationGCRF(x, y, z, jd)

	// 2. Atmospheric Drag with Dynamic Space Weather
	r := math.Sqrt(x*x + y*y + z*z)
	alt := r - physics.EARTH_R_EQUATORIAL

	if alt < 1200 && params.Mass > 0 {
		// Convert ECI to geodetic for atmosphere model
		latDeg, lonDeg := eciToGeodeticLatLon(x, y, z, jd)

		// Use dynamic density with space weather
		rho := physics.GetDensityDynamic(alt, latDeg, lonDeg, jd, sw)

		// Earth's rotation vector (rad/s): omega = [0, 0, 7.2921158553e-5]
		omegaZ := 7.2921158553e-5
		vxRel := vx - (-omegaZ * y)
		vyRel := vy - (omegaZ * x)
		vzRel := vz // No z-component for Earth rotation

		vRel2 := vxRel*vxRel + vyRel*vyRel + vzRel*vzRel
		vRelMag := math.Sqrt(vRel2)

		if vRelMag > 1e-6 {
			// Use per-object ballistic coefficient if available
			// BC = m/(Cd*A) in kg/m^2
			bc := params.BallisticCoeff
			if bc <= 0 {
				// Fallback: compute from Mass/DragArea/Cd
				if params.DragArea > 0 && params.Cd > 0 {
					bc = params.Mass / (params.Cd * params.DragArea)
				} else {
					bc = 100.0 // Nominal 100 kg/m^2
				}
			}

			// Drag acceleration: a = -0.5 * rho * v^2 / BC * (v/|v|)
			dragCoeff := 0.5 * rho / bc
			ax -= dragCoeff * vRelMag * vxRel
			ay -= dragCoeff * vRelMag * vyRel
			az -= dragCoeff * vRelMag * vzRel
		}
	}

	// 3. Solar Radiation Pressure (SRP)
	if params.SRPArea > 0 && params.Cr > 0 {
		sax, say, saz := SolarRadiationPressure(x, y, z, jd, params.SRPArea, params.Mass, params.Cr)
		ax += sax
		ay += say
		az += saz
	}

	// 4. 3rd Body: Moon (now with proper ephemeris including inclination)
	mx, my, mz := physics.MoonPosition(jd)
	rmoon2 := mx*mx + my*my + mz*mz
	rmoon := math.Sqrt(rmoon2)
	rmoon3 := rmoon2 * rmoon
	dmx, dmy, dmz := mx-x, my-y, mz-z
	drmoon2 := dmx*dmx + dmy*dmy + dmz*dmz
	drmoon3 := drmoon2 * math.Sqrt(drmoon2)

	if drmoon3 > 1e-6 && rmoon3 > 1e-6 {
		ax += physics.MOON_MU_IAU * (dmx/drmoon3 - mx/rmoon3)
		ay += physics.MOON_MU_IAU * (dmy/drmoon3 - my/rmoon3)
		az += physics.MOON_MU_IAU * (dmz/drmoon3 - mz/rmoon3)
	}

	// 5. 3rd Body: Sun (critical for GEO operations)
	sx, sy, sz := SunPosition(jd)
	rsun2 := sx*sx + sy*sy + sz*sz
	rsun := math.Sqrt(rsun2)
	rsun3 := rsun2 * rsun
	dsx, dsy, dsz := sx-x, sy-y, sz-z
	drsun2 := dsx*dsx + dsy*dsy + dsz*dsz
	drsun3 := drsun2 * math.Sqrt(drsun2)

	if drsun3 > 1e-6 && rsun3 > 1e-6 {
		ax += physics.SUN_MU_IAU * (dsx/drsun3 - sx/rsun3)
		ay += physics.SUN_MU_IAU * (dsy/drsun3 - sy/rsun3)
		az += physics.SUN_MU_IAU * (dsz/drsun3 - sz/rsun3)
	}

	return ax, ay, az
}

/**
 * eciToGeodeticLatLon converts ECI position to geodetic latitude/longitude
 * (simplified, neglecting polar motion and nutation for atmospheric density).
 *
 * @param x,y,z: ECI position in km
 * @param jd: Julian Date
 * @return: latitude, longitude in degrees
 */
func eciToGeodeticLatLon(x, y, z, jd float64) (latDeg, lonDeg float64) {
	// Greenwich Sidereal Time
	gst := greenwichSiderealTime(jd)

	// Longitude from GST
	lon := math.Atan2(y, x) - gst

	// Normalize longitude to -pi to pi
	for lon > math.Pi {
		lon -= 2 * math.Pi
	}
	for lon < -math.Pi {
		lon += 2 * math.Pi
	}

	// Distance from z-axis
	rho := math.Sqrt(x*x + y*y)

	// Geocentric latitude (approximation for atmosphere)
	lat := math.Atan2(z, rho)

	// Convert to degrees
	latDeg = lat * 180.0 / math.Pi
	lonDeg = lon * 180.0 / math.Pi

	return latDeg, lonDeg
}

/**
 * greenwichSiderealTime computes the Greenwich Sidereal Time in radians.
 * Simplified version for atmospheric density calculations.
 */
func greenwichSiderealTime(jd float64) float64 {
	// Julian centuries from J2000.0
	T := (jd - 2451545.0) / 36525.0

	// Greenwich Mean Sidereal Time at 0h UT (degrees)
	gmst0 := 280.46061837 + 360.98564736629*(jd-2451545.0) +
		0.000387933*T*T - T*T*T/38710000.0

	// Normalize to 0-360
	gmst0 = math.Mod(gmst0, 360.0)
	if gmst0 < 0 {
		gmst0 += 360.0
	}

	// Convert to radians
	return gmst0 * math.Pi / 180.0
}

/**
 * EstimatePropagationUncertainty computes the 1-sigma position uncertainty
 * given propagation time, drag model uncertainty, and ballistic coefficient uncertainty.
 *
 * Used for conjunction assessment confidence bounds and blind spot detection.
 *
 * @param initialAltKm: Initial altitude (km)
 * @param propHours: Propagation time (hours)
 * @param bcUncertainty: Relative uncertainty in ballistic coefficient (0.3 = 30%)
 * @param swForecastHours: Hours of space weather forecast uncertainty
 * @return: 1-sigma position uncertainty in km
 */
func EstimatePropagationUncertainty(
	initialAltKm, propHours, bcUncertainty float64,
	swForecastHours float64,
) float64 {
	// Drag uncertainty scales with:
	// 1. Ballistic coefficient uncertainty
	// 2. Space weather forecast uncertainty (typically ±20% per day)
	// 3. Propagation time (uncertainty grows with time)

	// Base drag uncertainty (altitude-dependent)
	baseUncertainty := 0.1 // 10% per orbit baseline

	if initialAltKm < 300 {
		// Low LEO: very sensitive to drag
		baseUncertainty = 0.3
	} else if initialAltKm < 500 {
		// Mid LEO
		baseUncertainty = 0.2
	} else if initialAltKm < 800 {
		// Higher LEO
		baseUncertainty = 0.15
	} else {
		// Near-GEO: minimal drag uncertainty
		baseUncertainty = 0.05
	}

	// Space weather forecast uncertainty: ±20% per day of forecast
	swUncertainty := 0.2 * (swForecastHours / 24.0)
	if swUncertainty > 0.5 {
		swUncertainty = 0.5 // Cap at 50%
	}

	// Combined uncertainty (RSS)
	totalRel := math.Sqrt(
		baseUncertainty*baseUncertainty +
			bcUncertainty*bcUncertainty +
			swUncertainty*swUncertainty,
	)

	// Position uncertainty grows with propagation time
	// Typical LEO orbit period = 90-120 minutes
	orbitsPerHour := 0.75 // Approximate
	numOrbits := propHours * orbitsPerHour

	// Uncertainty accumulates approximately as sqrt(N) for random errors
	// but linearly for systematic errors (like BC bias)
	growthFactor := math.Sqrt(numOrbits)

	// Typical along-track drift for 10% drag error:
	// ~1-5 km per orbit depending on altitude
	kmPerOrbit := 2.0 // Baseline
	if initialAltKm < 400 {
		kmPerOrbit = 5.0
	} else if initialAltKm > 600 {
		kmPerOrbit = 1.0
	}

	uncertaintyKm := kmPerOrbit * totalRel * growthFactor

	return uncertaintyKm
}
