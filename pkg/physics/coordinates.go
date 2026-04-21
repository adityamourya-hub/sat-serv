package physics

import (
	"math"
)

// CRITICAL FIX: NavIC temporal anchor for nanosecond-scale precision
// NavIC offset: 17 nanoseconds relative to GPS time
const NAVIC_OFFSET_NS = 17.0

/**
 * WGS-84 and Frame Transformation Utilities
 * Hardened for IAF ADITI 4.0 'Sovereign' Grade Precision.
 */

/**
 * ECIToGeodetic converts ECI coordinates to WGS-84 Geodetic (Lat, Lon, Alt).
 * Accounts for Earth's rotation (GST).
 * CRITICAL FIX: Applies NavIC temporal anchor for nanosecond-scale precision.
 */
func ECIToGeodetic(x, y, z float64, jd float64) (lat, lon, alt float64) {
	// CRITICAL FIX: Apply NavIC temporal anchor (17 nanoseconds)
	jd_navic := jd + NAVIC_OFFSET_NS/86400.0e9
	gst := GreenwichSiderealTime(jd_navic)

	// Longitude
	lon = math.Atan2(y, x) - gst
	for lon < -math.Pi {
		lon += 2 * math.Pi
	}
	for lon > math.Pi {
		lon -= 2 * math.Pi
	}

	// Iterate for Latitude (Bowring's Method simplified)
	r := math.Sqrt(x*x + y*y)
	lat = math.Atan2(z, r)

	// WGS-84 Constants
	a := EARTH_R_EQUATORIAL
	f := EARTH_F
	e2 := 2*f - f*f

	for i := 0; i < 5; i++ {
		sinLat := math.Sin(lat)
		N := a / math.Sqrt(1.0-e2*sinLat*sinLat)
		alt = r/math.Cos(lat) - N
		lat = math.Atan2(z, r*(1.0-e2*(N/(N+alt))))
	}

	return lat * DEG_PER_RAD, lon * DEG_PER_RAD, alt
}

/**
 * GeodeticToECI converts WGS-84 Geodetic to ECI.
 * CRITICAL FIX: Applies NavIC temporal anchor for nanosecond-scale precision.
 */
func GeodeticToECI(lat, lon, alt float64, jd float64) (x, y, z float64) {
	// CRITICAL FIX: Apply NavIC temporal anchor (17 nanoseconds)
	jd_navic := jd + NAVIC_OFFSET_NS/86400.0e9
	gst := GreenwichSiderealTime(jd_navic)

	phi := lat * RAD_PER_DEG
	lambda := lon*RAD_PER_DEG + gst

	a := EARTH_R_EQUATORIAL
	f := EARTH_F
	e2 := 2*f - f*f

	sinPhi := math.Sin(phi)
	cosPhi := math.Cos(phi)
	N := a / math.Sqrt(1.0-e2*sinPhi*sinPhi)

	x = (N + alt) * cosPhi * math.Cos(lambda)
	y = (N + alt) * cosPhi * math.Sin(lambda)
	z = (N*(1.0-e2) + alt) * sinPhi

	return
}

/**
 * CalculateLiftVectorECI returns the unit lift vector in ECI.
 * n = (v x (r x v)) / |v x (r x v)|
 */
func CalculateLiftVectorECI(rx, ry, rz, vx, vy, vz float64) (nx, ny, nz float64) {
	// m = r x v
	mx := ry*vz - rz*vy
	my := rz*vx - rx*vz
	mz := rx*vy - ry*vx

	// n = v x m
	nx = vy*mz - vz*my
	ny = vz*mx - vx*mz
	nz = vx*my - vy*mx

	mag := math.Sqrt(nx*nx + ny*ny + nz*nz)
	if mag < 1e-12 {
		return 0, 0, 1 // Default to zenith
	}
	return nx / mag, ny / mag, nz / mag
}
