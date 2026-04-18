package physics

import (
	"math"
)

/**
 * WGS-84 and Frame Transformation Utilities
 * Hardened for IAF ADITI 4.0 'Sovereign' Grade Precision.
 */

const (
	WGS84_A = 6378.137             // Semi-major axis (km)
	WGS84_F = 1.0 / 298.257223563  // Flattening
	WGS84_B = WGS84_A * (1.0 - WGS84_F)
	WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
	WGS84_EP2 = (WGS84_A*WGS84_A - WGS84_B*WGS84_B) / (WGS84_B * WGS84_B)
)

/**
 * ECEFToGeodeticWGS84 converts ECEF coordinates to Geodetic Lat, Lon, Alt.
 * Uses Bowring's iterative method for sub-meter precision.
 */
func ECEFToGeodeticWGS84(x, y, z float64) (lat, lon, alt float64) {
	p := math.Sqrt(x*x + y*y)
	lon = math.Atan2(y, x)

	// Bowring's Method Initial Guess
	phi := math.Atan2(z, p*(1.0-WGS84_E2))
	
	// Iteration for convergence
	for i := 0; i < 5; i++ {
		sinPhi := math.Sin(phi)
		n := WGS84_A / math.Sqrt(1.0-WGS84_E2*sinPhi*sinPhi)
		phi = math.Atan2(z+WGS84_E2*n*sinPhi, p)
	}

	lat = phi
	sinPhi := math.Sin(phi)
	n := WGS84_A / math.Sqrt(1.0-WGS84_E2*sinPhi*sinPhi)
	alt = p/math.Cos(phi) - n

	return
}

/**
 * GeodeticToECEFWGS84 converts Geodetic Lat, Lon, Alt to ECEF coordinates.
 */
func GeodeticToECEFWGS84(lat, lon, alt float64) (x, y, z float64) {
	sinLat := math.Sin(lat)
	cosLat := math.Cos(lat)
	sinLon := math.Sin(lon)
	cosLon := math.Cos(lon)

	n := WGS84_A / math.Sqrt(1.0-WGS84_E2*sinLat*sinLat)

	x = (n + alt) * cosLat * cosLon
	y = (n + alt) * cosLat * sinLon
	z = (n*(1.0-WGS84_E2) + alt) * sinLat
	return
}

/**
 * ECIToECEF approximates the rotation between ECI (Inertial) and ECEF (Fixed) 
 * using Greenwich Sidereal Time.
 */
func ECIToECEF(x, y, z, jd float64) (xf, yf, zf float64) {
	gst := GreenwichSiderealTime(jd)
	cosG := math.Cos(gst)
	sinG := math.Sin(gst)

	xf = x*cosG + y*sinG
	yf = -x*sinG + y*cosG
	zf = z
	return
}

/**
 * GreenwichSiderealTime computes the GST in radians at a given Julian Date.
 * Accurate for the current epoch (21st century).
 */
func GreenwichSiderealTime(jd float64) float64 {
	T := (jd - 2451545.0) / 36525.0
	// Degrees to Radians helper
	const deg2rad = math.Pi / 180.0
	// BMW / Vallado formula for Theta_GMST
	theta := 67310.54841 + (876600*3600+8640184.812866)*T + 0.093104*T*T - 6.2e-6*T*T*T
	gst := math.Mod(theta/240.0*deg2rad, 2*math.Pi)
	if gst < 0 {
		gst += 2 * math.Pi
	}
	return gst
}
