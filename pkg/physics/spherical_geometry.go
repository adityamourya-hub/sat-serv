package physics

import (
	"math"
)

// GreenwichSiderealTime removed (duplicated in frame_transforms.go)

/**
 * Spherical Geometry and Keplerian Utilities
 * Hardened for IAF ballistic impact precision.
 */

/**
 * SphericalProject calculates the destination coordinates (lat, lon) given an initial
 * point, an angular distance (psi), and an azimuth (beta).
 * Uses the Spherical Law of Cosines for great-circle navigation.
 */
func SphericalProject(lat1, lon1, psi, beta float64) (lat2, lon2 float64) {
	// Inputs in radians
	sinLat2 := math.Sin(lat1)*math.Cos(psi) + math.Cos(lat1)*math.Sin(psi)*math.Cos(beta)
	latRad2 := math.Asin(sinLat2)

	y := math.Sin(psi) * math.Sin(beta)
	x := math.Cos(lat1)*math.Cos(psi) - math.Sin(lat1)*math.Sin(psi)*math.Cos(beta)
	dLon := math.Atan2(y, x)

	return latRad2, lon1 + dLon
}

/**
 * CalculateSubOrbitalToF computes the Time of Flight for an elliptical arc from burnout to impact.
 * Uses Kepler's equation: TOF = sqrt(a^3/mu) * [(Ei - e*sinEi) - (Eb - e*sinEb)]
 */
func CalculateSubOrbitalToF(a, e, rBurnout, gamma, psi float64) float64 {
	if a <= 0 {
		return 0
	}

	// 1. Eccentric Anomaly at Burnout (Eb)
	// cos(nu_b) = (a(1-e^2)/r - 1)/e
	// More stable: tan(gamma) = e*sin(nu) / (1 + e*cos(nu))
	cosEb := (1.0 - rBurnout/a) / e
	if cosEb > 1 {
		cosEb = 1
	}
	if cosEb < -1 {
		cosEb = -1
	}
	eb := math.Acos(cosEb)

	// Adjust eb based on gamma (if gamma > 0, we are ascending)
	if gamma < 0 {
		eb = 2*math.Pi - eb
	}

	// 2. Eccentric Anomaly at Impact (Ei)
	// JD_EPOCH and SEC_PER_DAY removed (using constants.go)
	sinNuB := (math.Sqrt(a*a*(1-e*e)) * math.Sin(eb)) / rBurnout
	cosNuB := (math.Cos(eb) - e) / (1 - e*math.Cos(eb))
	nuB := math.Atan2(sinNuB, cosNuB)

	nuI := nuB + psi
	ei := 2.0 * math.Atan(math.Sqrt((1.0-e)/(1.0+e))*math.Tan(nuI/2.0))

	// Ensure Ei > Eb for monotonic time
	if ei < eb {
		ei += 2 * math.Pi
	}

	// 3. TOF = sqrt(a^3/mu) * diff(Mean Anomaly)
	sqrtA3Mu := math.Sqrt(math.Pow(a, 3) / EARTH_MU_IAU)
	maB := eb - e*math.Sin(eb)
	maI := ei - e*math.Sin(ei)

	return sqrtA3Mu * (maI - maB)
}

// ComputePCADispersion removed (duplicated in forces.go)

// CalculateLiftVectorECI removed (duplicated in coordinates.go)
