package physics

import (
	"math"
)

/**
 * OrbitalAcceleration calculates the total acceleration vector (km/s^2).
 * Includes J2 perturbation and basic 2-body gravity.
 */
func OrbitalAcceleration(rx, ry, rz, vx, vy, vz float64, jd float64, params PhysicsParams) (ax, ay, az float64) {
	r2 := rx*rx + ry*ry + rz*rz
	r := math.Sqrt(r2)
	r3 := r2 * r
	
	// Two-Body Acceleration
	mu := EARTH_MU_IAU
	ax = -mu * rx / r3
	ay = -mu * ry / r3
	az = -mu * rz / r3
	
	// J2 Perturbation (IAU Standard)
	re := EARTH_R_EQUATORIAL
	j2 := 1.08262668e-3
	
	z2r2 := (rz * rz) / r2
	factor := 1.5 * j2 * mu * (re * re) / (r2 * r3)
	
	ax += factor * (5.0*z2r2 - 1.0) * rx
	ay += factor * (5.0*z2r2 - 1.0) * ry
	az += factor * (5.0*z2r2 - 3.0) * rz
	
	return
}

/**
 * ComputePCADispersion calculates the major/minor axes and azimuth of a dispersion cloud.
 * Used for impact footprint analysis (Monte-Carlo PCA).
 */
func ComputePCADispersion(lats, lons []float64, sigma float64) (major, minor, azimuth float64) {
	n := len(lats)
	if n < 2 {
		return 0, 0, 0
	}
	
	meanLat, meanLon := 0.0, 0.0
	for i := 0; i < n; i++ {
		meanLat += lats[i]
		meanLon += lons[i]
	}
	meanLat /= float64(n)
	meanLon /= float64(n)
	
	// Covariance Matrix [C11 C12; C21 C22]
	var c11, c12, c22 float64
	for i := 0; i < n; i++ {
		dLat := lats[i] - meanLat
		dLon := (lons[i] - meanLon) * math.Cos(meanLat*math.Pi/180.0)
		c11 += dLat * dLat
		c12 += dLat * dLon
		c22 += dLon * dLon
	}
	c11 /= float64(n - 1)
	c12 /= float64(n - 1)
	c22 /= float64(n - 1)
	
	// Convert to km (approx)
	latToKm := 111.3
	lonToKm := 111.3 * math.Cos(meanLat*math.Pi/180.0)
	c11 *= latToKm * latToKm
	c12 *= latToKm * lonToKm
	c22 *= lonToKm * lonToKm
	
	// Eigenvalues
	trace := c11 + c22
	det := c11*c22 - c12*c12
	lambda1 := trace/2.0 + math.Sqrt(math.Max(0, (trace*trace/4.0)-det))
	lambda2 := trace/2.0 - math.Sqrt(math.Max(0, (trace*trace/4.0)-det))
	
	major = sigma * math.Sqrt(math.Max(0, lambda1))
	minor = sigma * math.Sqrt(math.Max(0, lambda2))
	azimuth = 0.5 * math.Atan2(2.0*c12, c11-c22)
	
	return
}
