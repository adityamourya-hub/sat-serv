package orbital

import (
	"math"
	"ssis/pkg/physics"
)

/**
 * High-fidelity Environmental Forces (iDEX ADITI 4.0 Industry Hardening)
 * Implements standard perturbations for production SSA/SDA.
 */

// Constants removed (using IAU standards from physics.constants)

/**
 * GravityAccelerationGCRF calculates the acceleration (km/s^2) due to Earth 
 * non-spherical mass distribution (J2, J3, J4 zonal harmonics).
 */
func GravityAccelerationGCRF(x, y, z float64) (ax, ay, az float64) {
	r2 := x*x + y*y + z*z
	r := math.Sqrt(r2)
	z2 := z * z
	invR := 1.0 / r
	invR2 := invR * invR
	
	// Base Point Mass
	invR3 := invR2 * invR
	muR3 := -physics.EARTH_MU_IAU * invR3
	ax = muR3 * x
	ay = muR3 * y
	az = muR3 * z

	// J2 Zonal Harmonic
	j2Fac := (1.5 * physics.J2_ADITI * physics.EARTH_MU_IAU * physics.EARTH_R_EQUATORIAL * physics.EARTH_R_EQUATORIAL) / (r2 * r2 * r)
	ax += j2Fac * x * (5*z2*invR2 - 1)
	ay += j2Fac * y * (5*z2*invR2 - 1)
	az += j2Fac * z * (5*z2*invR2 - 3)

	// J3 Zonal Harmonic
	j3Fac := (2.5 * physics.J3_ADITI * physics.EARTH_MU_IAU * math.Pow(physics.EARTH_R_EQUATORIAL, 3)) / (math.Pow(r, 7))
	ax += j3Fac * x * (7*z*z2*invR2*invR - 3*z*invR)
	ay += j3Fac * y * (7*z*z2*invR2*invR - 3*z*invR)
	az += j3Fac * (35.0/10.0*z2*z2*invR2*invR2 - 3.0*z2*invR2 + 0.3) // Simplified az J3

	// J4 Zonal Harmonic
	j4Fac := (1.875 * physics.J4_ADITI * physics.EARTH_MU_IAU * math.Pow(physics.EARTH_R_EQUATORIAL, 4)) / (math.Pow(r, 9))
	ax += j4Fac * x * (35*math.Pow(z, 4)/(r2*r2) - 30*z2/r2 + 3)
	ay += j4Fac * y * (35*math.Pow(z, 4)/(r2*r2) - 30*z2/r2 + 3)
	az += j4Fac * z * (35*math.Pow(z, 4)/(r2*r2) - 70*z2/(3*r2) + 5) // Simplified az J4

	return
}

/**
 * SolarRadiationPressure calculates the acceleration (km/s^2) due to solar flux.
 */
func SolarRadiationPressure(x, y, z float64, jd float64, area, mass, Cr float64) (ax, ay, az float64) {
	if mass <= 0 { return 0, 0, 0 }
	
	sx, sy, sz := SunPosition(jd)
	dx, dy, dz := sx-x, sy-y, sz-z
	d2 := dx*dx + dy*dy + dz*dz
	d := math.Sqrt(d2)

	// Shadow check: Is the Earth between the object and the Sun?
	// Simplified conical shadow model
	dotRS := x*sx + y*sy + z*sz
	if dotRS < 0 {
		h := math.Sqrt(r2(x, y, z) - pow2(dotRS/r(sx, sy, sz)))
		if h < physics.EARTH_R_EQUATORIAL {
			return 0, 0, 0 // In shadow
		}
	}

	// P = P0 * (1AU / d)^2 * (A/m) * Cr
	acc := (physics.SOLAR_PRESSURE * (physics.AU_KM / d) * (physics.AU_KM / d) * (area / mass) * Cr) / 1000.0 // N/kg to km/s^2
	ax = acc * dx / d
	ay = acc * dy / d
	az = acc * dz / d
	return
}

/**
 * SunPosition returns the ECI coordinates (km) of the Sun.
 * Low-precision analytical model suitable for SRP calculations.
 */
func SunPosition(jd float64) (x, y, z float64) {
	T := (jd - 2451545.0) / 36525.0
	L := (280.460 + 36000.771*T) * math.Pi / 180.0
	M := (357.527 + 35999.050*T) * math.Pi / 180.0
	ecliptic := (L + 1.91466*math.Sin(M) + 0.0199*math.Sin(2*M)) * math.Pi / 180.0
	obliquity := (23.439 - 0.013*T) * math.Pi / 180.0
	
	rSun := physics.AU_KM * (1.00014 - 0.01671*math.Cos(M) - 0.00014*math.Cos(2*M))
	x = rSun * math.Cos(ecliptic)
	y = rSun * math.Cos(obliquity) * math.Sin(ecliptic)
	z = rSun * math.Sin(obliquity) * math.Sin(ecliptic)
	return
}

func r(x, y, z float64) float64 { return math.Sqrt(x*x + y*y + z*z) }
func r2(x, y, z float64) float64 { return x*x + y*y + z*z }
func pow2(x float64) float64 { return x * x }

/**
 * ComputeJacobian calculates the 7x7 state transition matrix derivative (df/dx).
 * Indices 0-5: State (r, v). Index 6: Parameter (e.g. Ballistic Coeff or Drag Scale).
 */
func ComputeJacobian(x, y, z, vx, vy, vz float64, jd float64, params physics.PhysicsParams) [][]float64 {
	a := make([][]float64, 7)
	for i := range a {
		a[i] = make([]float64, 7)
	}

	// 1. Identity for upper right (dr/dt = v)
	a[0][3], a[1][4], a[2][5] = 1.0, 1.0, 1.0

	// 2. Gravity Gradient (da/dr)
	r2 := x*x + y*y + z*z
	r := math.Sqrt(r2)
	invR := 1.0 / r
	invR2 := invR * invR
	invR3 := invR * invR2
	muR5 := 3.0 * physics.EARTH_MU_IAU * invR3 * invR2

	// Simplified Point Mass Gradient
	a[3][0] = muR5*x*x - physics.EARTH_MU_IAU*invR3
	a[3][1] = muR5 * x * y
	a[3][2] = muR5 * x * z

	a[4][0] = muR5 * y * x
	a[4][1] = muR5*y*y - physics.EARTH_MU_IAU*invR3
	a[4][2] = muR5 * y * z

	a[5][0] = muR5 * z * x
	a[5][1] = muR5 * z * y
	a[5][2] = muR5*z*z - physics.EARTH_MU_IAU*invR3

	// J2 Gradient terms (approximate for STM stability)
	j2Fac := (1.5 * physics.J2_ADITI * physics.EARTH_MU_IAU * physics.EARTH_R_EQUATORIAL * physics.EARTH_R_EQUATORIAL) * invR3 * invR2
	a[3][0] += j2Fac * (1.0 - 5.0*z*z*invR2)
	a[4][1] += j2Fac * (1.0 - 5.0*z*z*invR2)
	a[5][2] += j2Fac * (3.0 - 5.0*z*z*invR2)

	// 3. Drag Gradient (da/dv)
	// a_drag = -0.5 * rho * v * v_rel * (A/m) * Cd
	v := math.Sqrt(vx*vx + vy*vy + vz*vz)
	if v > 1e-6 {
		rho := physics.GetDensityUnified(x, y, z, jd)
		dragConst := 0.5 * rho * (params.DragArea / params.Mass) * params.Cd
		a[3][3] = -dragConst * (2*vx*vx/v + v)
		a[4][4] = -dragConst * (2*vy*vy/v + v)
		a[5][5] = -dragConst * (2*vz*vz/v + v)
	}

	return a
}
