package physics

import (
	"math"
)

type Keplerian struct {
	A, E, I, RAAN, ARGP, M float64
}

/**
 * CartesianToKepler (RV2COE) converts state vectors to orbital elements.
 * Implements Vallado Algorithm 10 (RV2COE) with 3-1-3 Euler sequence.
 * @param s: Cartesian state in km and km/s.
 * @param mu: Gravitational parameter.
 * @return: 6-element Keplerian set.
 */
func CartesianToKepler(s State, mu float64) Keplerian {
	rVec := [3]float64{s.Pos[0], s.Pos[1], s.Pos[2]}
	vVec := [3]float64{s.Vel[0], s.Vel[1], s.Vel[2]}
	r := math.Sqrt(rVec[0]*rVec[0] + rVec[1]*rVec[1] + rVec[2]*rVec[2])
	v2 := vVec[0]*vVec[0] + vVec[1]*vVec[1] + vVec[2]*vVec[2]

	// 1. Angular momentum h = r x v
	hVec := [3]float64{
		rVec[1]*vVec[2] - rVec[2]*vVec[1],
		rVec[2]*vVec[0] - rVec[0]*vVec[2],
		rVec[0]*vVec[1] - rVec[1]*vVec[0],
	}
	h := math.Sqrt(hVec[0]*hVec[0] + hVec[1]*hVec[1] + hVec[2]*hVec[2])
	if h < 1e-12 {
		return Keplerian{A: r} // Degenerate case
	}

	// 2. Node vector n = k x h
	nVec := [3]float64{-hVec[1], hVec[0], 0}
	n := math.Sqrt(nVec[0]*nVec[0] + nVec[1]*nVec[1] + nVec[2]*nVec[2])

	// 3. Eccentricity vector e
	vDotR := rVec[0]*vVec[0] + rVec[1]*vVec[1] + rVec[2]*vVec[2]
	eVec := [3]float64{
		(1.0/mu) * ((v2-mu/r)*rVec[0] - vDotR*vVec[0]),
		(1.0/mu) * ((v2-mu/r)*rVec[1] - vDotR*vVec[1]),
		(1.0/mu) * ((v2-mu/r)*rVec[2] - vDotR*vVec[2]),
	}
	e := math.Sqrt(eVec[0]*eVec[0] + eVec[1]*eVec[1] + eVec[2]*eVec[2])

	// 4. Semi-major axis a (from energy)
	energy := v2/2.0 - mu/r
	a := -mu / (2.0 * energy)
	if math.Abs(e-1.0) < 1e-12 {
		a = math.Inf(1)
	}

	// 5. Inclination i
	i := math.Acos(clamp(hVec[2]/h, -1, 1))

	// 6. RAAN (Omega)
	longAscNode := 0.0
	if n > 1e-12 {
		longAscNode = math.Acos(clamp(nVec[0]/n, -1, 1))
		if nVec[1] < 0 {
			longAscNode = 2*math.Pi - longAscNode
		}
	}

	// 7. Argument of Perigee (omega)
	argPerigee := 0.0
	if n > 1e-12 && e > 1e-12 {
		dotNE := nVec[0]*eVec[0] + nVec[1]*eVec[1] + nVec[2]*eVec[2]
		argPerigee = math.Acos(clamp(dotNE/(n*e), -1, 1))
		if eVec[2] < 0 {
			argPerigee = 2*math.Pi - argPerigee
		}
	} else if e > 1e-12 {
		argPerigee = math.Atan2(eVec[1], eVec[0])
		if hVec[2] < 0 {
			argPerigee = 2*math.Pi - argPerigee
		}
	}

	// 8. True anomaly (nu)
	trueAnomaly := 0.0
	if e > 1e-12 {
		dotER := eVec[0]*rVec[0] + eVec[1]*rVec[1] + eVec[2]*rVec[2]
		trueAnomaly = math.Acos(clamp(dotER/(e*r), -1, 1))
		if vDotR < 0 {
			trueAnomaly = 2*math.Pi - trueAnomaly
		}
	} else {
		if n > 1e-12 {
			dotNR := nVec[0]*rVec[0] + nVec[1]*rVec[1] + nVec[2]*rVec[2]
			trueAnomaly = math.Acos(clamp(dotNR/(n*r), -1, 1))
			if rVec[2] < 0 {
				trueAnomaly = 2*math.Pi - trueAnomaly
			}
		} else {
			trueAnomaly = math.Atan2(rVec[1], rVec[0])
		}
	}

	// 9. Mean anomaly (M)
	cosE := (e + math.Cos(trueAnomaly)) / (1 + e*math.Cos(trueAnomaly))
	sinE := (math.Sqrt(1-e*e) * math.Sin(trueAnomaly)) / (1 + e*math.Cos(trueAnomaly))
	eccAnom := math.Atan2(sinE, cosE)
	meanAnom := eccAnom - e*math.Sin(eccAnom)

	return Keplerian{
		A: a, E: e, I: i,
		RAAN: normalizeAngle(longAscNode),
		ARGP: normalizeAngle(argPerigee),
		M:    normalizeAngle(meanAnom),
	}
}

/**
 * KeplerToECI (COE2RV) converts orbital elements back to Cartesian position.
 */
func KeplerToECI(k Keplerian, m, mu float64) [3]float64 {
	eAnom := solveKepler(normalizeAngle(m), k.E)
	nu := 2 * math.Atan2(math.Sqrt(1+k.E)*math.Sin(eAnom/2), math.Sqrt(1-k.E)*math.Cos(eAnom/2))
	p := k.A * (1 - k.E*k.E)
	r := p / (1 + k.E*math.Cos(nu))
	r_pqw := [3]float64{r * math.Cos(nu), r * math.Sin(nu), 0}

	cosW, sinW := math.Cos(k.ARGP), math.Sin(k.ARGP)
	cosI, sinI := math.Cos(k.I), math.Sin(k.I)
	cosO, sinO := math.Cos(k.RAAN), math.Sin(k.RAAN)

	r11 := cosO*cosW - sinO*sinW*cosI
	r12 := -cosO*sinW - sinO*cosW*cosI
	r13 := sinO * sinI
	r21 := sinO*cosW + cosO*sinW*cosI
	r22 := -sinO*sinW + cosO*cosW*cosI
	r23 := -cosO * sinI
	r31 := sinW * sinI
	r32 := cosW * sinI
	r33 := cosI

	return [3]float64{
		r11*r_pqw[0] + r12*r_pqw[1] + r13*r_pqw[2],
		r21*r_pqw[0] + r22*r_pqw[1] + r23*r_pqw[2],
		r31*r_pqw[0] + r32*r_pqw[1] + r33*r_pqw[2],
	}
}

func solveKepler(m, e float64) float64 {
	eAnom := m
	if e > 0.8 { eAnom = math.Pi }
	for i := 0; i < 10; i++ {
		delta := (eAnom - e*math.Sin(eAnom) - m) / (1 - e*math.Cos(eAnom))
		eAnom -= delta
		if math.Abs(delta) < 1e-12 { break }
	}
	return eAnom
}

func normalizeAngle(a float64) float64 {
	twoPi := 2 * math.Pi
	a = math.Mod(a, twoPi)
	if a < 0 { a += twoPi }
	return a
}

func clamp(v, min, max float64) float64 {
	if v < min { return min }
	if v > max { return max }
	return v
}
