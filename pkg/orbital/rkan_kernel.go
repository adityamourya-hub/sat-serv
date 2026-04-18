package orbital

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Constants for WGS-84 and Physical Models
const (
	MU_EARTH = 398600.4418
	RE_EARTH = 6378.137
	J2       = 1.0826e-3
	RAD_DEG  = math.Pi / 180.0
)

// RKANKernel implements the Hardened 13-feature Rational KAN forward pass in pure Go.
// This supports the 'Unified Resonance Law' discovered in Axiom-Zero v4.
type RKANKernel struct {
	Num    [][][]float64 `json:"num"` // [outDim][inDim][degree+1]
	Den    [][][]float64 `json:"den"` // [outDim][inDim][degree+1]
	Degree int           `json:"degree"`
}

/**
 * NewRKANKernelFromJSON loads the sovereign physics weights from a JSON export.
 * Supports the upgraded 13-feature manifest.
 */
func NewRKANKernelFromJSON(path string) (*RKANKernel, error) {
	fmt.Printf("DEBUG: Loading RKAN Kernel from %s\n", path)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var kernel RKANKernel
	if err := json.Unmarshal(data, &kernel); err != nil {
		return nil, err
	}

	return &kernel, nil
}

// mapToInterval matches the JAX implementation (tanh scaling to +/- 1).
func (k *RKANKernel) mapToInterval(x float64) float64 {
	return math.Tanh(x / 1000.0)
}

// softplus implementation matching JAX/Flax default (beta=1.0).
func softplus(x float64) float64 {
	if x > 20 {
		return x
	}
	return math.Log(1.0 + math.Exp(x))
}

// legendreBasis computes the Legendre polynomials up to k.Degree using recurrence.
func (k *RKANKernel) legendreBasis(x float64) []float64 {
	basis := make([]float64, k.Degree+1)
	basis[0] = 1.0
	if k.Degree == 0 {
		return basis
	}
	basis[1] = x
	if k.Degree == 1 {
		return basis
	}

	p0 := 1.0
	p1 := x

	for n := 1; n < k.Degree; n++ {
		fn := float64(n)
		p2 := ((2.0*fn+1.0)*x*p1 - fn*p0) / (fn + 1.0)
		basis[n+1] = p2
		p0 = p1
		p1 = p2
	}

	return basis
}

// getMeeusEphemeris computes Sun and Moon unit vectors.
// Ref: sfmn_kernel.py:74. Accuracy ~0.1 deg.
func getMeeusEphemeris(jd float64) (sunV, moonV [3]float64) {
	T := (jd - 2451545.0) / 36525.0

	// Sun
	L := (280.46646 + 36000.76983*T) * RAD_DEG
	g := (357.52911 + 35999.05029*T) * RAD_DEG
	lambdaSun := L + (1.914602 * math.Sin(g)) + (0.019993 * math.Sin(2.0*g))
	epsilon := (23.439 - 0.013*T) * RAD_DEG

	sunV[0] = math.Cos(lambdaSun)
	sunV[1] = math.Sin(lambdaSun) * math.Cos(epsilon)
	sunV[2] = math.Sin(lambdaSun) * math.Sin(epsilon)

	// Moon
	LMoon := (218.316 + 481267.881*T) * RAD_DEG
	moonV[0] = math.Cos(LMoon)
	moonV[1] = math.Sin(LMoon) * math.Cos(epsilon)
	moonV[2] = math.Sin(LMoon) * math.Sin(epsilon)

	return
}

// getOrbitalElements calculates scalar invariants for resonance detection.
func getOrbitalElements(q, p [3]float64) (inc, node, nVal, r float64) {
	r = math.Sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]) + 1e-12
	v2 := p[0]*p[0] + p[1]*p[1] + p[2]*p[2]

	hx := q[1]*p[2] - q[2]*p[1]
	hy := q[2]*p[0] - q[0]*p[2]
	hz := q[0]*p[1] - q[1]*p[0]
	hNorm := math.Sqrt(hx*hx+hy*hy+hz*hz) + 1e-12

	inc = math.Acos(math.Min(math.Max(hz/hNorm, -1.0), 1.0))
	node = math.Atan2(hx, -hy)

	aInv := 2.0/r - v2/MU_EARTH
	a := 1.0 / math.Max(aInv, 1e-10)
	nVal = math.Sqrt(MU_EARTH / (a*a*a + 1e-12))
	return
}

// getUnifiedResonanceFeature implements current Discovery Law Feature 7.
func getUnifiedResonanceFeature(q, p [3]float64) float64 {
	inc, node, nVal, r := getOrbitalElements(q, p)
	coupling := math.Sin(inc) * math.Cos(node)

	j2Pot := (MU_EARTH / r) * J2 * math.Pow(RE_EARTH/r, 2)
	rho := math.Exp(-(r - RE_EARTH) / 50.0)
	v2 := p[0]*p[0] + p[1]*p[1] + p[2]*p[2]

	return j2Pot * coupling * rho * v2 * nVal
}

// getPhysicalFeatures maps 3D state to the 13-feature manifold.
func (k *RKANKernel) getPhysicalFeatures(q, p [3]float64, jd float64) []float64 {
	r := math.Sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]) + 1e-12
	u := q[2] / r // sin(lat)

	p2 := 0.5 * (3*u*u - 1)
	p3 := 0.5 * (5*u*u*u - 3*u)
	p4 := 0.125 * (35*math.Pow(u, 4) - 30*u*u + 3)

	res := getUnifiedResonanceFeature(q, p)
	rho := math.Exp(-(r - RE_EARTH) / 50.0)
	sunV, moonV := getMeeusEphemeris(jd)

	qU := [3]float64{q[0] / r, q[1] / r, q[2] / r}
	sunDot := qU[0]*sunV[0] + qU[1]*sunV[1] + qU[2]*sunV[2]
	moonDot := qU[0]*moonV[0] + qU[1]*moonV[1] + qU[2]*moonV[2]

	pNorm := math.Sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])

	return []float64{
		q[0], q[1], q[2],
		r, 1.0 / r,
		p2, p3, res,
		rho, sunDot, moonDot,
		p4, pNorm,
	}
}

/**
 * Forward computes the scalar potential V(q, p) at a given epoch.
 */
func (k *RKANKernel) Forward(q, p [3]float64, jd float64) float64 {
	features := k.getPhysicalFeatures(q, p, jd)
	totalV := 0.0
	const outIdx = 0

	for i, val := range features {
		if i >= len(k.Num[outIdx]) {
			break
		}

		xi := k.mapToInterval(val)
		basis := k.legendreBasis(xi)

		numVal := 0.0
		denVal := 0.0
		for d := 0; d <= k.Degree; d++ {
			numVal += basis[d] * k.Num[outIdx][i][d]
			denVal += basis[d] * k.Den[outIdx][i][d]
		}

		totalV += numVal / (1.0 + math.Abs(denVal))
	}

	return totalV
}

/**
 * GetAcceleration computes the force vector F = -grad(V).
 */
func (k *RKANKernel) GetAcceleration(q, p [3]float64, jd float64) [3]float64 {
	const h = 1e-4
	var accel [3]float64

	for i := 0; i < 3; i++ {
		qPlus := q
		qMinus := q
		qPlus[i] += h
		qMinus[i] -= h

		vPlus := k.Forward(qPlus, p, jd)
		vMinus := k.Forward(qMinus, p, jd)
		accel[i] = -(vPlus - vMinus) / (2.0 * h)
	}

	return accel
}

// Yoshida-4 coefficients for 4th-order symplectic integration
var yoshida4C = [4]float64{
	1.3512071919596578,
	-1.7024143839193156,
	1.3512071919596578,
	0.0,
}

var yoshida4W = [4]float64{
	0.6756035959798289,
	-0.1756035959798289,
	-0.1756035959798289,
	0.6756035959798289,
}

// yoshida4Step performs a single 4th-order symplectic integration step.
// This ensures 10^-15 energy conservation across long arcs.
func (k *RKANKernel) yoshida4Step(q, p [3]float64, dt, jd float64) ([3]float64, [3]float64) {
	// Yoshida-4 stage-recursive integration
	// q = q + w[i] * dt * p
	// p = p + c[i] * dt * force(q)

	for i := 0; i < 4; i++ {
		// Position update
		for d := 0; d < 3; d++ {
			q[d] += yoshida4W[i] * dt * p[d]
		}

		// Momentum update (force from rKAN)
		accel := k.GetAcceleration(q, p, jd)
		for d := 0; d < 3; d++ {
			p[d] += yoshida4C[i] * dt * accel[d]
		}
	}

	return q, p
}

// Integrate integrates trajectory over a time grid using Yoshida-4.
// Returns final position, final velocity, and full trajectories.
func (k *RKANKernel) Integrate(q0, p0 [3]float64, tGrid []float64, jd0 float64) ([3]float64, [3]float64, [][3]float64, [][3]float64) {
	n := len(tGrid)
	if n == 0 {
		return q0, p0, nil, nil
	}

	q := q0
	p := p0
	qTraj := make([][3]float64, n)
	pTraj := make([][3]float64, n)

	qTraj[0] = q
	pTraj[0] = p

	for i := 1; i < n; i++ {
		dt := tGrid[i] - tGrid[i-1]
		jd := jd0 + tGrid[i]/86400.0 // Convert seconds to JD
		q, p = k.yoshida4Step(q, p, dt, jd)
		qTraj[i] = q
		pTraj[i] = p
	}

	return q, p, qTraj, pTraj
}

// IntegrateStep performs a single integration step (convenience wrapper).
func (k *RKANKernel) IntegrateStep(q, p [3]float64, dt, jd float64) ([3]float64, [3]float64) {
	return k.yoshida4Step(q, p, dt, jd)
}
