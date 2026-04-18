package proximity

import (
	"fmt"
	"math"
	"time"

	"gonum.org/v1/gonum/mat"
	"ssis/pkg/physics"
)

/**
 * ReachabilityRiskIndex (RRI) defines the operational boundary for neighborhood watch.
 * It encapsulates the reachability manifold of a potential non-cooperative neighbor 
 * based on its estimated delta-v budget.
 */
type RRI struct {
	ObjectID   int
	Ellipsoid  *ReachabilityEllipsoid
	HorizonSec float64
	Timestamp  time.Time
}

/**
 * ReachabilityEllipsoid represents a 3D volume in the relative Hill (RSW) frame 
 * that a satellite can reach within a specific time horizon.
 */
type ReachabilityEllipsoid struct {
	Center [3]float64    // Relative center (usually 0,0,0)
	Matrix *mat.SymDense // 3x3 shape matrix E where (x-c)T E (x-c) <= 1
}

/**
 * CalculateRRI computes the Reachability Risk Index for a given asset-threat pair.
 * It uses a hybrid approach:
 * 1. Clohessy-Wiltshire (CW) analytical solutions for T < 30m.
 * 2. Linearised State Transition Matrix (STM) with J2 for 30m < T < 2h.
 */
func CalculateRRI(asset, threat physics.State, deltaV float64, horizon float64) (*RRI, error) {
	fmt.Printf("[TRC] Entering CalculateRRI for Asset->Threat dv=%.2f horizon=%.2f\n", deltaV, horizon)
	defer fmt.Printf("[TRC] Exiting CalculateRRI\n")
	
	var ellipsoid *ReachabilityEllipsoid
	var err error

	if horizon < 1800.0 { // T < 30 minutes
		ellipsoid, err = computeCWReliability(asset, deltaV, horizon)
	} else {
		ellipsoid, err = computeSTMReachability(asset, deltaV, horizon)
	}

	if err != nil {
		return nil, err
	}

	rri := &RRI{
		ObjectID:   0, // To be mapped by caller
		Ellipsoid:  ellipsoid,
		HorizonSec: horizon,
		Timestamp:  time.Now(),
	}
	
	vol := 4.0 / 3.0 * math.Pi * math.Sqrt(1.0/mat.Det(ellipsoid.Matrix))
	fmt.Printf("[TRC] Ellipsoid Volume = %.2f km^3\n", vol)
	return rri, nil
}

/**
 * computeCWReliability implements the analytical Reachability Ellipsoid 
 * based on the Clohessy-Wiltshire (CW) equations for proximal relative motion.
 */
func computeCWReliability(ref physics.State, deltaV, t float64) (*ReachabilityEllipsoid, error) {
	fmt.Printf("[TRC] Entering computeCWReliability t=%.2f\n", t)
	defer fmt.Printf("[TRC] Exiting computeCWReliability\n")
	
	// 1. Calculate mean motion n
	r := math.Sqrt(ref.Pos[0]*ref.Pos[0] + ref.Pos[1]*ref.Pos[1] + ref.Pos[2]*ref.Pos[2])
	n := math.Sqrt(physics.EARTH_MU_IAU / (r * r * r))
	
	nt := n * t
	sinNT := math.Sin(nt)
	cosNT := math.Cos(nt)

	// 2. CW State Transition Matrix components (Phi_rv)
	props := make([]float64, 9)
	phi_rv := mat.NewDense(3, 3, props)
	
	phi_rv.Set(0, 0, (4*sinNT-3*nt)/n)
	phi_rv.Set(0, 1, (2/n)*(1-cosNT))
	phi_rv.Set(0, 2, 0)
	
	phi_rv.Set(1, 0, (2/n)*(cosNT-1))
	phi_rv.Set(1, 1, sinNT/n)
	phi_rv.Set(1, 2, 0)
	
	phi_rv.Set(2, 0, 0)
	phi_rv.Set(2, 1, 0)
	phi_rv.Set(2, 2, sinNT/n)

	// 3. Shape matrix C = deltaV^2 * (Phi_rv * Phi_rv^T)
	var matC mat.Dense
	matC.Mul(phi_rv, phi_rv.T())
	
	dvSq := deltaV * deltaV
	for i := 0; i < 9; i++ {
		props[i] *= dvSq
	}

	// 4. Matrix E is the inverse of C
	var invC mat.Dense
	if err := invC.Inverse(&matC); err != nil {
		return nil, fmt.Errorf("singular reachability matrix: %w", err)
	}

	symE := mat.NewSymDense(3, nil)
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			symE.SetSym(i, j, invC.At(i, j))
		}
	}

	return &ReachabilityEllipsoid{
		Center: [3]float64{0, 0, 0},
		Matrix: symE,
	}, nil
}

/**
 * computeSTMReachability implements a more robust reachability manifold 
 * using a linearised State Transition Matrix (STM) that accounts for 
 * J2 perturbations (Earth's oblateness).
 */
func computeSTMReachability(ref physics.State, deltaV, t float64) (*ReachabilityEllipsoid, error) {
	fmt.Printf("[TRC] Entering computeSTMReachability t=%.2f\n", t)
	defer fmt.Printf("[TRC] Exiting computeSTMReachability\n")
	
	r := math.Sqrt(ref.Pos[0]*ref.Pos[0] + ref.Pos[1]*ref.Pos[1] + ref.Pos[2]*ref.Pos[2])
	n := math.Sqrt(physics.EARTH_MU_IAU / (r * r * r))
	
	// J2 Correction for mean motion and drift
	j2Eff := 1.5 * physics.J2_ADITI * math.Pow(physics.EARTH_R_EQUATORIAL/r, 2)
	lat := math.Asin(ref.Pos[2]/r)
	n_bar := n * (1 + j2Eff*(1-1.5*math.Pow(math.Sin(lat), 2)))
	
	nt := n_bar * t
	sinNT := math.Sin(nt)
	cosNT := math.Cos(nt)

	phi_rv := mat.NewDense(3, 3, nil)
	phi_rv.Set(0, 0, (4*sinNT-3*nt)/n_bar)
	phi_rv.Set(0, 1, (2/n_bar)*(1-cosNT))
	phi_rv.Set(1, 0, (2/n_bar)*(cosNT-1))
	phi_rv.Set(1, 1, sinNT/n_bar)
	phi_rv.Set(2, 2, sinNT/n_bar)

	var matC mat.Dense
	matC.Mul(phi_rv, phi_rv.T())
	
	dvSq := deltaV * deltaV
	scaleMat(&matC, dvSq)

	var invC mat.Dense
	if err := invC.Inverse(&matC); err != nil {
		return nil, fmt.Errorf("singular STM reachability matrix: %w", err)
	}

	symE := mat.NewSymDense(3, nil)
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			symE.SetSym(i, j, invC.At(i, j))
		}
	}

	return &ReachabilityEllipsoid{
		Center: [3]float64{0, 0, 0},
		Matrix: symE,
	}, nil
}

/**
 * scaleMat is a helper to multiply all elements of a matrix by a scalar.
 */
func scaleMat(m *mat.Dense, s float64) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, m.At(i, j)*s)
		}
	}
}

/**
 * Intersects checks if a given relative position resides within the reachability volume.
 * Validates against the shape matrix quadratic form: xT E x <= 1.
 */
func (r *RRI) Intersects(relPos [3]float64) bool {
	fmt.Printf("[TRC] Entering RRI.Intersects pos=(%.2f, %.2f, %.2f)\n", relPos[0], relPos[1], relPos[2])
	defer fmt.Printf("[TRC] Exiting RRI.Intersects\n")
	
	x := mat.NewVecDense(3, []float64{
		relPos[0] - r.Ellipsoid.Center[0],
		relPos[1] - r.Ellipsoid.Center[1],
		relPos[2] - r.Ellipsoid.Center[2],
	})

	var tmp mat.VecDense
	tmp.MulVec(r.Ellipsoid.Matrix, x)
	val := mat.Dot(x, &tmp)

	return val <= 1.0
}

/**
 * Determinant3x3 calculates the determinant of a 3x3 symmetric matrix.
 * Specifically used for RRI volume calculations.
 */
func Determinant3x3(m *mat.SymDense) float64 {
	fmt.Printf("[TRC] Entering Determinant3x3\n")
	defer fmt.Printf("[TRC] Exiting Determinant3x3\n")
	a := m.At(0, 0)
	b := m.At(0, 1)
	c := m.At(0, 2)
	e := m.At(1, 1)
	f := m.At(1, 2)
	i := m.At(2, 2)

	det := a*(e*i-f*f) - b*(b*i-f*c) + c*(b*f-e*c)
	return det
}
