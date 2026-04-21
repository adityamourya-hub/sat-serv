package proximity

import (
	"fmt"
	"math"
	"sync"
	"time"

	"ssis/pkg/physics"

	"gonum.org/v1/gonum/mat"
)

// cacheKey represents a unique key for caching STM computations
type cacheKey struct {
	radius  float64
	horizon float64
}

// cacheEntry stores cached phi_rv matrix and related data
type cacheEntry struct {
	phiRv     *mat.Dense
	timestamp time.Time
}

// stmCache implements a simple LRU cache for STM computations
type stmCache struct {
	mu      sync.RWMutex
	cache   map[cacheKey]*cacheEntry
	maxSize int
	keys    []cacheKey
}

var (
	globalSTMCache *stmCache
	cacheOnce      sync.Once
)

// getSTMCache returns the singleton STM cache
func getSTMCache() *stmCache {
	cacheOnce.Do(func() {
		globalSTMCache = &stmCache{
			cache:   make(map[cacheKey]*cacheEntry),
			maxSize: 1000,
			keys:    make([]cacheKey, 0, 1000),
		}
	})
	return globalSTMCache
}

// get retrieves a cached phi_rv matrix
func (c *stmCache) get(key cacheKey) (*mat.Dense, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	entry, exists := c.cache[key]
	if !exists {
		return nil, false
	}
	return entry.phiRv, true
}

// put stores a phi_rv matrix in the cache with LRU eviction
func (c *stmCache) put(key cacheKey, phiRv *mat.Dense) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// If key already exists, update it
	if _, exists := c.cache[key]; exists {
		c.cache[key] = &cacheEntry{
			phiRv:     phiRv,
			timestamp: time.Now(),
		}
		return
	}

	// If at capacity, evict oldest entry
	if len(c.cache) >= c.maxSize {
		oldestKey := c.keys[0]
		delete(c.cache, oldestKey)
		c.keys = c.keys[1:]
	}

	// Add new entry
	c.cache[key] = &cacheEntry{
		phiRv:     phiRv,
		timestamp: time.Now(),
	}
	c.keys = append(c.keys, key)
}

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
 * 1. Clohessy-Wiltshire (CW) analytical solutions for T < threshold.
 * 2. Linearised State Transition Matrix (STM) with J2 for threshold < T < max horizon.
 */
func CalculateRRI(asset, threat physics.State, deltaV float64, horizon float64) (*RRI, error) {
	cfg := GetConfig()
	fmt.Printf("[TRC] Entering CalculateRRI for Asset->Threat dv=%.2f horizon=%.2f\n", deltaV, horizon)
	defer fmt.Printf("[TRC] Exiting CalculateRRI\n")

	var ellipsoid *ReachabilityEllipsoid
	var err error

	if horizon < cfg.CWHorizonThresholdSec {
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
 *
 * The CW equations describe relative motion in the Hill (RSW) frame for a circular
 * reference orbit. The State Transition Matrix Φ_rv(t) maps an impulsive delta-V
 * to the resulting relative position after time t:
 *
 *   Δr(t) = Φ_rv(t) * Δv
 *
 * Where Φ_rv(t) for the Hill frame is:
 *
 *   Φ_rv = | (4*sin(nt) - 3*nt)/n      (2/n)*(1-cos(nt))       0          |
 *          | (2/n)*(cos(nt) - 1)        sin(nt)/n                0          |
 *          | 0                           0                        sin(nt)/n  |
 *
 * Where:
 *   - n = sqrt(μ / r³) is the mean motion of the reference orbit
 *   - μ = Earth's gravitational parameter
 *   - r = Reference orbit radius
 *   - t = Time horizon
 *
 * The reachability ellipsoid shape matrix E is computed as:
 *   E = (deltaV² * Φ_rv * Φ_rv^T)⁻¹
 *
 * Reference: Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
 * Reference: Clohessy, W. H., & Wiltshire, R. S. (1960). Terminal guidance system
 *             for spacecraft rendezvous. Journal of the Astronautical Sciences.
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
 * Uses caching for phi_rv matrix computations to improve performance.
 *
 * The J2 perturbation accounts for Earth's oblateness (J2 ≈ 1.0826e-3) which causes
 * secular drift in the argument of latitude and right ascension of the ascending node.
 * This is particularly important for long-term reachability analysis (T > 30 minutes).
 *
 * The corrected mean motion n_bar includes J2 effects:
 *   n_bar = n * (1 + (3/2)*J2*(R_e/r)²*(1 - 1.5*sin²(lat)))
 *
 * Where:
 *   - n = sqrt(μ / r³) is the unperturbed mean motion
 *   - J2 = Earth's second zonal harmonic
 *   - R_e = Earth's equatorial radius
 *   - r = Orbital radius
 *   - lat = Orbital latitude
 *
 * The STM with J2 corrections provides more accurate reachability volumes for
 * extended time horizons where secular drift becomes significant.
 *
 * Reference: Vallado, "Fundamentals of Astrodynamics and Applications", Section 9.3
 * Reference: Schaub & Junkins, "Analytical Mechanics of Space Systems", Chapter 4
 */
func computeSTMReachability(ref physics.State, deltaV, t float64) (*ReachabilityEllipsoid, error) {
	fmt.Printf("[TRC] Entering computeSTMReachability t=%.2f\n", t)
	defer fmt.Printf("[TRC] Exiting computeSTMReachability\n")

	r := math.Sqrt(ref.Pos[0]*ref.Pos[0] + ref.Pos[1]*ref.Pos[1] + ref.Pos[2]*ref.Pos[2])

	// Check cache for phi_rv matrix
	cache := getSTMCache()
	key := cacheKey{radius: r, horizon: t}
	if cachedPhiRv, found := cache.get(key); found {
		fmt.Printf("[TRC] Cache HIT for phi_rv (r=%.2f, t=%.2f)\n", r, t)
		return computeEllipsoidFromPhiRv(cachedPhiRv, deltaV)
	}
	fmt.Printf("[TRC] Cache MISS for phi_rv (r=%.2f, t=%.2f)\n", r, t)

	n := math.Sqrt(physics.EARTH_MU_IAU / (r * r * r))

	// J2 Correction for mean motion and drift
	j2Eff := 1.5 * physics.J2_ADITI * math.Pow(physics.EARTH_R_EQUATORIAL/r, 2)
	lat := math.Asin(ref.Pos[2] / r)
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

	// Cache the phi_rv matrix for future use
	cache.put(key, phi_rv)

	return computeEllipsoidFromPhiRv(phi_rv, deltaV)
}

// computeEllipsoidFromPhiRv computes the reachability ellipsoid from a cached phi_rv matrix
func computeEllipsoidFromPhiRv(phi_rv *mat.Dense, deltaV float64) (*ReachabilityEllipsoid, error) {
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
