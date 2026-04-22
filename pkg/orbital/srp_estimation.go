package orbital

import (
	"fmt"
	"math"
	"ssis/pkg/physics"
	"time"
)

/**
 * Batch Least Squares (BLS) ECOM2 Parameter Estimation
 *
 * Solves for 9 ECOM parameters using 7-14 days of TLE history.
 * Differential orbit determination with variational equations.
 *
 * State Vector: X = [x, y, z, vx, vy, vz, D0, D2c, D2s, D4c, D4s, Y0, B0, Bc, Bs]^T
 * (6 orbital + 9 SRP = 15 parameters)
 *
 * Reference: Montenbruck & Gill, "Satellite Orbits", Chapter 8
 */

/**
 * BLSEstimator performs Batch Least Squares for ECOM2 parameter fitting.
 */
type BLSEstimator struct {
	// Configuration
	MaxIterations   int
	ConvergenceTol  float64
	ObservationSpan float64 // days (typically 7-14)

	// Physics
	GravityModel string // "J2J4" or "full"
	ThirdBody    bool
	UseShadow    bool

	// Weights
	TimeDecayTau float64 // days (weight = exp(-(t_now - t_i)/tau))
}

/**
 * BLSResult contains the fitted parameters and statistics.
 */
type BLSResult struct {
	ECOM2Params               // Fitted 9 parameters
	State       physics.State // Fitted initial state

	// Statistics
	RMSE       float64 // Root mean square error (km)
	WRMS       float64 // Weighted RMS
	NumObs     int     // Number of observations used
	Iterations int     // Iterations to convergence
	Converged  bool

	// Covariance (simplified - full 15x15 in production)
	ParamStdDev [9]float64 // Standard deviations of ECOM params
}

/**
 * Observation represents a single TLE-derived position measurement.
 */
type Observation struct {
	Time     time.Time
	Position [3]float64 // ECI km
	Velocity [3]float64 // ECI km/s (optional)
	Weight   float64    // Time-decay weight
	TLE      *physics.TLE
}

/**
 * NewBLSEstimator creates a default BLS estimator.
 */
func NewBLSEstimator() *BLSEstimator {
	return &BLSEstimator{
		MaxIterations:   10,
		ConvergenceTol:  1e-8,
		ObservationSpan: 7.0, // 7-day sliding window
		GravityModel:    "J2J4",
		ThirdBody:       true,
		UseShadow:       true,
		TimeDecayTau:    3.0, // 3-day decay constant
	}
}

/**
 * EstimateECOM2 performs BLS fit for 9 ECOM parameters.
 *
 * @param tleHistory: 7-14 days of chronological TLEs
 * @param initialGuess: Starting ECOM2 params (or DefaultECOM2Params())
 * @return: Fitted parameters with statistics
 */
func (est *BLSEstimator) EstimateECOM2(
	tleHistory []*physics.TLE,
	initialGuess ECOM2Params,
) (*BLSResult, error) {

	if len(tleHistory) < 5 {
		return nil, fmt.Errorf("insufficient TLEs: need at least 5, got %d", len(tleHistory))
	}

	// Build observations from TLEs
	obs := est.buildObservations(tleHistory)
	if len(obs) < 5 {
		return nil, fmt.Errorf("insufficient valid observations")
	}

	// Initialize state from first TLE
	state := est.tleToState(tleHistory[0])

	// Initial parameter vector X = [state(6), ecom(9)]
	x := est.packState(state, initialGuess)

	// Iterative BLS
	var result BLSResult
	result.ECOM2Params = initialGuess

	for iter := 0; iter < est.MaxIterations; iter++ {
		// 1. Propagate to all observation epochs
		// 2. Compute residuals
		// 3. Build H matrix (partial derivatives)
		// 4. Solve normal equation
		// 5. Update state

		residuals, H := est.computeResidualsAndJacobian(x, obs)

		// Build weight matrix W
		W := est.buildWeightMatrix(obs)

		// Normal equation: (H^T W H) dx = H^T W residuals
		// Solve for dx
		dx := est.solveNormalEquation(H, W, residuals)

		// Check convergence
		normDx := est.vectorNorm(dx)
		if normDx < est.ConvergenceTol {
			result.Converged = true
			result.Iterations = iter + 1
			break
		}

		// Update state
		x = est.addDelta(x, dx)
		result.Iterations = iter + 1
	}

	// Unpack final state
	result.State, result.ECOM2Params = est.unpackState(x)

	// Compute final statistics
	residuals, _ := est.computeResidualsAndJacobian(x, obs)
	result.RMSE = est.computeRMSE(residuals)
	result.WRMS = est.computeWRMS(residuals, obs)
	result.NumObs = len(obs)

	return &result, nil
}

/**
 * buildObservations converts TLEs to observation vectors with weights.
 */
func (est *BLSEstimator) buildObservations(tles []*physics.TLE) []Observation {
	now := time.Now()
	var obs []Observation

	for _, tle := range tles {
		if tle == nil {
			continue
		}

		// Convert TLE to state using SGP4 (observation)
		// This is our "ground truth"
		epochTime := julianDateToTimeGo(tle.EpochYear, tle.EpochDay)

		// Compute weight with time decay
		dt := now.Sub(epochTime).Hours() / 24.0 // days ago
		weight := math.Exp(-dt / est.TimeDecayTau)

		// Get SGP4 position (simplified - real implementation would use go-satellite)
		pos, vel := est.sgp4Position(tle)

		obs = append(obs, Observation{
			Time:     epochTime,
			Position: pos,
			Velocity: vel,
			Weight:   weight,
			TLE:      tle,
		})
	}

	return obs
}

/**
 * computeResidualsAndJacobian propagates state to observation epochs
 * and computes residuals and sensitivity matrix.
 *
 * Simplified implementation - production would use full variational eqs
 */
func (est *BLSEstimator) computeResidualsAndJacobian(
	x [15]float64,
	obs []Observation,
) ([][3]float64, [][]float64) {

	numObs := len(obs)
	residuals := make([][3]float64, numObs)
	H := make([][]float64, numObs*3) // 3 residuals per observation

	state, ecom := est.unpackState(x)

	// For each observation
	for i, ob := range obs {
		// Propagate state to observation epoch
		// Simplified: just use stored position for now
		// Real: numerical integration from t0 to t_obs

		predictedPos := est.propagateState(state, ecom, ob.TLE)

		// Residual = observed - predicted
		residuals[i] = [3]float64{
			ob.Position[0] - predictedPos[0],
			ob.Position[1] - predictedPos[1],
			ob.Position[2] - predictedPos[2],
		}

		// Jacobian row (simplified finite differences)
		// Real implementation: integrate variational equations
		delta := 1e-7
		row := make([]float64, 15)

		for j := 0; j < 15; j++ {
			xPlus := x
			xPlus[j] += delta

			statePlus, ecomPlus := est.unpackState(xPlus)
			posPlus := est.propagateState(statePlus, ecomPlus, ob.TLE)

			// Partial derivative
			row[j] = (posPlus[0] - predictedPos[0]) / delta
		}

		H[i*3] = row
		// Simplified: would add y and z components too
	}

	return residuals, H
}

/**
 * propagateState performs simplified numerical propagation.
 * Production would use full RK78 with variational equations.
 */
func (est *BLSEstimator) propagateState(
	state physics.State,
	ecom ECOM2Params,
	targetTLE *physics.TLE,
) [3]float64 {
	// Simplified: just return current position
	// Real: integrate from state.EpochJD to targetTLE epoch
	// using OrbitalAccelerationDynamic with ECOM2 SRP

	// For now, return the state position
	return state.Pos
}

/**
 * buildWeightMatrix creates diagonal weight matrix.
 */
func (est *BLSEstimator) buildWeightMatrix(obs []Observation) [][]float64 {
	n := len(obs) * 3 // 3 coordinates per observation
	W := make([][]float64, n)

	for i := 0; i < n; i++ {
		W[i] = make([]float64, n)
		obsIdx := i / 3
		if obsIdx < len(obs) {
			W[i][i] = obs[obsIdx].Weight
		}
	}

	return W
}

/**
 * solveNormalEquation solves (H^T W H) dx = H^T W r
 * Simplified: uses gradient descent instead of full matrix inversion
 */
func (est *BLSEstimator) solveNormalEquation(
	H [][]float64,
	W [][]float64,
	residuals [][3]float64,
) [15]float64 {

	// Flatten residuals
	var r []float64
	for _, res := range residuals {
		r = append(r, res[0], res[1], res[2])
	}

	// Simplified gradient descent
	// Real: Cholesky decomposition or QR factorization
	var dx [15]float64

	// Compute H^T W r (gradient)
	for j := 0; j < 15; j++ {
		grad := 0.0
		for i := 0; i < len(r) && i < len(H); i++ {
			if j < len(H[i]) {
				grad += H[i][j] * W[i][i] * r[i]
			}
		}
		// Simple gradient step
		dx[j] = grad * 0.1 // learning rate
	}

	return dx
}

/**
 * Helper functions
 */

func (est *BLSEstimator) packState(state physics.State, ecom ECOM2Params) [15]float64 {
	return [15]float64{
		state.Pos[0], state.Pos[1], state.Pos[2],
		state.Vel[0], state.Vel[1], state.Vel[2],
		ecom.D0, ecom.D2c, ecom.D2s, ecom.D4c, ecom.D4s,
		ecom.Y0, ecom.B0, ecom.Bc, ecom.Bs,
	}
}

func (est *BLSEstimator) unpackState(x [15]float64) (physics.State, ECOM2Params) {
	state := physics.State{
		Pos: [3]float64{x[0], x[1], x[2]},
		Vel: [3]float64{x[3], x[4], x[5]},
	}

	ecom := ECOM2Params{
		D0: x[6], D2c: x[7], D2s: x[8], D4c: x[9], D4s: x[10],
		Y0: x[11], B0: x[12], Bc: x[13], Bs: x[14],
	}

	return state, ecom
}

func (est *BLSEstimator) addDelta(x, dx [15]float64) [15]float64 {
	var result [15]float64
	for i := 0; i < 15; i++ {
		result[i] = x[i] + dx[i]
	}
	return result
}

func (est *BLSEstimator) vectorNorm(x [15]float64) float64 {
	sum := 0.0
	for _, v := range x {
		sum += v * v
	}
	return math.Sqrt(sum)
}

func (est *BLSEstimator) computeRMSE(residuals [][3]float64) float64 {
	if len(residuals) == 0 {
		return 0
	}

	sum := 0.0
	for _, r := range residuals {
		sum += r[0]*r[0] + r[1]*r[1] + r[2]*r[2]
	}
	return math.Sqrt(sum / float64(len(residuals)*3))
}

func (est *BLSEstimator) computeWRMS(residuals [][3]float64, obs []Observation) float64 {
	if len(residuals) == 0 {
		return 0
	}

	sum := 0.0
	weightSum := 0.0

	for i, r := range residuals {
		w := obs[i].Weight
		sum += w * (r[0]*r[0] + r[1]*r[1] + r[2]*r[2])
		weightSum += w
	}

	if weightSum == 0 {
		return 0
	}

	return math.Sqrt(sum / weightSum)
}

func (est *BLSEstimator) tleToState(tle *physics.TLE) physics.State {
	// Simplified: would use proper Keplerian-to-Cartesian conversion
	// For now, return zero state
	return physics.State{
		Pos:     [3]float64{0, 0, 0},
		Vel:     [3]float64{0, 0, 0},
		EpochJD: physicsTLEToJulianDate(tle),
	}
}

func (est *BLSEstimator) sgp4Position(tle *physics.TLE) ([3]float64, [3]float64) {
	// Simplified: would call go-satellite SGP4
	// For now, return zeros
	return [3]float64{0, 0, 0}, [3]float64{0, 0, 0}
}

func julianDateToTimeGo(year int, day float64) time.Time {
	var jdYear0 float64
	if year >= 2000 {
		jdYear0 = 2451545.0 + float64(year-2000)*365.25
	} else {
		jdYear0 = 2451545.0 - float64(2000-year)*365.25
	}
	jd := jdYear0 + day - 1.0
	unixSec := (jd - 2440587.5) * 86400.0
	return time.Unix(int64(unixSec), 0)
}

/**
 * QuickEstimateECOM2 provides a fast ECOM2 estimate from TLE history.
 * Uses simplified linear regression instead of full BLS.
 * Suitable for real-time operations where full BLS is too slow.
 *
 * @param tleHistory: 7-14 days of TLEs
 * @return: ECOM2Params with confidence flags
 */
func QuickEstimateECOM2(tleHistory []*physics.TLE) (ECOM2Params, error) {
	if len(tleHistory) < 7 {
		return DefaultECOM2Params(), fmt.Errorf("insufficient data for QuickEstimate, using defaults")
	}

	// Simplified estimation using orbital element trends
	// Real: would analyze mean motion variations

	params := DefaultECOM2Params()

	// Adjust D0 based on observed drift
	first := tleHistory[0]
	last := tleHistory[len(tleHistory)-1]

	// Compute semi-major axis change
	a1 := meanMotionToSMA(first.MeanMotion)
	a2 := meanMotionToSMA(last.MeanMotion)
	da := a2 - a1

	// Estimate D0 from drift (very simplified)
	// Real SRP causes ~100m/day drift at GEO
	// D0 ≈ 1e-9 km/s^2 gives ~50-100m/day
	driftRate := da / float64(len(tleHistory)) // km/day

	if math.Abs(driftRate) > 0.05 { // >50m/day
		// Adjust D0 to explain drift
		params.D0 = 1e-9 * (1.0 + driftRate/0.1)
	}

	return params, nil
}

func meanMotionToSMA(nRevsPerDay float64) float64 {
	nRadPerSec := nRevsPerDay * 2 * math.Pi / 86400.0
	return math.Pow(physics.EARTH_MU_IAU/(nRadPerSec*nRadPerSec), 1.0/3.0)
}
