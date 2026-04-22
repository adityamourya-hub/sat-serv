package od

/**
 * Batch Least Squares Orbit Determination
 *
 * Generates osculating state + 6x6 covariance from
 * a batch of sensor observations (radar/optical).
 *
 * This addresses the critical gap: TLEs don't have covariance,
 * but OD produces it from your own sensor data.
 *
 * Algorithm:
 * 1. Start with initial guess (TLE propagated or Gauss IOD)
 * 2. Compute expected observations from current state estimate
 * 3. Form residuals (O - C)
 * 4. Compute Jacobian (sensitivity matrix)
 * 5. Solve normal equations for state correction
 * 6. Iterate until convergence
 * 7. Output: State + Covariance = (AᵀWA)⁻¹
 *
 * References:
 * - Vallado: Batch Least Squares formulation
 * - Orekit: BatchLSEstimator implementation
 * - GorgiAstro/orbit-determination-examples
 */

import (
	"fmt"
	"math"

	"ssis/pkg/orbital"
	"ssis/pkg/physics"
)

// BatchLSEstimator performs Batch Least Squares OD
type BatchLSEstimator struct {
	MaxIterations        int
	ConvergenceThreshold float64 // Residual RMS change threshold
	DampingFactor        float64 // Levenberg-Marquardt damping
}

// NewBatchLSEstimator creates default estimator
func NewBatchLSEstimator() *BatchLSEstimator {
	return &BatchLSEstimator{
		MaxIterations:        10,
		ConvergenceThreshold: 1e-6, // 1e-6 km residual change = converged
		DampingFactor:        1e-3,
	}
}

// ODResult contains the output state and covariance
type ODResult struct {
	State       physics.State
	Covariance  [6][6]float64 // 6x6 position/velocity covariance (km, km/s)
	EpochJD     float64
	ResidualRMS float64 // Root mean square of residuals
	Iterations  int     // Number of iterations to convergence
	Converged   bool    // Whether solution converged
}

// Estimate performs Batch LS on observation batch
func (bls *BatchLSEstimator) Estimate(
	batch *ObservationBatch,
	initialState physics.State,
	initialJD float64,
	params physics.PhysicsParams,
) (*ODResult, error) {

	currentState := initialState
	currentJD := initialJD

	var lastResidualRMS float64 = 1e9

	for iter := 0; iter < bls.MaxIterations; iter++ {
		// Build normal equations
		// AᵀWA dx = AᵀW b
		// where A = Jacobian, W = weight matrix, b = residuals

		var AtWA [6][6]float64 // Normal matrix (6x6)
		var AtWb [6]float64    // Right-hand side (6)

		residuals := make([]float64, len(batch.Observations))

		for i, obs := range batch.Observations {
			// Propagate state to observation time
			obsJD := JulianDateFromTime(obs.Time)
			dt := (obsJD - currentJD) * 86400.0 // seconds

			propState := bls.propagateState(currentState, dt, currentJD, params)

			// Compute residual
			model := ObservationModel{
				ObsType: obs.Type,
				Station: obs.Station,
			}
			computed := model.ComputeExpected(propState, obsJD)
			residual := obs.Value - computed
			residuals[i] = residual

			// Weight = 1/σ²
			weight := 1.0 / (obs.Sigma * obs.Sigma)

			// Compute Jacobian row (sensitivity of this observation to state)
			jacobianRow := bls.computeJacobianRow(model, propState, obsJD, obs.Type)

			// Accumulate AᵀWA and AᵀWb
			for j := 0; j < 6; j++ {
				AtWb[j] += jacobianRow[j] * weight * residual
				for k := 0; k < 6; k++ {
					AtWA[j][k] += jacobianRow[j] * weight * jacobianRow[k]
				}
			}
		}

		// Compute residual RMS
		residualRMS := computeRMS(residuals)

		// Check convergence
		if math.Abs(residualRMS-lastResidualRMS) < bls.ConvergenceThreshold {
			// Compute covariance from normal matrix
			covariance := invertSymmetricMatrix(AtWA)

			return &ODResult{
				State:       currentState,
				Covariance:  covariance,
				EpochJD:     currentJD,
				ResidualRMS: residualRMS,
				Iterations:  iter + 1,
				Converged:   true,
			}, nil
		}

		lastResidualRMS = residualRMS

		// Solve for state correction: dx = (AᵀWA)⁻¹ AᵀWb
		// Add damping for stability (Levenberg-Marquardt)
		for i := 0; i < 6; i++ {
			AtWA[i][i] += bls.DampingFactor
		}

		dx := solveLinearSystem(AtWA, AtWb)

		// Update state
		currentState.Pos[0] += dx[0]
		currentState.Pos[1] += dx[1]
		currentState.Pos[2] += dx[2]
		currentState.Vel[0] += dx[3]
		currentState.Vel[1] += dx[4]
		currentState.Vel[2] += dx[5]
	}

	// Did not converge
	return &ODResult{
		State:       currentState,
		EpochJD:     currentJD,
		ResidualRMS: lastResidualRMS,
		Iterations:  bls.MaxIterations,
		Converged:   false,
	}, fmt.Errorf("batch LS did not converge after %d iterations", bls.MaxIterations)
}

// propagateState propagates cartesian state using RKF45
func (bls *BatchLSEstimator) propagateState(
	state physics.State,
	dt float64,
	jd float64,
	params physics.PhysicsParams,
) physics.State {

	x := []float64{
		state.Pos[0], state.Pos[1], state.Pos[2],
		state.Vel[0], state.Vel[1], state.Vel[2],
	}

	newX := orbital.PropagateCartesianState(x, dt, jd, params)

	return physics.State{
		Pos: [3]float64{newX[0], newX[1], newX[2]},
		Vel: [3]float64{newX[3], newX[4], newX[5]},
	}
}

// computeJacobianRow computes partial derivatives for one observation
// Returns ∂obs/∂state (1x6 row vector)
func (bls *BatchLSEstimator) computeJacobianRow(
	model ObservationModel,
	state physics.State,
	jd float64,
	obsType ObservationType,
) [6]float64 {

	// Numerical differentiation (finite differences)
	// ∂f/∂x ≈ (f(x+δx) - f(x-δx)) / (2δx)

	epsilon := 1e-6 // km for position, km/s for velocity

	var jacobian [6]float64

	for i := 0; i < 6; i++ {
		// Forward perturbation
		statePlus := state
		if i < 3 {
			statePlus.Pos[i] += epsilon
		} else {
			statePlus.Vel[i-3] += epsilon
		}
		obsPlus := model.ComputeExpected(statePlus, jd)

		// Backward perturbation
		stateMinus := state
		if i < 3 {
			stateMinus.Pos[i] -= epsilon
		} else {
			stateMinus.Vel[i-3] -= epsilon
		}
		obsMinus := model.ComputeExpected(stateMinus, jd)

		// Central difference
		jacobian[i] = (obsPlus - obsMinus) / (2.0 * epsilon)
	}

	return jacobian
}

// computeRMS calculates root mean square
func computeRMS(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, v := range values {
		sum += v * v
	}

	return math.Sqrt(sum / float64(len(values)))
}

// solveLinearSystem solves Ax = b for symmetric positive definite A
// Using Gaussian elimination with partial pivoting
func solveLinearSystem(A [6][6]float64, b [6]float64) [6]float64 {
	// Convert to float64 slices for easier manipulation
	n := 6
	a := make([][]float64, n)
	for i := range a {
		a[i] = make([]float64, n)
		copy(a[i], A[i][:])
	}

	x := make([]float64, n)
	copy(x, b[:])

	// Forward elimination
	for i := 0; i < n; i++ {
		// Partial pivoting
		maxVal := math.Abs(a[i][i])
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(a[k][i]) > maxVal {
				maxVal = math.Abs(a[k][i])
				maxRow = k
			}
		}

		// Swap rows
		for k := i; k < n; k++ {
			a[i][k], a[maxRow][k] = a[maxRow][k], a[i][k]
		}
		x[i], x[maxRow] = x[maxRow], x[i]

		// Eliminate column
		for k := i + 1; k < n; k++ {
			factor := a[k][i] / a[i][i]
			for j := i; j < n; j++ {
				a[k][j] -= factor * a[i][j]
			}
			x[k] -= factor * x[i]
		}
	}

	// Back substitution
	for i := n - 1; i >= 0; i-- {
		x[i] /= a[i][i]
		for k := i - 1; k >= 0; k-- {
			x[k] -= a[k][i] * x[i]
		}
	}

	var result [6]float64
	copy(result[:], x)
	return result
}

// invertSymmetricMatrix inverts a symmetric matrix using Cholesky
// Returns the inverse (which is also the covariance matrix)
func invertSymmetricMatrix(A [6][6]float64) [6][6]float64 {
	// For covariance computation, we need A⁻¹ where A = AᵀWA (normal matrix)
	// Using Gaussian elimination for simplicity

	n := 6

	// Create augmented matrix [A | I]
	aug := make([][]float64, n)
	for i := range aug {
		aug[i] = make([]float64, 2*n)
		for j := 0; j < n; j++ {
			aug[i][j] = A[i][j]
		}
		aug[i][n+i] = 1.0 // Identity
	}

	// Gaussian elimination
	for i := 0; i < n; i++ {
		// Pivot
		pivot := aug[i][i]
		for j := 0; j < 2*n; j++ {
			aug[i][j] /= pivot
		}

		// Eliminate other rows
		for k := 0; k < n; k++ {
			if k != i {
				factor := aug[k][i]
				for j := 0; j < 2*n; j++ {
					aug[k][j] -= factor * aug[i][j]
				}
			}
		}
	}

	// Extract inverse
	var inverse [6][6]float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			inverse[i][j] = aug[i][n+j]
		}
	}

	return inverse
}

// EstimateFromTLE performs OD starting from TLE initial guess
func (bls *BatchLSEstimator) EstimateFromTLE(
	batch *ObservationBatch,
	tle physics.TLE,
	params physics.PhysicsParams,
) (*ODResult, error) {

	// Convert TLE epoch to Julian Date
	// EpochYear is last 2 digits of year (e.g., 24 for 2024)
	year := tle.EpochYear
	if year < 57 {
		year += 2000 // 00-56 -> 2000-2056
	} else {
		year += 1900 // 57-99 -> 1957-1999
	}

	// JD for start of year + day of year
	epochJD := julianDayFromYMD(year, 1, 1) + tle.EpochDay - 1.0

	// Initial guess from TLE (simplified - would use full SGP4 in production)
	// TODO: Integrate pkg/orbital/sgp4_akhenakh.go for production-grade TLE propagation
	// Current implementation uses circular orbit approximation which can have ~100km error
	sma := math.Pow(physics.EARTH_MU_IAU/math.Pow(tle.MeanMotion*2*math.Pi/86400.0, 2), 1.0/3.0)

	// Position at epoch (simplified)
	pos := [3]float64{sma, 0, 0}
	vel := [3]float64{0, math.Sqrt(physics.EARTH_MU_IAU / sma), 0}

	initialState := physics.State{
		Pos: pos,
		Vel: vel,
	}

	return bls.Estimate(batch, initialState, epochJD, params)
}

// julianDayFromYMD computes Julian Day from year, month, day (simplified)
func julianDayFromYMD(year, month, day int) float64 {
	// Algorithm from Meeus
	if month <= 2 {
		year--
		month += 12
	}

	A := int(float64(year) / 100.0)
	B := 2 - A + int(float64(A)/4.0)

	jd := float64(int(365.25*float64(year+4716))) +
		float64(int(30.6001*float64(month+1))) +
		float64(day) + float64(B) - 1524.5

	return jd
}
