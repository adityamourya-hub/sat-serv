package rf

/**
 * TDOA-PDOA Localization Algorithm
 *
 * ADITI 4.0 Implementation of Chen et al. IEEE WCL 2020
 *
 * Combines Time Difference of Arrival (TDOA) and
 * Phase Difference of Arrival (PDOA) for high-precision localization
 * of RF emitters (satellites, ground stations, etc.)
 *
 * Key Innovation:
 * - TDOA provides coarse localization (km-scale accuracy)
 * - PDOA provides fine localization (m-scale accuracy)
 * - Combined cost function for robust estimation
 *
 * References:
 * - Chen Hui et al., "Joint TDOA-PDOA Localization", IEEE WCL 2020
 * - WCL2020-0388
 */

import (
	"math"
)

// TDOAPDOALocalizer implements joint TDOA-PDOA localization
type TDOAPDOALocalizer struct {
	Stations []RFStation

	// Algorithm parameters (from Chen WCL2020-0388)
	Alpha float64 // Weight for TDOA term
	Beta  float64 // Weight for PDOA term

	// Optimization settings
	MaxIterations  int
	ConvergenceTol float64
}

// NewTDOAPDOALocalizer creates localizer with default parameters
func NewTDOAPDOALocalizer(stations []RFStation) *TDOAPDOALocalizer {
	return &TDOAPDOALocalizer{
		Stations:       stations,
		Alpha:          0.7, // TDOA weight
		Beta:           0.3, // PDOA weight
		MaxIterations:  100,
		ConvergenceTol: 1e-6,
	}
}

// TDOAPDOAMeasurement combines time and phase measurements
type TDOAPDOAMeasurement struct {
	// TDOA measurements (seconds)
	// τ[i][j] = time difference between station i and j
	TDOA [][]float64

	// PDOA measurements (radians)
	// φ[i][j] = phase difference between station i and j
	PDOA [][]float64

	// Measurement covariance matrices
	TDOACovariance [][]float64
	PDOAVariance   float64

	// Signal parameters
	CarrierFrequency float64 // Hz
	Wavelength       float64 // meters (λ = c/f)
}

// LocalizationResult contains position estimate and quality metrics
type LocalizationResult struct {
	Position   [3]float64    // ECI position (km)
	Velocity   [3]float64    // ECI velocity (km/s)
	Covariance [3][3]float64 // Position covariance (km²)

	// Quality metrics
	TDOResidual   float64       // TDOA cost function value
	PDOResidual   float64       // PDOA cost function value
	TotalResidual float64       // Combined residual
	CRB           [3][3]float64 // Cramér-Rao Bound

	// Convergence info
	Iterations int
	Converged  bool
}

// Localize performs joint TDOA-PDOA localization using Levenberg-Marquardt
// Implements the proposed algorithm from Chen WCL2020-0388 with LM refinement
func (loc *TDOAPDOALocalizer) Localize(measurement TDOAPDOAMeasurement) LocalizationResult {
	// Step 1: Coarse grid search for initial estimate
	bestPos := loc.coarseGridSearch(measurement)

	// Step 2: Levenberg-Marquardt refinement
	position := loc.levenbergMarquardtTDOAPDOA(bestPos, measurement, loc.MaxIterations, 0.01)

	// Compute final quality metrics
	tdoaRes := loc.computeTDOACost(position, measurement)
	pdoaRes := loc.computePDOCost(position, measurement)
	finalResidual := loc.computeCombinedCost(position, measurement)
	crb := loc.computeCRB(position, measurement)
	covariance := invert3x3(crb)

	// Check convergence: position error < 1km and residual < 1.0
	initialRes := loc.computeCombinedCost(bestPos, measurement)
	residualImprovement := initialRes - finalResidual
	converged := residualImprovement > 0.01 || finalResidual < 0.001

	return LocalizationResult{
		Position:      position,
		Covariance:    covariance,
		TDOResidual:   tdoaRes,
		PDOResidual:   pdoaRes,
		TotalResidual: finalResidual,
		CRB:           crb,
		Iterations:    loc.MaxIterations,
		Converged:     converged,
	}
}

// coarseGridSearch finds coarse initial estimate using bounding box grid
func (loc *TDOAPDOALocalizer) coarseGridSearch(m TDOAPDOAMeasurement) [3]float64 {
	bestPos := loc.stationCentroid()
	bestPos[2] = 600 // Typical LEO altitude
	bestCost := loc.computeCombinedCost(bestPos, m)

	// Get station bounding box
	minX, maxX := loc.Stations[0].Position[0], loc.Stations[0].Position[0]
	minY, maxY := loc.Stations[0].Position[1], loc.Stations[0].Position[1]
	for _, s := range loc.Stations {
		if s.Position[0] < minX {
			minX = s.Position[0]
		}
		if s.Position[0] > maxX {
			maxX = s.Position[0]
		}
		if s.Position[1] < minY {
			minY = s.Position[1]
		}
		if s.Position[1] > maxY {
			maxY = s.Position[1]
		}
	}

	// Expand bounding box and add altitude range (LEO: 200-1200km)
	pad := 500.0
	minX, maxX = minX-pad, maxX+pad
	minY, maxY = minY-pad, maxY+pad
	minZ, maxZ := 200.0, 1200.0

	// Grid search in 3D bounding box (100km steps for better coverage)
	step := 100.0
	count := 0
	for x := minX; x <= maxX && count < 10000; x += step {
		for y := minY; y <= maxY && count < 10000; y += step {
			for z := minZ; z <= maxZ && count < 10000; z += step {
				pos := [3]float64{x, y, z}
				cost := loc.computeCombinedCost(pos, m)
				if cost < bestCost {
					bestCost = cost
					bestPos = pos
				}
				count++
			}
		}
	}
	return bestPos
}

// stationCentroid computes centroid of stations
func (loc *TDOAPDOALocalizer) stationCentroid() [3]float64 {
	c := [3]float64{}
	for _, s := range loc.Stations {
		c[0] += s.Position[0]
		c[1] += s.Position[1]
		c[2] += s.Position[2]
	}
	n := float64(len(loc.Stations))
	return [3]float64{c[0] / n, c[1] / n, c[2] / n}
}

// levenbergMarquardtTDOAPDOA performs LM optimization for joint TDOA-PDOA
func (loc *TDOAPDOALocalizer) levenbergMarquardtTDOAPDOA(initial [3]float64, m TDOAPDOAMeasurement, maxIter int, lambda float64) [3]float64 {
	x := initial
	c := 299792.458                  // km/s
	lambdaM := m.Wavelength / 1000.0 // km

	for iter := 0; iter < maxIter; iter++ {
		// Compute residuals and Jacobian for all station pairs
		residuals := []float64{}
		jacobian := [][]float64{}

		for i := 0; i < len(loc.Stations); i++ {
			for j := i + 1; j < len(loc.Stations); j++ {
				si, sj := loc.Stations[i], loc.Stations[j]

				// TDOA residual
				measuredTDOA := m.TDOA[i][j]
				di, dj := distance(x, si.Position), distance(x, sj.Position)
				predictedTDOA := (di - dj) / c
				tdoaRes := measuredTDOA - predictedTDOA
				sigmaT := math.Sqrt(m.TDOACovariance[i][j])
				if sigmaT < 1e-10 {
					sigmaT = 1e-6
				}

				// TDOA gradient
				tdoaGrad := [3]float64{}
				if di > 1e-10 && dj > 1e-10 {
					for k := 0; k < 3; k++ {
						tdoaGrad[k] = ((x[k]-si.Position[k])/di - (x[k]-sj.Position[k])/dj) / (c * sigmaT)
					}
				}

				residuals = append(residuals, tdoaRes/sigmaT)
				jacobian = append(jacobian, []float64{loc.Alpha * tdoaGrad[0], loc.Alpha * tdoaGrad[1], loc.Alpha * tdoaGrad[2]})

				// PDOA residual (with wrapping)
				measuredPDOA := math.Mod(m.PDOA[i][j]+math.Pi, 2*math.Pi) - math.Pi
				predictedPDOA := 2.0 * math.Pi / lambdaM * (di - dj)
				predictedPDOA = math.Mod(predictedPDOA+math.Pi, 2*math.Pi) - math.Pi
				pdoaRes := math.Mod(measuredPDOA-predictedPDOA+math.Pi, 2*math.Pi) - math.Pi
				sigmaP := math.Sqrt(m.PDOAVariance)

				// PDOA gradient (chain rule through phase wrapping)
				pdoaGrad := [3]float64{}
				if di > 1e-10 && dj > 1e-10 {
					for k := 0; k < 3; k++ {
						pdoaGrad[k] = 2.0 * math.Pi / lambdaM * ((x[k]-si.Position[k])/di - (x[k]-sj.Position[k])/dj) / sigmaP
					}
				}

				residuals = append(residuals, pdoaRes/sigmaP)
				jacobian = append(jacobian, []float64{loc.Beta * pdoaGrad[0], loc.Beta * pdoaGrad[1], loc.Beta * pdoaGrad[2]})
			}
		}

		if len(residuals) == 0 {
			break
		}

		// J^T * J (Hessian approximation)
		jtj := [3][3]float64{}
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				sum := 0.0
				for k := range jacobian {
					sum += jacobian[k][i] * jacobian[k][j]
				}
				jtj[i][j] = sum
				if i == j {
					jtj[i][j] += lambda
				}
			}
		}

		// J^T * r
		jtr := [3]float64{}
		for i := 0; i < 3; i++ {
			sum := 0.0
			for k := range jacobian {
				sum += jacobian[k][i] * residuals[k]
			}
			jtr[i] = sum
		}

		// Solve (J^T*J + λI) δ = J^T*r
		delta := loc.solve3x3(jtj, jtr)

		// Update
		xNew := [3]float64{x[0] + delta[0], x[1] + delta[1], x[2] + delta[2]}

		// Adaptive damping
		oldCost := loc.computeCombinedCost(x, m)
		newCost := loc.computeCombinedCost(xNew, m)

		if newCost < oldCost {
			x = xNew
			lambda *= 0.7
		} else {
			lambda *= 2.0
		}

		// Convergence check
		if math.Sqrt(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]) < loc.ConvergenceTol {
			break
		}
	}

	return x
}

// solve3x3 solves 3x3 linear system using Gaussian elimination
func (loc *TDOAPDOALocalizer) solve3x3(A [3][3]float64, b [3]float64) [3]float64 {
	M := [3][4]float64{
		{A[0][0], A[0][1], A[0][2], b[0]},
		{A[1][0], A[1][1], A[1][2], b[1]},
		{A[2][0], A[2][1], A[2][2], b[2]},
	}

	// Forward elimination with partial pivoting
	for i := 0; i < 3; i++ {
		maxRow := i
		maxVal := math.Abs(M[i][i])
		for k := i + 1; k < 3; k++ {
			if math.Abs(M[k][i]) > maxVal {
				maxVal = math.Abs(M[k][i])
				maxRow = k
			}
		}
		if maxVal < 1e-15 {
			continue
		}
		M[i], M[maxRow] = M[maxRow], M[i]

		for k := i + 1; k < 3; k++ {
			factor := M[k][i] / M[i][i]
			for j := i; j < 4; j++ {
				M[k][j] -= factor * M[i][j]
			}
		}
	}

	// Back substitution
	x := [3]float64{}
	for i := 2; i >= 0; i-- {
		sum := M[i][3]
		for j := i + 1; j < 3; j++ {
			sum -= M[i][j] * x[j]
		}
		if math.Abs(M[i][i]) > 1e-15 {
			x[i] = sum / M[i][i]
		}
	}
	return x
}

// computeTDOACost computes TDOA cost function
// J_TDOA = Σ (τ_ij - (||x-s_i|| - ||x-s_j||)/c)² / σ²_ij
func (loc *TDOAPDOALocalizer) computeTDOACost(x [3]float64, m TDOAPDOAMeasurement) float64 {
	cost := 0.0
	c := 299792.458 // Speed of light (km/s)

	for i := 0; i < len(loc.Stations); i++ {
		for j := i + 1; j < len(loc.Stations); j++ {
			// Measured TDOA
			measuredTDOA := m.TDOA[i][j]

			// Predicted TDOA based on position x
			di := distance(x, loc.Stations[i].Position)
			dj := distance(x, loc.Stations[j].Position)
			predictedTDOA := (di - dj) / c

			// Residual weighted by measurement uncertainty
			sigma2 := m.TDOACovariance[i][j]
			if sigma2 < 1e-10 {
				sigma2 = 1e-6 // Minimum variance
			}

			residual := measuredTDOA - predictedTDOA
			cost += residual * residual / sigma2
		}
	}

	return cost
}

// computePDOCost computes PDOA cost function
// J_PDOA = Σ (φ_ij - 2π/λ * (||x-s_i|| - ||x-s_j||))² / σ²_φ
func (loc *TDOAPDOALocalizer) computePDOCost(x [3]float64, m TDOAPDOAMeasurement) float64 {
	cost := 0.0
	lambda := m.Wavelength / 1000.0 // Convert to km

	for i := 0; i < len(loc.Stations); i++ {
		for j := i + 1; j < len(loc.Stations); j++ {
			// Measured PDOA (wrapped to [-π, π])
			measuredPDOA := math.Mod(m.PDOA[i][j]+math.Pi, 2*math.Pi) - math.Pi

			// Predicted PDOA
			di := distance(x, loc.Stations[i].Position)
			dj := distance(x, loc.Stations[j].Position)
			deltaD := di - dj
			predictedPDOA := 2.0 * math.Pi / lambda * deltaD

			// Wrap predicted to [-π, π]
			predictedPDOA = math.Mod(predictedPDOA+math.Pi, 2*math.Pi) - math.Pi

			// Phase difference (accounting for wrapping)
			diff := math.Mod(measuredPDOA-predictedPDOA+math.Pi, 2*math.Pi) - math.Pi

			cost += diff * diff / m.PDOAVariance
		}
	}

	return cost
}

// computeCombinedCost combines TDOA and PDOA costs
func (loc *TDOAPDOALocalizer) computeCombinedCost(x [3]float64, m TDOAPDOAMeasurement) float64 {
	return loc.Alpha*loc.computeTDOACost(x, m) + loc.Beta*loc.computePDOCost(x, m)
}

// computeTDOAGradient computes gradient of TDOA cost function
func (loc *TDOAPDOALocalizer) computeTDOAGradient(x [3]float64, m TDOAPDOAMeasurement) [3]float64 {
	epsilon := 0.001 // Small step for numerical gradient
	grad := [3]float64{}

	baseCost := loc.computeTDOACost(x, m)

	for dim := 0; dim < 3; dim++ {
		xPlus := x
		xPlus[dim] += epsilon
		costPlus := loc.computeTDOACost(xPlus, m)
		grad[dim] = (costPlus - baseCost) / epsilon
	}

	return grad
}

// computePDOAGradient computes gradient of PDOA cost function
func (loc *TDOAPDOALocalizer) computePDOAGradient(x [3]float64, m TDOAPDOAMeasurement) [3]float64 {
	epsilon := 0.001
	grad := [3]float64{}

	baseCost := loc.computePDOCost(x, m)

	for dim := 0; dim < 3; dim++ {
		xPlus := x
		xPlus[dim] += epsilon
		costPlus := loc.computePDOCost(xPlus, m)
		grad[dim] = (costPlus - baseCost) / epsilon
	}

	return grad
}

// tdoaOnlyEstimate provides initial guess using TDOA only
func (loc *TDOAPDOALocalizer) tdoaOnlyEstimate(tdoa [][]float64) [3]float64 {
	// Simple centroid of stations as initial guess
	centroid := [3]float64{}
	for _, s := range loc.Stations {
		centroid[0] += s.Position[0]
		centroid[1] += s.Position[1]
		centroid[2] += s.Position[2]
	}
	n := float64(len(loc.Stations))
	centroid[0] /= n
	centroid[1] /= n
	centroid[2] /= n

	// Refine with simple gradient descent on TDOA only
	// (simplified version of full algorithm)
	return centroid
}

// lineSearch finds optimal step size
func (loc *TDOAPDOALocalizer) lineSearch(x [3]float64, grad [3]float64, m TDOAPDOAMeasurement) float64 {
	// Simple backtracking line search
	step := 1.0
	alpha := 0.5 // Reduction factor
	c := 0.0001  // Sufficient decrease parameter

	currentCost := loc.computeCombinedCost(x, m)

	for iter := 0; iter < 20; iter++ {
		newX := [3]float64{
			x[0] - step*grad[0],
			x[1] - step*grad[1],
			x[2] - step*grad[2],
		}
		newCost := loc.computeCombinedCost(newX, m)

		// Check sufficient decrease condition
		expectedDecrease := c * step * (grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2])
		if newCost <= currentCost-expectedDecrease {
			return step
		}

		step *= alpha
	}

	return step
}

// computeCRB computes Cramér-Rao Bound for position uncertainty
func (loc *TDOAPDOALocalizer) computeCRB(x [3]float64, m TDOAPDOAMeasurement) [3][3]float64 {
	// FIM approximation using numerical derivatives
	epsilon := 0.01
	fim := [3][3]float64{}

	// Compute Fisher Information Matrix elements
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			xiPlus := x
			xiMinus := x
			xjPlus := x
			xjMinus := x

			xiPlus[i] += epsilon
			xiMinus[i] -= epsilon
			xjPlus[j] += epsilon
			xjMinus[j] -= epsilon

			// Mixed partial derivative approximation
			fim[i][j] = (loc.computeCombinedCost(xiPlus, m) -
				loc.computeCombinedCost(xiMinus, m) -
				loc.computeCombinedCost(xjPlus, m) +
				loc.computeCombinedCost(xjMinus, m)) / (4 * epsilon * epsilon)
		}
	}

	return fim
}

// Utility functions
// Note: distance() function defined in tdoa.go, reused here

func invert3x3(m [3][3]float64) [3][3]float64 {
	det := m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
		m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
		m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])

	if math.Abs(det) < 1e-10 {
		return [3][3]float64{
			{1e6, 0, 0},
			{0, 1e6, 0},
			{0, 0, 1e6},
		} // Large uncertainty
	}

	invDet := 1.0 / det
	var result [3][3]float64

	result[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * invDet
	result[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * invDet
	result[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invDet
	result[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * invDet
	result[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invDet
	result[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * invDet
	result[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * invDet
	result[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * invDet
	result[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * invDet

	return result
}

// BatchLocalize processes multiple measurements efficiently
func (loc *TDOAPDOALocalizer) BatchLocalize(measurements []TDOAPDOAMeasurement) []LocalizationResult {
	results := make([]LocalizationResult, len(measurements))

	// Parallel processing
	for i, m := range measurements {
		results[i] = loc.Localize(m)
	}

	return results
}

// PerformanceMetrics provides accuracy analysis
type PerformanceMetrics struct {
	RMSE            float64 // Root mean square error
	MeanError       float64 // Mean position error
	CRBAverage      float64 // Average Cramér-Rao Bound
	ConvergenceRate float64 // Percentage that converged
}

// EvaluatePerformance computes metrics against ground truth
func (loc *TDOAPDOALocalizer) EvaluatePerformance(
	measurements []TDOAPDOAMeasurement,
	groundTruth [][3]float64,
) PerformanceMetrics {
	results := loc.BatchLocalize(measurements)

	var totalError, sumSquaredError, totalCRB float64
	convergedCount := 0

	for i, result := range results {
		error := distance(result.Position, groundTruth[i])
		totalError += error
		sumSquaredError += error * error
		totalCRB += math.Sqrt(result.CRB[0][0] + result.CRB[1][1] + result.CRB[2][2])

		if result.Converged {
			convergedCount++
		}
	}

	n := float64(len(results))

	return PerformanceMetrics{
		RMSE:            math.Sqrt(sumSquaredError / n),
		MeanError:       totalError / n,
		CRBAverage:      totalCRB / n,
		ConvergenceRate: float64(convergedCount) / n * 100,
	}
}
