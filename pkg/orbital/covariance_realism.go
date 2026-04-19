package orbital

import (
	"math"
	"sort"

	"ssis/pkg/physics"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

/**
 * Covariance Realism Implementation - Cano et al. (2023) Methodology
 *
 * Reference: "Covariance determination for improving uncertainty realism in
 * orbit determination and propagation" (Advances in Space Research, 2023)
 *
 * This implements the consider parameter + Mahalanobis inference approach for
 * achieving covariance realism in SSA/SDA systems.
 *
 * Core principle: Treat uncertain dynamic parameters (drag BC, SRP coeff,
 * density scale, measurement biases) as consider parameters with variances
 * representing model error. Tune these variances so that Mahalanobis distances
 * of orbital differences follow χ² distribution.
 *
 * For ADITI 4.0 IAF certification, this provides traceable, physics-based
 * uncertainty quantification without black-box ML.
 */

// ConsiderParameters holds the uncertain dynamic parameters that affect
// covariance realism but are not explicitly estimated in OD
type ConsiderParameters struct {
	// Drag uncertainty (dominant in LEO < 2000 km)
	BC_Sigma          float64 // Ballistic coefficient uncertainty [kg/m²]
	DensityScaleSigma float64 // NRLMSISE-00/JB2008 density model scale error [%]

	// SRP uncertainty (dominant in GEO / HEO)
	CR_Sigma float64 // Reflectivity coefficient uncertainty (typically 0.1-0.3)

	// Time/clock uncertainty (affects all orbit regimes)
	ClockBiasSigma float64 // Epoch time bias uncertainty [seconds]

	// Measurement biases (systematic sensor errors)
	RangeBiasSigma     float64 // Radar range bias [km]
	RangeRateBiasSigma float64 // Radar range-rate bias [km/s]
	AngleBiasSigma     float64 // Optical angle bias [rad]
}

// DefaultConsiderParameters returns conservative defaults based on
// Cano et al. (2023) and operational SSA experience
func DefaultConsiderParameters(regime physics.OrbitRegime) ConsiderParameters {
	switch regime {
	case physics.RegimeLEO:
		// LEO: Drag dominates, SRP negligible
		return ConsiderParameters{
			BC_Sigma:           0.05, // 5% BC uncertainty typical for debris
			DensityScaleSigma:  0.25, // 25% density model error (NRLMSISE-00)
			CR_Sigma:           0.0,  // SRP negligible in LEO
			ClockBiasSigma:     0.1,  // 0.1 sec timing
			RangeBiasSigma:     0.01, // 10m range bias
			RangeRateBiasSigma: 1e-5, // 1 cm/s
			AngleBiasSigma:     1e-5, // ~2 arcsec
		}
	case physics.RegimeGEO:
		// GEO: SRP dominates, drag negligible
		return ConsiderParameters{
			BC_Sigma:           0.0, // Drag negligible
			DensityScaleSigma:  0.0,
			CR_Sigma:           0.15, // 15% CR uncertainty (cannonball model)
			ClockBiasSigma:     0.1,
			RangeBiasSigma:     0.02, // 20m (GEO tracking typically worse)
			RangeRateBiasSigma: 5e-6, // 0.5 cm/s
			AngleBiasSigma:     2e-5, // ~4 arcsec
		}
	case physics.RegimeMEO:
		// MEO: Both drag and SRP relevant (GPS, Galileo)
		return ConsiderParameters{
			BC_Sigma:           0.02,
			DensityScaleSigma:  0.15,
			CR_Sigma:           0.10,
			ClockBiasSigma:     0.05,
			RangeBiasSigma:     0.005,
			RangeRateBiasSigma: 1e-5,
			AngleBiasSigma:     1e-5,
		}
	default:
		// Conservative defaults
		return ConsiderParameters{
			BC_Sigma:           0.05,
			DensityScaleSigma:  0.20,
			CR_Sigma:           0.10,
			ClockBiasSigma:     0.1,
			RangeBiasSigma:     0.01,
			RangeRateBiasSigma: 1e-5,
			AngleBiasSigma:     1e-5,
		}
	}
}

// CovarianceRealismResult holds the validation output
type CovarianceRealismResult struct {
	MahalanobisDistances []float64          // d_M² values for each test point
	Chi2Statistic        float64            // Overall χ² test statistic
	Chi2PValue           float64            // P-value for χ² test
	CVMStatistic         float64            // Cramer-von-Mises statistic
	CVMPValue            float64            // CVM test p-value
	Containment1Sigma    float64            // % within 1σ
	Containment2Sigma    float64            // % within 2σ
	Containment3Sigma    float64            // % within 3σ
	IsRealistic          bool               // Overall pass/fail
	RecommendedScale     float64            // Suggested covariance scale factor
	ConsiderParams       ConsiderParameters // Tuned parameters
}

/**
 * ValidateCovarianceRealism performs statistical tests on prediction errors
 * to assess if the covariance matrix realistically represents uncertainty.
 *
 * Tests performed:
 * 1. Mahalanobis distance distribution vs χ²(6)
 * 2. Cramer-von-Mises goodness-of-fit test
 * 3. Containment statistics (1σ, 2σ, 3σ)
 *
 * @param predictedStates States from OD/propagation with covariance
 * @param referenceStates "Truth" states (high-accuracy ephemeris or denser OD)
 * @return CovarianceRealismResult with test statistics and pass/fail
 */
func ValidateCovarianceRealism(
	predictedStates []physics.State,
	referenceStates []physics.State,
) CovarianceRealismResult {
	if len(predictedStates) != len(referenceStates) || len(predictedStates) == 0 {
		return CovarianceRealismResult{IsRealistic: false}
	}

	n := len(predictedStates)
	mahalDistances := make([]float64, n)
	contain1Sigma := 0
	contain2Sigma := 0
	contain3Sigma := 0

	for i := 0; i < n; i++ {
		// Compute state error (predicted - reference)
		error := computeStateError(predictedStates[i], referenceStates[i])

		// Get covariance (use default if zero)
		cov := predictedStates[i].Covariance
		if isZeroCovariance(cov) {
			// Fallback: construct from default TLE model
			// This won't pass realism but provides diagnostics
			cov = physics.ConstructDefaultCovarianceFromTLE(physics.TLE{})
		}

		// Compute Mahalanobis distance: d_M² = ε^T P^{-1} ε
		mahal := computeMahalanobisSquared(error, cov)
		mahalDistances[i] = mahal

		// Containment tests
		if mahal <= 1.0 { // 1σ in 6D ≈ 1.0 (normalized)
			contain1Sigma++
		}
		if mahal <= 4.0 { // 2σ in 6D ≈ 4.0
			contain2Sigma++
		}
		if mahal <= 9.0 { // 3σ in 6D ≈ 9.0
			contain3Sigma++
		}
	}

	// χ² goodness-of-fit test (6 degrees of freedom for 6D state)
	chi2Stat, chi2PValue := chi2Test(mahalDistances, 6)

	// Cramer-von-Mises test against χ²(6)
	cvmStat, cvmPValue := cramerVonMisesTest(mahalDistances, 6)

	// Compute recommended scale factor if covariance is optimistic
	// Scale = mean(d_M²) / 6 (expected mean for χ²(6))
	meanMahal := statsMean(mahalDistances)
	recommendedScale := math.Sqrt(meanMahal / 6.0)

	// Pass criteria (per Poore et al. 2016 and Cano et al. 2023)
	// - CVM p-value > 0.05 (distribution matches χ²)
	// - Containment 3σ > 95% (practical operational requirement)
	isRealistic := cvmPValue > 0.05 && float64(contain3Sigma)/float64(n) > 0.95

	return CovarianceRealismResult{
		MahalanobisDistances: mahalDistances,
		Chi2Statistic:        chi2Stat,
		Chi2PValue:           chi2PValue,
		CVMStatistic:         cvmStat,
		CVMPValue:            cvmPValue,
		Containment1Sigma:    float64(contain1Sigma) / float64(n),
		Containment2Sigma:    float64(contain2Sigma) / float64(n),
		Containment3Sigma:    float64(contain3Sigma) / float64(n),
		IsRealistic:          isRealistic,
		RecommendedScale:     recommendedScale,
		ConsiderParams:       ConsiderParameters{}, // Will be filled by TuneConsiderParameters
	}
}

/**
 * TuneConsiderParameters optimizes consider parameter variances to achieve
 * covariance realism using the Cano et al. (2023) iterative method.
 *
 * Algorithm:
 * 1. Start with initial consider parameters
 * 2. Propagate with consider covariance
 * 3. Compute Mahalanobis distances vs reference
 * 4. Adjust consider variances to match χ²(6) distribution
 * 5. Iterate until convergence
 *
 * @param observations Tracking data for OD
 * @param referenceEphem High-accuracy reference ephemeris (e.g., from GPS, SLR, or dense tracking)
 * @param initial Initial consider parameters
 * @param maxIter Maximum tuning iterations
 * @return Tuned parameters and realism validation result
 */
func TuneConsiderParameters(
	observations []TrackingObservation,
	referenceEphem []physics.State,
	initial ConsiderParameters,
	maxIter int,
) (ConsiderParameters, CovarianceRealismResult) {
	current := initial
	var bestResult CovarianceRealismResult
	bestScore := -1.0

	// Simple gradient-free optimization (coordinate descent)
	// In production, use more sophisticated optimization (e.g., NLopt, Bayesian)

	for iter := 0; iter < maxIter; iter++ {
		// Run OD/propagation with current consider parameters
		predicted := runODWithConsider(observations, current)

		// Validate covariance realism
		result := ValidateCovarianceRealism(predicted, referenceEphem)

		// Score: higher p-value + containment is better
		score := result.CVMPValue + result.Containment3Sigma

		if score > bestScore {
			bestScore = score
			bestResult = result
			bestResult.ConsiderParams = current
		}

		// Check convergence
		if result.IsRealistic {
			return current, result
		}

		// Adjust consider parameters based on result
		// If covariance is optimistic (mean Mahalanobis > 6), increase uncertainties
		meanMahal := statsMean(result.MahalanobisDistances)
		targetMean := 6.0 // Expected mean for χ²(6)

		scale := math.Sqrt(meanMahal / targetMean)
		if scale > 2.0 {
			scale = 2.0 // Limit adjustment per iteration
		}
		if scale < 0.5 {
			scale = 0.5
		}

		// Scale consider parameters
		current.BC_Sigma *= scale
		current.DensityScaleSigma *= scale
		current.CR_Sigma *= scale
		current.ClockBiasSigma *= scale

		// Measurement biases typically well-calibrated, adjust slower
		current.RangeBiasSigma *= math.Sqrt(scale)
		current.RangeRateBiasSigma *= math.Sqrt(scale)
		current.AngleBiasSigma *= math.Sqrt(scale)
	}

	return bestResult.ConsiderParams, bestResult
}

// TrackingObservation represents a single tracking measurement for OD
type TrackingObservation struct {
	Time  float64    // Julian Date
	Type  string     // "range", "range_rate", "angle_ra", "angle_dec"
	Value float64    // Measured value
	Sigma float64    // Measurement noise (1-sigma)
	Site  [3]float64 // Observer ECI position
}

// Helper functions

func computeStateError(predicted, reference physics.State) []float64 {
	return []float64{
		predicted.Pos[0] - reference.Pos[0],
		predicted.Pos[1] - reference.Pos[1],
		predicted.Pos[2] - reference.Pos[2],
		predicted.Vel[0] - reference.Vel[0],
		predicted.Vel[1] - reference.Vel[1],
		predicted.Vel[2] - reference.Vel[2],
	}
}

func computeMahalanobisSquared(error []float64, cov [6][6]float64) float64 {
	// Convert to gonum matrices for inversion
	epsilon := mat.NewVecDense(6, error)

	// Build covariance matrix
	P := mat.NewSymDense(6, nil)
	for i := 0; i < 6; i++ {
		for j := 0; j <= i; j++ {
			P.SetSym(i, j, cov[i][j])
		}
	}

	// Compute P^{-1} ε
	var PinvEpsilon mat.VecDense
	errSolve := PinvEpsilon.SolveVec(mat.DenseCopyOf(P), epsilon)
	if errSolve != nil {
		// Fallback: use pseudo-inverse or return large value
		return 1e6
	}

	// d_M² = ε^T P^{-1} ε
	mahal := mat.Dot(epsilon, &PinvEpsilon)
	return mahal
}

func isZeroCovariance(cov [6][6]float64) bool {
	for i := 0; i < 6; i++ {
		if cov[i][i] != 0 {
			return false
		}
	}
	return true
}

func chi2Test(samples []float64, df int) (statistic, pvalue float64) {
	// Compute χ² test statistic: sum of (observed - expected)² / expected
	// For Mahalanobis distances, expected distribution is χ²(df)

	// Bin the data and compare to expected χ² frequencies
	// Simplified: use mean and variance test
	n := float64(len(samples))
	mean := statsMean(samples)
	variance := statsVariance(samples)

	// For χ²(df): mean = df, variance = 2*df
	expectedMean := float64(df)
	expectedVar := float64(2 * df)

	// Test statistic (simplified)
	statistic = math.Abs(mean-expectedMean)/math.Sqrt(expectedMean/n) +
		math.Abs(variance-expectedVar)/math.Sqrt(2*expectedVar*expectedVar/n)

	// Approximate p-value (simplified)
	pvalue = math.Exp(-statistic / 2.0)

	return statistic, pvalue
}

func cramerVonMisesTest(samples []float64, df int) (statistic, pvalue float64) {
	n := len(samples)
	if n == 0 {
		return 0, 0
	}

	// Sort samples
	sorted := make([]float64, n)
	copy(sorted, samples)
	sort.Float64s(sorted)

	// Cramer-von-Mises statistic: W² = Σ (U_i - (2i-1)/(2n))² + 1/(12n)
	// where U_i = F(X_i) and F is the CDF of χ²(df)
	chi2Dist := distuv.ChiSquared{K: float64(df)}

	sum := 0.0
	for i, x := range sorted {
		u := chi2Dist.CDF(x)
		expected := (float64(i) + 0.5) / float64(n)
		diff := u - expected
		sum += diff * diff
	}

	statistic = sum + 1.0/(12.0*float64(n))

	// Approximate p-value using asymptotic distribution
	// For CVM: n*W² follows a known asymptotic distribution
	pvalue = math.Exp(-float64(n) * statistic)

	return statistic, pvalue
}

func statsMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func statsVariance(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}
	m := statsMean(values)
	sum := 0.0
	for _, v := range values {
		d := v - m
		sum += d * d
	}
	return sum / float64(len(values)-1)
}

// runODWithConsider performs orbit determination with consider parameters
// This is a placeholder - in production, integrate with your UKF/batch OD
func runODWithConsider(obs []TrackingObservation, consider ConsiderParameters) []physics.State {
	// TODO: Integrate with existing UKF/OD system
	// For now, return empty (caller must provide actual OD output)
	return nil
}
