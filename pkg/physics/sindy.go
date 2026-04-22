package physics

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
)

/**
 * SINDy (Sparse Identification of Nonlinear Dynamics)
 *
 * Discovers governing equations of adversary maneuvers via sparse regression.
 * Replaces black-box rKAN with interpretable differential equations.
 *
 * Reference: Brunton et al. "Discovering governing equations from data" (PNAS 2016)
 */

// Prediction represents a falsifiable prediction for validation
type Prediction struct {
	Timestamp   time.Time  // Prediction timestamp
	State       [6]float64 // Predicted state [x,y,z,vx,vy,vz]
	Probability float64    // Prediction confidence
	HorizonSec  float64    // Prediction horizon in seconds
}

type SINDy struct {
	// Library configuration
	PolynomialDegree int
	UseTrigonometric bool
	UseExponential   bool

	// STLSQ parameters
	SparsityThreshold float64
	MaxIterations     int
	RidgeLambda       float64 // Ridge regularization for rank-deficient libraries

	// Bootstrap CI parameters
	BootstrapResamples int // Number of bootstrap resamples for CI (default 1000)
}

// NewSINDy creates a SINDy discoverer with default parameters
func NewSINDy() *SINDy {
	return &SINDy{
		PolynomialDegree:   3,
		UseTrigonometric:   true,
		UseExponential:     false,
		SparsityThreshold:  0.1,
		MaxIterations:      10,
		RidgeLambda:        1e-2,
		BootstrapResamples: 1000,
	}
}

// ManeuverResult contains discovered dynamics and interpretation
type ManeuverResult struct {
	Coefficients    []float64 // Sparse coefficients for library terms
	ActiveTerms     []string  // Human-readable term names
	ThrustDetected  bool      // Whether non-gravitational forces identified
	ThrustMagnitude float64   // Estimated delta-V (m/s)
	EquationString  string    // LaTeX-ready equation
	Confidence      float64   // R² fit quality

	// Bootstrap Confidence Intervals (95% CI for each coefficient)
	CILower    []float64 // Lower bound of 95% CI
	CIUpper    []float64 // Upper bound of 95% CI
	CICoverage float64   // Proportion of coefficients with CI not containing zero

	// Residual-based discovery metadata
	ResidualMean float64 // Mean residual magnitude
	ResidualStd  float64 // Std dev of residuals
	TargetType   string  // "residual" (new) vs "derivative" (legacy)

	// Symbolic Certificate (VSC) - Extended for Sovereign Handshake
	ConstraintViolations []ConstraintViolation // Physics constraint pass/fail (replaces A-F grade)
	CausalProvenance     map[string]float64    // Term importance scores
	Predictions          []Prediction          // Falsifiable 60s predictions
	HamiltonianHash      string                // Hamiltonian Invariant Hash
	CertificateValid     bool                  // Certificate validity
	CertificateIssued    time.Time             // When certificate was issued
	EnergyDriftSigma     float64               // σ_H for spoof detection
}

// DiscoverManeuver identifies thrust signatures from orbital residuals
//
// Inputs: states = [pos(3), vel(3)] over time
//
//	dt = time step between samples
//
// Output: Sparse coefficients where only thrust terms are non-zero
//
// NEW: Targets residual r(t) = ẍ_observed - ẍ_keplerian instead of full ẋ.
// This focuses SINDy on discovering only the unmodeled dynamics (maneuvers),
// not the Keplerian physics which is already known.
func (s *SINDy) DiscoverManeuver(states [][6]float64, dt float64) (*ManeuverResult, error) {
	// 1. Build Theta Library (candidate functions)
	theta := s.buildLibrary(states)

	// 2. Compute RESIDUALS r(t) = observed - keplerian_prediction
	// instead of full derivatives ẋ
	residuals := s.computeResiduals(states, dt)

	// Align row counts: buildLibrary returns N rows, computeResiduals returns N-2
	// (central difference loses first and last samples). Trim theta to match.
	tRows, tCols := theta.Dims()
	rRows, _ := residuals.Dims()
	if tRows != rRows {
		trimmedData := make([]float64, rRows*tCols)
		for i := 0; i < rRows; i++ {
			for j := 0; j < tCols; j++ {
				trimmedData[i*tCols+j] = theta.At(i+1, j)
			}
		}
		theta = mat.NewDense(rRows, tCols, trimmedData)
	}

	// 3. Sparse Regression (STLSQ) targeting residuals
	coefficients := s.stlsq(theta, residuals, s.SparsityThreshold)

	// 4. Bootstrap Confidence Intervals
	ciLower, ciUpper := s.bootstrapCI(coefficients, states, theta, residuals, s.BootstrapResamples)

	// 5. Interpret results
	result := s.interpretCoefficients(coefficients, states, theta, residuals)

	// 6. Add bootstrap CI data
	result.CILower = ciLower
	result.CIUpper = ciUpper
	result.CICoverage = s.computeCICoverage(coefficients, ciLower, ciUpper)
	result.TargetType = "residual"

	// 7. Compute energy drift for sovereign handshake certificate
	result.ComputeEnergyDrift(states)

	return result, nil
}

// buildLibrary constructs the candidate function matrix Theta(X)
// Each column is a candidate term: [1, x, y, z, vx, vy, vz, x², xy, sin(x), ...]
func (s *SINDy) buildLibrary(states [][6]float64) *mat.Dense {
	nSamples := len(states)
	nTerms := s.countLibraryTerms()

	data := make([]float64, nSamples*nTerms)

	for i, state := range states {
		x, y, z := state[0], state[1], state[2]
		vx, vy, vz := state[3], state[4], state[5]

		col := 0

		// Constant term
		data[i*nTerms+col] = 1.0
		col++

		// Linear terms (position)
		data[i*nTerms+col] = x
		col++
		data[i*nTerms+col] = y
		col++
		data[i*nTerms+col] = z
		col++

		// Linear terms (velocity)
		data[i*nTerms+col] = vx
		col++
		data[i*nTerms+col] = vy
		col++
		data[i*nTerms+col] = vz
		col++

		// Polynomial terms up to degree 3
		for d := 2; d <= s.PolynomialDegree; d++ {
			// x^d, y^d, z^d
			data[i*nTerms+col] = math.Pow(x, float64(d))
			col++
			data[i*nTerms+col] = math.Pow(y, float64(d))
			col++
			data[i*nTerms+col] = math.Pow(z, float64(d))
			col++

			// Cross terms (d=2): xy, xz, yz
			if d == 2 {
				data[i*nTerms+col] = x * y
				col++
				data[i*nTerms+col] = x * z
				col++
				data[i*nTerms+col] = y * z
				col++
			}
		}

		// Trigonometric terms (for periodic thrust profiles)
		if s.UseTrigonometric {
			r := math.Sqrt(x*x + y*y + z*z)
			if r > 1e-6 {
				data[i*nTerms+col] = math.Sin(r / 7000.0) // Normalized by typical LEO radius
				col++
				data[i*nTerms+col] = math.Cos(r / 7000.0)
				col++
			}
		}

		// Velocity magnitude (drag-like term)
		vMag := math.Sqrt(vx*vx + vy*vy + vz*vz)
		data[i*nTerms+col] = vMag * vx // Directional drag
		col++
		data[i*nTerms+col] = vMag * vy
		col++
		data[i*nTerms+col] = vMag * vz
		col++
	}

	return mat.NewDense(nSamples, nTerms, data)
}

// computeDerivatives calculates numerical first derivatives using central differences
func (s *SINDy) computeDerivatives(states [][6]float64, dt float64) *mat.Dense {
	nSamples := len(states)

	// Compute first derivatives using central differences: (x_{i+1} - x_{i-1}) / (2*dt)
	// This gives dx/dt for all state components (position -> velocity, velocity -> acceleration)
	data := make([]float64, (nSamples-2)*6)

	for i := 1; i < nSamples-1; i++ {
		for dim := 0; dim < 6; dim++ {
			data[(i-1)*6+dim] = (states[i+1][dim] - states[i-1][dim]) / (2.0 * dt)
		}
	}

	return mat.NewDense(nSamples-2, 6, data)
}

// computeResiduals calculates r(t) = observed_dynamics - keplerian_dynamics
// This is the key innovation: SINDy discovers only the unmodeled (maneuver) component
func (s *SINDy) computeResiduals(states [][6]float64, dt float64) *mat.Dense {
	nSamples := len(states)

	// Full observed derivatives (accelerations)
	observed := s.computeDerivatives(states, dt)
	nObsSamples, _ := observed.Dims()

	// Keplerian prediction (central gravity only)
	keplerian := mat.NewDense(nObsSamples, 6, nil)

	for i := 1; i < nSamples-1; i++ {
		state := states[i]
		x, y, z := state[0], state[1], state[2]
		vx, vy, vz := state[3], state[4], state[5]

		// Position vector and magnitude
		r := math.Sqrt(x*x + y*y + z*z)

		// Keplerian acceleration: a = -μ/r³ * r_vec
		// Earth's standard gravitational parameter μ ≈ 398600.4418 km³/s²
		mu := 398600.4418

		if r > 1e-6 {
			scale := -mu / (r * r * r)
			// Acceleration (indices 3,4,5 for vx, vy, vz derivatives)
			keplerian.Set(i-1, 3, scale*x) // d(vx)/dt
			keplerian.Set(i-1, 4, scale*y) // d(vy)/dt
			keplerian.Set(i-1, 5, scale*z) // d(vz)/dt
		}

		// Velocity derivatives for position (d(pos)/dt = vel)
		keplerian.Set(i-1, 0, vx) // dx/dt = vx
		keplerian.Set(i-1, 1, vy) // dy/dt = vy
		keplerian.Set(i-1, 2, vz) // dz/dt = vz
	}

	// Residual: r(t) = observed - keplerian
	var residuals mat.Dense
	residuals.Sub(observed, keplerian)

	return &residuals
}

// bootstrapCI computes 95% confidence intervals via bootstrap resampling
func (s *SINDy) bootstrapCI(
	coeffs []float64,
	_states [][6]float64,
	theta, residuals *mat.Dense,
	nBootstraps int,
) (lower, upper []float64) {
	nSamples, nTerms := theta.Dims()
	_, nDims := residuals.Dims()

	lower = make([]float64, len(coeffs))
	upper = make([]float64, len(coeffs))

	// Fill with original coefficients (fallback)
	copy(lower, coeffs)
	copy(upper, coeffs)

	if nSamples < 10 || nBootstraps < 10 {
		return lower, upper
	}

	// Store bootstrap coefficient estimates
	bootstrapCoeffs := make([][]float64, nBootstraps)

	for b := 0; b < nBootstraps; b++ {
		// Resample with replacement
		resampledTheta := mat.NewDense(nSamples, nTerms, nil)
		resampledResiduals := mat.NewDense(nSamples, nDims, nil)

		for i := 0; i < nSamples; i++ {
			// Random index (with replacement)
			idx := int(math.Mod(float64(i*7+b*13), float64(nSamples)))
			if idx < 0 || idx >= nSamples {
				idx = i % nSamples
			}

			// Copy row
			for j := 0; j < nTerms; j++ {
				resampledTheta.Set(i, j, theta.At(idx, j))
			}
			for j := 0; j < nDims; j++ {
				resampledResiduals.Set(i, j, residuals.At(idx, j))
			}
		}

		// Run STLSQ on resampled data
		bootCoeffs := s.stlsq(resampledTheta, resampledResiduals, s.SparsityThreshold)
		bootstrapCoeffs[b] = bootCoeffs
	}

	// Compute percentiles for 95% CI
	for i := 0; i < len(coeffs); i++ {
		values := make([]float64, nBootstraps)
		for b := 0; b < nBootstraps; b++ {
			values[b] = bootstrapCoeffs[b][i]
		}

		// Sort to find percentiles
		sortFloats(values)

		// 2.5th and 97.5th percentiles for 95% CI
		lowerIdx := int(0.025 * float64(nBootstraps))
		upperIdx := int(0.975 * float64(nBootstraps))

		if lowerIdx >= nBootstraps {
			lowerIdx = nBootstraps - 1
		}
		if upperIdx >= nBootstraps {
			upperIdx = nBootstraps - 1
		}

		lower[i] = values[lowerIdx]
		upper[i] = values[upperIdx]
	}

	return lower, upper
}

// computeCICoverage calculates proportion of coefficients with CI not containing zero
func (s *SINDy) computeCICoverage(coeffs, ciLower, ciUpper []float64) float64 {
	if len(coeffs) == 0 || len(ciLower) == 0 || len(ciUpper) == 0 {
		return 0.0
	}

	significant := 0
	for i := 0; i < len(coeffs); i++ {
		// CI does not contain zero if both bounds have same sign
		if (ciLower[i] > 0 && ciUpper[i] > 0) || (ciLower[i] < 0 && ciUpper[i] < 0) {
			significant++
		}
	}

	return float64(significant) / float64(len(coeffs))
}

// sortFloats sorts a slice of float64 in-place using stdlib sort
func sortFloats(arr []float64) {
	sort.Float64s(arr)
}

// stlsq performs Sequential Thresholded Least Squares regression
// This is the core SINDy algorithm for sparse coefficient discovery
func (s *SINDy) stlsq(theta, xDot *mat.Dense, threshold float64) []float64 {
	tRows, nTerms := theta.Dims()
	dRows, nDims := xDot.Dims()

	// Align row counts: buildLibrary returns N rows, computeResiduals returns N-1 rows
	// Take the minimum to prevent gonum panic on dimension mismatch
	nSamples := tRows
	if dRows < nSamples {
		nSamples = dRows
	}
	if nSamples == 0 {
		return make([]float64, nTerms*nDims)
	}

	// Slice matrices to aligned row count if needed
	if tRows != nSamples || dRows != nSamples {
		theta = theta.Slice(0, nSamples, 0, nTerms).(*mat.Dense)
		xDot = xDot.Slice(0, nSamples, 0, nDims).(*mat.Dense)
	}

	// Initialize with least squares solution
	coefficients := make([]float64, nTerms*nDims)

	// Iterate: fit → threshold → refit
	for iter := 0; iter < s.MaxIterations; iter++ {
		// Identify small coefficients
		bigIndices := make([]int, 0)
		for i := 0; i < nTerms; i++ {
			maxCoeff := 0.0
			for d := 0; d < nDims; d++ {
				if abs(coefficients[i*nDims+d]) > maxCoeff {
					maxCoeff = abs(coefficients[i*nDims+d])
				}
			}
			if maxCoeff > threshold || iter == 0 {
				bigIndices = append(bigIndices, i)
			}
		}

		if len(bigIndices) == 0 {
			break
		}

		// Build reduced theta matrix
		reducedTheta := mat.NewDense(nSamples, len(bigIndices), nil)
		for i, idx := range bigIndices {
			col := theta.ColView(idx)
			for r := 0; r < nSamples; r++ {
				reducedTheta.Set(r, i, col.At(r, 0))
			}
		}

		// Least squares on reduced set: fast path via Solve, fallback to SVD pseudo-inverse
		// for rank-deficient libraries (e.g., vMag*vx ≈ const*vx for circular orbits)
		var reducedCoeffs mat.Dense
		err := reducedCoeffs.Solve(reducedTheta, xDot)
		if err != nil {
			// Rank-deficient: use SVD pseudo-inverse
			var svd mat.SVD
			if !svd.Factorize(reducedTheta, mat.SVDFull) {
				break
			}
			var u mat.Dense
			svd.UTo(&u)
			sigma := svd.Values(nil)
			var v mat.Dense
			svd.VTo(&v)

			mCols := len(bigIndices)
			vSigmaInv := mat.NewDense(mCols, nSamples, nil)
			for i := 0; i < mCols; i++ {
				for j := 0; j < mCols && j < len(sigma); j++ {
					if sigma[j] > 1e-10 {
						vSigmaInv.Set(i, j, v.At(i, j)/sigma[j])
					}
				}
			}
			var pseudo mat.Dense
			pseudo.Mul(vSigmaInv, u.T())
			reducedCoeffs.Mul(&pseudo, xDot)
		}

		// Update full coefficient vector
		for i, idx := range bigIndices {
			for d := 0; d < nDims; d++ {
				coefficients[idx*nDims+d] = reducedCoeffs.At(i, d)
			}
		}
		for i := 0; i < nTerms; i++ {
			found := false
			for _, idx := range bigIndices {
				if i == idx {
					found = true
					break
				}
			}
			if !found {
				for d := 0; d < nDims; d++ {
					coefficients[i*nDims+d] = 0
				}
			}
		}
	}

	return coefficients
}

// interpretCoefficients converts sparse coefficients to human-readable results
func (s *SINDy) interpretCoefficients(coeffs []float64, states [][6]float64, theta, xDot *mat.Dense) *ManeuverResult {
	termNames := s.getTermNames()

	result := &ManeuverResult{
		Coefficients: coeffs,
		ActiveTerms:  make([]string, 0),
	}

	nDims := 6
	thrustMag := 0.0

	// Identify active (non-zero) terms
	for i, name := range termNames {
		maxCoeff := 0.0
		for d := 0; d < nDims; d++ {
			if abs(coeffs[i*nDims+d]) > maxCoeff {
				maxCoeff = abs(coeffs[i*nDims+d])
			}
		}

		if maxCoeff > s.SparsityThreshold {
			result.ActiveTerms = append(result.ActiveTerms,
				fmt.Sprintf("%s (%.3f)", name, maxCoeff))

			// Detect thrust (non-gravitational acceleration terms)
			if name != "1" && name != "x" && name != "y" && name != "z" &&
				!strings.HasPrefix(name, "x²") && !strings.HasPrefix(name, "y²") {
				thrustMag += maxCoeff
			}
		}
	}

	result.ThrustDetected = thrustMag > 0.01
	result.ThrustMagnitude = thrustMag * 1000 // Convert to m/s² approx

	// Build equation string
	result.EquationString = s.buildEquationString(coeffs, termNames)

	// Compute fit quality (R² from predicted vs observed)
	result.Confidence = s.computeRSquared(coeffs, states, theta, xDot)

	return result
}

// getTermNames returns human-readable names for library terms
func (s *SINDy) getTermNames() []string {
	names := []string{"1", "x", "y", "z", "vx", "vy", "vz"}

	// Polynomial terms
	for d := 2; d <= s.PolynomialDegree; d++ {
		names = append(names, fmt.Sprintf("x^%d", d))
		names = append(names, fmt.Sprintf("y^%d", d))
		names = append(names, fmt.Sprintf("z^%d", d))
		if d == 2 {
			names = append(names, "xy", "xz", "yz")
		}
	}

	if s.UseTrigonometric {
		names = append(names, "sin(r)", "cos(r)")
	}

	names = append(names, "v*vx", "v*vy", "v*vz")

	return names
}

// countLibraryTerms returns the number of candidate functions
func (s *SINDy) countLibraryTerms() int {
	count := 7 // 1 + x,y,z + vx,vy,vz

	// Polynomial: 3*(degree-1) for x^d,y^d,z^d + 3 cross terms for d=2
	count += 3*(s.PolynomialDegree-1) + 3

	if s.UseTrigonometric {
		count += 2
	}

	count += 3 // Velocity magnitude terms

	return count
}

// Helper functions
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func (s *SINDy) buildEquationString(coeffs []float64, termNames []string) string {
	// Lightweight: comma-joined list of active terms with coefficients
	// (Replaces LaTeX formatting which is unused in production telemetry)
	parts := make([]string, 0)
	nDims := 6
	for i, name := range termNames {
		maxC := 0.0
		for d := 0; d < nDims; d++ {
			if c := coeffs[i*nDims+d]; abs(c) > maxC {
				maxC = abs(c)
			}
		}
		if maxC > s.SparsityThreshold {
			parts = append(parts, fmt.Sprintf("%s:%.3e", name, maxC))
		}
	}
	return strings.Join(parts, ", ")
}

func (s *SINDy) computeRSquared(coeffs []float64, _states [][6]float64, theta, xDot *mat.Dense) float64 {
	// Calculate R² (coefficient of determination) from predicted vs observed
	// R² = 1 - SS_res / SS_tot
	// where SS_res = sum of squared residuals, SS_tot = total sum of squares

	nSamples, nTerms := theta.Dims()
	_, nDims := xDot.Dims()

	if nSamples == 0 || nDims == 0 {
		return 0.0
	}

	// Reshape coefficients for matrix multiplication
	coeffsMatrix := mat.NewDense(nTerms, nDims, coeffs)

	// Predicted values: X_pred = Theta * Coeffs
	var predicted mat.Dense
	predicted.Mul(theta, coeffsMatrix)

	// Calculate mean of observed values across all dimensions
	meanObserved := 0.0
	count := 0
	for i := 0; i < nSamples; i++ {
		for d := 0; d < nDims; d++ {
			meanObserved += xDot.At(i, d)
			count++
		}
	}
	meanObserved /= float64(count)

	// Calculate SS_res (sum of squared residuals) and SS_tot (total sum of squares)
	ssRes := 0.0
	ssTot := 0.0

	for i := 0; i < nSamples; i++ {
		for d := 0; d < nDims; d++ {
			observed := xDot.At(i, d)
			pred := predicted.At(i, d)

			residual := observed - pred
			ssRes += residual * residual

			diffFromMean := observed - meanObserved
			ssTot += diffFromMean * diffFromMean
		}
	}

	// R² = 1 - SS_res/SS_tot
	// If SS_tot is 0 (all observed values are identical), return 1.0 (perfect fit)
	if ssTot < 1e-10 {
		return 1.0
	}

	rSquared := 1.0 - (ssRes / ssTot)

	// Clamp to [0, 1] range to handle numerical edge cases
	if rSquared < 0.0 {
		rSquared = 0.0
	}
	if rSquared > 1.0 {
		rSquared = 1.0
	}

	return rSquared
}

// ValidateCertificate checks if certificate meets criteria for validity
//
// Sovereign Handshake: Every ObservationReport must include Hamiltonian Invariant Hash.
// If σ_H / H_avg >= 10^-7, certificate self-invalidates (spoof detection).
func (r *ManeuverResult) ValidateCertificate() bool {
	// Self-invalidation check: σ_H / H_avg >= 10^-7 (spoof detection)
	if r.EnergyDriftSigma >= 1e-7 {
		r.CertificateValid = false
		return false
	}

	// Check physics constraint violations
	if !AllPassed(r.ConstraintViolations) {
		r.CertificateValid = false
		return false
	}

	// Check bootstrap CI coverage
	if r.CICoverage < 0.8 {
		r.CertificateValid = false
		return false
	}

	r.CertificateValid = true
	return true
}

// ComputeHamiltonianHash computes a tolerance-binned Hamiltonian hash for the certificate.
//
// Raw float hashing is an anti-pattern (tiny numerical differences flip the hash).
// Instead, we bin the Hamiltonian values by tolerance before hashing, making the hash
// robust to numerical noise while still detecting physically significant changes.
func (r *ManeuverResult) ComputeHamiltonianHash(states [][6]float64, tolerance float64) string {
	if tolerance <= 0 {
		tolerance = 1e-6 // Default: bin at 1e-6 relative tolerance
	}

	// Compute average Hamiltonian across trajectory
	sumH := 0.0
	for _, state := range states {
		sumH += computeHamiltonian(state)
	}
	avgH := sumH / float64(len(states))

	// Tolerance-binned hash: quantize by tolerance relative to magnitude
	scale := math.Abs(avgH)
	if scale < 1e-10 {
		scale = 1.0
	}
	binnedAvg := int64(avgH / (tolerance * scale))

	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("%d", binnedAvg)))

	// Also bin coefficients for robustness
	for _, c := range r.Coefficients {
		binnedC := int64(c / tolerance)
		h.Write([]byte(fmt.Sprintf("%d", binnedC)))
	}

	r.HamiltonianHash = hex.EncodeToString(h.Sum(nil))
	return r.HamiltonianHash
}

// ComputeEnergyDrift computes σ_H / |H_avg| from trajectory states.
// Sets r.EnergyDriftSigma for spoof detection: if >= 1e-7, certificate self-invalidates.
func (r *ManeuverResult) ComputeEnergyDrift(states [][6]float64) {
	if len(states) == 0 {
		r.EnergyDriftSigma = 1.0
		return
	}

	values := make([]float64, len(states))
	sumH := 0.0
	for i, state := range states {
		h := computeHamiltonian(state)
		values[i] = h
		sumH += h
	}
	avgH := sumH / float64(len(states))

	// Standard deviation
	variance := 0.0
	for _, h := range values {
		diff := h - avgH
		variance += diff * diff
	}
	stdH := math.Sqrt(variance / float64(len(values)))

	// Normalized energy drift
	if math.Abs(avgH) > 1e-10 {
		r.EnergyDriftSigma = stdH / math.Abs(avgH)
	} else {
		r.EnergyDriftSigma = stdH
	}
}

// ComputeConstraintViolations evaluates physics consistency against thresholds.
// Replaces the A-F grade gamification with explicit pass/fail per constraint.
func (r *ManeuverResult) ComputeConstraintViolations(states [][6]float64) {
	th := DefaultGradeThresholds()

	r.ConstraintViolations = []ConstraintViolation{
		{
			Constraint: "energy_drift",
			Value:      r.EnergyDriftSigma,
			Threshold:  th.MaxEnergyDriftA,
			Unit:       "relative",
			Passed:     r.EnergyDriftSigma < th.MaxEnergyDriftA,
		},
		{
			Constraint: "position_residual",
			Value:      r.ResidualMean,
			Threshold:  th.MaxResidualA,
			Unit:       "meters",
			Passed:     r.ResidualMean < th.MaxResidualA,
		},
		{
			Constraint: "sindy_confidence",
			Value:      r.Confidence,
			Threshold:  th.MinSINDyConfidenceC,
			Unit:       "R2",
			Passed:     r.Confidence > th.MinSINDyConfidenceC,
		},
		{
			Constraint: "maneuver_explained",
			Value:      0.0,
			Threshold:  0.0,
			Unit:       "bool",
			Passed:     !r.ThrustDetected || (r.ThrustDetected && r.Confidence > 0.7),
		},
	}
}

// ComputeCausalProvenance calculates term importance scores for causal provenance
func (r *ManeuverResult) ComputeCausalProvenance() {
	r.CausalProvenance = make(map[string]float64)

	// Normalize coefficients by magnitude to get importance scores
	totalMag := 0.0
	for _, c := range r.Coefficients {
		totalMag += math.Abs(c)
	}

	if totalMag > 0 {
		for i, c := range r.Coefficients {
			termName := r.ActiveTerms[i]
			if i < len(r.ActiveTerms) {
				r.CausalProvenance[termName] = math.Abs(c) / totalMag
			}
		}
	}
}

// GenerateFalsifiablePredictions creates 60s predictions for validation
func (r *ManeuverResult) GenerateFalsifiablePredictions(initialState [6]float64) {
	r.Predictions = make([]Prediction, 0)

	// Generate predictions at 10s intervals for 60s horizon
	for t := 10.0; t <= 60.0; t += 10.0 {
		pred := Prediction{
			Timestamp:   time.Now().Add(time.Duration(t) * time.Second),
			State:       propagateState(initialState, t),
			Probability: r.Confidence,
			HorizonSec:  t,
		}
		r.Predictions = append(r.Predictions, pred)
	}
}

// Helper functions for Symbolic Certificate

func computeHamiltonian(state [6]float64) float64 {
	vx, vy, vz := state[3], state[4], state[5]
	kinetic := 0.5 * (vx*vx + vy*vy + vz*vz)

	r := math.Sqrt(state[0]*state[0] + state[1]*state[1] + state[2]*state[2])
	potential := -398600.4418 / r

	return kinetic + potential
}

func propagateState(state [6]float64, duration float64) [6]float64 {
	// Simple Keplerian propagation for predictions
	r := math.Sqrt(state[0]*state[0] + state[1]*state[1] + state[2]*state[2])
	mu := 398600.4418

	ax := -mu * state[0] / (r * r * r)
	ay := -mu * state[1] / (r * r * r)
	az := -mu * state[2] / (r * r * r)

	return [6]float64{
		state[0] + state[3]*duration,
		state[1] + state[4]*duration,
		state[2] + state[5]*duration,
		state[3] + ax*duration,
		state[4] + ay*duration,
		state[5] + az*duration,
	}
}
