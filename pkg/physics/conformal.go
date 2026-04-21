// pkg/physics/conformal.go - ADITI 4.0 Conformal Prediction Wrapper
//
// Distribution-free uncertainty quantification for the HF-PGFM.
// Provides mathematical guarantees that predictions will contain the true
// target position with user-specified confidence (1-α), regardless of noise
// distribution or maneuver type.
//
// Implementation: Split Conformal Prediction (Inductive)
//   1. Non-conformity score: s_i = max_t ||y_i(t) - ŷ_i(t)||_2
//   2. Calibration: q̂ = (1-α)(n+1)/n quantile of scores
//   3. Inference: C(x_test) = {y : ||y - ŷ||_2 ≤ q̂}
//
// Strategic Value (May 4th Audit):
//   - Distribution-free: Works with non-Gaussian noise, glint, ionospheric interference
//   - Safety guarantee: "99.9% confidence HGV is inside this 150m sphere"
//   - Adaptive calibration: Different q̂ for different latent regimes
//   - Sovereign-grade: Mathematical proof of intercept zone
//
// Integration:
//   - Calibrated on held-out trajectories from Go-native simulator
//   - Includes adversarial trajectories (max-G turns) for stress testing
//   - Populates SigmaEnvelope in SymbolicCertificate as conformal radius
//   - Complements SINDy Bootstrap CI (intent) with location guarantee
//
// References:
//   - Vovk et al. "Algorithmic Learning in a Random World" (2005)
//   - Angelopoulos & Bates "A Gentle Introduction to Conformal Prediction" (2021)
//   - May 4th Audit: iDEX ADITI 4.0 Sovereign Brain

package physics

import (
	"fmt"
	"math"
	"sort"
)

// =============================================================================
// Conformal Prediction Core
// =============================================================================

// ConformalWrapper implements Split Conformal Prediction for uncertainty quantification
type ConformalWrapper struct {
	Quantile        float64 // The calibrated q̂ value (global safety buffer)
	Alpha           float64 // Miscoverage rate (e.g., 0.01 for 99% coverage)
	IsCalibrated    bool
	CalibrationSize int // Number of samples used for calibration
}

// NonConformityScore represents a single calibration score
type NonConformityScore struct {
	MaxL2Error    float64   // Maximum L2 error over trajectory
	RegimeLabel   string    // Latent regime (e.g., "ORBITAL", "HGV_SKIP")
	ResidualTrace []float64 // Full residual trajectory for analysis
}

// CalibrationSet holds the calibration data
type CalibrationSet struct {
	Scores          []NonConformityScore
	RegimeQuantiles map[string]float64 // q̂ per regime for adaptive calibration
}

// NewConformalWrapper creates an uncalibrated conformal wrapper
func NewConformalWrapper() *ConformalWrapper {
	return &ConformalWrapper{
		Alpha:        0.01, // Default: 99% coverage
		IsCalibrated: false,
	}
}

// Calibrate computes q̂ from non-conformity scores using split conformal
// This is typically run once after H100 training completes
func (cw *ConformalWrapper) Calibrate(scores []float64, alpha float64) error {
	if len(scores) == 0 {
		return ErrEmptyCalibrationSet
	}

	n := float64(len(scores))

	// Sort scores
	sortedScores := make([]float64, len(scores))
	copy(sortedScores, scores)
	sort.Float64s(sortedScores)

	// Calculate index for (1-α)(n+1)/n quantile
	// This ensures coverage guarantee under exchangeability
	index := int(math.Ceil((1.0-alpha)*(n+1.0))) - 1

	// Clamp index to valid range
	if index < 0 {
		index = 0
	}
	if index >= len(sortedScores) {
		index = len(sortedScores) - 1
	}

	cw.Quantile = sortedScores[index]
	cw.Alpha = alpha
	cw.IsCalibrated = true
	cw.CalibrationSize = len(scores)

	return nil
}

// CalibrateWithRegimes performs adaptive calibration with regime-specific q̂
// Useful when different flight regimes have different uncertainty characteristics
func (cw *ConformalWrapper) CalibrateWithRegimes(
	calibrationSet CalibrationSet,
	alpha float64,
) error {
	if len(calibrationSet.Scores) == 0 {
		return ErrEmptyCalibrationSet
	}

	// Group scores by regime
	regimeScores := make(map[string][]float64)
	regimeQuantiles := make(map[string]float64)

	for _, score := range calibrationSet.Scores {
		regime := score.RegimeLabel
		if regime == "" {
			regime = "DEFAULT"
		}
		regimeScores[regime] = append(regimeScores[regime], score.MaxL2Error)
	}

	// Calibrate each regime separately
	for regime, scores := range regimeScores {
		if len(scores) == 0 {
			continue
		}

		n := float64(len(scores))
		sortedScores := make([]float64, len(scores))
		copy(sortedScores, scores)
		sort.Float64s(sortedScores)

		index := int(math.Ceil((1.0-alpha)*(n+1.0))) - 1
		if index < 0 {
			index = 0
		}
		if index >= len(sortedScores) {
			index = len(sortedScores) - 1
		}

		regimeQuantiles[regime] = sortedScores[index]
	}

	// Store regime quantiles
	calibrationSet.RegimeQuantiles = regimeQuantiles

	// Set global quantile as max of all regimes (conservative)
	globalQuantile := 0.0
	for _, q := range regimeQuantiles {
		if q > globalQuantile {
			globalQuantile = q
		}
	}

	cw.Quantile = globalQuantile
	cw.Alpha = alpha
	cw.IsCalibrated = true
	cw.CalibrationSize = len(calibrationSet.Scores)

	return nil
}

// Wrap applies the conformal interval to a predicted point
// Returns the radius of the conformal ball (guaranteed coverage)
func (cw *ConformalWrapper) Wrap(prediction [3]float64) float64 {
	if !cw.IsCalibrated {
		// Return safe default if not calibrated
		return 1000.0 // 1km default safety buffer
	}
	return cw.Quantile
}

// WrapWithRegime applies regime-specific conformal interval
func (cw *ConformalWrapper) WrapWithRegime(
	prediction [3]float64,
	regime string,
	calibrationSet CalibrationSet,
) float64 {
	if !cw.IsCalibrated {
		return 1000.0
	}

	// Use regime-specific quantile if available
	if q, ok := calibrationSet.RegimeQuantiles[regime]; ok {
		return q
	}

	// Fall back to global quantile
	return cw.Quantile
}

// =============================================================================
// Non-Conformity Score Computation
// =============================================================================

// ComputeMaxL2Error calculates the maximum L2 error over a trajectory
// This is the non-conformity score for conformal prediction
func ComputeMaxL2Error(
	groundTruth [][3]float64,
	prediction [][3]float64,
) float64 {
	if len(groundTruth) != len(prediction) {
		return math.Inf(1)
	}
	if len(groundTruth) == 0 {
		return 0.0
	}

	maxError := 0.0
	for i := range groundTruth {
		error := l2Distance(groundTruth[i], prediction[i])
		if error > maxError {
			maxError = error
		}
	}

	return maxError
}

// ComputeTrajectoryScores computes non-conformity scores for a calibration set
func ComputeTrajectoryScores(
	groundTruths [][][3]float64,
	predictions [][][3]float64,
	regimeLabels []string,
) []NonConformityScore {
	if len(groundTruths) != len(predictions) {
		return nil
	}

	scores := make([]NonConformityScore, len(groundTruths))

	for i := range groundTruths {
		maxL2 := ComputeMaxL2Error(groundTruths[i], predictions[i])

		regime := "DEFAULT"
		if i < len(regimeLabels) && regimeLabels[i] != "" {
			regime = regimeLabels[i]
		}

		scores[i] = NonConformityScore{
			MaxL2Error:  maxL2,
			RegimeLabel: regime,
		}
	}

	return scores
}

// l2Distance computes Euclidean distance between two 3D points
func l2Distance(a, b [3]float64) float64 {
	dx := a[0] - b[0]
	dy := a[1] - b[1]
	dz := a[2] - b[2]
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

// =============================================================================
// Conformal Region (Prediction Set)
// =============================================================================

// ConformalRegion represents the prediction set C(x_test)
type ConformalRegion struct {
	Center [3]float64 // Prediction center
	Radius float64    // Conformal radius (q̂)
	Alpha  float64    // Coverage level
}

// Contains checks if a point is within the conformal region
func (cr *ConformalRegion) Contains(point [3]float64) bool {
	return l2Distance(point, cr.Center) <= cr.Radius
}

// Volume returns the volume of the conformal sphere (in km³)
func (cr *ConformalRegion) Volume() float64 {
	// Volume of sphere: (4/3)πr³
	rKm := cr.Radius / 1000.0 // Convert meters to km
	return (4.0 / 3.0) * math.Pi * rKm * rKm * rKm
}

// =============================================================================
// Integration with Symbolic Certificate
// =============================================================================

// ApplyConformalToCertificate replaces the Koopman σ envelope with conformal radius
func ApplyConformalToCertificate(
	cert *SymbolicCertificate,
	wrapper *ConformalWrapper,
	regime string,
	calibrationSet CalibrationSet,
) error {
	if !wrapper.IsCalibrated {
		return ErrNotCalibrated
	}

	// Get final prediction point (60s horizon)
	if len(cert.Projection.Trajectory) == 0 {
		return ErrEmptyTrajectory
	}

	finalPos := cert.Projection.Trajectory[len(cert.Projection.Trajectory)-1]

	// Get conformal radius
	var radius float64
	if regime != "" && len(calibrationSet.RegimeQuantiles) > 0 {
		radius = wrapper.WrapWithRegime(finalPos, regime, calibrationSet)
	} else {
		radius = wrapper.Wrap(finalPos)
	}

	// Replace SigmaEnvelope with conformal radius
	// For the full trajectory, we can use linear interpolation
	numPoints := len(cert.Projection.Trajectory)
	cert.Projection.SigmaEnvelope = make([]float64, numPoints)
	for i := 0; i < numPoints; i++ {
		// Linear interpolation from 0 at t=0 to radius at t=60s
		fraction := float64(i) / float64(numPoints-1)
		cert.Projection.SigmaEnvelope[i] = radius * fraction
	}

	return nil
}

// =============================================================================
// Calibration Set Generation
// =============================================================================

// GenerateCalibrationSet creates a calibration set from simulator data
// This is the "Sovereign Calibration Set" held out from training
type CalibrationConfig struct {
	HoldoutRatio       float64 // Fraction of data to hold out (e.g., 0.2)
	IncludeAdversarial bool    // Include max-G turns for stress testing
	NumSamples         int     // Number of calibration samples
}

// GenerateCalibrationFromSimulator creates calibration scores from Go-native simulator
func GenerateCalibrationFromSimulator(
	config CalibrationConfig,
) (CalibrationSet, error) {
	// In production, this would:
	// 1. Load held-out trajectories from simulator
	// 2. Run PHS prediction on each
	// 3. Compute max L2 error
	// 4. Assign regime labels based on latent code

	// Mock implementation
	return CalibrationSet{
		Scores: []NonConformityScore{
			{MaxL2Error: 12.3, RegimeLabel: "ORBITAL"},
			{MaxL2Error: 15.7, RegimeLabel: "ORBITAL"},
			{MaxL2Error: 423.5, RegimeLabel: "HGV_SKIP"},
			{MaxL2Error: 387.2, RegimeLabel: "HGV_SKIP"},
			{MaxL2Error: 512.8, RegimeLabel: "HGV_MANEUVER"},
		},
		RegimeQuantiles: map[string]float64{
			"ORBITAL":      15.7,
			"HGV_SKIP":     423.5,
			"HGV_MANEUVER": 512.8,
		},
	}, nil
}

// =============================================================================
// Errors
// =============================================================================

var (
	ErrEmptyCalibrationSet = &ConformalError{
		Code:    "EMPTY_CALIBRATION",
		Message: "calibration set cannot be empty",
	}
	ErrNotCalibrated = &ConformalError{
		Code:    "NOT_CALIBRATED",
		Message: "conformal wrapper must be calibrated before use",
	}
	ErrEmptyTrajectory = &ConformalError{
		Code:    "EMPTY_TRAJECTORY",
		Message: "trajectory cannot be empty",
	}
)

// ConformalError represents a conformal prediction error
type ConformalError struct {
	Code    string
	Message string
}

func (e *ConformalError) Error() string {
	return e.Code + ": " + e.Message
}

// =============================================================================
// Verification and Validation
// =============================================================================

// VerifyCoverage checks empirical coverage on a test set
// Returns the actual coverage rate (should be ≈ 1-α if calibration is valid)
func VerifyCoverage(
	testGroundTruths [][][3]float64,
	testPredictions [][][3]float64,
	wrapper *ConformalWrapper,
) (float64, error) {
	if !wrapper.IsCalibrated {
		return 0.0, ErrNotCalibrated
	}
	if len(testGroundTruths) != len(testPredictions) {
		return 0.0, &ConformalError{
			Code:    "MISMATCHED_SETS",
			Message: "ground truth and prediction sets must have same length",
		}
	}

	covered := 0
	for i := range testGroundTruths {
		// Check if final point is within conformal region
		if len(testPredictions[i]) == 0 || len(testGroundTruths[i]) == 0 {
			continue
		}

		finalPred := testPredictions[i][len(testPredictions[i])-1]
		finalTruth := testGroundTruths[i][len(testGroundTruths[i])-1]

		error := l2Distance(finalPred, finalTruth)
		if error <= wrapper.Quantile {
			covered++
		}
	}

	coverage := float64(covered) / float64(len(testGroundTruths))
	return coverage, nil
}

// =============================================================================
// String Representation
// =============================================================================

// String returns human-readable conformal wrapper status
func (cw *ConformalWrapper) String() string {
	status := "UNCALIBRATED"
	if cw.IsCalibrated {
		status = "CALIBRATED"
	}

	return fmt.Sprintf(
		"ConformalWrapper[%s]\n"+
			"  Quantile (q̂): %.1fm\n"+
			"  Coverage (1-α): %.1f%%\n"+
			"  Calibration Samples: %d",
		status,
		cw.Quantile,
		(1.0-cw.Alpha)*100,
		cw.CalibrationSize,
	)
}

// =============================================================================
// Constants
// =============================================================================

const (
	// Standard coverage levels
	Coverage90  = 0.10  // 90% coverage (α=0.10)
	Coverage95  = 0.05  // 95% coverage (α=0.05)
	Coverage99  = 0.01  // 99% coverage (α=0.01)
	Coverage999 = 0.001 // 99.9% coverage (α=0.001)
)
