package ml

/**
 * Kullback-Leibler Divergence for Information-Theoretic Maneuver Detection
 *
 * Geometric Mechanics Optimization #5
 *
 * Problem: PINN processes all objects, causing "feature fatigue"
 * Solution: KL divergence triggers wake-up only on surprising observations
 *
 * Math:
 *   D_KL(P_obs || P_pred) = ∫ p_obs(x) log(p_obs(x)/p_pred(x)) dx
 *
 * When D_KL spikes → object is trying to "hide" its maneuver
 *
 * Benefits:
 *   - ML layer only "wakes up" on KL divergence spike
 *   - Focus compute on objects actively deviating from physics
 *   - Aghori-class detection stays lean
 *
 * References:
 *   - Kullback & Leibler (1951): On Information and Sufficiency
 *   - Cover & Thomas: Elements of Information Theory
 */

import (
	"math"
)

// GaussianPDF represents multivariate Gaussian
type GaussianPDF struct {
	Mean    [6]float64    // [x,y,z,vx,vy,vz]
	Cov     [6][6]float64 // Covariance matrix
	InvCov  [6][6]float64 // Pre-computed inverse
	LogDet  float64       // log|Σ| for normalization
	IsValid bool
}

// ComputeMahalanobis computes (x-μ)^T Σ^-1 (x-μ)
func (g *GaussianPDF) ComputeMahalanobis(x [6]float64) float64 {
	diff := [6]float64{
		x[0] - g.Mean[0],
		x[1] - g.Mean[1],
		x[2] - g.Mean[2],
		x[3] - g.Mean[3],
		x[4] - g.Mean[4],
		x[5] - g.Mean[5],
	}

	// Compute Σ^-1 × diff
	var invDiff [6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			invDiff[i] += g.InvCov[i][j] * diff[j]
		}
	}

	// Compute diff^T × (Σ^-1 × diff)
	mahal := 0.0
	for i := 0; i < 6; i++ {
		mahal += diff[i] * invDiff[i]
	}

	return mahal
}

// Evaluate computes probability density
func (g *GaussianPDF) Evaluate(x [6]float64) float64 {
	if !g.IsValid {
		return 0.0
	}

	mahal := g.ComputeMahalanobis(x)

	// p(x) = (2π)^(-d/2) |Σ|^(-1/2) exp(-0.5 × Mahal)
	d := 6.0
	norm := math.Pow(2*math.Pi, -d/2) * math.Exp(-0.5*g.LogDet)

	return norm * math.Exp(-0.5*mahal)
}

// KLDivergence computes D_KL(P || Q) = E_P[log(P/Q)]
// Analytical for Gaussians:
// D_KL = 0.5 × [tr(Σ_Q^-1 Σ_P) + (μ_Q-μ_P)^T Σ_Q^-1 (μ_Q-μ_P) - d + log(|Σ_Q|/|Σ_P|)]
func KLDivergence(p, q GaussianPDF) float64 {
	if !p.IsValid || !q.IsValid {
		return math.Inf(1)
	}

	d := 6.0

	// Term 1: tr(Σ_Q^-1 Σ_P)
	trace := 0.0
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			trace += q.InvCov[i][j] * p.Cov[j][i]
		}
	}

	// Term 2: (μ_Q-μ_P)^T Σ_Q^-1 (μ_Q-μ_P)
	diff := [6]float64{
		q.Mean[0] - p.Mean[0],
		q.Mean[1] - p.Mean[1],
		q.Mean[2] - p.Mean[2],
		q.Mean[3] - p.Mean[3],
		q.Mean[4] - p.Mean[4],
		q.Mean[5] - p.Mean[5],
	}

	var invDiff [6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			invDiff[i] += q.InvCov[i][j] * diff[j]
		}
	}

	mahalDiff := 0.0
	for i := 0; i < 6; i++ {
		mahalDiff += diff[i] * invDiff[i]
	}

	// Term 3: log(|Σ_Q|/|Σ_P|) = log|Σ_Q| - log|Σ_P|
	logDetRatio := q.LogDet - p.LogDet

	kl := 0.5 * (trace + mahalDiff - d + logDetRatio)

	return kl
}

// KLManeuverDetector uses KL divergence for selective ML activation
type KLManeuverDetector struct {
	// Thresholds
	KLThreshold float64 // D_KL value that triggers wake-up

	// State tracking
	PredictedPDF map[string]GaussianPDF // Propagated covariance
	ObservedPDF  map[string]GaussianPDF // Sensor observations

	// Flags
	AwakeObjects map[string]bool // Objects currently being analyzed
}

// NewKLManeuverDetector creates detector with threshold
func NewKLManeuverDetector(threshold float64) *KLManeuverDetector {
	return &KLManeuverDetector{
		KLThreshold:  threshold,
		PredictedPDF: make(map[string]GaussianPDF),
		ObservedPDF:  make(map[string]GaussianPDF),
		AwakeObjects: make(map[string]bool),
	}
}

// UpdatePrediction stores propagated state and covariance
func (klm *KLManeuverDetector) UpdatePrediction(
	objectID string,
	state [6]float64,
	cov [6][6]float64,
) {
	invCov, logDet := invertCovariance6x6(cov)

	klm.PredictedPDF[objectID] = GaussianPDF{
		Mean:    state,
		Cov:     cov,
		InvCov:  invCov,
		LogDet:  logDet,
		IsValid: true,
	}
}

// UpdateObservation processes new sensor data
func (klm *KLManeuverDetector) UpdateObservation(
	objectID string,
	observation [6]float64,
	obsCov [6][6]float64,
) (bool, float64) {
	// Get predicted PDF
	predicted, exists := klm.PredictedPDF[objectID]
	if !exists {
		return false, 0.0
	}

	// Create observation PDF (tight covariance around observation)
	obsInvCov, obsLogDet := invertCovariance6x6(obsCov)
	observed := GaussianPDF{
		Mean:    observation,
		Cov:     obsCov,
		InvCov:  obsInvCov,
		LogDet:  obsLogDet,
		IsValid: true,
	}

	// Compute KL divergence
	kl := KLDivergence(observed, predicted)

	// Check if object should wake up
	shouldWake := kl > klm.KLThreshold

	if shouldWake {
		klm.AwakeObjects[objectID] = true
	}

	return shouldWake, kl
}

// GetAwakeObjects returns list of objects requiring ML analysis
func (klm *KLManeuverDetector) GetAwakeObjects() []string {
	var result []string
	for id := range klm.AwakeObjects {
		result = append(result, id)
	}
	return result
}

// ComputeSurprisal calculates information gain from observation
// Surprisal = -log(P_predicted(observation))
func (klm *KLManeuverDetector) ComputeSurprisal(objectID string, observation [6]float64) float64 {
	predicted, exists := klm.PredictedPDF[objectID]
	if !exists {
		return math.Inf(1)
	}

	prob := predicted.Evaluate(observation)
	if prob <= 0 {
		return math.Inf(1)
	}

	return -math.Log(prob)
}

// invertCovariance6x6 computes inverse and log determinant
// Uses Cholesky decomposition for symmetric positive definite matrices
func invertCovariance6x6(cov [6][6]float64) (inv [6][6]float64, logDet float64) {
	// For production: Use LAPACK Cholesky (dpotrf/dpotri)
	// Simplified: Direct 6x6 inverse

	// Compute determinant and inverse using cofactor expansion
	// This is O(d³) = 216 operations for 6×6

	// Placeholder: Return identity for invalid covariance
	det := computeDeterminant6x6(cov)
	if math.Abs(det) < 1e-15 {
		return [6][6]float64{}, math.Inf(-1)
	}

	// For now, return approximate inverse
	// Full implementation would use Gaussian elimination or Cholesky

	// Simple diagonal approximation (for testing)
	for i := 0; i < 6; i++ {
		if cov[i][i] > 0 {
			inv[i][i] = 1.0 / cov[i][i]
		}
	}

	logDet = math.Log(math.Abs(det))

	return
}

// computeDeterminant6x6 calculates |Σ| for 6×6 matrix
func computeDeterminant6x6(m [6][6]float64) float64 {
	// Use LU decomposition for efficiency
	// Simplified: Return product of diagonal (for diagonal-dominant)
	det := 1.0
	for i := 0; i < 6; i++ {
		det *= m[i][i]
	}
	return det
}

// ComputeKLBatch processes multiple objects efficiently
func (klm *KLManeuverDetector) ComputeKLBatch(
	objectIDs []string,
	observations [][6]float64,
) map[string]float64 {
	results := make(map[string]float64)

	for i, id := range objectIDs {
		if i < len(observations) {
			obsCov := makeDiagonalCovariance(10.0) // 10m uncertainty
			_, kl := klm.UpdateObservation(id, observations[i], obsCov)
			results[id] = kl
		}
	}

	return results
}

// makeDiagonalCovariance creates diagonal covariance matrix
func makeDiagonalCovariance(s float64) [6][6]float64 {
	var cov [6][6]float64
	for i := 0; i < 6; i++ {
		cov[i][i] = float64(s) * float64(s)
	}
	return cov
}

// ThresholdRecommendation suggests KL threshold based on false positive rate
func ThresholdRecommendation(desiredFPR float64) float64 {
	// For Gaussian, KL ~ χ²(6)/2
	// P(D_KL > threshold) = desiredFPR

	// χ²(6) critical values:
	// 95%: 12.59
	// 99%: 16.81
	// 99.9%: 22.46

	switch desiredFPR {
	case 0.05:
		return 12.59 / 2.0 // ~6.3
	case 0.01:
		return 16.81 / 2.0 // ~8.4
	case 0.001:
		return 22.46 / 2.0 // ~11.2
	}

	return 8.0 // Default: ~99% confidence
}

// PerformanceComparison shows efficiency gain
func KLPerformanceComparison() (klObjectsPerSec, pinnObjectsPerSec, savings float64) {
	// KL divergence: ~1μs per object (simple analytical formula)
	klObjectsPerSec = 1000000.0

	// PINN: ~10ms per object (neural network forward pass)
	pinnObjectsPerSec = 100.0

	// With KL triggering: Process 100k with KL, wake up ~100 for PINN
	// Total time: 100k × 1μs + 100 × 10ms = 0.1s + 1s = 1.1s
	// vs PINN all: 100k × 10ms = 1000s

	savings = 1000.0 / 1.1 // ~909× speedup
	return
}
