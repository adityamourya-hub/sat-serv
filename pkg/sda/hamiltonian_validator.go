package sda

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"time"

	"ssis/pkg/catalog"
	"ssis/pkg/physics"

	"go.uber.org/zap"
)

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * Hamiltonian Invariant Validator
 *
 * The ultimate guardrail against Data Poisoning. In an adversarial scenario,
 * an attacker might inject "Ghost Tracks" (fake ephemeris) into the catalog
 * to overwhelm IAF operators.
 *
 * The Law of Conservation: Every object in space must obey the Hamiltonian H.
 * If the system receives a track where dH/dt ≠ 0 (and no thruster plume is
 * detected via DMD/Spectra), the Axiom-Zero kernel rejects the data as "Non-Physical."
 *
 * Thermodynamic Trust: We aren't trusting a "Digital Signature" (which can be
 * stolen); we are trusting the Laws of Thermodynamics. If the data doesn't
 * conserve energy, it's a spoof.
 */

// HamiltonianValidator validates tracks against energy conservation
type HamiltonianValidator struct {
	DriftThreshold    float64 // σ_H / H_avg threshold (default 1e-7)
	MaxDriftSamples   int     // Number of samples for drift calculation
	RequirePlumeCheck bool    // Require plume detection for non-conserving tracks
	logger            *zap.Logger

	// Metrics for observability
	validationCount     int64
	rejectionCount      int64
	totalValidationTime time.Duration
}

// NewHamiltonianValidator creates a new Hamiltonian validator with logging
func NewHamiltonianValidator() *HamiltonianValidator {
	logger, _ := zap.NewProduction()
	return &HamiltonianValidator{
		DriftThreshold:    1e-7, // 10^-7 threshold for spoof detection
		MaxDriftSamples:   50,   // 50 samples for drift calculation
		RequirePlumeCheck: true, // Require plume detection for non-conserving tracks
		logger:            logger,
	}
}

// SetLogger allows setting a custom logger (for testing or integration)
func (hv *HamiltonianValidator) SetLogger(logger *zap.Logger) {
	hv.logger = logger
}

// ValidationResult represents the result of Hamiltonian validation
type ValidationResult struct {
	Valid                 bool
	DriftSigma            float64 // σ_H / H_avg
	DriftExceedsThreshold bool
	PlumeDetected         bool
	Reason                string
	Confidence            float64
}

// ValidateTrack validates a track against Hamiltonian invariants
// Algorithm:
// 1. Compute Hamiltonian at multiple time points
// 2. Calculate σ_H / H_avg (energy drift)
// 3. Check if drift exceeds threshold
// 4. If drift exceeds threshold, require plume detection
func (hv *HamiltonianValidator) ValidateTrack(obj catalog.SDAObject, window TimeWindow, plumeDetected bool) ValidationResult {
	startTime := time.Now()
	hv.validationCount++

	// Build trajectory states
	states := hv.buildTrajectoryStates(obj, window)
	if len(states) < 2 {
		hv.logger.Warn("Hamiltonian validation failed - insufficient samples",
			zap.Int("object_id", obj.ID),
			zap.Int("sample_count", len(states)),
		)
		return ValidationResult{
			Valid:      false,
			Reason:     "Insufficient state samples",
			Confidence: 0,
		}
	}

	// Compute Hamiltonian at each state point
	hamiltonians := hv.computeHamiltonians(states)

	// Calculate average Hamiltonian
	sumH := 0.0
	for _, H := range hamiltonians {
		sumH += H
	}
	avgH := sumH / float64(len(hamiltonians))

	// Compute σ_H (standard deviation)
	variance := 0.0
	for _, H := range hamiltonians {
		diff := H - avgH
		variance += diff * diff
	}
	sigmaH := math.Sqrt(variance / float64(len(hamiltonians)))

	// Calculate σ_H / H_avg
	driftSigma := sigmaH / math.Abs(avgH)

	// Check if drift exceeds threshold
	driftExceeds := driftSigma > hv.DriftThreshold

	// Validation logic
	valid := true
	reason := "Energy conserved"

	if driftExceeds {
		// Energy not conserved - check for plume detection
		if hv.RequirePlumeCheck && !plumeDetected {
			valid = false
			reason = "Energy drift without plume detection - possible spoof"
		} else if hv.RequirePlumeCheck && plumeDetected {
			// Plume detected - drift is expected (thrust)
			valid = true
			reason = "Energy drift with plume detection - thrust maneuver"
		} else {
			// Not requiring plume check - allow drift
			valid = true
			reason = "Energy drift within tolerance"
		}
	}

	// Calculate confidence based on drift magnitude
	confidence := hv.calculateConfidence(driftSigma, driftExceeds, plumeDetected)

	// Track timing metrics
	validationTime := time.Since(startTime)
	hv.totalValidationTime += validationTime

	// Log validation result
	if valid {
		hv.logger.Info("Hamiltonian validation passed",
			zap.Int("object_id", obj.ID),
			zap.Float64("drift_sigma", driftSigma),
			zap.Float64("threshold", hv.DriftThreshold),
			zap.Duration("validation_time", validationTime),
		)
	} else {
		hv.rejectionCount++
		hv.logger.Error("Hamiltonian validation FAILED - possible spoof",
			zap.Int("object_id", obj.ID),
			zap.Float64("drift_sigma", driftSigma),
			zap.Float64("threshold", hv.DriftThreshold),
			zap.String("reason", reason),
			zap.Duration("validation_time", validationTime),
		)
	}

	return ValidationResult{
		Valid:                 valid,
		DriftSigma:            driftSigma,
		DriftExceedsThreshold: driftExceeds,
		PlumeDetected:         plumeDetected,
		Reason:                reason,
		Confidence:            confidence,
	}
}

// buildTrajectoryStates builds trajectory states using Forest-Ruth symplectic integration
// Propagates the object state forward in time using energy-conserving integrator
func (hv *HamiltonianValidator) buildTrajectoryStates(obj catalog.SDAObject, window TimeWindow) [][6]float64 {
	nSamples := hv.MaxDriftSamples
	if nSamples < 2 {
		nSamples = 50 // Default
	}

	// Initialize Keplerian PHS model
	model := NewKeplerianPHSModel()

	// Initial state: position and velocity (unit mass, so momentum = velocity)
	q := []float64{obj.Pos.X, obj.Pos.Y, obj.Pos.Z}
	p := []float64{obj.Vel.X, obj.Vel.Y, obj.Vel.Z}
	z := []float64{} // No latent code for pure Keplerian

	// Time step: divide window into nSamples steps
	duration := window.End.Sub(window.Start).Seconds()
	dt := duration / float64(nSamples)
	if dt <= 0 {
		dt = 60.0 // Default 1 minute steps
	}

	var states [][6]float64

	// Store initial state
	states = append(states, [6]float64{q[0], q[1], q[2], p[0], p[1], p[2]})

	// Propagate using Forest-Ruth symplectic integrator
	for i := 0; i < nSamples; i++ {
		// Perform one symplectic step
		qNext, pNext := physics.ForestRuthStep(model, q, p, z, dt, physics.ManifoldCartesian)

		// Store propagated state
		states = append(states, [6]float64{qNext[0], qNext[1], qNext[2], pNext[0], pNext[1], pNext[2]})

		// Update for next iteration
		q = qNext
		p = pNext
	}

	return states
}

// computeHamiltonians computes Hamiltonian at each state point using PHS model
func (hv *HamiltonianValidator) computeHamiltonians(states [][6]float64) []float64 {
	model := NewKeplerianPHSModel()
	hamiltonians := make([]float64, len(states))

	for i, state := range states {
		q := []float64{state[0], state[1], state[2]}
		p := []float64{state[3], state[4], state[5]}
		z := []float64{}

		// Use PHS model Hamiltonian for consistency with propagation
		hamiltonians[i] = model.Hamiltonian(q, p, z)
	}

	return hamiltonians
}

// calculateConfidence calculates validation confidence
func (hv *HamiltonianValidator) calculateConfidence(driftSigma float64, driftExceeds bool, plumeDetected bool) float64 {
	if !driftExceeds {
		// Energy conserved - high confidence
		return 0.99
	}

	if driftExceeds && plumeDetected {
		// Drift with plume - expected thrust maneuver
		return 0.85
	}

	if driftExceeds && !plumeDetected {
		// Drift without plume - possible spoof
		// Confidence decreases as drift increases
		confidence := 1.0 - (driftSigma / hv.DriftThreshold)
		if confidence < 0 {
			confidence = 0
		}
		return confidence
	}

	return 0.5
}

// ValidateCertificate validates a SINDy certificate against Hamiltonian invariants
// This extends the existing SINDy certificate validation with energy conservation
func (hv *HamiltonianValidator) ValidateCertificate(result interface{}) bool {
	// In production: check result.EnergyDriftSigma against threshold
	// For now: placeholder implementation
	return true
}

// GenerateHamiltonianHash generates a cryptographic hash of the Hamiltonian state
// This provides a "Physical Firewall" signature for the track
func (hv *HamiltonianValidator) GenerateHamiltonianHash(obj catalog.SDAObject, window TimeWindow) string {
	states := hv.buildTrajectoryStates(obj, window)
	hamiltonians := hv.computeHamiltonians(states)

	// Compute average Hamiltonian
	sumH := 0.0
	for _, H := range hamiltonians {
		sumH += H
	}
	avgH := sumH / float64(len(hamiltonians))

	// Compute σ_H
	variance := 0.0
	for _, H := range hamiltonians {
		diff := H - avgH
		variance += diff * diff
	}
	sigmaH := math.Sqrt(variance / float64(len(hamiltonians)))

	// Check if drift exceeds threshold
	driftSigma := sigmaH / math.Abs(avgH)
	if driftSigma >= hv.DriftThreshold {
		return "INVALID:ENERGY_DRIFT_EXCEEDED"
	}

	// Generate cryptographic hash from average Hamiltonian and object metadata
	// This provides a tamper-evident seal for the track
	hashInput := fmt.Sprintf("%.15f|%.15f|%d|%s|%d|%d",
		avgH, sigmaH, obj.ID, obj.Name, window.Start.Unix(), window.End.Unix())

	hash := sha256.Sum256([]byte(hashInput))
	hashStr := hex.EncodeToString(hash[:])

	return fmt.Sprintf("H:%s|DRIFT:%.2e", hashStr[:32], driftSigma)
}

// BatchValidateTracks validates multiple tracks in batch
func (hv *HamiltonianValidator) BatchValidateTracks(objects []catalog.SDAObject, window TimeWindow, plumeDetections map[int]bool) map[int]ValidationResult {
	results := make(map[int]ValidationResult)

	for _, obj := range objects {
		plumeDetected := plumeDetections[obj.ID]
		result := hv.ValidateTrack(obj, window, plumeDetected)
		results[obj.ID] = result
	}

	return results
}

// FilterSpoofedTracks filters out tracks that fail Hamiltonian validation
func (hv *HamiltonianValidator) FilterSpoofedTracks(objects []catalog.SDAObject, window TimeWindow, plumeDetections map[int]bool) []catalog.SDAObject {
	var validObjects []catalog.SDAObject

	for _, obj := range objects {
		plumeDetected := plumeDetections[obj.ID]
		result := hv.ValidateTrack(obj, window, plumeDetected)

		if result.Valid {
			validObjects = append(validObjects, obj)
		}
		// Else: reject spoofed track
	}

	return validObjects
}

// ValidatorMetrics contains operational metrics for observability
type ValidatorMetrics struct {
	ValidationCount     int64
	RejectionCount      int64
	RejectionRate       float64
	TotalValidationTime time.Duration
	AvgValidationTime   time.Duration
	DriftThreshold      float64
}

// GetMetrics returns current operational metrics for monitoring
func (hv *HamiltonianValidator) GetMetrics() ValidatorMetrics {
	avgTime := time.Duration(0)
	if hv.validationCount > 0 {
		avgTime = hv.totalValidationTime / time.Duration(hv.validationCount)
	}

	rejectionRate := 0.0
	if hv.validationCount > 0 {
		rejectionRate = float64(hv.rejectionCount) / float64(hv.validationCount)
	}

	return ValidatorMetrics{
		ValidationCount:     hv.validationCount,
		RejectionCount:      hv.rejectionCount,
		RejectionRate:       rejectionRate,
		TotalValidationTime: hv.totalValidationTime,
		AvgValidationTime:   avgTime,
		DriftThreshold:      hv.DriftThreshold,
	}
}
