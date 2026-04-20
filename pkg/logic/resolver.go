// Package logic provides the Conflict Resolver for ADITI 4.0
//
// The Conflict Resolver acts as the "Supreme Court" of the software stack,
// arbitrating between the Go-native physics engine (Heart) and the JAX-based
// AI brain (Brain). It ensures that physical truth always prevails over
// model predictions when safety is at stake.
//
// This module implements a Bayesian Trust Arbiter that calculates a
// "Physical Consistency Score" for the Brain's output. If the Brain's
// predicted trajectory violates laws of motion or diverges from the
// Go-native SGP4/J2 propagator beyond tolerance, the system defaults to
// the "Sovereign Safe" Go logic.
//
// Strategic Purpose:
//   - Prevents AI "hallucinations" during sensor jamming/EW
//   - Detects "Manifold Breach" (aerodynamically impossible maneuvers)
//   - Maintains commander trust via transparent arbitration
//
// Resolution Modes:
//   - SOVEREIGN_SAFE: Default to Go Physics (Brain unreliable)
//   - COGNITIVE_LEAD: Trust JAX Brain (high confidence, consistent)
//   - ANOMALY: Total disagreement (potential spoofing/manifold breach)
//
// References:
//   - ADITI 4.0 HF-PGFM Architecture
//   - iDEX May 4th Demo Requirements
//   - Bayesian Trust Theory for Safety-Critical Systems

package logic

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// ResolveMode indicates the arbitration decision
type ResolveMode string

const (
	// ModeSovereignSafe defaults to Go Physics when Brain is unreliable
	// This is the safest mode - never trust AI over physics
	ModeSovereignSafe ResolveMode = "SOVEREIGN_SAFE"

	// ModeCognitiveLead trusts the JAX Brain when predictions are
	// physically consistent and confidence is high
	ModeCognitiveLead ResolveMode = "COGNITIVE_LEAD"

	// ModeAnomalyAlert indicates total disagreement between models
	// Potential causes: sensor spoofing, manifold breach, or EW attack
	ModeAnomalyAlert ResolveMode = "ANOMALY"

	// Default tolerance for L2 residual between models (50 meters)
	DefaultTolerance = 50.0

	// Structural limits for "Manifold Breach" detection
	MaxSustainableG = 15.0   // Maximum G-force for HGV (aerodynamic limit)
	MaxThermalFlux  = 5000.0 // W/cm² (re-entry heating limit)
)

// ResolutionResult contains the arbitration decision and metadata
type ResolutionResult struct {
	SelectedPosition [3]float64  // The chosen position [km]
	Mode             ResolveMode // Resolution mode
	TrustScore       float64     // Current trust in Brain [0, 1]
	Residual         float64     // L2 distance between models [km]
	Consistency      float64     // Physical consistency score [0, 1]
	Timestamp        int64       // Unix nanoseconds
	Reason           string      // Human-readable explanation
}

// ConflictResolver is the "Supreme Court" of ADITI 4.0
// It arbitrates between Go-native physics and JAX-based AI
type ConflictResolver struct {
	mu              sync.RWMutex
	Tolerance       float64 // Max allowed L2 residual between models [km]
	ConfidenceAlpha float64 // Smoothing factor for trust scores (EMA)
	CurrentTrust    float64 // Current trust in JAX Brain [0, 1]

	// Historical tracking
	TrustHistory   []float64     // Trust score over time
	ResolutionLog  []ResolveMode // Mode history for analysis
	MaxHistorySize int

	// Performance metrics
	TotalResolutions   int64
	SovereignSafeCount int64
	CognitiveLeadCount int64
	AnomalyCount       int64

	// Last resolution time
	LastResolutionTime time.Time
}

// NewConflictResolver creates a new resolver with specified tolerance
func NewConflictResolver(tolerance float64) *ConflictResolver {
	if tolerance <= 0 {
		tolerance = DefaultTolerance
	}

	return &ConflictResolver{
		Tolerance:          tolerance,
		ConfidenceAlpha:    0.8, // EMA smoothing: 80% historical, 20% current
		CurrentTrust:       1.0, // Start with full trust (optimistic)
		TrustHistory:       make([]float64, 0, 1000),
		ResolutionLog:      make([]ResolveMode, 0, 1000),
		MaxHistorySize:     10000,
		LastResolutionTime: time.Now(),
	}
}

// Resolve decides which state to broadcast to the C2 HUD
// This is the core arbitration function - the "Judge"
func (cr *ConflictResolver) Resolve(
	goPos, goVel [3]float64, // Go-native physics (SGP4/J2)
	jaxPos, jaxVel [3]float64, // JAX brain prediction (SE3/MBPO)
	jaxConfidence float64, // Brain's self-reported confidence [0, 1]
	jaxEpistemicUnc float64, // Brain's epistemic uncertainty
) ResolutionResult {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	cr.TotalResolutions++
	cr.LastResolutionTime = time.Now()

	// 1. Calculate the Euclidean (L2) Residual between models
	residual := cr.calculateResidual(goPos, jaxPos)

	// 2. Check for "Manifold Breach" (aerodynamically impossible)
	breachDetected, breachReason := cr.detectManifoldBreach(goPos, goVel, jaxPos, jaxVel)

	// 3. Calculate Physical Consistency Score
	// How well does JAX prediction match Go physics?
	consistency := cr.calculateConsistency(residual, jaxEpistemicUnc)

	// 4. Bayesian Update of Trust Score
	// We penalize trust if:
	//   - Residual is high (models disagree)
	//   - Epistemic uncertainty is high (Brain unsure)
	//   - Manifold breach detected (physically impossible)
	trustUpdate := jaxConfidence * consistency
	if breachDetected {
		trustUpdate *= 0.1 // Severe penalty for physics violations
	}

	// Exponential moving average: trust = α*trust + (1-α)*update
	cr.CurrentTrust = cr.ConfidenceAlpha*cr.CurrentTrust + (1.0-cr.ConfidenceAlpha)*trustUpdate

	// Clamp trust to [0, 1]
	if cr.CurrentTrust < 0 {
		cr.CurrentTrust = 0
	}
	if cr.CurrentTrust > 1 {
		cr.CurrentTrust = 1
	}

	// Record history
	cr.recordTrust(cr.CurrentTrust)

	// 5. The Arbiter Decision
	var result ResolutionResult

	if breachDetected {
		// PHYSICS ANOMALY: Brain predicts impossible maneuver
		// This could be:
		//   - Sensor spoofing (Electronic Warfare)
		//   - Manifold breach (target doing 50G turn)
		//   - Brain hallucination (high uncertainty + high G)
		cr.AnomalyCount++
		cr.recordMode(ModeAnomalyAlert)

		result = ResolutionResult{
			SelectedPosition: goPos,
			Mode:             ModeAnomalyAlert,
			TrustScore:       cr.CurrentTrust,
			Residual:         residual,
			Consistency:      consistency,
			Timestamp:        time.Now().UnixNano(),
			Reason:           fmt.Sprintf("MANIFOLD BREACH: %s | Brain hallucinating physics violation", breachReason),
		}

	} else if cr.CurrentTrust < 0.4 {
		// LOW TRUST: Brain is unreliable or diverging wildly
		// Default to sovereign physics for safety
		cr.SovereignSafeCount++
		cr.recordMode(ModeSovereignSafe)

		result = ResolutionResult{
			SelectedPosition: goPos,
			Mode:             ModeSovereignSafe,
			TrustScore:       cr.CurrentTrust,
			Residual:         residual,
			Consistency:      consistency,
			Timestamp:        time.Now().UnixNano(),
			Reason:           fmt.Sprintf("SOVEREIGN SAFE: Trust=%.2f below threshold | Residual=%.1fkm", cr.CurrentTrust, residual),
		}

	} else if residual > cr.Tolerance*5 {
		// MASSIVE DISAGREEMENT: Potential sensor spoofing
		// Even if trust is high, models are 250km apart - something is wrong
		cr.AnomalyCount++
		cr.recordMode(ModeAnomalyAlert)

		result = ResolutionResult{
			SelectedPosition: goPos,
			Mode:             ModeAnomalyAlert,
			TrustScore:       cr.CurrentTrust,
			Residual:         residual,
			Consistency:      consistency,
			Timestamp:        time.Now().UnixNano(),
			Reason:           fmt.Sprintf("ANOMALY: Models diverged %.1fkm | Potential EW attack", residual),
		}

	} else {
		// COGNITIVE LEAD: Trust the JAX Brain
		// High confidence, physically consistent, models agree
		cr.CognitiveLeadCount++
		cr.recordMode(ModeCognitiveLead)

		result = ResolutionResult{
			SelectedPosition: jaxPos,
			Mode:             ModeCognitiveLead,
			TrustScore:       cr.CurrentTrust,
			Residual:         residual,
			Consistency:      consistency,
			Timestamp:        time.Now().UnixNano(),
			Reason:           fmt.Sprintf("COGNITIVE LEAD: Trust=%.2f | JAX imagination active", cr.CurrentTrust),
		}
	}

	return result
}

// calculateResidual computes L2 distance between Go and JAX positions
func (cr *ConflictResolver) calculateResidual(goPos, jaxPos [3]float64) float64 {
	residual := 0.0
	for i := 0; i < 3; i++ {
		diff := goPos[i] - jaxPos[i]
		residual += diff * diff
	}
	return math.Sqrt(residual)
}

// detectManifoldBreach checks if JAX prediction violates physics laws
// Returns true if maneuver is aerodynamically impossible
func (cr *ConflictResolver) detectManifoldBreach(
	goPos, goVel [3]float64,
	jaxPos, jaxVel [3]float64,
) (bool, string) {
	// Calculate acceleration implied by JAX prediction
	// a = (v_jax - v_go) / Δt (assuming 10s propagation step)
	dt := 10.0 // seconds

	acc := [3]float64{
		(jaxVel[0] - goVel[0]) / dt,
		(jaxVel[1] - goVel[1]) / dt,
		(jaxVel[2] - goVel[2]) / dt,
	}

	accMagnitude := math.Sqrt(acc[0]*acc[0] + acc[1]*acc[1] + acc[2]*acc[2])

	// Convert to G-forces (1 G = 9.81 m/s² = 0.00981 km/s²)
	gForces := accMagnitude / 0.00981

	// Check against structural limits
	if gForces > MaxSustainableG {
		return true, fmt.Sprintf("%.1fG exceeds %dG structural limit", gForces, int(MaxSustainableG))
	}

	// Check for instantaneous velocity changes (impossible without magic)
	velChange := cr.calculateResidual(goVel, jaxVel)
	if velChange > 2.0 { // 2 km/s instantaneous change is impossible
		return true, fmt.Sprintf("%.1f km/s instantaneous Δv impossible", velChange)
	}

	// Check altitude bounds (can't go underground or to moon)
	altitude := math.Sqrt(jaxPos[0]*jaxPos[0] + jaxPos[1]*jaxPos[1] + jaxPos[2]*jaxPos[2])
	if altitude < 6371 { // Below Earth surface
		return true, fmt.Sprintf("Altitude %.1f km below Earth surface", altitude)
	}
	if altitude > 100000 { // Beyond Lunar orbit
		return true, fmt.Sprintf("Altitude %.1f km beyond operational envelope", altitude)
	}

	return false, ""
}

// calculateConsistency computes physical consistency score [0, 1]
func (cr *ConflictResolver) calculateConsistency(residual, epistemicUnc float64) float64 {
	// Consistency decreases as residual increases
	residualFactor := 1.0 / (1.0 + (residual / cr.Tolerance))

	// Consistency decreases as epistemic uncertainty increases
	uncertaintyFactor := 1.0 - math.Min(epistemicUnc, 1.0)

	// Combined consistency (geometric mean)
	consistency := math.Sqrt(residualFactor * uncertaintyFactor)

	return consistency
}

// recordTrust adds trust score to history (with circular buffer)
func (cr *ConflictResolver) recordTrust(trust float64) {
	cr.TrustHistory = append(cr.TrustHistory, trust)
	if len(cr.TrustHistory) > cr.MaxHistorySize {
		cr.TrustHistory = cr.TrustHistory[1:]
	}
}

// recordMode adds resolution mode to history
func (cr *ConflictResolver) recordMode(mode ResolveMode) {
	cr.ResolutionLog = append(cr.ResolutionLog, mode)
	if len(cr.ResolutionLog) > cr.MaxHistorySize {
		cr.ResolutionLog = cr.ResolutionLog[1:]
	}
}

// GetCurrentTrust returns the current trust score (thread-safe)
func (cr *ConflictResolver) GetCurrentTrust() float64 {
	cr.mu.RLock()
	defer cr.mu.RUnlock()
	return cr.CurrentTrust
}

// GetStatistics returns resolution statistics (thread-safe)
func (cr *ConflictResolver) GetStatistics() map[string]interface{} {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	total := cr.TotalResolutions
	if total == 0 {
		total = 1 // Avoid divide by zero
	}

	return map[string]interface{}{
		"total_resolutions":  cr.TotalResolutions,
		"sovereign_safe_pct": float64(cr.SovereignSafeCount) / float64(total) * 100,
		"cognitive_lead_pct": float64(cr.CognitiveLeadCount) / float64(total) * 100,
		"anomaly_pct":        float64(cr.AnomalyCount) / float64(total) * 100,
		"current_trust":      cr.CurrentTrust,
		"tolerance_km":       cr.Tolerance,
		"trust_history_len":  len(cr.TrustHistory),
	}
}

// GetResolutionLog returns recent resolution modes for analysis
func (cr *ConflictResolver) GetResolutionLog(n int) []ResolveMode {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	if n > len(cr.ResolutionLog) {
		n = len(cr.ResolutionLog)
	}

	// Return last n modes (newest first)
	result := make([]ResolveMode, n)
	for i := 0; i < n; i++ {
		result[i] = cr.ResolutionLog[len(cr.ResolutionLog)-1-i]
	}
	return result
}

// ResetTrust resets the trust score to maximum (e.g., after manual intervention)
func (cr *ConflictResolver) ResetTrust() {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	cr.CurrentTrust = 1.0
	cr.TrustHistory = cr.TrustHistory[:0]
}

// SetTolerance updates the tolerance threshold dynamically
func (cr *ConflictResolver) SetTolerance(tolerance float64) {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	if tolerance > 0 {
		cr.Tolerance = tolerance
	}
}

// String returns human-readable resolver status
func (cr *ConflictResolver) String() string {
	stats := cr.GetStatistics()
	return fmt.Sprintf(
		"ConflictResolver[Trust=%.2f | Mode=%s | ResidualTol=%.1fkm | Resolutions=%d]",
		stats["current_trust"],
		cr.GetLastMode(),
		stats["tolerance_km"],
		stats["total_resolutions"],
	)
}

// GetLastMode returns the most recent resolution mode
func (cr *ConflictResolver) GetLastMode() ResolveMode {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	if len(cr.ResolutionLog) == 0 {
		return ModeSovereignSafe // Default
	}
	return cr.ResolutionLog[len(cr.ResolutionLog)-1]
}
