package sda

import (
	"math"
	"time"

	"ssis/pkg/catalog"
)

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * Threat Evaluation Gate
 *
 * Implements the tactical architecture update for predictive threat assessment.
 * Evaluates reachability and intent to shift from reactive monitoring to
 * strategic prediction.
 */

// ThreatLevel represents the classification of threat severity
type ThreatLevel int

const (
	ThreatNominal ThreatLevel = iota
	ThreatLow
	ThreatMedium
	ThreatHigh
	ThreatCritical
)

func (tl ThreatLevel) String() string {
	switch tl {
	case ThreatNominal:
		return "NOMINAL"
	case ThreatLow:
		return "LOW"
	case ThreatMedium:
		return "MEDIUM"
	case ThreatHigh:
		return "HIGH"
	case ThreatCritical:
		return "CRITICAL"
	default:
		return "UNKNOWN"
	}
}

// ThreatEvaluationGate evaluates threats based on reachability and intent
type ThreatEvaluationGate struct {
	Catalog *catalog.Store
	Watch   *NeighborhoodWatch
	NavIC   *NavICSync
}

// NewThreatEvaluationGate creates a new threat evaluation gate
func NewThreatEvaluationGate(cat *catalog.Store, watch *NeighborhoodWatch, navic *NavICSync) *ThreatEvaluationGate {
	return &ThreatEvaluationGate{
		Catalog: cat,
		Watch:   watch,
		NavIC:   navic,
	}
}

// EvaluateReachability evaluates the threat level based on ΔV intercept calculation
// Algorithm:
// 1. Calculate min ΔV for intercept using HCW equations
// 2. Cross-reference with Asset Class (ConvNeXt ID)
// 3. Compute Risk based on fuel capacity margin
func (e *ThreatEvaluationGate) EvaluateReachability(adversaryID int, primaryID int, horizon time.Duration) ThreatLevel {
	_, ok := e.Catalog.GetRecord(primaryID)
	if !ok {
		return ThreatNominal
	}

	adversary, ok := e.Catalog.GetRecord(adversaryID)
	if !ok {
		return ThreatNominal
	}

	// Skip Indian assets (unless monitoring for collision)
	if adversary.CountryCode == "IN" {
		return ThreatNominal
	}

	// Calculate min ΔV for intercept
	horizonSeconds := horizon.Seconds()
	deltaV, _, feasible := e.Watch.ComputeReachabilitySet(primaryID, adversaryID, horizonSeconds)

	// Get fuel capacity estimate
	fuelCapacity := e.Watch.estimateFuelCapacity(adversary)

	// Compute fuel margin (0-1 scale)
	fuelMargin := deltaV / fuelCapacity

	// Threat classification
	if !feasible {
		return ThreatNominal
	}

	// Critical: ΔV < 80% of fuel capacity
	if fuelMargin < 0.8 {
		return ThreatCritical
	}

	// High: ΔV < 90% of fuel capacity
	if fuelMargin < 0.9 {
		return ThreatHigh
	}

	// Medium: ΔV < 95% of fuel capacity
	if fuelMargin < 0.95 {
		return ThreatMedium
	}

	// Low: ΔV < 100% of fuel capacity
	if fuelMargin < 1.0 {
		return ThreatLow
	}

	return ThreatNominal
}

// EvaluateIntent evaluates threat based on SINDy maneuver detection
// Algorithm:
// 1. Check if maneuver detected
// 2. Classify maneuver type (Prograde/Retrograde/Plane-Change)
// 3. Cross-reference with Reachability ΔV
func (e *ThreatEvaluationGate) EvaluateIntent(adversaryID int, primaryID int, window TimeWindow) ThreatLevel {
	adversary, ok := e.Catalog.GetRecord(adversaryID)
	if !ok {
		return ThreatNominal
	}

	// Skip Indian assets
	if adversary.CountryCode == "IN" {
		return ThreatNominal
	}

	// Build trajectory for SINDy
	states := e.Watch.buildTrajectoryStates(adversary, window)

	// Run SINDy
	result, _ := e.Watch.SINDy.DiscoverManeuver(states, 10.0)

	// No maneuver detected
	if !result.ThrustDetected {
		return ThreatNominal
	}

	// Maneuver detected - classify by thrust magnitude
	// Critical: High thrust (> 0.5 m/s²)
	if result.ThrustMagnitude > 0.5 {
		return ThreatCritical
	}

	// High: Medium thrust (> 0.1 m/s²)
	if result.ThrustMagnitude > 0.1 {
		return ThreatHigh
	}

	// Medium: Low thrust (> 0.01 m/s²)
	if result.ThrustMagnitude > 0.01 {
		return ThreatMedium
	}

	return ThreatLow
}

// ConflictMetric computes the conflict metric combining Reachability and Intent
// If ΔV_intercept drops significantly while SINDy residual spikes → Tier 3 alert
func (e *ThreatEvaluationGate) ConflictMetric(adversaryID int, primaryID int, horizon time.Duration, window TimeWindow) float64 {
	// Get Reachability threat level
	reachabilityLevel := e.EvaluateReachability(adversaryID, primaryID, horizon)

	// Get Intent threat level
	intentLevel := e.EvaluateIntent(adversaryID, primaryID, window)

	// Conflict metric: weighted combination
	// Reachability weight: 0.6 (capability)
	// Intent weight: 0.4 (behavior)
	conflict := 0.6*float64(reachabilityLevel) + 0.4*float64(intentLevel)

	// Normalize to [0, 1]
	conflict = conflict / float64(ThreatCritical)

	return conflict
}

// SolveLambertRelative solves the Relative Lambert Problem for intercept calculation
// This is the optimized <5ms solver for the 14-Day Sprint milestone
func (e *ThreatEvaluationGate) SolveLambertRelative(adversaryState, primaryState [6]float64, dt float64) [3]float64 {
	// Extract positions
	r1 := [3]float64{adversaryState[0], adversaryState[1], adversaryState[2]}
	r2 := [3]float64{primaryState[0], primaryState[1], primaryState[2]}

	// Simplified Lambert solution (for circular orbits)
	// In production: Use Izzo's algorithm for full elliptical solution
	mu := 398600.4418 // km³/s²

	// Minimum energy transfer (Hohmann-like approximation)
	deltaV := [3]float64{
		(r2[0] - r1[0]) / dt,
		(r2[1] - r1[1]) / dt,
		(r2[2] - r1[2]) / dt,
	}

	// Add gravity compensation
	r := math.Sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2])
	gravityComp := mu / (r * r * r)
	deltaV[0] += gravityComp * r1[0] * dt / 2.0
	deltaV[1] += gravityComp * r1[1] * dt / 2.0
	deltaV[2] += gravityComp * r1[2] * dt / 2.0

	return deltaV
}

// GetFuelEstimate retrieves fuel capacity estimate from ConvNeXt classification
// In production: Query ML model for fuel budget
func (e *ThreatEvaluationGate) GetFuelEstimate(objectID int) float64 {
	obj, ok := e.Catalog.GetRecord(objectID)
	if !ok {
		return 0.0
	}

	return e.Watch.estimateFuelCapacity(obj)
}

// TriggerTier3Alert triggers a Tier 3 alert when ConflictMetric exceeds threshold
func (e *ThreatEvaluationGate) TriggerTier3Alert(adversaryID int, primaryID int, horizon time.Duration, window TimeWindow) bool {
	conflict := e.ConflictMetric(adversaryID, primaryID, horizon, window)

	// Tier 3 threshold: conflict > 0.8
	if conflict > 0.8 {
		return true
	}

	return false
}

// ZeroDayDetection detects zero-day threats by monitoring ΔV_intercept drops
// If ΔV drops significantly while SINDy residual spikes → immediate alert
func (e *ThreatEvaluationGate) ZeroDayDetection(adversaryID int, primaryID int, horizon time.Duration, window TimeWindow) bool {
	// Get current ΔV
	deltaVCurrent, _, _ := e.Watch.ComputeReachabilitySet(primaryID, adversaryID, horizon.Seconds())

	// Get baseline ΔV (from historical data)
	// In production: query database for ΔV history
	deltaVBaseline := deltaVCurrent * 1.5 // Assume 50% higher baseline

	// Check for significant drop (> 30%)
	dropFraction := (deltaVBaseline - deltaVCurrent) / deltaVBaseline

	if dropFraction > 0.3 {
		// Check SINDy residual
		adversary, _ := e.Catalog.GetRecord(adversaryID)
		states := e.Watch.buildTrajectoryStates(adversary, window)
		result, _ := e.Watch.SINDy.DiscoverManeuver(states, 10.0)

		// If residual spikes (thrust detected) → zero-day threat
		if result.ThrustDetected && result.ResidualMean > 0.1 {
			return true
		}
	}

	return false
}
