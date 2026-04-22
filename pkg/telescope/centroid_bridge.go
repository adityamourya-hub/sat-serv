// pkg/telescope/centroid_bridge.go
// Centroid-to-PHS Feedback Bridge
//
// Implements closed-loop feedback between MFBD vision output and AI-IMM EFE solver.
// The centroid observation feeds in as an Observation Prior to resolve epistemic
// ambiguity when PHS predictions deviate from camera observations.
//
// Logic flow:
//   MFBD → centroid (x,y) → Az/El → Observation Prior → EFE Conflict Metric → Intent inference
//
// When centroid deviates from PHS prediction, the Conflict Metric spikes, helping
// the system decide if the deviation is a maneuver or a sensor glitch.

package telescope

import (
	"fmt"
	"math"

	"ssis/pkg/ml/ai_imm"
	"ssis/pkg/physics"
)

// CentroidBridge bridges MFBD vision output to AI-IMM EFE solver
type CentroidBridge struct {
	visionEngine *VisionEngine
	EFESolver    *ai_imm.EFESolver

	// Observation prior state
	lastObsAz   float64
	lastObsEl   float64
	lastObsTime int64

	// Conflict tracking
	conflictHistory    []float64
	maxConflictSamples int
}

// NewCentroidBridge creates a new centroid feedback bridge
func NewCentroidBridge(visionEngine *VisionEngine, EFESolver *ai_imm.EFESolver) *CentroidBridge {
	return &CentroidBridge{
		visionEngine:       visionEngine,
		EFESolver:          EFESolver,
		conflictHistory:    make([]float64, 0, 100),
		maxConflictSamples: 100,
	}
}

// ProcessObservation processes MFBD geometry and feeds centroid to AI-IMM as Observation Prior
//
// Steps:
// 1. Extract centroid from reconstructed image
// 2. Transform pixel coordinates to Topocentric Az/El
// 3. Package as Observation Prior for EFE solver
// 4. Compute Conflict Metric if PHS prediction available
func (cb *CentroidBridge) ProcessObservation(geometry *ObjectGeometry, predictedAz, predictedEl float64) (conflict float64, err error) {
	if cb.visionEngine == nil {
		return 0.0, fmt.Errorf("vision engine not initialized")
	}

	// 1. Extract centroid
	centroidX, centroidY := cb.visionEngine.GetCentroid(geometry)

	// 2. Transform to Az/El
	obsAz, obsEl := cb.visionEngine.PixelToAzEl(centroidX, centroidY)

	// Update observation state
	cb.lastObsAz = obsAz
	cb.lastObsEl = obsEl
	cb.lastObsTime = geometry.Timestamp

	// 3. Compute Conflict Metric if PHS prediction available
	if predictedAz >= 0 && predictedEl >= 0 {
		conflict = cb.computeConflictMetric(predictedAz, predictedEl, obsAz, obsEl)
		cb.trackConflict(conflict)
	}

	// 4. Feed as Observation Prior to EFE solver (if available)
	if cb.EFESolver != nil {
		// Create observation state
		obsState := &physics.State{
			// Note: This is a simplified observation state
			// In production, full ECI position would be reconstructed from Az/El + range
			EpochJD: float64(geometry.Timestamp)/86400.0 + 2440587.5,
		}

		// The centroid observation modifies the Adversarial Prior
		// by adding a "local" observation likelihood that competes with
		// the PHS prediction
		cb.updateObservationPrior(obsState, obsAz, obsEl)
	}

	return conflict, nil
}

// computeConflictMetric computes angular distance between PHS prediction and centroid observation
//
// Returns conflict metric in arcseconds
func (cb *CentroidBridge) computeConflictMetric(predAz, predEl, obsAz, obsEl float64) float64 {
	// Convert to radians
	predAzRad := predAz * math.Pi / 180.0
	predElRad := predEl * math.Pi / 180.0
	obsAzRad := obsAz * math.Pi / 180.0
	obsElRad := obsEl * math.Pi / 180.0

	// Unit vectors for predicted
	predX := math.Cos(predElRad) * math.Cos(predAzRad)
	predY := math.Cos(predElRad) * math.Sin(predAzRad)
	predZ := math.Sin(predElRad)

	// Unit vectors for observed
	obsX := math.Cos(obsElRad) * math.Cos(obsAzRad)
	obsY := math.Cos(obsElRad) * math.Sin(obsAzRad)
	obsZ := math.Sin(obsElRad)

	// Dot product
	dot := predX*obsX + predY*obsY + predZ*obsZ

	// Clamp to [-1, 1]
	if dot > 1.0 {
		dot = 1.0
	}
	if dot < -1.0 {
		dot = -1.0
	}

	// Angular separation in radians
	sepRad := math.Acos(dot)

	// Convert to arcseconds
	conflictArcsec := sepRad * 180.0 / math.Pi * 3600.0

	return conflictArcsec
}

// trackConflict maintains conflict history for anomaly detection
func (cb *CentroidBridge) trackConflict(conflict float64) {
	cb.conflictHistory = append(cb.conflictHistory, conflict)
	if len(cb.conflictHistory) > cb.maxConflictSamples {
		cb.conflictHistory = cb.conflictHistory[1:]
	}
}

// GetAverageConflict returns average conflict over history
func (cb *CentroidBridge) GetAverageConflict() float64 {
	if len(cb.conflictHistory) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, c := range cb.conflictHistory {
		sum += c
	}
	return sum / float64(len(cb.conflictHistory))
}

// IsConflictHigh checks if conflict metric indicates significant deviation
// Threshold: > 10 arcsec indicates potential maneuver or sensor glitch
func (cb *CentroidBridge) IsConflictHigh() bool {
	avgConflict := cb.GetAverageConflict()
	return avgConflict > 10.0 // 10 arcsec threshold
}

// updateObservationPrior updates the EFE solver's Adversarial Prior with centroid observation
//
// Centroid-to-PHS Feedback - wires camera to Active Inference IMM
// When camera observation deviates from PHS prediction, Conflict Metric spikes,
// helping system decide if deviation is maneuver or sensor glitch.
func (cb *CentroidBridge) updateObservationPrior(obsState *physics.State, obsAz, obsEl float64) {
	if cb.EFESolver == nil {
		fmt.Printf("[CENTROID] Observation Prior: Az=%.2f°, El=%.2f° (EFE solver not connected)\n", obsAz, obsEl)
		return
	}

	// Compute camera confidence based on observation quality
	// In production, this would use actual SNR from MFBD
	cameraConfidence := 0.85 // Default confidence

	// Set observation prior in EFE solver
	// This creates epistemic ambiguity that drives policy selection
	cb.EFESolver.SetObservationPrior(obsAz, obsEl, cameraConfidence)

	fmt.Printf("[CENTROID] Observation Prior set: Az=%.2f°, El=%.2f°, conf=%.2f\n",
		obsAz, obsEl, cameraConfidence)
}

// SetEFESolver connects the EFE solver for closed-loop feedback
// Call this after EFE solver is initialized to complete the wiring
func (cb *CentroidBridge) SetEFESolver(solver *ai_imm.EFESolver) {
	cb.EFESolver = solver
	fmt.Printf("[CENTROID] EFE solver connected for closed-loop feedback\n")
}

// GetObservation returns the last centroid observation
func (cb *CentroidBridge) GetObservation() (az, el float64, timestamp int64) {
	return cb.lastObsAz, cb.lastObsEl, cb.lastObsTime
}

// Reset clears observation state (for new target acquisition)
func (cb *CentroidBridge) Reset() {
	cb.lastObsAz = 0
	cb.lastObsEl = 0
	cb.lastObsTime = 0
	cb.conflictHistory = make([]float64, 0, cb.maxConflictSamples)
}
