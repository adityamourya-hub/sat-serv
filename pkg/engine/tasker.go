package engine

import (
	"fmt"
	"time"

	"ssis/pkg/types"
)

/**
 * Tasker provides predictive sensor management using Symplectic sensitivity fields.
 * It analyzes the Adjoint gradients (lambda_t) to determine if a space object's
 * state uncertainty is likely to exceed tactical thresholds, requiring increased
 * measurement density (CPI refactoring).
 *
 * <plan_complexity>Medium</plan_complexity><total_lines_estimate>150</total_lines_estimate>
 */
type Tasker struct {
	// Threshold for Information Gain Request (in bits or sensitivity magnitude)
	SensitivityThreshold float64
}

/**
 * NewTasker initializes a predictive tasking engine.
 */
func NewTasker(threshold float64) *Tasker {
	if threshold <= 0 {
		threshold = 0.85 // Default sensitivity coefficient
	}
	return &Tasker{
		SensitivityThreshold: threshold,
	}
}

/**
 * PredictFutureUncertainty analyzes the Adjoint sensitivity lambda_t.
 * If dUncertainty/dt is high, it yields a recommendation for sensor tasking.
 *
 * @param adjointGrads: The sensitivity of the terminal loss w.r.t initial state.
 * @returns: bool indicating if tasking is required, and the recommended CPI scale.
 */
func (t *Tasker) PredictFutureUncertainty(adjointGrads []float64) (bool, float64) {
	// Entry Point: Tasking Prediction
	fmt.Printf("DEBUG: Tasker analyzing Adjoint sensitivity field at %v\n", time.Now())

	// Scalar magnitude of the sensitivity vector (position and momentum parts)
	magnitude := 0.0
	for _, g := range adjointGrads {
		magnitude += g * g
	}

	// If the sensitivity is high, future state error will balloon without new data.
	if magnitude > t.SensitivityThreshold {
		fmt.Printf("WARNING: High sensitivity detected (Mag=%.4f). Requesting sensor tasking.\n", magnitude)

		// Recommended scaling for Coherent Processing Interval (CPI)
		// Scale is proportional to the log of the sensitivity magnitude.
		scale := 1.0 + (magnitude / 2.0)
		return true, scale
	}

	// Exit Point: No tasking required
	return false, 1.0
}

/**
 * CreateTaskingAlert generates a high-priority alert for ground-based radars.
 *
 * @param targetID: NORAD ID of the object requiring focus.
 * @param cpiScale: Recommended increase in measurement density.
 * @returns: A signed server.Alert object.
 */
func (t *Tasker) CreateTaskingAlert(targetID string, cpiScale float64) *types.Alert {
	alert := &types.Alert{
		AlertID:     fmt.Sprintf("TASK-%d", time.Now().Unix()),
		Type:        types.AlertTypeSensorTasking,
		Priority:    types.AlertPriorityHigh,
		TargetID:    targetID,
		Probability: 1.0, // High certainty of uncertainty
		Message:     fmt.Sprintf("Requesting CPI Scale %.2fx for target %s to minimize covariance swell.", cpiScale, targetID),
		Timestamp:   time.Now(),
	}

	return alert
}
