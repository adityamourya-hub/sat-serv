package nasa42

/**
 * NASA 42 Offline Validation Runner
 *
 * Periodically validates sovereign propagation against NASA 42 ground truth.
 *
 * Strategy:
 * - Sample 1% of high-risk conjunctions for ground truth comparison
 * - Detect propagation drift before operational impact
 * - Build confidence for DPHNO transition
 *
 * Use Case:
 * 1. Sovereign stack detects high-risk conjunction (Pc > 1e-4)
 * 2. ValidationRunner samples 1% of these events
 * 3. Propagate both objects with NASA 42 via IPC
 * 4. Compare sovereign vs 42 results
 * 5. Log deviations >100m threshold
 * 6. Export metrics for Prometheus/Grafana monitoring
 */

import (
	"fmt"
	"math"
	"sync"
	"time"

	"ssis/pkg/physics"
)

// ValidationResult captures comparison between sovereign stack and NASA 42 propagation
type ValidationResult struct {
	Timestamp           time.Time
	ObjectID            string
	SovereignPosition   [3]float64 // km
	SovereignVelocity   [3]float64 // km/s
	FortyTwoPosition    [3]float64 // km
	FortyTwoVelocity    [3]float64 // km/s
	PositionErrorMeters float64
	VelocityErrorMmS    float64
	WithinTolerance     bool

	// DPHNO comparison (future)
	DPHNOPredicted   [3]float64
	DPHNOErrorMeters float64
}

// ValidationMetrics aggregates validation statistics
type ValidationMetrics struct {
	mu                  sync.Mutex
	TotalComparisons    int64
	WithinTolerance     int64
	MeanDeviationMeters float64
	MaxDeviationMeters  float64
	DPHNOAccuracy       float64 // % within tolerance
}

// ValidationRunner manages periodic validation against NASA 42
type ValidationRunner struct {
	client             *IPCClient
	enabled            bool
	sampleRate         float64 // 0.01 = 1%
	deviationThreshold float64 // 100.0 meters
	metrics            ValidationMetrics
	stopChan           chan struct{}
}

// NewValidationRunner creates a validation runner with NASA 42 client
func NewValidationRunner(client *IPCClient) *ValidationRunner {
	return &ValidationRunner{
		client:             client,
		enabled:            true,
		sampleRate:         0.01,  // Sample 1% of high-risk conjunctions
		deviationThreshold: 100.0, // 100m threshold
		stopChan:           make(chan struct{}),
	}
}

// ValidateConjunction validates a conjunction against NASA 42 ground truth
// Returns ValidationResult and error if validation fails
func (vr *ValidationRunner) ValidateConjunction(
	objectID string,
	primary physics.State,
	duration float64, // seconds
) (*ValidationResult, error) {
	if !vr.enabled {
		return nil, fmt.Errorf("validation runner disabled")
	}

	// Sample check: only validate sampleRate fraction
	// In production, use rand.Float64() < sampleRate
	// For now, always validate in test mode

	// Step 1: Propagate primary with NASA 42
	primaryPos := [3]float64{primary.Pos[0], primary.Pos[1], primary.Pos[2]}
	primaryVel := [3]float64{primary.Vel[0], primary.Vel[1], primary.Vel[2]}
	primaryQuat := [4]float64{1, 0, 0, 0} // Identity quaternion (simplified)

	// Assume 1000kg mass (should be passed from object catalog)
	primaryResult, err := vr.client.HandoffObject(
		objectID,
		primaryPos,
		primaryVel,
		primaryQuat,
		1000.0,
		duration,
	)
	if err != nil {
		return nil, fmt.Errorf("NASA 42 primary propagation failed: %v", err)
	}

	// Step 2: Calculate position error (meters)
	dx := (primary.Pos[0] - primaryResult.PosX) * 1000.0
	dy := (primary.Pos[1] - primaryResult.PosY) * 1000.0
	dz := (primary.Pos[2] - primaryResult.PosZ) * 1000.0
	positionError := math.Sqrt(dx*dx + dy*dy + dz*dz)

	// Step 3: Calculate velocity error (mm/s)
	dvx := (primary.Vel[0] - primaryResult.VelX) * 1000.0
	dvy := (primary.Vel[1] - primaryResult.VelY) * 1000.0
	dvz := (primary.Vel[2] - primaryResult.VelZ) * 1000.0
	velocityError := math.Sqrt(dvx*dvx + dvy*dvy + dvz*dvz)

	// Step 4: Check tolerance
	withinTolerance := positionError <= vr.deviationThreshold

	// Step 5: Build result
	result := &ValidationResult{
		Timestamp:           time.Now(),
		ObjectID:            objectID,
		SovereignPosition:   [3]float64{primary.Pos[0], primary.Pos[1], primary.Pos[2]},
		SovereignVelocity:   [3]float64{primary.Vel[0], primary.Vel[1], primary.Vel[2]},
		FortyTwoPosition:    [3]float64{primaryResult.PosX, primaryResult.PosY, primaryResult.PosZ},
		FortyTwoVelocity:    [3]float64{primaryResult.VelX, primaryResult.VelY, primaryResult.VelZ},
		PositionErrorMeters: positionError,
		VelocityErrorMmS:    velocityError,
		WithinTolerance:     withinTolerance,
	}

	// Step 6: Update metrics
	vr.updateMetrics(result)

	return result, nil
}

// updateMetrics aggregates validation statistics
func (vr *ValidationRunner) updateMetrics(result *ValidationResult) {
	vr.metrics.mu.Lock()
	defer vr.metrics.mu.Unlock()

	vr.metrics.TotalComparisons++

	if result.WithinTolerance {
		vr.metrics.WithinTolerance++
	}

	// Update mean deviation (running average)
	n := float64(vr.metrics.TotalComparisons)
	oldMean := vr.metrics.MeanDeviationMeters
	vr.metrics.MeanDeviationMeters = oldMean + (result.PositionErrorMeters-oldMean)/n

	// Update max deviation
	if result.PositionErrorMeters > vr.metrics.MaxDeviationMeters {
		vr.metrics.MaxDeviationMeters = result.PositionErrorMeters
	}
}

// GetMetrics returns current validation metrics
func (vr *ValidationRunner) GetMetrics() ValidationMetrics {
	vr.metrics.mu.Lock()
	defer vr.metrics.mu.Unlock()

	// Calculate accuracy percentage
	accuracy := 0.0
	if vr.metrics.TotalComparisons > 0 {
		accuracy = float64(vr.metrics.WithinTolerance) / float64(vr.metrics.TotalComparisons) * 100.0
	}

	return ValidationMetrics{
		TotalComparisons:    vr.metrics.TotalComparisons,
		WithinTolerance:     vr.metrics.WithinTolerance,
		MeanDeviationMeters: vr.metrics.MeanDeviationMeters,
		MaxDeviationMeters:  vr.metrics.MaxDeviationMeters,
		DPHNOAccuracy:       accuracy,
	}
}

// Enable enables the validation runner
func (vr *ValidationRunner) Enable() {
	vr.enabled = true
}

// Disable disables the validation runner
func (vr *ValidationRunner) Disable() {
	vr.enabled = false
}

// SetSampleRate configures the sampling rate (0.0-1.0)
func (vr *ValidationRunner) SetSampleRate(rate float64) {
	if rate >= 0.0 && rate <= 1.0 {
		vr.sampleRate = rate
	}
}

// SetDeviationThreshold configures the tolerance threshold in meters
func (vr *ValidationRunner) SetDeviationThreshold(threshold float64) {
	vr.deviationThreshold = threshold
}

// Close shuts down the validation runner
func (vr *ValidationRunner) Close() error {
	close(vr.stopChan)
	if vr.client != nil {
		return vr.client.Disconnect()
	}
	return nil
}
