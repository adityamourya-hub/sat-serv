// Package onnx provides Axiom-Zero ONNX model integration for SSIS.
//
// This file integrates three key Axiom-Zero models:
//   1. rKAN Residual Covariance Head - Physics-informed uncertainty quantification
//   2. Delta-rKAN Maneuver Attributor - Symbolic maneuver detection
//   3. POL Transformer - Pattern-of-Life behavior classification
//
// The integration follows the jax2tf→ONNX export path for air-gap compatibility.
// When ONNX runtime is unavailable, stub implementations provide graceful fallback.
//
//go:build onnx

package onnx

import (
	"fmt"
	"path/filepath"
	"sync"
	"time"

	"ssis/pkg/catalog"
	"ssis/pkg/physics"
)

/**
 * AxiomZeroEngine provides unified access to all Axiom-Zero ONNX models.
 *
 * This is the primary integration point between the Python/JAX research substrate
 * and the Go operational core. Models are exported via jax2tf→ONNX and loaded
 * at runtime for inference.
 */
type AxiomZeroEngine struct {
	mu sync.RWMutex

	// Individual model instances
	CovarianceModel   *RKANCovarianceModel
	ManeuverModel     *DeltaRKANAttributorModel
	BehaviorModel     *POLTransformerModel

	// Configuration
	modelDir    string
	enabled     bool
	loadedAt    time.Time
	loadErrors  []string
}

// AxiomZeroConfig configures the Axiom-Zero engine.
type AxiomZeroConfig struct {
	ModelDir string // Directory containing ONNX models

	// Model-specific paths (relative to ModelDir or absolute)
	CovarianceModelPath   string
	ManeuverModelPath     string
	BehaviorModelPath     string

	// Inference parameters
	ManeuverTimesteps     int     // History length for maneuver detection (default 20)
	BehaviorSeqLen        int     // Sequence length for POL analysis (default 50)
	ManeuverThreshold     float64 // Confidence threshold for maneuver detection

	// Feature flags
	EnableCovariance   bool
	EnableManeuver     bool
	EnableBehavior     bool
}

// DefaultAxiomZeroConfig returns sensible defaults.
func DefaultAxiomZeroConfig(modelDir string) AxiomZeroConfig {
	return AxiomZeroConfig{
		ModelDir:            modelDir,
		CovarianceModelPath: "rkan_residual_covariance.onnx",
		ManeuverModelPath:   "delta_rkan_attributor.onnx",
		BehaviorModelPath:   "pol_transformer.onnx",
		ManeuverTimesteps:   20,
		BehaviorSeqLen:      50,
		ManeuverThreshold:   0.7,
		EnableCovariance:    true,
		EnableManeuver:      true,
		EnableBehavior:      true,
	}
}

// NewAxiomZeroEngine creates and initializes the Axiom-Zero engine.
func NewAxiomZeroEngine(config AxiomZeroConfig) (*AxiomZeroEngine, error) {
	engine := &AxiomZeroEngine{
		modelDir:   config.ModelDir,
		enabled:    false,
		loadErrors: []string{},
	}

	// Load covariance model
	if config.EnableCovariance && config.CovarianceModelPath != "" {
		covPath := config.CovarianceModelPath
		if !filepath.IsAbs(covPath) {
			covPath = filepath.Join(config.ModelDir, covPath)
		}

		model, err := NewRKANCovarianceModel(covPath)
		if err != nil {
			engine.loadErrors = append(engine.loadErrors,
				fmt.Sprintf("covariance model: %v", err))
		} else {
			engine.CovarianceModel = model
		}
	}

	// Load maneuver model
	if config.EnableManeuver && config.ManeuverModelPath != "" {
		manPath := config.ManeuverModelPath
		if !filepath.IsAbs(manPath) {
			manPath = filepath.Join(config.ModelDir, manPath)
		}

		model, err := NewDeltaRKANAttributorModel(manPath, config.ManeuverTimesteps)
		if err != nil {
			engine.loadErrors = append(engine.loadErrors,
				fmt.Sprintf("maneuver model: %v", err))
		} else {
			model.SetConfidenceThreshold(config.ManeuverThreshold)
			engine.ManeuverModel = model
		}
	}

	// Load behavior model
	if config.EnableBehavior && config.BehaviorModelPath != "" {
		behPath := config.BehaviorModelPath
		if !filepath.IsAbs(behPath) {
			behPath = filepath.Join(config.ModelDir, behPath)
		}

		model, err := NewPOLTransformerModel(behPath, config.BehaviorSeqLen, 8)
		if err != nil {
			engine.loadErrors = append(engine.loadErrors,
				fmt.Sprintf("behavior model: %v", err))
		} else {
			engine.BehaviorModel = model
		}
	}

	// Mark as enabled if at least one model loaded successfully
	if engine.CovarianceModel != nil || engine.ManeuverModel != nil || engine.BehaviorModel != nil {
		engine.enabled = true
		engine.loadedAt = time.Now()
	}

	return engine, nil
}

// IsEnabled returns true if at least one model is loaded and functional.
func (e *AxiomZeroEngine) IsEnabled() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.enabled
}

// LoadErrors returns any errors encountered during model loading.
func (e *AxiomZeroEngine) LoadErrors() []string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return append([]string{}, e.loadErrors...)
}

// ModelStatus returns the loading status of each model.
func (e *AxiomZeroEngine) ModelStatus() map[string]bool {
	e.mu.RLock()
	defer e.mu.RUnlock()

	return map[string]bool{
		"covariance": e.CovarianceModel != nil,
		"maneuver":   e.ManeuverModel != nil,
		"behavior":   e.BehaviorModel != nil,
	}
}

// GetPhysicsInformedCovariance returns covariance inflation for an object.
// This integrates with the collision package to provide learned uncertainty.
func (e *AxiomZeroEngine) GetPhysicsInformedCovariance(
	state physics.State,
	environmentalFactors map[string]float64,
) (RKANCovarianceOutput, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.CovarianceModel == nil {
		return RKANCovarianceOutput{}, fmt.Errorf("covariance model not loaded")
	}

	// Build input from state
	input := RKANCovarianceInput{
		PositionX: state.Pos[0],
		PositionY: state.Pos[1],
		PositionZ: state.Pos[2],
		VelocityX: state.Vel[0],
		VelocityY: state.Vel[1],
		VelocityZ: state.Vel[2],
	}

	// Add environmental factors if provided
	if f107, ok := environmentalFactors["f107"]; ok {
		input.SolarFluxF107 = f107
	}
	if ap, ok := environmentalFactors["ap"]; ok {
		input.GeomagneticAp = ap
	}

	// Compute orbital elements for additional features
	// Simplified calculations
	r := []float64{state.Pos[0], state.Pos[1], state.Pos[2]}
	v := []float64{state.Vel[0], state.Vel[1], state.Vel[2]}

	// Semi-major axis (vis-viva)
	rMag := 0.0
	for _, x := range r {
		rMag += x * x
	}
	rMag = sqrt(rMag)

	vMag := 0.0
	for _, x := range v {
		vMag += x * x
	}
	vMag = sqrt(vMag)

	mu := 398600.4418 // km³/s²
	sma := 1.0 / (2.0/rMag - vMag*vMag/mu)

	input.SemiMajorAxis = sma
	// Eccentricity and other elements would need full orbital element computation

	return e.CovarianceModel.Predict(input)
}

// sqrt is a helper for square root.
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Newton's method
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// DetectAndAttributeManeuver analyzes state history and returns maneuver details.
func (e *AxiomZeroEngine) DetectAndAttributeManeuver(
	objectID int,
	states []physics.State,
) (DeltaRKANOutput, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.ManeuverModel == nil {
		return DeltaRKANOutput{}, fmt.Errorf("maneuver model not loaded")
	}

	history := StateHistory{
		ObjectID: objectID,
		States:   states,
	}

	return e.ManeuverModel.Predict(history)
}

// ClassifyBehavior analyzes an object's state sequence and returns behavior class.
func (e *AxiomZeroEngine) ClassifyBehavior(
	objectID int,
	states []physics.State,
) (POLTransformerOutput, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.BehaviorModel == nil {
		return POLTransformerOutput{}, fmt.Errorf("behavior model not loaded")
	}

	sequence := StateSequence{
		ObjectID: objectID,
		States:   states,
		Metadata: make(map[string]string),
	}

	return e.BehaviorModel.Predict(sequence)
}

// ProcessCatalogObjects runs POL analysis on catalog objects.
func (e *AxiomZeroEngine) ProcessCatalogObjects(objects []catalog.ObjectRecord) ([]POLTransformerOutput, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.BehaviorModel == nil {
		return nil, fmt.Errorf("behavior model not loaded")
	}

	// Convert catalog objects to state sequences
	// This requires historical data which may not be in ObjectRecord
	// For now, return error indicating need for state history
	return nil, fmt.Errorf("state history required for POL analysis; use ProcessStateSequences instead")
}

// ProcessStateSequences runs POL analysis on state sequences.
func (e *AxiomZeroEngine) ProcessStateSequences(sequences []StateSequence) ([]POLTransformerOutput, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.BehaviorModel == nil {
		return nil, fmt.Errorf("behavior model not loaded")
	}

	return e.BehaviorModel.PredictBatch(sequences)
}

// Close releases all model resources.
func (e *AxiomZeroEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.enabled = false

	var errs []error

	if e.CovarianceModel != nil {
		if err := e.CovarianceModel.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if e.ManeuverModel != nil {
		if err := e.ManeuverModel.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if e.BehaviorModel != nil {
		if err := e.BehaviorModel.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing models: %v", errs)
	}

	return nil
}

// HealthCheck returns the health status of the engine.
func (e *AxiomZeroEngine) HealthCheck() map[string]interface{} {
	e.mu.RLock()
	defer e.mu.RUnlock()

	status := map[string]interface{}{
		"enabled":    e.enabled,
		"loaded_at":  e.loadedAt,
		"model_dir":  e.modelDir,
		"load_errors": e.loadErrors,
	}

	models := map[string]map[string]interface{}{}

	if e.CovarianceModel != nil {
		info := e.CovarianceModel.ModelInfo()
		models["covariance"] = map[string]interface{}{
			"id":      info.ModelID,
			"version": info.Version,
			"status":  "loaded",
		}
	} else {
		models["covariance"] = map[string]interface{}{"status": "not_loaded"}
	}

	if e.ManeuverModel != nil {
		info := e.ManeuverModel.ModelInfo()
		models["maneuver"] = map[string]interface{}{
			"id":      info.ModelID,
			"version": info.Version,
			"status":  "loaded",
		}
	} else {
		models["maneuver"] = map[string]interface{}{"status": "not_loaded"}
	}

	if e.BehaviorModel != nil {
		info := e.BehaviorModel.ModelInfo()
		models["behavior"] = map[string]interface{}{
			"id":      info.ModelID,
			"version": info.Version,
			"status":  "loaded",
		}
	} else {
		models["behavior"] = map[string]interface{}{"status": "not_loaded"}
	}

	status["models"] = models

	return status
}
