// STUB: This file is a build stub. All functions return "not yet implemented" until the real backend is bound.
// Package onnx provides Axiom-Zero ONNX model integration for SSIS.
// This is the stub implementation for when ONNX runtime is not available.
//
//go:build !onnx

package onnx

import (
	"fmt"
	"time"

	"ssis/pkg/catalog"
	"ssis/pkg/physics"
)

// AxiomZeroEngine stub implementation.
type AxiomZeroEngine struct {
	modelDir   string
	enabled    bool
	loadErrors []string
}

// AxiomZeroConfig configures the Axiom-Zero engine.
type AxiomZeroConfig struct {
	ModelDir string
	CovarianceModelPath   string
	ManeuverModelPath     string
	BehaviorModelPath     string
	ManeuverTimesteps     int
	BehaviorSeqLen        int
	ManeuverThreshold     float64
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

// NewAxiomZeroEngine creates a stub engine.
func NewAxiomZeroEngine(config AxiomZeroConfig) (*AxiomZeroEngine, error) {
	return &AxiomZeroEngine{
		modelDir:   config.ModelDir,
		enabled:    false,
		loadErrors: []string{"ONNX runtime not available - stub mode"},
	}, nil
}

// IsEnabled returns false in stub mode.
func (e *AxiomZeroEngine) IsEnabled() bool {
	return false
}

// LoadErrors returns the stub error.
func (e *AxiomZeroEngine) LoadErrors() []string {
	return append([]string{}, e.loadErrors...)
}

// ModelStatus returns all false in stub mode.
func (e *AxiomZeroEngine) ModelStatus() map[string]bool {
	return map[string]bool{
		"covariance": false,
		"maneuver":   false,
		"behavior":   false,
	}
}

// GetPhysicsInformedCovariance returns error in stub mode.
func (e *AxiomZeroEngine) GetPhysicsInformedCovariance(
	state physics.State,
	environmentalFactors map[string]float64,
) (RKANCovarianceOutput, error) {
	return RKANCovarianceOutput{}, fmt.Errorf("Axiom-Zero engine disabled: ONNX runtime not available")
}

// DetectAndAttributeManeuver returns error in stub mode.
func (e *AxiomZeroEngine) DetectAndAttributeManeuver(
	objectID int,
	states []physics.State,
) (DeltaRKANOutput, error) {
	return DeltaRKANOutput{}, fmt.Errorf("Axiom-Zero engine disabled: ONNX runtime not available")
}

// ClassifyBehavior returns error in stub mode.
func (e *AxiomZeroEngine) ClassifyBehavior(
	objectID int,
	states []physics.State,
) (POLTransformerOutput, error) {
	return POLTransformerOutput{}, fmt.Errorf("Axiom-Zero engine disabled: ONNX runtime not available")
}

// ProcessCatalogObjects returns error in stub mode.
func (e *AxiomZeroEngine) ProcessCatalogObjects(objects []catalog.ObjectRecord) ([]POLTransformerOutput, error) {
	return nil, fmt.Errorf("Axiom-Zero engine disabled: ONNX runtime not available")
}

// ProcessStateSequences returns error in stub mode.
func (e *AxiomZeroEngine) ProcessStateSequences(sequences []StateSequence) ([]POLTransformerOutput, error) {
	return nil, fmt.Errorf("Axiom-Zero engine disabled: ONNX runtime not available")
}

// Close is a no-op in stub mode.
func (e *AxiomZeroEngine) Close() error {
	return nil
}

// HealthCheck returns stub status.
func (e *AxiomZeroEngine) HealthCheck() map[string]interface{} {
	return map[string]interface{}{
		"enabled":    false,
		"loaded_at":  time.Time{},
		"model_dir":  e.modelDir,
		"load_errors": e.loadErrors,
		"models": map[string]interface{}{
			"covariance": map[string]string{"status": "stub"},
			"maneuver":   map[string]string{"status": "stub"},
			"behavior":   map[string]string{"status": "stub"},
		},
	}
}
