package onnx

import (
	"fmt"
	"path/filepath"
	"sync"
	"time"
)

// ModelSlot defines a registered model in the session pool.
type ModelSlot struct {
	ModelID     string
	ModelPath   string
	Version     string
	Session     *Session
	Enabled     bool
	LoadedAt    time.Time
	Description string
	InputShape  []int
	OutputShape []int
}

// SessionPool manages multiple ONNX inference sessions.
type SessionPool struct {
	mu          sync.RWMutex
	sessions    map[string]*ModelSlot // modelID -> ModelSlot
	registry    *VersionRegistry
	modelDir    string
	initialized bool
}

// NewSessionPool creates a new session pool with the given model directory.
func NewSessionPool(modelDir string) *SessionPool {
	return &SessionPool{
		sessions:    make(map[string]*ModelSlot),
		registry:    NewVersionRegistry(),
		modelDir:    modelDir,
		initialized: false,
	}
}

// RegisterModel registers a new model slot in the pool.
func (p *SessionPool) RegisterModel(modelID, modelPath, version string, inputShape, outputShape []int, description string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if _, exists := p.sessions[modelID]; exists {
		return fmt.Errorf("model %s already registered", modelID)
	}

	fullPath := filepath.Join(p.modelDir, modelPath)

	// Create session config
	config := DefaultSessionConfig(modelID, fullPath, version)
	config.EnableLogging = true

	// For now, use mock inferrer - will be replaced with real ONNX inferrer
	// when onnxruntime-go is integrated
	inferrer := NewMockInferrer(modelID, version)
	session := NewSession(config, inferrer)

	slot := &ModelSlot{
		ModelID:     modelID,
		ModelPath:   fullPath,
		Version:     version,
		Session:     session,
		Enabled:     true,
		LoadedAt:    time.Now(),
		Description: description,
		InputShape:  inputShape,
		OutputShape: outputShape,
	}

	p.sessions[modelID] = slot

	// Register in version registry
	modelVersion := ModelVersion{
		ID:          version,
		ModelID:     modelID,
		ModelPath:   fullPath,
		DeployedAt:  time.Now(),
		Description: description,
	}

	checksum, err := ComputeChecksum(fullPath)
	if err == nil {
		modelVersion.Checksum = checksum
	}

	p.registry.Register(modelVersion)

	return nil
}

// GetSession retrieves a session by model ID.
func (p *SessionPool) GetSession(modelID string) (*Session, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	slot, exists := p.sessions[modelID]
	if !exists {
		return nil, fmt.Errorf("model %s not found in pool", modelID)
	}

	if !slot.Enabled {
		return nil, fmt.Errorf("model %s is disabled", modelID)
	}

	return slot.Session, nil
}

// GetModelSlot retrieves a model slot by model ID.
func (p *SessionPool) GetModelSlot(modelID string) (*ModelSlot, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	slot, exists := p.sessions[modelID]
	if !exists {
		return nil, fmt.Errorf("model %s not found in pool", modelID)
	}

	return slot, nil
}

// ListModels returns all registered model IDs.
func (p *SessionPool) ListModels() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	models := make([]string, 0, len(p.sessions))
	for modelID := range p.sessions {
		models = append(models, modelID)
	}
	return models
}

// EnableModel enables a model slot.
func (p *SessionPool) EnableModel(modelID string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	slot, exists := p.sessions[modelID]
	if !exists {
		return fmt.Errorf("model %s not found", modelID)
	}

	slot.Enabled = true
	return nil
}

// DisableModel disables a model slot.
func (p *SessionPool) DisableModel(modelID string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	slot, exists := p.sessions[modelID]
	if !exists {
		return fmt.Errorf("model %s not found", modelID)
	}

	slot.Enabled = false
	return nil
}

// HealthCheck returns health status of all registered models.
func (p *SessionPool) HealthCheck() map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()

	status := make(map[string]interface{})
	models := make(map[string]interface{})

	for modelID, slot := range p.sessions {
		modelInfo := map[string]interface{}{
			"id":           slot.ModelID,
			"version":      slot.Version,
			"path":         slot.ModelPath,
			"enabled":      slot.Enabled,
			"loaded_at":    slot.LoadedAt.Format(time.RFC3339),
			"description":  slot.Description,
			"input_shape":  slot.InputShape,
			"output_shape": slot.OutputShape,
		}

		if slot.Session != nil {
			metrics := slot.Session.Metrics()
			modelInfo["metrics"] = metrics
		}

		models[modelID] = modelInfo
	}

	status["enabled"] = true
	status["model_dir"] = p.modelDir
	status["models"] = models
	status["total_models"] = len(p.sessions)

	return status
}

// Close shuts down all sessions in the pool.
func (p *SessionPool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	var lastErr error
	for modelID, slot := range p.sessions {
		if slot.Session != nil {
			if err := slot.Session.Close(); err != nil {
				lastErr = fmt.Errorf("error closing %s: %w", modelID, err)
			}
		}
	}

	p.sessions = make(map[string]*ModelSlot)
	p.initialized = false
	return lastErr
}

// InitializeAxiomZeroModels registers the Axiom-Zero physics models.
func (p *SessionPool) InitializeAxiomZeroModels() error {
	// Register rKAN Covariance Model
	err := p.RegisterModel(
		"axiom_rkan_residual",
		"axiom_rkan_residual.onnx",
		"3.0.0",
		[]int{13}, // Input: 13 orbital features
		[]int{6},  // Output: 6 covariance inflation factors
		"rKAN residual covariance head for physics-informed uncertainty quantification",
	)
	if err != nil {
		return fmt.Errorf("failed to register axiom_rkan_residual: %w", err)
	}

	// Register Delta-rKAN Maneuver Attributor
	err = p.RegisterModel(
		"axiom_delta_rkan_attributor",
		"axiom_delta_rkan_attributor.onnx",
		"3.0.0",
		[]int{20, 6}, // Input: 20 timesteps, 6 state features
		[]int{8},     // Output: maneuver_prob(1) + delta_v(3) + maneuver_type(4)
		"Delta-rKAN maneuver attributor with symbolic LaTeX output",
	)
	if err != nil {
		return fmt.Errorf("failed to register axiom_delta_rkan_attributor: %w", err)
	}

	// Register POL Transformer
	err = p.RegisterModel(
		"axiom_pol_transformer",
		"axiom_pol_transformer.onnx",
		"3.0.0",
		[]int{50, 6}, // Input: 50 timesteps, 6 state features
		[]int{264},   // Output: behavior_class(8) + anomaly_score(1) + embedding(256)
		"Pattern-of-Life transformer for UCT classification",
	)
	if err != nil {
		return fmt.Errorf("failed to register axiom_pol_transformer: %w", err)
	}

	p.initialized = true
	return nil
}

// GetRegistry returns the version registry.
func (p *SessionPool) GetRegistry() *VersionRegistry {
	return p.registry
}
