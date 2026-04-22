// pkg/onnx/model_registry.go - Centralized Model Management
//
// Manages ONNX model lifecycle with Hessian Hash validation.
// Provides thread-safe access to loaded models for the 100Hz reflex loop.

package onnx

import (
	"crypto/sha256"
	"fmt"
	"sync"
)

// ONNXSession is a placeholder for the actual ONNX runtime session.
// The actual implementation uses pkg/ml/onnx.NativeInferrer with onnxruntime-go.
// This stub exists for API compatibility during the transition.
type ONNXSession struct {
	modelPath string
	device    string
}

// NewONNXSession creates a new ONNX session (placeholder).
// In production, this would initialize the actual onnxruntime-go session.
func NewONNXSession(modelPath, device string) (*ONNXSession, error) {
	return &ONNXSession{
		modelPath: modelPath,
		device:    device,
	}, nil
}

// Close releases the ONNX session resources (placeholder).
func (s *ONNXSession) Close() error {
	return nil
}

// ModelType represents the type of ML model
type ModelType string

const (
	ModelTypeHamiltonian ModelType = "hamiltonian" // HamiltonianNet for energy conservation
	ModelTypeTransition  ModelType = "transition"  // TransitionNet for IMM matrices
	ModelTypeSINDy       ModelType = "sindy"       // SINDyNet for latent force discovery
)

// ModelMetadata contains model configuration and validation data
type ModelMetadata struct {
	Name        string  // Human-readable name
	Path        string  // Path to .onnx file
	HessianHash string  // SHA-256 hash of model's Hessian matrix (hex string)
	InputShape  []int64 // Input tensor shape
	OutputShape []int64 // Output tensor shape
	ModelType   ModelType
	Device      string // CPU|GPU|ANE (Apple Neural Engine)
	Enabled     bool   // Whether model is active
}

// ModelRegistry manages all registered ONNX models
type ModelRegistry struct {
	models map[string]*ONNXSession // name -> session
	meta   map[string]ModelMetadata
	mu     sync.RWMutex
}

// Global model registry instance
var globalRegistry = &ModelRegistry{
	models: make(map[string]*ONNXSession),
	meta:   make(map[string]ModelMetadata),
}

// RegisterModel loads and registers an ONNX model
func RegisterModel(metadata ModelMetadata) error {
	if !metadata.Enabled {
		return nil // Skip disabled models
	}

	// Load ONNX session
	session, err := NewONNXSession(metadata.Path, metadata.Device)
	if err != nil {
		return fmt.Errorf("failed to load model %s: %w", metadata.Name, err)
	}

	// Validate Hessian hash if provided
	if metadata.HessianHash != "" {
		if err := ValidateHessianHash(session, metadata.HessianHash); err != nil {
			session.Close()
			return fmt.Errorf("Hessian hash validation failed for %s: %w", metadata.Name, err)
		}
	}

	globalRegistry.mu.Lock()
	defer globalRegistry.mu.Unlock()
	globalRegistry.models[metadata.Name] = session
	globalRegistry.meta[metadata.Name] = metadata

	return nil
}

// GetModel retrieves a model session by name
func GetModel(name string) (*ONNXSession, error) {
	globalRegistry.mu.RLock()
	defer globalRegistry.mu.RUnlock()

	session, ok := globalRegistry.models[name]
	if !ok {
		return nil, fmt.Errorf("model not found: %s", name)
	}

	return session, nil
}

// GetMetadata retrieves model metadata by name
func GetMetadata(name string) (ModelMetadata, error) {
	globalRegistry.mu.RLock()
	defer globalRegistry.mu.RUnlock()

	meta, ok := globalRegistry.meta[name]
	if !ok {
		return ModelMetadata{}, fmt.Errorf("model metadata not found: %s", name)
	}

	return meta, nil
}

// ListModels returns all registered model names
func ListModels() []string {
	globalRegistry.mu.RLock()
	defer globalRegistry.mu.RUnlock()

	names := make([]string, 0, len(globalRegistry.models))
	for name := range globalRegistry.models {
		names = append(names, name)
	}
	return names
}

// Shutdown closes all model sessions
func Shutdown() {
	globalRegistry.mu.Lock()
	defer globalRegistry.mu.Unlock()

	for _, session := range globalRegistry.models {
		_ = session.Close()
	}
	globalRegistry.models = make(map[string]*ONNXSession)
}

// ValidateHessianHash computes and validates the Hessian hash
// This is a simplified version - in production, would extract actual Hessian from model weights
func ValidateHessianHash(session *ONNXSession, expectedHash string) error {
	// TODO: Implement actual Hessian extraction from ONNX model
	// For now, compute hash of model file path as placeholder
	h := sha256.New()
	h.Write([]byte(session.modelPath))
	computedHash := fmt.Sprintf("%x", h.Sum(nil))

	if computedHash != expectedHash {
		return fmt.Errorf("hash mismatch: computed %s, expected %s", computedHash, expectedHash)
	}

	return nil
}
