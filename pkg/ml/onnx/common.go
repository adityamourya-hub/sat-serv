package onnx

import (
	"fmt"
	"math/rand"
	"time"
)

// ModelInfo contains metadata about a loaded ONNX model.
type ModelInfo struct {
	ModelID      string
	Version      string
	InputNames   []string
	OutputNames  []string
	LoadedAt     time.Time
	RequestCount int64
}

// MockInferrer is a test implementation of Inferrer that simulates ONNX inference.
type MockInferrer struct {
	modelInfo  ModelInfo
	latencyMS  float64
	errorRate  float64
	rng        *rand.Rand
	requestCnt int64
}

// NewMockInferrer creates a mock inferrer for testing.
func NewMockInferrer(modelID, version string) *MockInferrer {
	return &MockInferrer{
		modelInfo: ModelInfo{
			ModelID:     modelID,
			Version:     version,
			InputNames:  []string{"input_0", "input_1"},
			OutputNames: []string{"output_0"},
			LoadedAt:    time.Now(),
		},
		latencyMS: 5.0,
		errorRate: 0.0,
		rng:       rand.New(rand.NewSource(42)),
	}
}

// InferBatch processes a batch of requests with mock inference.
func (m *MockInferrer) InferBatch(requests []InferenceRequest) []InferenceResult {
	results := make([]InferenceResult, len(requests))

	for i, req := range requests {
		// Simulate latency
		if m.latencyMS > 0 {
			jitter := m.rng.Float64()*m.latencyMS*0.3 - m.latencyMS*0.15
			time.Sleep(time.Duration((m.latencyMS+jitter)*1000) * time.Microsecond)
		}

		m.requestCnt++

		// Simulate occasional errors
		if m.rng.Float64() < m.errorRate {
			results[i] = InferenceResult{
				RequestID: req.ID,
				Error:     fmt.Errorf("mock inference error"),
				ModelID:   req.ModelID,
				Version:   req.Version,
			}
			continue
		}

		// Generate mock outputs
		outputs := map[string]interface{}{
			"output_0": make([]float32, 10),
		}

		results[i] = InferenceResult{
			RequestID:  req.ID,
			Outputs:    outputs,
			Confidence: 0.85 + m.rng.Float64()*0.15,
			Error:      nil,
			ModelID:    req.ModelID,
			Version:    req.Version,
		}
	}

	m.modelInfo.RequestCount = m.requestCnt
	return results
}

// Close releases mock inferrer resources.
func (m *MockInferrer) Close() error {
	return nil
}

// ModelInfo returns mock model metadata.
func (m *MockInferrer) ModelInfo() ModelInfo {
	m.modelInfo.RequestCount = m.requestCnt
	return m.modelInfo
}

// SetLatency sets simulated inference latency.
func (m *MockInferrer) SetLatency(ms float64) {
	m.latencyMS = ms
}

// SetErrorRate sets the probability of inference errors (0.0 to 1.0).
func (m *MockInferrer) SetErrorRate(rate float64) {
	m.errorRate = rate
}

// --- Common Helpers ---

// TensorToFloat32 converts a generic output tensor to float32 slice.
func TensorToFloat32(tensor interface{}) ([]float32, error) {
	if tensor == nil {
		return nil, fmt.Errorf("nil tensor")
	}

	switch v := tensor.(type) {
	case []float32:
		return v, nil
	case []float64:
		result := make([]float32, len(v))
		for i, f := range v {
			result[i] = float32(f)
		}
		return result, nil
	default:
		return nil, fmt.Errorf("cannot convert %T to []float32", v)
	}
}

// TensorToFloat64 converts a generic output tensor to float64 slice.
func TensorToFloat64(tensor interface{}) ([]float64, error) {
	if tensor == nil {
		return nil, fmt.Errorf("nil tensor")
	}

	switch v := tensor.(type) {
	case []float64:
		return v, nil
	case []float32:
		result := make([]float64, len(v))
		for i, f := range v {
			result[i] = float64(f)
		}
		return result, nil
	default:
		return nil, fmt.Errorf("cannot convert %T to []float64", v)
	}
}

// ExtractConfidence extracts scalar confidence from output tensors.
func ExtractConfidence(outputs map[string]interface{}, key string) (float64, error) {
	if outputs == nil {
		return 0.0, fmt.Errorf("nil outputs")
	}

	val, exists := outputs[key]
	if !exists {
		return 0.0, fmt.Errorf("key %s not found in outputs", key)
	}

	switch v := val.(type) {
	case float32:
		return float64(v), nil
	case float64:
		return v, nil
	case []float32:
		if len(v) > 0 {
			return float64(v[0]), nil
		}
	case []float64:
		if len(v) > 0 {
			return v[0], nil
		}
	}

	return 0.0, fmt.Errorf("cannot extract confidence from %T", val)
}

// BatchTensorShape represents the shape metadata for a batched tensor.
type BatchTensorShape struct {
	BatchSize int
	Dims      []int // non-batch dimensions
}

// ReshapeTensor reshapes a flat tensor according to the given shape.
// This is a placeholder; real implementation would use unsafe pointers or cgo.
func ReshapeTensor(data []float32, shape BatchTensorShape) (map[string]interface{}, error) {
	if shape.BatchSize <= 0 {
		return nil, fmt.Errorf("invalid batch size: %d", shape.BatchSize)
	}

	return map[string]interface{}{
		"data":  data,
		"shape": shape,
	}, nil
}
