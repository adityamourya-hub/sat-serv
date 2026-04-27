package physics

import (
	"fmt"
	"math"
)

// EdgeInference handles ONNX model inference on restricted hardware
// (M2 Air, tactical servers) with thermal-aware tiering
type EdgeInference struct {
	onnxRuntime *ONNXRuntime
	
	// Model availability flags
	denoiserAvailable   bool
	regressorAvailable  bool
	segmenterAvailable  bool
}

// NewEdgeInference creates edge inference handler
func NewEdgeInference(modelDir string) (*EdgeInference, error) {
	runtime, err := NewONNXRuntime(modelDir)
	if err != nil {
		return nil, fmt.Errorf("failed to init ONNX runtime: %w", err)
	}
	
	return &EdgeInference{
		onnxRuntime: runtime,
	}, nil
}

// SetModelAvailability marks which models are loaded
func (ei *EdgeInference) SetModelAvailability(denoiser, regressor, segmenter bool) {
	ei.denoiserAvailable = denoiser
	ei.regressorAvailable = regressor
	ei.segmenterAvailable = segmenter
}

// InferOrbitalState predicts orbital state [x,y,z,vx,vy,vz] from current state
// Falls back to analytical propagation if model unavailable or thermal throttled
func (ei *EdgeInference) InferOrbitalState(current State, dt float64, propagator func(State, float64) State) (State, error) {
	// Check thermal
	if ei.onnxRuntime.thermalEnabled && !ei.onnxRuntime.canExecute() {
		// Fallback to analytical
		return propagator(current, dt), nil
	}
	
	if !ei.regressorAvailable {
		// Fallback to analytical propagation
		return propagator(current, dt), nil
	}
	
	// Stub: In production, this would run ONNX inference
	// For now, use analytical propagation as fallback
	return propagator(current, dt), nil
}

// InferDenoise removes noise from orbital state measurements
func (ei *EdgeInference) InferDenoise(noisyState State) (State, error) {
	if !ei.denoiserAvailable {
		return noisyState, nil
	}
	
	// Stub: Would denoise via ONNX
	return noisyState, nil
}

// TierSelection determines which tier to use based on workload and thermal
func (ei *EdgeInference) TierSelection(inferenceCost float64) int {
	temp := ei.onnxRuntime.GetThermal()
	
	// Select tier based on thermal state and inference cost
	if temp > 70.0 || inferenceCost > 100.0 {
		// High cost or high temp -> Tier 1 (reflex)
		return 1
	} else if temp > 60.0 || inferenceCost > 10.0 {
		// Moderate -> Tier 2 (pruned ONNX)
		return 2
	}
	// Low cost, low temp -> Tier 3 (full ONNX)
	return 3
}

// EstimateInferenceCost estimates compute cost in ms for a given state
func (ei *EdgeInference) EstimateInferenceCost(state State, model string) float64 {
	// Simple heuristic based on model type
	switch model {
	case "regressor":
		return 1.0 // 1ms for small MLP
	case "denoiser":
		return 5.0 // 5ms for 1D conv
	case "segmenter":
		return 50.0 // 50ms for vision model
	default:
		return 10.0
	}
}

// ExecuteWithTier runs inference with automatic tier selection
func (ei *EdgeInference) ExecuteWithTier(input State, model string, propagator func(State, float64) State) (State, error) {
	cost := ei.EstimateInferenceCost(input, model)
	tier := ei.TierSelection(cost)
	
	// Update ONNX runtime tier
	ei.onnxRuntime.SetTier(tier)
	
	if tier == 1 {
		// Tier 1: Pure analytical, no ML
		return propagator(input, 0), nil
	}
	
	// Tier 2/3: ONNX inference (stub for now)
	return ei.InferOrbitalState(input, 0, propagator)
}

// EdgeMetrics tracks inference performance on edge hardware
type EdgeMetrics struct {
	TotalInferences   int64
	FallbackCount     int64
	ThermalThrottles  int64
	AvgLatencyMs      float64
	MemoryUsageMB     float64
}

func (ei *EdgeInference) GetMetrics() EdgeMetrics {
	// Stub: Would track actual metrics in production
	return EdgeMetrics{
		TotalInferences:  0,
		FallbackCount:    0,
		ThermalThrottles: 0,
		AvgLatencyMs:     0,
		MemoryUsageMB:    0,
	}
}

// PowerEstimate estimates power consumption for inference
func (ei *EdgeInference) PowerEstimate(model string, inferenceTimeMs float64) float64 {
	// M2 Air power estimates (watts)
	basePower := 2.0 // Base system power
	inferencePower := 0.0
	
	switch model {
	case "regressor":
		inferencePower = 1.5 // Small model
	case "denoiser":
		inferencePower = 3.0 // Medium model
	case "segmenter":
		inferencePower = 8.0 // Large vision model
	default:
		inferencePower = 2.0
	}
	
	// Power scales with time
	powerFactor := math.Min(inferenceTimeMs/1000.0, 1.0)
	
	return basePower + inferencePower*powerFactor
}
