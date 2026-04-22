//go:build onnx

// pkg/onnx/session.go - ONNX Runtime Wrapper
//
// Zero-copy ONNX session wrapper for M2 Neural Engine inference.
// Provides <2ms inference latency for 100Hz reflex loop.

package onnx

import (
	"fmt"

	onnx "github.com/yalue/onnxruntime_go"
)

// ONNXSession wraps onnxruntime_go.AdvancedSession with zero-copy optimizations
type ONNXSession struct {
	session      *onnx.AdvancedSession
	modelPath    string
	inputName    string
	outputName   string
	inputShape   []int64
	outputShape  []int64
	inputTensor  *onnx.Tensor
	outputTensor *onnx.Tensor
}

// NewONNXSession creates a new ONNX session from a model file
// Note: This is a simplified version. For production, use the pattern from pkg/ml/onnx/
func NewONNXSession(modelPath, device string) (*ONNXSession, error) {
	// Initialize ONNX runtime (call once per process)
	err := onnx.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	// Create tensors with proper type parameterization
	// Use float64 for physics precision
	inputShape := []int64{1, 6}
	inputTensor, err := onnx.NewEmptyTensor[float64](inputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	outputShape := []int64{1, 1}
	outputTensor, err := onnx.NewEmptyTensor[float64](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Create session with default input/output names
	session, err := onnx.NewAdvancedSession(
		modelPath,
		[]string{"input"},  // input name
		[]string{"output"}, // output name
		[]*onnx.Tensor{inputTensor},
		[]*onnx.Tensor{outputTensor},
		nil, // session options
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &ONNXSession{
		session:      session,
		modelPath:    modelPath,
		inputName:    "input",
		outputName:   "output",
		inputShape:   inputShape,
		outputShape:  outputShape,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}, nil
}

// Run performs single inference
func (s *ONNXSession) Run(input []float64) ([]float64, error) {
	// Copy input data to tensor
	data := s.inputTensor.GetData()
	for i, v := range input {
		data[i] = v
	}

	// Run inference
	err := s.session.Run()
	if err != nil {
		return nil, fmt.Errorf("ONNX inference failed: %w", err)
	}

	// Read output data
	outputData := s.outputTensor.GetData()
	result := make([]float64, len(outputData))
	copy(result, outputData)

	return result, nil
}

// RunBatch performs batch inference for multiple inputs
func (s *ONNXSession) RunBatch(inputs [][]float64) ([][]float64, error) {
	results := make([][]float64, len(inputs))

	for i, input := range inputs {
		output, err := s.Run(input)
		if err != nil {
			return nil, fmt.Errorf("batch inference failed at index %d: %w", i, err)
		}
		results[i] = output
	}

	return results, nil
}

// GetInputShape returns the input tensor shape
func (s *ONNXSession) GetInputShape() []int64 {
	return s.inputShape
}

// GetOutputShape returns the output tensor shape
func (s *ONNXSession) GetOutputShape() []int64 {
	return s.outputShape
}

// Close releases the ONNX session resources
func (s *ONNXSession) Close() error {
	if s.session != nil {
		err := s.session.Destroy()
		s.session = nil
		return err
	}
	return nil
}
