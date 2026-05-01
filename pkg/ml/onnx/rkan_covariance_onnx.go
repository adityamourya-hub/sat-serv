//go:build onnx

package onnx

import (
	"fmt"
	"math"

	onnx "github.com/yalue/onnxruntime_go"
)

/**
 * RKANCovarianceModel wraps the Axiom-Zero rKAN Residual Covariance Head ONNX model.
 *
 * This model provides physics-informed uncertainty quantification by learning
 * Hamiltonian residuals. It inflates the nominal process noise covariance based
 * on learned perturbation structure (solar flux, drag, unmodeled third-body).
 *
 * Input:  (batch, 13) orbital features
 * Output: (batch, 6) diagonal covariance inflation [sigma_xx, sigma_yy, sigma_zz, 
 *                     sigma_vxvx, sigma_vyvy, sigma_vzvz] in km² and (km/s)²
 */
type RKANCovarianceModel struct {
	session *onnx.AdvancedSession
	input   *onnx.Tensor[float32]
	output  *onnx.Tensor[float32]
}

// RKANCovarianceInput represents the 13 orbital features expected by the model.
type RKANCovarianceInput struct {
	// Positional features (3)
	PositionX, PositionY, PositionZ float64 // km, ECI

	// Velocity features (3)  
	VelocityX, VelocityY, VelocityZ float64 // km/s, ECI

	// Orbital element features (4)
	SemiMajorAxis float64 // km
	Eccentricity  float64
	Inclination   float64 // rad
	RAAN          float64 // rad

	// Environmental features (3)
	SolarFluxF107 float64 // Solar activity index
	GeomagneticAp float64 // Geomagnetic activity index
	LocalTime     float64 // Hours (0-24)
}

// RKANCovarianceOutput represents the model's predicted covariance inflation.
type RKANCovarianceOutput struct {
	// Position covariance inflation (km²)
	PosCovXX, PosCovYY, PosCovZZ float64

	// Velocity covariance inflation ((km/s)²)
	VelCovXX, VelCovYY, VelCovZZ float64

	// Realism score (0-1) indicating confidence in the learned residual
	RealismScore float64
}

// NewRKANCovarianceModel loads the rKAN covariance model from the given ONNX path.
func NewRKANCovarianceModel(modelPath string) (*RKANCovarianceModel, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required for rKAN covariance model")
	}

	// Create tensors for input/output
	// Input: (batch=1, 13 features)
	inputShape := []int64{1, 13}
	inputTensor, err := onnx.NewEmptyTensor[float64](inputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	// Output: (batch=1, 6 covariance components)
	outputShape := []int64{1, 6}
	outputTensor, err := onnx.NewEmptyTensor[float64](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Create ONNX session
	session, err := onnx.NewAdvancedSession(
		modelPath,
		[]string{"StatefulPartitionedCall:0"}, // Output names from TF SavedModel
		[]string{"args_0"},                    // Input name
		[]*onnx.Tensor{inputTensor},
		[]*onnx.Tensor{outputTensor},
		nil, // No additional options
	)
	if err != nil {
		// Fallback: try common alternative names
		session, err = onnx.NewAdvancedSession(
			modelPath,
			[]string{"output"},
			[]string{"input"},
			[]*onnx.Tensor{inputTensor},
			[]*onnx.Tensor{outputTensor},
			nil,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create ONNX session: %w", err)
		}
	}

	return &RKANCovarianceModel{
		session: session,
		input:   inputTensor,
		output:  outputTensor,
	}, nil
}

// Name returns the model identifier.
func (m *RKANCovarianceModel) Name() string {
	return "axiom_rkan_covariance"
}

// Version returns the model version.
func (m *RKANCovarianceModel) Version() string {
	return "3.0.0"
}

// Predict runs inference and returns covariance inflation.
func (m *RKANCovarianceModel) Predict(input RKANCovarianceInput) (RKANCovarianceOutput, error) {
	// Populate input tensor
	data := m.input.GetData()
	data[0] = input.PositionX
	data[1] = input.PositionY
	data[2] = input.PositionZ
	data[3] = input.VelocityX
	data[4] = input.VelocityY
	data[5] = input.VelocityZ
	data[6] = input.SemiMajorAxis
	data[7] = input.Eccentricity
	data[8] = input.Inclination
	data[9] = input.RAAN
	data[10] = input.SolarFluxF107
	data[11] = input.GeomagneticAp
	data[12] = input.LocalTime

	// Run inference
	if err := m.session.Run(); err != nil {
		return RKANCovarianceOutput{}, fmt.Errorf("inference failed: %w", err)
	}

	// Extract output
	outputData := m.output.GetData()
	if len(outputData) < 6 {
		return RKANCovarianceOutput{}, fmt.Errorf("unexpected output size: %d", len(outputData))
	}

	// Compute realism score based on output magnitude
	// High inflation in unexpected directions indicates low confidence
	avgPosCov := (outputData[0] + outputData[1] + outputData[2]) / 3.0
	avgVelCov := (outputData[3] + outputData[4] + outputData[5]) / 3.0

	// Realism score: higher when inflation is moderate (not too low, not extreme)
	realism := 0.5
	if avgPosCov > 1e-8 && avgPosCov < 1e-4 {
		realism = 0.8
	} else if avgPosCov > 0 {
		realism = 0.3 + 0.5*math.Exp(-avgPosCov/1e-4)
	}

	return RKANCovarianceOutput{
		PosCovXX:     outputData[0],
		PosCovYY:     outputData[1],
		PosCovZZ:     outputData[2],
		VelCovXX:     outputData[3],
		VelCovYY:     outputData[4],
		VelCovZZ:     outputData[5],
		RealismScore: realism,
	}, nil
}

// PredictBatch runs batch inference for multiple objects.
func (m *RKANCovarianceModel) PredictBatch(inputs []RKANCovarianceInput) ([]RKANCovarianceOutput, error) {
	batchSize := len(inputs)
	if batchSize == 0 {
		return []RKANCovarianceOutput{}, nil
	}

	// For now, process sequentially (batch support requires tensor reshaping)
	outputs := make([]RKANCovarianceOutput, batchSize)
	for i, inp := range inputs {
		out, err := m.Predict(inp)
		if err != nil {
			return nil, fmt.Errorf("batch item %d failed: %w", i, err)
		}
		outputs[i] = out
	}

	return outputs, nil
}

// InflateCovariance applies the learned residual to a nominal covariance matrix.
// The nominal covariance is expected as a 6x6 matrix (position + velocity states).
func (m *RKANCovarianceModel) InflateCovariance(nominalCov [6][6]float64, 
	residual RKANCovarianceOutput) [6][6]float64 {
	var inflated [6][6]float64

	// Copy nominal covariance
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			inflated[i][j] = nominalCov[i][j]
		}
	}

	// Add diagonal inflation from rKAN residual
	inflated[0][0] += residual.PosCovXX
	inflated[1][1] += residual.PosCovYY
	inflated[2][2] += residual.PosCovZZ
	inflated[3][3] += residual.VelCovXX
	inflated[4][4] += residual.VelCovYY
	inflated[5][5] += residual.VelCovZZ

	return inflated
}

// Close releases the ONNX session.
func (m *RKANCovarianceModel) Close() error {
	if m.session != nil {
		return m.session.Destroy()
	}
	return nil
}

// ModelInfo returns metadata about the loaded model.
func (m *RKANCovarianceModel) ModelInfo() ModelInfo {
	return ModelInfo{
		ModelID:     m.Name(),
		Version:     m.Version(),
		InputNames:  []string{"orbital_features"},
		OutputNames: []string{"covariance_inflation"},
	}
}

// InferBatch implements the Inferrer interface for integration with Session.
func (m *RKANCovarianceModel) InferBatch(requests []InferenceRequest) []InferenceResult {
	results := make([]InferenceResult, len(requests))

	for i, req := range requests {
		// Extract inputs from request
		input := RKANCovarianceInput{}
		if pos, ok := req.Inputs["position"].([3]float64); ok {
			input.PositionX, input.PositionY, input.PositionZ = pos[0], pos[1], pos[2]
		}
		if vel, ok := req.Inputs["velocity"].([3]float64); ok {
			input.VelocityX, input.VelocityY, input.VelocityZ = vel[0], vel[1], vel[2]
		}

		output, err := m.Predict(input)
		results[i] = InferenceResult{
			RequestID:  req.ID,
			ModelID:    m.Name(),
			Version:    m.Version(),
			Confidence: output.RealismScore,
			Outputs: map[string]interface{}{
				"covariance": output,
			},
			Error: err,
		}
	}

	return results
}
