//go:build !onnx
// STUB: This file is a build stub. All functions return "not yet implemented" until the real backend is bound.

package onnx

/*
*

  - RKANCovarianceModel stub implementation.
  - This is used when the onnx build tag is not set.
*/
type RKANCovarianceModel struct{}

type RKANCovarianceInput struct {
	PositionX, PositionY, PositionZ float64
	VelocityX, VelocityY, VelocityZ float64
	SemiMajorAxis                   float64
	Eccentricity                    float64
	Inclination                     float64
	RAAN                            float64
	SolarFluxF107                   float64
	GeomagneticAp                   float64
	LocalTime                       float64
}

type RKANCovarianceOutput struct {
	PosCovXX, PosCovYY, PosCovZZ float64
	VelCovXX, VelCovYY, VelCovZZ float64
	RealismScore                 float64
}

func NewRKANCovarianceModel(modelPath string) (*RKANCovarianceModel, error) {
	return &RKANCovarianceModel{}, nil
}

func (m *RKANCovarianceModel) Name() string {
	return "axiom_rkan_covariance_stub"
}

func (m *RKANCovarianceModel) Version() string {
	return "3.0.0-stub"
}

func (m *RKANCovarianceModel) Predict(input RKANCovarianceInput) (RKANCovarianceOutput, error) {
	// Return nominal covariance inflation (no learned residuals)
	return RKANCovarianceOutput{
		PosCovXX:     1e-6,
		PosCovYY:     1e-6,
		PosCovZZ:     1e-6,
		VelCovXX:     1e-8,
		VelCovYY:     1e-8,
		VelCovZZ:     1e-8,
		RealismScore: 0.0, // No ML confidence
	}, nil
}

func (m *RKANCovarianceModel) PredictBatch(inputs []RKANCovarianceInput) ([]RKANCovarianceOutput, error) {
	outputs := make([]RKANCovarianceOutput, len(inputs))
	for i, inp := range inputs {
		out, _ := m.Predict(inp)
		outputs[i] = out
	}
	return outputs, nil
}

func (m *RKANCovarianceModel) InflateCovariance(nominalCov [6][6]float64, residual RKANCovarianceOutput) [6][6]float64 {
	var inflated [6][6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			inflated[i][j] = nominalCov[i][j]
		}
	}
	inflated[0][0] += residual.PosCovXX
	inflated[1][1] += residual.PosCovYY
	inflated[2][2] += residual.PosCovZZ
	inflated[3][3] += residual.VelCovXX
	inflated[4][4] += residual.VelCovYY
	inflated[5][5] += residual.VelCovZZ
	return inflated
}

func (m *RKANCovarianceModel) Close() error {
	return nil
}

func (m *RKANCovarianceModel) ModelInfo() ModelInfo {
	return ModelInfo{
		ModelID:     m.Name(),
		Version:     m.Version(),
		InputNames:  []string{"orbital_features"},
		OutputNames: []string{"covariance_inflation"},
	}
}

func (m *RKANCovarianceModel) InferBatch(requests []InferenceRequest) []InferenceResult {
	results := make([]InferenceResult, len(requests))
	for i, req := range requests {
		results[i] = InferenceResult{
			RequestID:  req.ID,
			ModelID:    m.Name(),
			Version:    m.Version(),
			Confidence: 0.0,
			Outputs: map[string]interface{}{
				"note": "stub implementation - no onnx runtime",
			},
		}
	}
	return results
}
