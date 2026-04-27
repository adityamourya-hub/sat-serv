//go:build !onnx
// STUB: This file is a build stub. All functions return "not yet implemented" until the real backend is bound.

package onnx

import (
	"fmt"

	"ssis/pkg/spatial"
)

// BlindSpotForecastONNXModel is a default-build stub for TCN forecasting.
type BlindSpotForecastONNXModel struct {
	ModelPath string
}

func NewBlindSpotForecastONNXModel(modelPath string) (*BlindSpotForecastONNXModel, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}
	return &BlindSpotForecastONNXModel{ModelPath: modelPath}, nil
}

func (m *BlindSpotForecastONNXModel) Name() string {
	return "onnx_tcn_blindspot_stub"
}

func (m *BlindSpotForecastONNXModel) Forecast(history []spatial.BlindSpotSample, horizonHours int) (spatial.BlindSpotForecastResult, error) {
	if len(history) == 0 {
		return spatial.BlindSpotForecastResult{}, fmt.Errorf("history is required")
	}
	return spatial.BlindSpotForecastResult{}, fmt.Errorf("onnx blindspot backend not enabled in this build; model=%s", m.ModelPath)
}
