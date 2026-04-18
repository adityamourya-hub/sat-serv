//go:build onnx

package onnx

import (
	"fmt"

	_ "github.com/yalue/onnxruntime_go"

	"ssis/pkg/spatial"
)

// Built with -tags onnx.
// Expected ONNX signature:
//
//	inputs:
//	 - history_series: [B, T, F]
//	 - horizon_hours: [B, 1]
//	outputs:
//	 - forecast_ratio: [B, H]
//	 - uncertainty_low: [B, H]
//	 - uncertainty_high: [B, H]
//	 - model_confidence: [B, 1]
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
	return "onnx_tcn_blindspot"
}

func (m *BlindSpotForecastONNXModel) Forecast(history []spatial.BlindSpotSample, horizonHours int) (spatial.BlindSpotForecastResult, error) {
	if len(history) == 0 {
		return spatial.BlindSpotForecastResult{}, fmt.Errorf("history is required")
	}

	// TODO: bind tensor conversion and output mapping in deployment image with native ORT libs.
	return spatial.BlindSpotForecastResult{}, fmt.Errorf("onnx blindspot inference not yet bound")
}
