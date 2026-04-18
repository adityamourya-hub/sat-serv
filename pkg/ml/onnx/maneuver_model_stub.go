//go:build !onnx

package onnx

import (
	"fmt"

	"ssis/pkg/maneuver"
)

// ManeuverONNXModel is a stub in default builds.
// Build with native ONNX Runtime integration in a dedicated target for deployment.
type ManeuverONNXModel struct {
	ModelPath string
}

func NewManeuverONNXModel(modelPath string) (*ManeuverONNXModel, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}
	return &ManeuverONNXModel{ModelPath: modelPath}, nil
}

func (m *ManeuverONNXModel) Name() string {
	return "onnx_lstm_attention_stub"
}

func (m *ManeuverONNXModel) Predict(window []maneuver.ManeuverFrame) (maneuver.ManeuverInference, error) {
	if len(window) == 0 {
		return maneuver.ManeuverInference{}, fmt.Errorf("empty maneuver frame window")
	}

	return maneuver.ManeuverInference{}, fmt.Errorf(
		"onnx runtime backend is not enabled in this build; model=%s", m.ModelPath,
	)
}
