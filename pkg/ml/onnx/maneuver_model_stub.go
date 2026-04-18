//go:build !onnx

package onnx

import (
	"fmt"

	"ssis/pkg/physics"
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

func (m *ManeuverONNXModel) Predict(window []physics.ManeuverFrame) (physics.ManeuverInference, error) {
	if len(window) == 0 {
		return physics.ManeuverInference{}, fmt.Errorf("empty maneuver frame window")
	}

	return physics.ManeuverInference{}, fmt.Errorf(
		"onnx runtime backend is not enabled in this build; model=%s", m.ModelPath,
	)
}
