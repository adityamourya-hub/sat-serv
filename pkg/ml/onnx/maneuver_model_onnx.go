//go:build onnx

package onnx

import (
	"fmt"
	"ssis/pkg/physics"

	_ "github.com/yalue/onnxruntime_go"
)

// This implementation is compiled when built with -tags onnx.
// It intentionally keeps the runtime adapter thin: preprocessing happens in physics,
// while this layer maps ONNX tensors to maneuver inference payloads.
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
	return "onnx_lstm_attention"
}

func (m *ManeuverONNXModel) Predict(window []physics.ManeuverFrame) (physics.ManeuverInference, error) {
	if len(window) == 0 {
		return physics.ManeuverInference{}, fmt.Errorf("empty maneuver frame window")
	}

	// TODO: wire exact tensor mapping once deployment image ships with native runtime.
	// Expected outputs from exported model:
	//  1) maneuver_probability: [1, 1]
	//  2) maneuver_type_logits: [1, 2] (impulsive, continuous)
	//  3) delta_v_estimate: [1, 1]
	//  4) confidence_score: [1, 1]
	return physics.ManeuverInference{}, fmt.Errorf("onnx runtime inference not yet bound")
}
