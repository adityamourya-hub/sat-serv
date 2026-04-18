//go:build onnx

package onnx

import (
	"fmt"

	_ "github.com/yalue/onnxruntime_go"

	"ssis/pkg/orbital"
)

// This implementation is compiled with -tags onnx.
// Expected model signature (from ai/collision/train_collision_surrogate.py):
//
//	inputs:
//	 - node_features: [B, N, F]
//	 - adjacency: [B, N, N]
//	outputs:
//	 - risk_score: [B, 1]
//	 - uncertainty_low: [B, 1]
//	 - uncertainty_high: [B, 1]
//	 - node_attention: [B, N]
type CollisionONNXModel struct {
	ModelPath string
}

func NewCollisionONNXModel(modelPath string) (*CollisionONNXModel, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}
	return &CollisionONNXModel{ModelPath: modelPath}, nil
}

func (m *CollisionONNXModel) Name() string {
	return "onnx_gatv2_surrogate"
}

func (m *CollisionONNXModel) PredictRisk(input orbital.CollisionRiskInput, graph orbital.CollisionNeighborhoodGraph) (orbital.CollisionGraphInference, error) {
	if len(graph.Nodes) == 0 {
		return orbital.CollisionGraphInference{}, fmt.Errorf("empty graph")
	}

	// TODO: bind tensors and parse outputs once runtime deployment image includes native ORT libs.
	return orbital.CollisionGraphInference{}, fmt.Errorf("onnx runtime inference not yet bound")
}
