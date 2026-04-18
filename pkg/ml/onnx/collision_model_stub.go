//go:build !onnx

package onnx

import (
	"fmt"

	"ssis/pkg/proximity"
)

// CollisionONNXModel is a default-build stub for GATv2 surrogate collision scoring.
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
	return "onnx_gatv2_surrogate_stub"
}

func (m *CollisionONNXModel) PredictRisk(input proximity.CollisionRiskInput, graph proximity.CollisionNeighborhoodGraph) (proximity.CollisionGraphInference, error) {
	if len(graph.Nodes) == 0 {
		return proximity.CollisionGraphInference{}, fmt.Errorf("empty graph")
	}
	return proximity.CollisionGraphInference{}, fmt.Errorf(
		"onnx collision backend not enabled in this build; model=%s", m.ModelPath,
	)
}
