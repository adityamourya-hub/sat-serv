//go:build !onnx
// STUB: This file is a build stub. All functions return "not yet implemented" until the real backend is bound.

package onnx

import (
	"fmt"

	"ssis/pkg/catalog"
)

// TrackAssociationONNXModel is a default-build stub for Transformer/DeepSORT association models.
type TrackAssociationONNXModel struct {
	ModelPath string
}

func NewTrackAssociationONNXModel(modelPath string) (*TrackAssociationONNXModel, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}
	return &TrackAssociationONNXModel{ModelPath: modelPath}, nil
}

func (m *TrackAssociationONNXModel) Name() string {
	return "onnx_transformer_association_stub"
}

func (m *TrackAssociationONNXModel) Associate(obs catalog.FusedTrackObservation, candidates []catalog.ObjectRecord) (catalog.TrackAssociationInference, error) {
	if len(candidates) == 0 {
		return catalog.TrackAssociationInference{}, fmt.Errorf("no candidates")
	}
	return catalog.TrackAssociationInference{}, fmt.Errorf("onnx tracker backend not enabled in this build; model=%s", m.ModelPath)
}
