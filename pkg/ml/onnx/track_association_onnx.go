//go:build onnx

package onnx

import (
	"fmt"

	_ "github.com/yalue/onnxruntime_go"

	"ssis/pkg/catalog"
)

// Built with -tags onnx.
// Expected association ONNX signature:
//
//	inputs:
//	 - observation: [B, Fo]
//	 - candidates: [B, N, Fc]
//	outputs:
//	 - association_scores: [B, N]
//	 - model_confidence: [B, 1]
//	 - attention: [B, N]
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
	return "onnx_transformer_association"
}

func (m *TrackAssociationONNXModel) Associate(obs catalog.FusedTrackObservation, candidates []catalog.ObjectRecord) (catalog.TrackAssociationInference, error) {
	if len(candidates) == 0 {
		return catalog.TrackAssociationInference{}, fmt.Errorf("no candidates")
	}

	// TODO: bind tensor I/O in deployment build with runtime libs.
	return catalog.TrackAssociationInference{}, fmt.Errorf("onnx association inference not yet bound")
}
