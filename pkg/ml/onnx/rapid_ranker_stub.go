//go:build !onnx
// STUB: This file is a build stub. All functions return "not yet implemented" until the real backend is bound.

package onnx

import (
	"fmt"

	"ssis/pkg/physics"
)

// RapidRankerONNXModel is a default-build stub for XGBoost/LightGBM ranker inference.
type RapidRankerONNXModel struct {
	ModelPath string
}

func NewRapidRankerONNXModel(modelPath string) (*RapidRankerONNXModel, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}
	return &RapidRankerONNXModel{ModelPath: modelPath}, nil
}

func (m *RapidRankerONNXModel) Name() string {
	return "onnx_xgboost_ranker_stub"
}

func (m *RapidRankerONNXModel) Rank(track physics.RapidTrack, class physics.ThreatClass, reachable bool, baselinePriority int) (physics.RapidRankInference, error) {
	return physics.RapidRankInference{}, fmt.Errorf(
		"onnx rapid ranker backend not enabled in this build; model=%s", m.ModelPath,
	)
}

// RapidPolicyONNXModel is a default-build stub for optional PPO policy inference.
type RapidPolicyONNXModel struct {
	ModelPath string
}

func NewRapidPolicyONNXModel(modelPath string) (*RapidPolicyONNXModel, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}
	return &RapidPolicyONNXModel{ModelPath: modelPath}, nil
}

func (m *RapidPolicyONNXModel) Name() string {
	return "onnx_ppo_policy_stub"
}

func (m *RapidPolicyONNXModel) Recommend(track physics.RapidTrack, class physics.ThreatClass, score float64, reachable bool) (physics.RapidPolicyInference, error) {
	return physics.RapidPolicyInference{}, fmt.Errorf(
		"onnx rapid policy backend not enabled in this build; model=%s", m.ModelPath,
	)
}
