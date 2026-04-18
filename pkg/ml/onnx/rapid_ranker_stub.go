//go:build !onnx

package onnx

import (
	"fmt"

	"ssis/pkg/hypersonic"
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

func (m *RapidRankerONNXModel) Rank(track hypersonic.RapidTrack, class hypersonic.ThreatClass, reachable bool, baselinePriority int) (hypersonic.RapidRankInference, error) {
	return hypersonic.RapidRankInference{}, fmt.Errorf(
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

func (m *RapidPolicyONNXModel) Recommend(track hypersonic.RapidTrack, class hypersonic.ThreatClass, score float64, reachable bool) (hypersonic.RapidPolicyInference, error) {
	return hypersonic.RapidPolicyInference{}, fmt.Errorf(
		"onnx rapid policy backend not enabled in this build; model=%s", m.ModelPath,
	)
}
