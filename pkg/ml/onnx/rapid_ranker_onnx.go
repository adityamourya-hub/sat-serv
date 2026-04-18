//go:build onnx

package onnx

import (
	"fmt"

	_ "github.com/yalue/onnxruntime_go"

	"ssis/pkg/physics"
)

// Built with -tags onnx.
// Expected ranker ONNX signature:
//
//	input: rapid_features [B, F]
//	outputs: dynamic_score [B,1], confidence [B,1], low [B,1], high [B,1]
//
// Expected policy ONNX signature:
//
//	input: policy_features [B, F]
//	outputs: action_logits [B, K], confidence [B,1]
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
	return "onnx_xgboost_ranker"
}

func (m *RapidRankerONNXModel) Rank(track physics.RapidTrack, class physics.ThreatClass, reachable bool, baselinePriority int) (physics.RapidRankInference, error) {
	return physics.RapidRankInference{}, fmt.Errorf("onnx runtime inference not yet bound")
}

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
	return "onnx_ppo_policy"
}

func (m *RapidPolicyONNXModel) Recommend(track physics.RapidTrack, class physics.ThreatClass, score float64, reachable bool) (physics.RapidPolicyInference, error) {
	return physics.RapidPolicyInference{}, fmt.Errorf("onnx runtime inference not yet bound")
}
