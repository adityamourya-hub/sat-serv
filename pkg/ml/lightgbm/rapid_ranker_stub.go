//go:build !lightgbm

package lightgbm

import (
	"fmt"

	"ssis/pkg/hypersonic"
)

// RapidRanker is a default-build stub; compile with -tags lightgbm to enable C API backend.
type RapidRanker struct {
	ModelPath string
}

func NewRapidRanker(modelPath string) (*RapidRanker, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}
	return &RapidRanker{ModelPath: modelPath}, nil
}

func (r *RapidRanker) Name() string {
	return "lightgbm_capi_stub"
}

func (r *RapidRanker) Rank(track hypersonic.RapidTrack, class hypersonic.ThreatClass, reachable bool, baselinePriority int) (hypersonic.RapidRankInference, error) {
	return hypersonic.RapidRankInference{}, fmt.Errorf("lightgbm backend not enabled; rebuild with -tags lightgbm")
}

func (r *RapidRanker) Close() error {
	return nil
}
