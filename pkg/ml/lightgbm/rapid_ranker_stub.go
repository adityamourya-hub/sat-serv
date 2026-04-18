//go:build !lightgbm

package lightgbm

import (
	"fmt"

	"ssis/pkg/physics"
)

// RapidRanker is a default-build stub; compile with -tags lightgbm to enable C API backend.
type RapidRanker struct {
	ModelPath string
}

func NewRapidRanker(modelPath string) (*RapidRanker, error) {
	fmt.Printf("[TRC] Entering NewRapidRanker for %s\n", modelPath)
	defer fmt.Printf("[TRC] Exiting NewRapidRanker\n")
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}
	return &RapidRanker{ModelPath: modelPath}, nil
}

func (r *RapidRanker) Name() string {
	return "lightgbm_capi_stub"
}

func (r *RapidRanker) Rank(track physics.HypersonicState, class physics.ThreatClass, reachable bool, baselinePriority int) (physics.RapidRankInference, error) {
	return physics.RapidRankInference{}, fmt.Errorf("lightgbm backend not enabled; rebuild with -tags lightgbm")
}

func (r *RapidRanker) Close() error {
	return nil
}
