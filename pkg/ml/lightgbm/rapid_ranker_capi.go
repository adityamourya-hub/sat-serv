//go:build lightgbm

package lightgbm

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -llightgbm
#include <stdlib.h>
#include <stdint.h>

// Keep C symbols opaque to avoid binding to specific API version in this scaffold.
typedef void* BoosterHandle;
*/
import "C"

import (
	"fmt"
	"unsafe"

	"ssis/pkg/physics"
)

// RapidRanker provides LightGBM C API inference for rapid threat scoring.
// This scaffold intentionally keeps API calls minimal and conservative for portability.
type RapidRanker struct {
	ModelPath string
	handle    C.BoosterHandle
}

func NewRapidRanker(modelPath string) (*RapidRanker, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}

	r := &RapidRanker{ModelPath: modelPath}
	if err := r.loadModel(); err != nil {
		return nil, err
	}
	return r, nil
}

func (r *RapidRanker) Name() string {
	return "lightgbm_capi"
}

func (r *RapidRanker) loadModel() error {
	// NOTE: This is a placeholder to keep the build contract ready.
	// Integrate LGBM_BoosterCreateFromModelfile + prediction entrypoints
	// once deployment runtime and versioned LightGBM SDK are finalized.
	cPath := C.CString(r.ModelPath)
	defer C.free(unsafe.Pointer(cPath))
	_ = cPath
	return fmt.Errorf("lightgbm C API model loading not yet bound")
}

func (r *RapidRanker) Rank(track physics.RapidTrack, class physics.ThreatClass, reachable bool, baselinePriority int) (physics.RapidRankInference, error) {
	if r.handle == nil {
		return physics.RapidRankInference{}, fmt.Errorf("lightgbm model is not loaded")
	}

	// Placeholder mapping contract:
	// features := [mach, altitude, burnout_velocity, flight_path_angle, distance_to_asset, rcs, baseline]
	return physics.RapidRankInference{}, fmt.Errorf("lightgbm C API inference not yet bound")
}

func (r *RapidRanker) Close() error {
	r.handle = nil
	return nil
}
