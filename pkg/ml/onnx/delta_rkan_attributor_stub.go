//go:build !onnx

package onnx

import (
	"ssis/pkg/physics"
	"time"
)

type DeltaRKANAttributorModel struct {
	confidenceThreshold float64
}

type ManeuverType int

const (
	ManeuverNone        ManeuverType = 0
	ManeuverStationKeep ManeuverType = 1
	ManeuverPhasing     ManeuverType = 2
	ManeuverPlaneChange ManeuverType = 3
)

func (mt ManeuverType) String() string {
	switch mt {
	case ManeuverNone:
		return "none"
	case ManeuverStationKeep:
		return "station_keep"
	case ManeuverPhasing:
		return "phasing"
	case ManeuverPlaneChange:
		return "plane_change"
	default:
		return "unknown"
	}
}

type DeltaVRTN struct {
	Radial     float64
	Transverse float64
	Normal     float64
}

func (dv DeltaVRTN) Magnitude() float64 {
	return 0
}

type DeltaRKANOutput struct {
	ManeuverDetected bool
	ManeuverProb     float64
	ManeuverType     ManeuverType
	TypeConfidence   float64
	DeltaV           DeltaVRTN
	Confidence       float64
	LaTeXExplanation string
	DetectionTime    time.Time
}

type StateHistory struct {
	ObjectID int
	States   []physics.State
}

func NewDeltaRKANAttributorModel(modelPath string, timesteps int) (*DeltaRKANAttributorModel, error) {
	return &DeltaRKANAttributorModel{confidenceThreshold: 0.7}, nil
}

func (m *DeltaRKANAttributorModel) Name() string {
	return "axiom_delta_rkan_attributor_stub"
}

func (m *DeltaRKANAttributorModel) Version() string {
	return "3.0.0-stub"
}

func (m *DeltaRKANAttributorModel) SetConfidenceThreshold(threshold float64) {
	m.confidenceThreshold = threshold
}

func (m *DeltaRKANAttributorModel) Predict(history StateHistory) (DeltaRKANOutput, error) {
	return DeltaRKANOutput{
		ManeuverDetected: false,
		ManeuverProb:     0,
		ManeuverType:     ManeuverNone,
		TypeConfidence:   0,
		DeltaV:           DeltaVRTN{},
		Confidence:       0,
		LaTeXExplanation: "\\text{Stub: No ONNX runtime}",
		DetectionTime:    time.Now(),
	}, nil
}

func (m *DeltaRKANAttributorModel) Close() error {
	return nil
}

func (m *DeltaRKANAttributorModel) ModelInfo() ModelInfo {
	return ModelInfo{
		ModelID:     m.Name(),
		Version:     m.Version(),
		InputNames:  []string{"state_history"},
		OutputNames: []string{"maneuver_prob", "delta_v", "maneuver_type"},
	}
}

func (m *DeltaRKANAttributorModel) InferBatch(requests []InferenceRequest) []InferenceResult {
	results := make([]InferenceResult, len(requests))
	for i, req := range requests {
		results[i] = InferenceResult{
			RequestID:  req.ID,
			ModelID:    m.Name(),
			Version:    m.Version(),
			Confidence: 0,
			Outputs: map[string]interface{}{
				"note": "stub implementation - no onnx runtime",
			},
		}
	}
	return results
}

func ComputeRTNFrame(state physics.State) ([3]float64, [3]float64, [3]float64) {
	return [3]float64{1, 0, 0}, [3]float64{0, 1, 0}, [3]float64{0, 0, 1}
}
