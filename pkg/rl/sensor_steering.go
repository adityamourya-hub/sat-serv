package rl

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/yalue/onnxruntime_go"

	"ssis/pkg/common"
)

// SDAObjectFeature represents the input features for a single object in the RL model.
type SDAObjectFeature struct {
	ObjectID         int
	CovarianceTrace  float32
	TimeSinceLastObs float32
	IsProtected      float32 // 1.0 or 0.0
	PriorityClass    float32 // 0 to 5
	Eccentricity     float32
	Altitude         float32
	SNREstimate      float32
	Reserved         float32
}

// RadarConstraints defines operational guardrails for sensor tasking output.
type RadarConstraints struct {
	MaxTaskings           int
	MinSNR                float32
	MaxTrackStalenessMin  float32
	ReserveProtectedShare float32
}

// TaskingRecommendation is a single prioritized steering action.
type TaskingRecommendation struct {
	ObjectID           int
	PriorityScore      float32
	Tasking            string
	Reason             string
	ConstraintFiltered bool
}

func DefaultRadarConstraints() RadarConstraints {
	return RadarConstraints{
		MaxTaskings:           8,
		MinSNR:                6.0,
		MaxTrackStalenessMin:  180.0,
		ReserveProtectedShare: 0.40,
	}
}

// SteeringAgent wraps the PPO-based sensor steering model.
type SteeringAgent struct {
	session *onnxruntime_go.DynamicAdvancedSession
}

// NewSteeringAgent initializes the RL steering agent.
func NewSteeringAgent(modelPath string) (*SteeringAgent, error) {
	strict := common.StrictModeEnabled()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		common.EmitFallbackEvent("rl", "sensor_steering", "model_missing", "use_heuristic_policy", 0.0, strict, map[string]interface{}{"model_path": modelPath})
		if strict {
			return nil, fmt.Errorf("strict mode: RL steering model missing: %s", modelPath)
		}
		return &SteeringAgent{session: nil}, nil // Fallback mode
	}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{"sda_state"},
		[]string{"priority_scores"},
		nil, // Default options
	)
	if err != nil {
		common.EmitFallbackEvent("rl", "sensor_steering", "onnx_init_failed", "use_heuristic_policy", 0.0, strict, map[string]interface{}{
			"model_path": modelPath,
			"error":      err.Error(),
		})
		if strict {
			return nil, fmt.Errorf("strict mode: RL steering ONNX init failed: %w", err)
		}
		// If initialization failed (like missing runtime), return fallback agent
		return &SteeringAgent{session: nil}, nil
	}
	common.EmitInfoEvent("rl", "sensor_steering", "onnx_initialized", "use_onnx_policy", strict, map[string]interface{}{"model_path": modelPath})

	return &SteeringAgent{session: session}, nil
}

func (a *SteeringAgent) inferPriorityScores(features []SDAObjectFeature) ([]float32, error) {
	if a.session == nil {
		// Heuristic fallback approximating PPO behavior with uncertainty + radar observability terms.
		scores := make([]float32, len(features))
		for i, f := range features {
			score :=
				f.CovarianceTrace*40.0 +
					f.TimeSinceLastObs*0.03 +
					f.IsProtected*10.0 +
					f.PriorityClass*2.5 +
					(20.0-f.SNREstimate)*0.20 +
					f.Reserved*1.5
			scores[i] = score
		}
		return scores, nil
	}

	const nObjects = 128
	const nFeatures = 8
	// Flatten features into tensor format [1, 128, 8]
	data := make([]float32, nObjects*nFeatures)
	for i := 0; i < len(features) && i < nObjects; i++ {
		f := features[i]
		base := i * nFeatures
		data[base+0] = f.CovarianceTrace
		data[base+1] = f.TimeSinceLastObs
		data[base+2] = f.IsProtected
		data[base+3] = f.PriorityClass
		data[base+4] = f.Eccentricity
		data[base+5] = f.Altitude
		data[base+6] = f.SNREstimate
		data[base+7] = f.Reserved
	}

	// Prepare input/output tensors
	inputShape := onnxruntime_go.NewShape(1, int64(nObjects), int64(nFeatures))
	inputTensor, err := onnxruntime_go.NewTensor(inputShape, data)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	inputs := []onnxruntime_go.Value{inputTensor}
	outputs := []onnxruntime_go.Value{nil} // One output expected (priority_scores)

	if err := a.session.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("RL inference failure: %w", err)
	}
	defer func() {
		for _, v := range outputs {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	// Extract score data from the first output tensor
	// outputs[0] is priority_scores [1, 128]
	scoreTensor, ok := outputs[0].(*onnxruntime_go.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output tensor type for priority scores")
	}
	outputData := scoreTensor.GetData()

	limit := len(features)
	if limit > nObjects {
		limit = nObjects
	}
	scores := make([]float32, limit)
	for i := 0; i < limit; i++ {
		scores[i] = outputData[i]
	}
	return scores, nil
}

// RecommendTasking runs PPO policy inference and returns a prioritized tasking list.
func (a *SteeringAgent) RecommendTasking(features []SDAObjectFeature, constraints RadarConstraints) ([]TaskingRecommendation, error) {
	if len(features) == 0 {
		return nil, nil
	}
	if constraints.MaxTaskings <= 0 {
		constraints.MaxTaskings = DefaultRadarConstraints().MaxTaskings
	}
	if constraints.MinSNR <= 0 {
		constraints.MinSNR = DefaultRadarConstraints().MinSNR
	}
	if constraints.MaxTrackStalenessMin <= 0 {
		constraints.MaxTrackStalenessMin = DefaultRadarConstraints().MaxTrackStalenessMin
	}
	if constraints.ReserveProtectedShare < 0 || constraints.ReserveProtectedShare > 1 {
		constraints.ReserveProtectedShare = DefaultRadarConstraints().ReserveProtectedShare
	}

	scores, err := a.inferPriorityScores(features)
	if err != nil {
		return nil, err
	}

	taskings := make([]TaskingRecommendation, 0, len(scores))
	for i := 0; i < len(scores); i++ {
		f := features[i]
		score := scores[i]
		reasons := make([]string, 0, 3)
		task := "task_radar_standard"
		filtered := false

		if f.CovarianceTrace >= 0.03 {
			task = "task_radar_high_priority"
			reasons = append(reasons, "uncertainty_hotspot")
		}
		if f.TimeSinceLastObs >= constraints.MaxTrackStalenessMin {
			score *= 1.25
			task = "task_reacquire_track"
			reasons = append(reasons, "track_stale")
		}
		if f.SNREstimate < constraints.MinSNR {
			score *= 0.45
			filtered = true
			if task != "task_reacquire_track" {
				task = "defer_low_snr"
			}
			reasons = append(reasons, "snr_constraint")
		}
		if f.IsProtected > 0.5 {
			score *= 1.20
			reasons = append(reasons, "protected_asset")
		}
		if f.Reserved > 0.8 {
			score *= 0.80
			reasons = append(reasons, "radar_busy")
		}
		if len(reasons) == 0 {
			reasons = append(reasons, "ppo_priority")
		}

		taskings = append(taskings, TaskingRecommendation{
			ObjectID:           f.ObjectID,
			PriorityScore:      score,
			Tasking:            task,
			Reason:             strings.Join(reasons, "+"),
			ConstraintFiltered: filtered,
		})
	}

	sort.Slice(taskings, func(i, j int) bool {
		return taskings[i].PriorityScore > taskings[j].PriorityScore
	})

	if constraints.MaxTaskings < len(taskings) {
		taskings = taskings[:constraints.MaxTaskings]
	}
	enforceProtectedShare(taskings, features, constraints)

	return taskings, nil
}

// RankObjects preserves backward compatibility with existing call sites.
func (a *SteeringAgent) RankObjects(features []SDAObjectFeature) ([]int, error) {
	taskings, err := a.RecommendTasking(features, DefaultRadarConstraints())
	if err != nil {
		return nil, err
	}
	result := make([]int, len(taskings))
	for i, t := range taskings {
		result[i] = t.ObjectID
	}
	return result, nil
}

func enforceProtectedShare(taskings []TaskingRecommendation, features []SDAObjectFeature, constraints RadarConstraints) {
	if len(taskings) == 0 || constraints.ReserveProtectedShare <= 0 {
		return
	}
	need := int(math.Ceil(float64(constraints.ReserveProtectedShare) * float64(len(taskings))))
	if need <= 0 {
		return
	}
	protectedByID := make(map[int]bool, len(features))
	for _, f := range features {
		if f.IsProtected > 0.5 {
			protectedByID[f.ObjectID] = true
		}
	}
	have := 0
	for _, t := range taskings {
		if protectedByID[t.ObjectID] {
			have++
		}
	}
	if have >= need {
		return
	}

	// Promote protected assets from the current list if possible.
	for i := len(taskings) - 1; i >= 0 && have < need; i-- {
		if protectedByID[taskings[i].ObjectID] {
			continue
		}
		bestIdx := -1
		bestScore := float32(-1e9)
		for j := i + 1; j < len(taskings); j++ {
			if protectedByID[taskings[j].ObjectID] && taskings[j].PriorityScore > bestScore {
				bestIdx = j
				bestScore = taskings[j].PriorityScore
			}
		}
		if bestIdx >= 0 {
			taskings[i], taskings[bestIdx] = taskings[bestIdx], taskings[i]
			have++
		}
	}
}

func (a *SteeringAgent) Close() {
	if a.session != nil {
		a.session.Destroy()
	}
}
