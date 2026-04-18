package ml

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"ssis/pkg/common"
	"ssis/pkg/hypersonic"
	"ssis/pkg/maneuver"
	"ssis/pkg/proximity"
)

// InferenceEngine orchestrates all ML model predictions with fallback logic.
type InferenceEngine interface {
	PredictRapidScore(track hypersonic.RapidTrack) (hypersonic.RapidRankInference, error)
	PredictManeuver(window []maneuver.ManeuverFrame) (maneuver.ManeuverInference, error)
	PredictCollisionRisk(input proximity.CollisionRiskInput, graph proximity.CollisionNeighborhoodGraph) (proximity.CollisionGraphInference, error)
	IsHealthy() bool
}

// DummyInferenceEngine remains as fallback when ONNX is unavailable in air-gapped setups.
type DummyInferenceEngine struct {
	Name string
}

func NewDummyInferenceEngine() *DummyInferenceEngine {
	return &DummyInferenceEngine{Name: "DummyInferenceEngine"}
}

func (e *DummyInferenceEngine) PredictRapidScore(track hypersonic.RapidTrack) (hypersonic.RapidRankInference, error) {
	baseScore := 50.0
	reachable := track.DistanceToAssetKm <= 1500.0
	if reachable {
		baseScore += 30.0
	}
	relVel := math.Max(track.BurnoutVelocityKmS, 0.1)
	timeToThreatMin := track.DistanceToAssetKm / relVel / 60.0
	if timeToThreatMin < 30 {
		baseScore += 20.0
	}
	distanceFactor := math.Max(0, 1.0-(track.DistanceToAssetKm/3000.0))
	baseScore += distanceFactor * 20.0
	baseScore = math.Min(100, math.Max(0, baseScore))
	return hypersonic.RapidRankInference{
		Score:           baseScore,
		Confidence:      0.45,
		UncertaintyLow:  math.Max(0, baseScore-15.0),
		UncertaintyHigh: math.Min(100, baseScore+15.0),
		FeatureWeights: map[string]float64{
			"is_reachable":   float64(boolToInt(reachable)),
			"time_to_threat": timeToThreatMin,
			"distance":       track.DistanceToAssetKm,
			"mach":           track.Mach,
		},
	}, nil
}

func (e *DummyInferenceEngine) PredictManeuver(window []maneuver.ManeuverFrame) (maneuver.ManeuverInference, error) {
	if len(window) == 0 {
		return maneuver.ManeuverInference{}, fmt.Errorf("empty window")
	}
	var meanRes, varRes float64
	for _, frame := range window {
		meanRes += frame.Residual
	}
	meanRes /= float64(len(window))
	for _, frame := range window {
		diff := frame.Residual - meanRes
		varRes += diff * diff
	}
	varRes /= float64(len(window))
	stdRes := math.Sqrt(varRes)
	shap := maneuver.HeuristicSHAP(window)
	probability := 0.0
	manType := maneuver.ManeuverNone
	if stdRes > 2.0 {
		probability = 0.7
		manType = maneuver.ManeuverImpulsive
	} else if stdRes > 1.0 {
		probability = 0.4
		manType = maneuver.ManeuverLowThrust
	}
	return maneuver.ManeuverInference{
		Probability: probability,
		Type:        manType,
		DeltaV:      stdRes * 0.1,
		Confidence:  0.5,
		SHAP:        shap,
	}, nil
}

func (e *DummyInferenceEngine) PredictCollisionRisk(input proximity.CollisionRiskInput, graph proximity.CollisionNeighborhoodGraph) (proximity.CollisionGraphInference, error) {
	pc := proximity.AlfanoMaxPc(input.MissDistanceKm, input.HardbodyRadiusKm)
	uncertainty := input.SigmaRadialKm + input.SigmaIntrackKm + input.SigmaCrosstrackKm
	att := make(map[int]float64, len(graph.Nodes))
	if len(graph.Nodes) > 0 {
		w := 1.0 / float64(len(graph.Nodes))
		for _, n := range graph.Nodes {
			att[n.ID] = w
		}
	}
	return proximity.CollisionGraphInference{
		RiskScore:        pc,
		UncertaintyLower: math.Max(0, pc-uncertainty),
		UncertaintyUpper: math.Min(1.0, pc+uncertainty),
		Confidence:       0.55,
		NodeAttention:    att,
	}, nil
}

func (e *DummyInferenceEngine) IsHealthy() bool { return true }

type OnnxInferenceEngine struct {
	ModelDir string

	maneuverSession  *ort.DynamicAdvancedSession
	collisionSession *ort.DynamicAdvancedSession
	rapidSession     *ort.DynamicAdvancedSession
	maneuverOutputs  int
	collisionOutputs int
	rapidOutputs     int
	rapidInputWidth  int

	healthy bool
	mu      sync.Mutex
}

var (
	onnxEnvOnce sync.Once
	onnxEnvErr  error
)

func initializeOnnxEnvironment(modelDir string) error {
	onnxEnvOnce.Do(func() {
		path, discoverErr := discoverOnnxSharedLibrary(modelDir)
		if path != "" {
			ort.SetSharedLibraryPath(path)
		}

		onnxEnvErr = ort.InitializeEnvironment()
		if onnxEnvErr != nil && discoverErr != nil {
			onnxEnvErr = fmt.Errorf("%w; runtime library auto-discovery also failed: %v", onnxEnvErr, discoverErr)
		}
	})
	return onnxEnvErr
}

func discoverOnnxSharedLibrary(modelDir string) (string, error) {
	if explicit := strings.TrimSpace(os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")); explicit != "" {
		return explicit, nil
	}

	candidates := []string{
		filepath.Join(modelDir, "onnxruntime.so"),
		filepath.Join(modelDir, "libonnxruntime.so"),
		filepath.Join(modelDir, "libonnxruntime.dylib"),
		"onnxruntime.so",
		"libonnxruntime.so",
		"libonnxruntime.dylib",
	}

	home, _ := os.UserHomeDir()
	globPatterns := []string{
		"/opt/homebrew/lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.dylib",
		"/usr/local/lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.dylib",
		"/opt/homebrew/lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.so",
		"/usr/local/lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.so",
	}
	if home != "" {
		globPatterns = append(globPatterns,
			filepath.Join(home, "Library/Python/*/lib/python/site-packages/onnxruntime/capi/libonnxruntime*.dylib"),
			filepath.Join(home, ".virtualenvs/*/lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.dylib"),
			filepath.Join(home, "Library/Python/*/lib/python/site-packages/onnxruntime/capi/libonnxruntime*.so"),
			filepath.Join(home, ".virtualenvs/*/lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.so"),
		)
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}

	for _, pattern := range globPatterns {
		matches, _ := filepath.Glob(pattern)
		sort.Strings(matches)
		for _, m := range matches {
			if _, err := os.Stat(m); err == nil {
				return m, nil
			}
		}
	}

	if runtime.GOOS == "darwin" {
		return "", fmt.Errorf("no ONNX Runtime dylib found; set ONNXRUNTIME_SHARED_LIBRARY_PATH")
	}
	return "", fmt.Errorf("no ONNX Runtime shared library found; set ONNXRUNTIME_SHARED_LIBRARY_PATH")
}

func NewOnnxInferenceEngine(modelDir string) (*OnnxInferenceEngine, error) {
	if modelDir == "" {
		return nil, fmt.Errorf("model dir required")
	}
	if err := initializeOnnxEnvironment(modelDir); err != nil {
		return nil, fmt.Errorf("init onnx environment: %w", err)
	}

	maneuverPath := filepath.Join(modelDir, "maneuver_detector.onnx")
	collisionPath := filepath.Join(modelDir, "collision_risk.onnx")
	rapidPath := filepath.Join(modelDir, "rapid_ranker.onnx")

	for _, p := range []string{maneuverPath, collisionPath, rapidPath} {
		if _, err := os.Stat(p); err != nil {
			return nil, fmt.Errorf("model missing %s: %w", p, err)
		}
	}

	maneuverSession, maneuverOutputs, _, err := newDynamicSessionFromModelIO(maneuverPath)
	if err != nil {
		return nil, fmt.Errorf("create maneuver session: %w", err)
	}

	collisionSession, collisionOutputs, _, err := newDynamicSessionFromModelIO(collisionPath)
	if err != nil {
		_ = maneuverSession.Destroy()
		return nil, fmt.Errorf("create collision session: %w", err)
	}

	rapidSession, rapidOutputs, rapidInputWidth, err := newDynamicSessionFromModelIO(rapidPath)
	if err != nil {
		_ = maneuverSession.Destroy()
		_ = collisionSession.Destroy()
		return nil, fmt.Errorf("create rapid session: %w", err)
	}

	return &OnnxInferenceEngine{
		ModelDir:         modelDir,
		maneuverSession:  maneuverSession,
		collisionSession: collisionSession,
		rapidSession:     rapidSession,
		maneuverOutputs:  maneuverOutputs,
		collisionOutputs: collisionOutputs,
		rapidOutputs:     rapidOutputs,
		rapidInputWidth:  rapidInputWidth,
		healthy:          true,
	}, nil
}

func (e *OnnxInferenceEngine) PredictManeuver(window []maneuver.ManeuverFrame) (maneuver.ManeuverInference, error) {
	if len(window) == 0 {
		return maneuver.ManeuverInference{}, fmt.Errorf("empty maneuver window")
	}
	e.mu.Lock()
	defer e.mu.Unlock()

	flat := make([]float32, 0, len(window)*2)
	for _, f := range window {
		flat = append(flat, float32(f.Residual), float32(f.DeltaV))
	}
	in, err := ort.NewTensor(ort.NewShape(1, int64(len(window)), 2), flat)
	if err != nil {
		return maneuver.ManeuverInference{}, fmt.Errorf("maneuver input tensor: %w", err)
	}
	defer in.Destroy()

	inputs := []ort.Value{in}
	outputs := make([]ort.Value, e.maneuverOutputs)
	if err := e.maneuverSession.Run(inputs, outputs); err != nil {
		return maneuver.ManeuverInference{}, fmt.Errorf("run maneuver session: %w", err)
	}
	defer destroyValues(outputs)

	prob := firstFloatAt(outputs, 0, 0)
	typeIdx := int(firstFloatAt(outputs, 1, 0) + 0.5)
	deltaV := firstFloatAt(outputs, 2, 0)
	conf := firstFloatAt(outputs, 3, 0)

	mType := maneuver.ManeuverNone
	switch typeIdx {
	case 1:
		mType = maneuver.ManeuverImpulsive
	case 2:
		mType = maneuver.ManeuverLowThrust
	case 3:
		mType = maneuver.ManeuverHypersonic
	}
	return maneuver.ManeuverInference{
		Probability: clamp01(prob),
		Type:        mType,
		DeltaV:      math.Max(0, deltaV),
		Confidence:  clamp01(conf),
	}, nil
}

func (e *OnnxInferenceEngine) PredictCollisionRisk(input proximity.CollisionRiskInput, graph proximity.CollisionNeighborhoodGraph) (proximity.CollisionGraphInference, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	features := []float32{
		float32(input.MissDistanceKm),
		float32(input.HardbodyRadiusKm),
		float32(input.RelativeVelocityKmS),
		float32(input.TimeToTCASeconds),
		float32(input.SigmaRadialKm),
		float32(input.SigmaIntrackKm),
		float32(input.SigmaCrosstrackKm),
	}
	in, err := ort.NewTensor(ort.NewShape(1, int64(len(features))), features)
	if err != nil {
		return proximity.CollisionGraphInference{}, fmt.Errorf("collision input tensor: %w", err)
	}
	defer in.Destroy()

	inputs := []ort.Value{in}
	outputs := make([]ort.Value, e.collisionOutputs)
	if err := e.collisionSession.Run(inputs, outputs); err != nil {
		return proximity.CollisionGraphInference{}, fmt.Errorf("run collision session: %w", err)
	}
	defer destroyValues(outputs)

	att := make(map[int]float64, len(graph.Nodes))
	if len(graph.Nodes) > 0 {
		w := 1.0 / float64(len(graph.Nodes))
		for _, n := range graph.Nodes {
			att[n.ID] = w
		}
	}

	return proximity.CollisionGraphInference{
		RiskScore:        clamp01(firstFloatAt(outputs, 0, 0)),
		UncertaintyLower: clamp01(firstFloatAt(outputs, 1, 0)),
		UncertaintyUpper: clamp01(firstFloatAt(outputs, 2, 0)),
		Confidence:       clamp01(firstFloatAt(outputs, 3, 0)),
		NodeAttention:    att,
	}, nil
}

func (e *OnnxInferenceEngine) PredictRapidScore(track hypersonic.RapidTrack) (hypersonic.RapidRankInference, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	features := []float32{
		float32(track.Mach),
		float32(track.AltitudeKm),
		float32(track.BurnoutVelocityKmS),
		float32(track.FlightPathAngleRad),
		float32(track.LaunchLat),
		float32(track.LaunchLon),
		float32(track.DistanceToAssetKm),
		float32(track.RadarCrossSection),
		float32(track.ManeuverIntentScore),
	}
	if e.rapidInputWidth > 0 && len(features) < e.rapidInputWidth {
		extras := []float32{
			float32(track.DistanceToAssetKm / math.Max(track.BurnoutVelocityKmS, 0.1)),
			float32(track.Mach * track.BurnoutVelocityKmS),
			float32(track.AltitudeKm * math.Cos(track.FlightPathAngleRad)),
			float32(track.RadarCrossSection * (1 + track.ManeuverIntentScore)),
			float32(track.DistanceToAssetKm * math.Sin(track.FlightPathAngleRad)),
			float32(track.LaunchLat - track.LaunchLon),
		}
		for i := 0; len(features) < e.rapidInputWidth; i++ {
			features = append(features, extras[i%len(extras)])
		}
	} else if e.rapidInputWidth > 0 && len(features) > e.rapidInputWidth {
		features = features[:e.rapidInputWidth]
	}
	in, err := ort.NewTensor(ort.NewShape(1, int64(len(features))), features)
	if err != nil {
		return hypersonic.RapidRankInference{}, fmt.Errorf("rapid input tensor: %w", err)
	}
	defer in.Destroy()

	inputs := []ort.Value{in}
	outputs := make([]ort.Value, e.rapidOutputs)
	if err := e.rapidSession.Run(inputs, outputs); err != nil {
		return hypersonic.RapidRankInference{}, fmt.Errorf("run rapid session: %w", err)
	}
	defer destroyValues(outputs)

	score := firstFloatAt(outputs, 0, 0)
	conf := firstFloatAt(outputs, 1, 0)
	low := firstFloatAt(outputs, 2, 0)
	high := firstFloatAt(outputs, 3, 0)
	if low > high {
		low, high = high, low
	}

	return hypersonic.RapidRankInference{
		Score:           clampScore(score),
		Confidence:      clamp01(conf),
		UncertaintyLow:  clampScore(low),
		UncertaintyHigh: clampScore(high),
		FeatureWeights: map[string]float64{
			"mach":     track.Mach,
			"distance": track.DistanceToAssetKm,
			"altitude": track.AltitudeKm,
			"intent":   track.ManeuverIntentScore,
		},
	}, nil
}

func (e *OnnxInferenceEngine) IsHealthy() bool {
	return e != nil && e.healthy
}

func NewInferenceEngine(modelDir string) (InferenceEngine, error) {
	onnx, err := NewOnnxInferenceEngine(modelDir)
	if err != nil {
		return nil, fmt.Errorf("init ONNX inference engine from %s: %w", modelDir, err)
	}
	common.EmitInfoEvent("ml", "inference_engine", "onnx_initialized", "use_onnx", common.StrictModeEnabled(), map[string]interface{}{"model_dir": modelDir})
	fmt.Printf("[ML] Using ONNX InferenceEngine from %s\n", modelDir)
	return onnx, nil
}

func NewInferenceEngineWithFallback(modelDir string) (InferenceEngine, error) {
	engine, err := NewInferenceEngine(modelDir)
	if err == nil {
		return engine, nil
	}
	strict := common.StrictModeEnabled()
	common.EmitFallbackEvent("ml", "inference_engine", "onnx_unavailable", "use_dummy_engine", 0.0, strict, map[string]interface{}{
		"model_dir": modelDir,
		"error":     err.Error(),
	})
	if strict {
		return nil, fmt.Errorf("strict mode: ONNX inference required: %w", err)
	}
	fmt.Printf("[ML] ONNX unavailable (%v); using DummyInferenceEngine (heuristic fallback)\n", err)
	return NewDummyInferenceEngine(), nil
}

func newDynamicSessionFromModelIO(modelPath string) (*ort.DynamicAdvancedSession, int, int, error) {
	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("inspect io: %w", err)
	}
	inputNames := make([]string, 0, len(inputs))
	outputNames := make([]string, 0, len(outputs))
	inputWidth := 0
	for _, in := range inputs {
		if in.OrtValueType == ort.ONNXTypeTensor {
			inputNames = append(inputNames, in.Name)
			if inputWidth == 0 {
				inputWidth = inferInputWidth(in.Dimensions)
			}
		}
	}
	for _, out := range outputs {
		if out.OrtValueType == ort.ONNXTypeTensor {
			outputNames = append(outputNames, out.Name)
		}
	}
	if len(inputNames) == 0 || len(outputNames) == 0 {
		return nil, 0, 0, fmt.Errorf("no tensor inputs/outputs discovered")
	}
	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, nil)
	if err != nil {
		return nil, 0, 0, err
	}
	return session, len(outputNames), inputWidth, nil
}

func inferInputWidth(shape ort.Shape) int {
	if len(shape) == 0 {
		return 0
	}
	if len(shape) == 2 && shape[0] == 1 && shape[1] > 0 {
		return int(shape[1])
	}
	prod := 1
	for _, d := range shape {
		if d <= 0 {
			continue
		}
		prod *= int(d)
	}
	if prod <= 1 {
		return 0
	}
	return prod
}

func firstFloatAt(values []ort.Value, outputIdx int, tensorIdx int) float64 {
	if len(values) == 0 {
		return 0
	}
	if outputIdx < 0 {
		outputIdx = 0
	}
	if outputIdx >= len(values) {
		outputIdx = len(values) - 1
	}
	return firstFloat(values[outputIdx], tensorIdx)
}

func firstFloat(v ort.Value, idx int) float64 {
	if v == nil {
		return 0
	}
	switch t := v.(type) {
	case *ort.Tensor[float32]:
		data := t.GetData()
		if idx >= 0 && idx < len(data) {
			return float64(data[idx])
		}
	case *ort.Tensor[float64]:
		data := t.GetData()
		if idx >= 0 && idx < len(data) {
			return data[idx]
		}
	case *ort.Tensor[int64]:
		data := t.GetData()
		if idx >= 0 && idx < len(data) {
			return float64(data[idx])
		}
	case *ort.Tensor[int32]:
		data := t.GetData()
		if idx >= 0 && idx < len(data) {
			return float64(data[idx])
		}
	}
	return 0
}

func destroyValues(values []ort.Value) {
	for _, value := range values {
		if value != nil {
			_ = value.Destroy()
		}
	}
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func clampScore(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 200 {
		return 200
	}
	return v
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}
