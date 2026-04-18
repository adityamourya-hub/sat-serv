package catalog

import (
	"fmt"
	"math"
	"os"

	"github.com/yalue/onnxruntime_go"

	"ssis/pkg/octree"
)

// PatternOfLifeInference is a compact score bundle used to augment MHT likelihoods.
type PatternOfLifeInference struct {
	PatternScore float64 // [0,1] recurrent pattern consistency
	IntentProxy  float64 // [0,1] subtle maneuver/intent proxy
	Confidence   float64 // [0,1] model confidence in this inference
}

// PatternOfLifeModel augments MHT with recurrent temporal behavior scoring.
type PatternOfLifeModel interface {
	Name() string
	Infer(trackID int, history []TrackObservation, obs TrackObservation, candidate ObjectRecord, maneuverFlag bool) PatternOfLifeInference
	Close()
}

// SmallITransformerPOLModel provides a lightweight pattern-of-life scorer.
// If ONNX is available, it blends ONNX output with deterministic autocorrelation features.
// If ONNX is unavailable, it still provides useful autocorrelation/trend scoring.
type SmallITransformerPOLModel struct {
	session *onnxruntime_go.DynamicAdvancedSession
	enabled bool
	seqLen  int
}

func NewSmallITransformerPOLModel(modelPath string) (*SmallITransformerPOLModel, error) {
	m := &SmallITransformerPOLModel{seqLen: 16}
	if modelPath == "" {
		return m, nil
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return m, nil
	}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{"track_sequence"},
		[]string{"pattern_score", "intent_proxy", "confidence"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("load small iTransformer model: %w", err)
	}
	m.session = session
	m.enabled = true
	return m, nil
}

func (m *SmallITransformerPOLModel) Name() string {
	if m.enabled {
		return "small_itransformer_onnx"
	}
	return "autocorrelation_fallback"
}

func (m *SmallITransformerPOLModel) Infer(trackID int, history []TrackObservation, obs TrackObservation, candidate ObjectRecord, maneuverFlag bool) PatternOfLifeInference {
	base := fallbackPatternOfLifeInference(history, obs, candidate, maneuverFlag)
	if !m.enabled || m.session == nil {
		return base
	}
	if len(history) < 4 {
		return base
	}

	seq := history
	if len(seq) > m.seqLen {
		seq = seq[len(seq)-m.seqLen:]
	}

	// Features per step: dx, dy, dz to candidate, speed proxy, dt(min), sigma
	data := make([]float32, m.seqLen*6)
	for i := 0; i < m.seqLen; i++ {
		idx := i
		if i >= len(seq) {
			idx = len(seq) - 1
		}
		h := seq[idx]
		baseIdx := i * 6
		dx := h.Pos.X - candidate.Pos.X
		dy := h.Pos.Y - candidate.Pos.Y
		dz := h.Pos.Z - candidate.Pos.Z
		data[baseIdx+0] = float32(dx)
		data[baseIdx+1] = float32(dy)
		data[baseIdx+2] = float32(dz)

		speed := 0.0
		dtMin := 1.0
		if idx > 0 {
			prev := seq[idx-1]
			dtSec := float64(h.TimestampUnix - prev.TimestampUnix)
			if dtSec < 1 {
				dtSec = 1
			}
			dtMin = dtSec / 60.0
			speed = distance3(h.Pos, prev.Pos) / dtSec
		}
		data[baseIdx+3] = float32(speed)
		data[baseIdx+4] = float32(dtMin)
		sig := h.SigmaKm
		if sig <= 0 {
			sig = obs.SigmaKm
		}
		data[baseIdx+5] = float32(sig)
	}

	inShape := onnxruntime_go.NewShape(1, int64(m.seqLen), 6)
	inTensor, err := onnxruntime_go.NewTensor(inShape, data)
	if err != nil {
		return base
	}
	defer inTensor.Destroy()

	inputs := []onnxruntime_go.Value{inTensor}
	outputs := []onnxruntime_go.Value{nil, nil, nil}
	if err := m.session.Run(inputs, outputs); err != nil {
		return base
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	onnxPattern := firstTensorScalar(outputs[0])
	onnxIntent := firstTensorScalar(outputs[1])
	onnxConf := firstTensorScalar(outputs[2])

	w := clamp01POL(0.3 + 0.7*onnxConf)
	return PatternOfLifeInference{
		PatternScore: clamp01POL((1.0-w)*base.PatternScore + w*onnxPattern),
		IntentProxy:  clamp01POL((1.0-w)*base.IntentProxy + w*onnxIntent),
		Confidence:   clamp01POL((1.0-w)*base.Confidence + w*onnxConf),
	}
}

func (m *SmallITransformerPOLModel) Close() {
	if m.session != nil {
		m.session.Destroy()
	}
}

func fallbackPatternOfLifeInference(history []TrackObservation, obs TrackObservation, candidate ObjectRecord, maneuverFlag bool) PatternOfLifeInference {
	if len(history) == 0 {
		return PatternOfLifeInference{PatternScore: 0.5, IntentProxy: boolToFloat(maneuverFlag), Confidence: 0.35}
	}

	seq := history
	if len(seq) > 32 {
		seq = seq[len(seq)-32:]
	}

	residualSeries := make([]float64, 0, len(seq)+1)
	for _, h := range seq {
		residualSeries = append(residualSeries, distance3(h.Pos, candidate.Pos))
	}
	residualSeries = append(residualSeries, distance3(obs.Pos, candidate.Pos))

	// Trend consistency: compare observed residual to 1-step prediction.
	predResidual := residualSeries[len(residualSeries)-1]
	if len(residualSeries) >= 3 {
		rn1 := residualSeries[len(residualSeries)-2]
		rn2 := residualSeries[len(residualSeries)-3]
		predResidual = rn1 + (rn1 - rn2)
	}
	trendErr := math.Abs(residualSeries[len(residualSeries)-1] - predResidual)
	trendScore := math.Exp(-trendErr / 15.0)

	// Autocorrelation of residual deltas as pattern-of-life stability proxy.
	deltas := make([]float64, 0, len(residualSeries)-1)
	for i := 1; i < len(residualSeries); i++ {
		deltas = append(deltas, residualSeries[i]-residualSeries[i-1])
	}
	auto := lag1Autocorrelation(deltas)
	autoScore := clamp01POL(0.5 + 0.5*auto)

	// Subtle intent proxy: persistent low-amplitude but coherent drift during maneuver periods.
	driftMean := meanAbs(deltas)
	intent := clamp01POL(0.4*autoScore + 0.4*(1.0-math.Exp(-driftMean/2.0)) + 0.2*boolToFloat(maneuverFlag))

	pattern := clamp01POL(0.6*trendScore + 0.4*autoScore)
	conf := clamp01POL(0.35 + 0.4*float64(len(seq))/32.0 + 0.25*autoScore)

	return PatternOfLifeInference{
		PatternScore: pattern,
		IntentProxy:  intent,
		Confidence:   conf,
	}
}

func firstTensorScalar(v onnxruntime_go.Value) float64 {
	if v == nil {
		return 0
	}
	switch t := v.(type) {
	case *onnxruntime_go.Tensor[float32]:
		d := t.GetData()
		if len(d) > 0 {
			return float64(d[0])
		}
	case *onnxruntime_go.Tensor[float64]:
		d := t.GetData()
		if len(d) > 0 {
			return d[0]
		}
	}
	return 0
}

func lag1Autocorrelation(x []float64) float64 {
	if len(x) < 3 {
		return 0
	}
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))

	num := 0.0
	den := 0.0
	for i := 1; i < len(x); i++ {
		num += (x[i] - mean) * (x[i-1] - mean)
	}
	for _, v := range x {
		d := v - mean
		den += d * d
	}
	if den <= 1e-12 {
		return 0
	}
	return clampRange(num/den, -1, 1)
}

func meanAbs(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	s := 0.0
	for _, v := range x {
		s += math.Abs(v)
	}
	return s / float64(len(x))
}

func distance3(a, b octree.Vector3) float64 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	dz := a.Z - b.Z
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

func clamp01POL(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func clampRange(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func boolToFloat(v bool) float64 {
	if v {
		return 1
	}
	return 0
}
