//go:build !onnx
// STUB: This file is a build stub. All functions return "not yet implemented" until the real backend is bound.

package onnx

import (
	"ssis/pkg/physics"
	"time"
)

type POLTransformerModel struct {
	seqLen   int
	nClasses int
}

type ObjectBehaviorClass int

const (
	ClassUnknown      ObjectBehaviorClass = 0
	ClassLEOComms     ObjectBehaviorClass = 1
	ClassMEONav       ObjectBehaviorClass = 2
	ClassGEOStation   ObjectBehaviorClass = 3
	ClassHEOMolniya   ObjectBehaviorClass = 4
	ClassLEOImaging   ObjectBehaviorClass = 5
	ClassLEOScience   ObjectBehaviorClass = 6
	ClassDebris       ObjectBehaviorClass = 7
)

func (obc ObjectBehaviorClass) String() string {
	switch obc {
	case ClassUnknown:
		return "unknown"
	case ClassLEOComms:
		return "leo_communications"
	case ClassMEONav:
		return "meo_navigation"
	case ClassGEOStation:
		return "geo_stationary"
	case ClassHEOMolniya:
		return "heo_molniya"
	case ClassLEOImaging:
		return "leo_imaging"
	case ClassLEOScience:
		return "leo_science"
	case ClassDebris:
		return "debris"
	default:
		return "unclassified"
	}
}

type POLClassProbability struct {
	Class       ObjectBehaviorClass
	Probability float64
}

type POLTransformerOutput struct {
	ObjectID        int
	BehaviorClass   ObjectBehaviorClass
	ClassConfidence float64
	ClassProbs      []POLClassProbability
	AnomalyScore    float64
	IsAnomaly       bool
	Embedding       [256]float64
	InferenceTime   time.Time
}

type StateSequence struct {
	ObjectID int
	States   []physics.State
	Metadata map[string]string
}

func NewPOLTransformerModel(modelPath string, seqLen int, nClasses int) (*POLTransformerModel, error) {
	if seqLen <= 0 {
		seqLen = 50
	}
	if nClasses <= 0 {
		nClasses = 8
	}
	return &POLTransformerModel{seqLen: seqLen, nClasses: nClasses}, nil
}

func (m *POLTransformerModel) Name() string {
	return "axiom_pol_transformer_stub"
}

func (m *POLTransformerModel) Version() string {
	return "3.0.0-stub"
}

func (m *POLTransformerModel) Predict(sequence StateSequence) (POLTransformerOutput, error) {
	return POLTransformerOutput{
		ObjectID:        sequence.ObjectID,
		BehaviorClass:   ClassUnknown,
		ClassConfidence: 0,
		ClassProbs:      []POLClassProbability{{Class: ClassUnknown, Probability: 1.0}},
		AnomalyScore:    0,
		IsAnomaly:       false,
		Embedding:       [256]float64{},
		InferenceTime:   time.Now(),
	}, nil
}

func (m *POLTransformerModel) PredictBatch(sequences []StateSequence) ([]POLTransformerOutput, error) {
	results := make([]POLTransformerOutput, len(sequences))
	for i, seq := range sequences {
		out, _ := m.Predict(seq)
		results[i] = out
	}
	return results, nil
}

func (m *POLTransformerModel) ClassifyUCT(sequence StateSequence) (ObjectBehaviorClass, float64, error) {
	return ClassUnknown, 0, nil
}

func (m *POLTransformerModel) ComputeSimilarity(emb1, emb2 [256]float64) float64 {
	return 0
}

func (m *POLTransformerModel) FindSimilarObjects(
	reference POLTransformerOutput,
	candidates []POLTransformerOutput,
	threshold float64,
	topK int,
) []POLTransformerOutput {
	return []POLTransformerOutput{}
}

func (m *POLTransformerModel) Close() error {
	return nil
}

func (m *POLTransformerModel) ModelInfo() ModelInfo {
	return ModelInfo{
		ModelID:     m.Name(),
		Version:     m.Version(),
		InputNames:  []string{"state_sequence"},
		OutputNames: []string{"behavior_class", "anomaly_score", "embedding"},
	}
}

func (m *POLTransformerModel) InferBatch(requests []InferenceRequest) []InferenceResult {
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
