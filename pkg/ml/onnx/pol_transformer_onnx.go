//go:build onnx

package onnx

import (
	"fmt"
	"math"
	"time"

	onnx "github.com/yalue/onnxruntime_go"

	"ssis/pkg/orbital"
)

/**
 * POLTransformerModel wraps the Axiom-Zero Pattern-of-Life Transformer ONNX model.
 *
 * This model provides self-supervised behavior classification for space objects:
 * - Learns "normal" behavior per object class (LEO-comms, MEO-nav, GEO-station, etc.)
 * - Predicts anomaly scores for uncorrelated track (UCT) classification
 * - Provides latent embeddings for similarity-based catalog association
 *
 * Architecture: 6-layer Transformer, 256-dim, 4-head attention
 *
 * Input:  (batch, seq_len, 6) orbital state sequence [x,y,z,vx,vy,vz]
 * Output:
 *   - behavior_class: (batch, n_classes) softmax over behavior types
 *   - anomaly_score: (batch, 1) 0=normal, 1=anomalous
 *   - embedding: (batch, 256) latent representation
 */
type POLTransformerModel struct {
	session      *onnx.AdvancedSession
	input        *onnx.Tensor
	outputClass  *onnx.Tensor
	outputAnomaly *onnx.Tensor
	outputEmbed  *onnx.Tensor
	seqLen       int
	nClasses     int
}

// ObjectBehaviorClass represents the classified behavior type.
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

// POLClassProbability represents classification probabilities.
type POLClassProbability struct {
	Class       ObjectBehaviorClass
	Probability float64
}

// POLTransformerOutput represents the complete model output.
type POLTransformerOutput struct {
	ObjectID        int
	BehaviorClass   ObjectBehaviorClass
	ClassConfidence float64 // Top-1 probability
	ClassProbs      []POLClassProbability // All class probabilities
	AnomalyScore    float64 // 0=normal, 1=anomalous
	IsAnomaly       bool
	Embedding       [256]float64 // Latent representation
	InferenceTime   time.Time
}

// StateSequence represents orbital state history for POL analysis.
type StateSequence struct {
	ObjectID int
	States   []orbital.State // Should be sorted by time
	Metadata map[string]string
}

// NewPOLTransformerModel loads the POL Transformer from the given ONNX path.
func NewPOLTransformerModel(modelPath string, seqLen int, nClasses int) (*POLTransformerModel, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required for POL transformer")
	}
	if seqLen <= 0 {
		seqLen = 50 // Default: 50 timesteps
	}
	if nClasses <= 0 {
		nClasses = 8 // Default: 8 behavior classes
	}

	// Input: (batch=1, seq_len, 6)
	inputShape := []int64{1, int64(seqLen), 6}
	inputTensor, err := onnx.NewEmptyTensor[float64](inputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	// Output 1: behavior class probabilities (batch=1, n_classes)
	outputClassShape := []int64{1, int64(nClasses)}
	outputClass, err := onnx.NewEmptyTensor[float64](outputClassShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output class tensor: %w", err)
	}

	// Output 2: anomaly score (batch=1, 1)
	outputAnomalyShape := []int64{1, 1}
	outputAnomaly, err := onnx.NewEmptyTensor[float64](outputAnomalyShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output anomaly tensor: %w", err)
	}

	// Output 3: embedding (batch=1, 256)
	outputEmbedShape := []int64{1, 256}
	outputEmbed, err := onnx.NewEmptyTensor[float64](outputEmbedShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output embedding tensor: %w", err)
	}

	// Create ONNX session
	session, err := onnx.NewAdvancedSession(
		modelPath,
		[]string{"behavior_class", "anomaly_score", "embedding"}, // Output names
		[]string{"state_sequence"},                               // Input name
		[]*onnx.Tensor{inputTensor},
		[]*onnx.Tensor{outputClass, outputAnomaly, outputEmbed},
		nil,
	)
	if err != nil {
		// Fallback: try alternative naming
		session, err = onnx.NewAdvancedSession(
			modelPath,
			[]string{"output_0", "output_1", "output_2"},
			[]string{"input"},
			[]*onnx.Tensor{inputTensor},
			[]*onnx.Tensor{outputClass, outputAnomaly, outputEmbed},
			nil,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create ONNX session: %w", err)
		}
	}

	return &POLTransformerModel{
		session:       session,
		input:         inputTensor,
		outputClass:   outputClass,
		outputAnomaly: outputAnomaly,
		outputEmbed:   outputEmbed,
		seqLen:        seqLen,
		nClasses:      nClasses,
	}, nil
}

// Name returns the model identifier.
func (m *POLTransformerModel) Name() string {
	return "axiom_pol_transformer"
}

// Version returns the model version.
func (m *POLTransformerModel) Version() string {
	return "3.0.0"
}

// Predict runs inference on a state sequence.
func (m *POLTransformerModel) Predict(sequence StateSequence) (POLTransformerOutput, error) {
	if len(sequence.States) < m.seqLen {
		return POLTransformerOutput{}, fmt.Errorf("insufficient states: got %d, need %d",
			len(sequence.States), m.seqLen)
	}

	// Populate input tensor
	data := m.input.GetData()
	startIdx := len(sequence.States) - m.seqLen

	for t := 0; t < m.seqLen; t++ {
		state := sequence.States[startIdx+t]
		baseIdx := t * 6
		data[baseIdx+0] = state.X
		data[baseIdx+1] = state.Y
		data[baseIdx+2] = state.Z
		data[baseIdx+3] = state.VX
		data[baseIdx+4] = state.VY
		data[baseIdx+5] = state.VZ
	}

	// Run inference
	if err := m.session.Run(); err != nil {
		return POLTransformerOutput{}, fmt.Errorf("inference failed: %w", err)
	}

	// Extract outputs
	classData := m.outputClass.GetData()
	anomalyData := m.outputAnomaly.GetData()
	embedData := m.outputEmbed.GetData()

	// Find top class
	maxProb := classData[0]
	maxIdx := 0
	for i := 1; i < m.nClasses && i < len(classData); i++ {
		if classData[i] > maxProb {
			maxProb = classData[i]
			maxIdx = i
		}
	}

	// Build class probabilities
	classProbs := make([]POLClassProbability, 0, m.nClasses)
	for i := 0; i < m.nClasses && i < len(classData); i++ {
		classProbs = append(classProbs, POLClassProbability{
			Class:       ObjectBehaviorClass(i),
			Probability: classData[i],
		})
	}

	// Sort by probability descending
	for i := range classProbs {
		for j := i + 1; j < len(classProbs); j++ {
			if classProbs[j].Probability > classProbs[i].Probability {
				classProbs[i], classProbs[j] = classProbs[j], classProbs[i]
			}
		}
	}

	// Anomaly detection
	anomalyScore := anomalyData[0]
	if anomalyScore < 0 {
		anomalyScore = 0
	}
	if anomalyScore > 1 {
		anomalyScore = 1
	}

	// Extract embedding
	var embedding [256]float64
	copyLen := len(embedData)
	if copyLen > 256 {
		copyLen = 256
	}
	for i := 0; i < copyLen; i++ {
		embedding[i] = embedData[i]
	}

	return POLTransformerOutput{
		ObjectID:        sequence.ObjectID,
		BehaviorClass:   ObjectBehaviorClass(maxIdx),
		ClassConfidence: maxProb,
		ClassProbs:      classProbs,
		AnomalyScore:    anomalyScore,
		IsAnomaly:       anomalyScore > 0.7, // Threshold for anomaly flag
		Embedding:       embedding,
		InferenceTime:   time.Now(),
	}, nil
}

// PredictBatch runs batch inference.
func (m *POLTransformerModel) PredictBatch(sequences []StateSequence) ([]POLTransformerOutput, error) {
	results := make([]POLTransformerOutput, len(sequences))
	for i, seq := range sequences {
		out, err := m.Predict(seq)
		if err != nil {
			return nil, fmt.Errorf("sequence %d (object %d) failed: %w", i, seq.ObjectID, err)
		}
		results[i] = out
	}
	return results, nil
}

// ClassifyUCT classifies an uncorrelated track (UCT) based on POL analysis.
// Returns the classification and a confidence score.
func (m *POLTransformerModel) ClassifyUCT(sequence StateSequence) (ObjectBehaviorClass, float64, error) {
	output, err := m.Predict(sequence)
	if err != nil {
		return ClassUnknown, 0, err
	}

	// If anomaly score is high, the track is unusual
	if output.IsAnomaly {
		// Still return the top class but with reduced confidence
		return output.BehaviorClass, output.ClassConfidence * (1 - output.AnomalyScore), nil
	}

	return output.BehaviorClass, output.ClassConfidence, nil
}

// ComputeSimilarity computes cosine similarity between two object embeddings.
func (m *POLTransformerModel) ComputeSimilarity(emb1, emb2 [256]float64) float64 {
	dot := 0.0
	norm1 := 0.0
	norm2 := 0.0

	for i := 0; i < 256; i++ {
		dot += emb1[i] * emb2[i]
		norm1 += emb1[i] * emb1[i]
		norm2 += emb2[i] * emb2[i]
	}

	if norm1 == 0 || norm2 == 0 {
		return 0
	}

	return dot / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

// FindSimilarObjects finds objects with similar behavior patterns.
func (m *POLTransformerModel) FindSimilarObjects(
	reference POLTransformerOutput,
	candidates []POLTransformerOutput,
	threshold float64,
	topK int,
) []POLTransformerOutput {
	type scored struct {
		obj   POLTransformerOutput
		score float64
	}

	scoredObjs := make([]scored, len(candidates))
	for i, candidate := range candidates {
		sim := m.ComputeSimilarity(reference.Embedding, candidate.Embedding)
		scoredObjs[i] = scored{obj: candidate, score: sim}
	}

	// Filter by threshold and sort
	filtered := make([]scored, 0)
	for _, so := range scoredObjs {
		if so.score >= threshold {
			filtered = append(filtered, so)
		}
	}

	// Sort by score descending
	for i := range filtered {
		for j := i + 1; j < len(filtered); j++ {
			if filtered[j].score > filtered[i].score {
				filtered[i], filtered[j] = filtered[j], filtered[i]
			}
		}
	}

	// Return top K
	if topK > len(filtered) {
		topK = len(filtered)
	}

	result := make([]POLTransformerOutput, topK)
	for i := 0; i < topK; i++ {
		result[i] = filtered[i].obj
	}

	return result
}

// Close releases the ONNX session.
func (m *POLTransformerModel) Close() error {
	if m.session != nil {
		return m.session.Destroy()
	}
	return nil
}

// ModelInfo returns metadata.
func (m *POLTransformerModel) ModelInfo() ModelInfo {
	return ModelInfo{
		ModelID:     m.Name(),
		Version:     m.Version(),
		InputNames:  []string{"state_sequence"},
		OutputNames: []string{"behavior_class", "anomaly_score", "embedding"},
	}
}

// InferBatch implements the Inferrer interface.
func (m *POLTransformerModel) InferBatch(requests []InferenceRequest) []InferenceResult {
	results := make([]InferenceResult, len(requests))

	for i, req := range requests {
		sequence := StateSequence{ObjectID: -1}

		if states, ok := req.Inputs["states"].([]orbital.State); ok {
			sequence.States = states
		}
		if objID, ok := req.Inputs["object_id"].(int); ok {
			sequence.ObjectID = objID
		}

		output, err := m.Predict(sequence)
		results[i] = InferenceResult{
			RequestID:  req.ID,
			ModelID:    m.Name(),
			Version:    m.Version(),
			Confidence: output.ClassConfidence,
			Outputs: map[string]interface{}{
				"behavior_class":   output.BehaviorClass.String(),
				"class_confidence": output.ClassConfidence,
				"anomaly_score":    output.AnomalyScore,
				"is_anomaly":       output.IsAnomaly,
			},
			Error: err,
		}
	}

	return results
}
