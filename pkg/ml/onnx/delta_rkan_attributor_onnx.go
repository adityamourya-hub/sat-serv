//go:build onnx

package onnx

import (
	"fmt"
	"math"
	"time"

	onnx "github.com/yalue/onnxruntime_go"

	"ssis/pkg/physics"
)

/**
 * DeltaRKANAttributorModel wraps the Axiom-Zero Delta-rKAN Maneuver Attributor ONNX model.
 *
 * This model provides symbolic maneuver detection and attribution by:
 * 1. Processing state history through LSTM encoder
 * 2. Applying Delta-rKAN residual learning to identify Δv events
 * 3. Classifying maneuver type and estimating delta-V in RTN frame
 * 4. Generating human-readable LaTeX explanations
 *
 * Input:  (batch, timesteps, 6) state history [x,y,z,vx,vy,vz]
 * Output:
 *   - maneuver_prob: (batch, 1) probability of maneuver
 *   - delta_v: (batch, 3) estimated delta-V in RTN frame (m/s)
 *   - maneuver_type: (batch, 4) one-hot [none, station_keep, phasing, plane_change]
 */
type DeltaRKANAttributorModel struct {
	session             *onnx.AdvancedSession
	input               *onnx.Tensor[float32]
	outputProb          *onnx.Tensor[float32]
	outputDeltaV        *onnx.Tensor[float32]
	outputType          *onnx.Tensor[float32]
	timesteps           int
	confidenceThreshold float64
}

// ManeuverType represents the classified maneuver type.
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

// DeltaVRTN represents delta-V components in Radial-Transverse-Normal frame.
type DeltaVRTN struct {
	Radial     float64 // m/s, along position vector
	Transverse float64 // m/s, along velocity direction in orbital plane
	Normal     float64 // m/s, perpendicular to orbital plane
}

// Magnitude returns the total delta-V magnitude.
func (dv DeltaVRTN) Magnitude() float64 {
	return math.Sqrt(dv.Radial*dv.Radial + dv.Transverse*dv.Transverse + dv.Normal*dv.Normal)
}

// DeltaRKANOutput represents the complete maneuver attribution result.
type DeltaRKANOutput struct {
	ManeuverDetected bool
	ManeuverProb     float64
	ManeuverType     ManeuverType
	TypeConfidence   float64
	DeltaV           DeltaVRTN // m/s
	Confidence       float64   // Overall confidence 0-1
	LaTeXExplanation string    // Human-readable LaTeX formula
	DetectionTime    time.Time
}

// StateHistory represents a sequence of orbital states for maneuver detection.
type StateHistory struct {
	ObjectID int
	States   []physics.State // Must be sorted by time, ascending
}

// NewDeltaRKANAttributorModel loads the Delta-rKAN model from the given ONNX path.
func NewDeltaRKANAttributorModel(modelPath string, timesteps int) (*DeltaRKANAttributorModel, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required for Delta-rKAN attributor")
	}
	if timesteps <= 0 {
		timesteps = 20 // Default: 20 timesteps of history
	}

	// Create input tensor: (batch=1, timesteps, 6)
	inputShape := []int64{1, int64(timesteps), 6}
	inputTensor, err := onnx.NewEmptyTensor[float64](inputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	// Create output tensors
	// Output 1: maneuver probability (batch=1, 1)
	outputProbShape := []int64{1, 1}
	outputProb, err := onnx.NewEmptyTensor[float64](outputProbShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output prob tensor: %w", err)
	}

	// Output 2: delta-V (batch=1, 3)
	outputDeltaVShape := []int64{1, 3}
	outputDeltaV, err := onnx.NewEmptyTensor[float64](outputDeltaVShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output delta-v tensor: %w", err)
	}

	// Output 3: maneuver type (batch=1, 4)
	outputTypeShape := []int64{1, 4}
	outputType, err := onnx.NewEmptyTensor[float64](outputTypeShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output type tensor: %w", err)
	}

	// Create ONNX session
	session, err := onnx.NewAdvancedSession(
		modelPath,
		[]string{"maneuver_prob", "delta_v", "maneuver_type"}, // Output names
		[]string{"state_history"},                             // Input name
		[]*onnx.Tensor{inputTensor},
		[]*onnx.Tensor{outputProb, outputDeltaV, outputType},
		nil,
	)
	if err != nil {
		// Fallback: try alternative naming
		session, err = onnx.NewAdvancedSession(
			modelPath,
			[]string{"output_0", "output_1", "output_2"},
			[]string{"input"},
			[]*onnx.Tensor{inputTensor},
			[]*onnx.Tensor{outputProb, outputDeltaV, outputType},
			nil,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create ONNX session: %w", err)
		}
	}

	return &DeltaRKANAttributorModel{
		session:             session,
		input:               inputTensor,
		outputProb:          outputProb,
		outputDeltaV:        outputDeltaV,
		outputType:          outputType,
		timesteps:           timesteps,
		confidenceThreshold: 0.7,
	}, nil
}

// Name returns the model identifier.
func (m *DeltaRKANAttributorModel) Name() string {
	return "axiom_delta_rkan_attributor"
}

// Version returns the model version.
func (m *DeltaRKANAttributorModel) Version() string {
	return "3.0.0"
}

// SetConfidenceThreshold sets the minimum confidence for maneuver detection.
func (m *DeltaRKANAttributorModel) SetConfidenceThreshold(threshold float64) {
	m.confidenceThreshold = threshold
}

// Predict runs inference on a state history and returns maneuver attribution.
func (m *DeltaRKANAttributorModel) Predict(history StateHistory) (DeltaRKANOutput, error) {
	if len(history.States) < m.timesteps {
		return DeltaRKANOutput{}, fmt.Errorf("insufficient history: got %d, need %d",
			len(history.States), m.timesteps)
	}

	// Populate input tensor with most recent timesteps
	data := m.input.GetData()
	startIdx := len(history.States) - m.timesteps

	for t := 0; t < m.timesteps; t++ {
		state := history.States[startIdx+t]
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
		return DeltaRKANOutput{}, fmt.Errorf("inference failed: %w", err)
	}

	// Extract outputs
	probData := m.outputProb.GetData()
	deltaVData := m.outputDeltaV.GetData()
	typeData := m.outputType.GetData()

	maneuverProb := probData[0]

	// Determine maneuver type from one-hot output
	typeProbs := []float64{typeData[0], typeData[1], typeData[2], typeData[3]}
	maxProb := typeProbs[0]
	maxIdx := 0
	for i := 1; i < 4; i++ {
		if typeProbs[i] > maxProb {
			maxProb = typeProbs[i]
			maxIdx = i
		}
	}

	maneuverType := ManeuverType(maxIdx)
	typeConfidence := maxProb

	// Calculate overall confidence
	confidence := maneuverProb * typeConfidence
	if confidence < 0 {
		confidence = 0
	}
	if confidence > 1 {
		confidence = 1
	}

	// Build output
	output := DeltaRKANOutput{
		ManeuverDetected: maneuverProb > m.confidenceThreshold,
		ManeuverProb:     maneuverProb,
		ManeuverType:     maneuverType,
		TypeConfidence:   typeConfidence,
		DeltaV: DeltaVRTN{
			Radial:     deltaVData[0],
			Transverse: deltaVData[1],
			Normal:     deltaVData[2],
		},
		Confidence:    confidence,
		DetectionTime: time.Now(),
	}

	// Generate LaTeX explanation
	output.LaTeXExplanation = m.generateLaTeX(output)

	return output, nil
}

// generateLaTeX creates a human-readable LaTeX formula describing the maneuver.
func (m *DeltaRKANAttributorModel) generateLaTeX(output DeltaRKANOutput) string {
	dv := output.DeltaV
	mag := dv.Magnitude()

	switch output.ManeuverType {
	case ManeuverNone:
		return "\\text{No maneuver detected}"

	case ManeuverStationKeep:
		return fmt.Sprintf("\\Delta v = %.2f \\text{ m/s (station-keep)}", mag)

	case ManeuverPlaneChange:
		// Estimate inclination change from normal component
		diDeg := math.Abs(dv.Normal) / 7500.0 * 180.0 / math.Pi
		return fmt.Sprintf("\\Delta v_N = %+.2f \\text{ m/s} \\Rightarrow \\Delta i \\approx %.2f°",
			dv.Normal, diDeg)

	case ManeuverPhasing:
		// Estimate semi-major axis change from transverse component
		// dv_t ≈ 0.5 * da * sqrt(mu/a^3)
		// For LEO: da ≈ 2 * dv_t / 7.5 km (approximate)
		daKm := 2.0 * math.Abs(dv.Transverse) / 7.5
		direction := "raising"
		if dv.Transverse < 0 {
			direction = "lowering"
		}
		return fmt.Sprintf("\\Delta v_T = %+.2f \\text{ m/s} \\Rightarrow \\Delta a \\approx %.1f \\text{ km (%s)}",
			dv.Transverse, daKm, direction)

	default:
		return fmt.Sprintf("\\vec{\\Delta v} = (%.1f, %.1f, %.1f) \\text{ m/s (RTN)}",
			dv.Radial, dv.Transverse, dv.Normal)
	}
}

// ComputeRTNFrame computes the RTN frame unit vectors from a state.
// Returns (R_hat, T_hat, N_hat) as 3-element arrays.
func ComputeRTNFrame(state physics.State) ([3]float64, [3]float64, [3]float64) {
	// R: Radial (position direction)
	r := [3]float64{state.X, state.Y, state.Z}
	rMag := math.Sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])
	var rHat [3]float64
	if rMag > 0 {
		rHat[0] = r[0] / rMag
		rHat[1] = r[1] / rMag
		rHat[2] = r[2] / rMag
	}

	// N: Normal (angular momentum direction)
	v := [3]float64{state.VX, state.VY, state.VZ}
	h := [3]float64{
		r[1]*v[2] - r[2]*v[1],
		r[2]*v[0] - r[0]*v[2],
		r[0]*v[1] - r[1]*v[0],
	}
	hMag := math.Sqrt(h[0]*h[0] + h[1]*h[1] + h[2]*h[2])
	var nHat [3]float64
	if hMag > 0 {
		nHat[0] = h[0] / hMag
		nHat[1] = h[1] / hMag
		nHat[2] = h[2] / hMag
	}

	// T: Transverse (N × R, completes right-handed frame)
	tHat := [3]float64{
		nHat[1]*rHat[2] - nHat[2]*rHat[1],
		nHat[2]*rHat[0] - nHat[0]*rHat[2],
		nHat[0]*rHat[1] - nHat[1]*rHat[0],
	}
	tMag := math.Sqrt(tHat[0]*tHat[0] + tHat[1]*tHat[1] + tHat[2]*tHat[2])
	if tMag > 0 {
		tHat[0] /= tMag
		tHat[1] /= tMag
		tHat[2] /= tMag
	}

	return rHat, tHat, nHat
}

// Close releases the ONNX session.
func (m *DeltaRKANAttributorModel) Close() error {
	if m.session != nil {
		return m.session.Destroy()
	}
	return nil
}

// ModelInfo returns metadata about the loaded model.
func (m *DeltaRKANAttributorModel) ModelInfo() ModelInfo {
	return ModelInfo{
		ModelID:     m.Name(),
		Version:     m.Version(),
		InputNames:  []string{"state_history"},
		OutputNames: []string{"maneuver_prob", "delta_v", "maneuver_type"},
	}
}

// InferBatch implements the Inferrer interface.
func (m *DeltaRKANAttributorModel) InferBatch(requests []InferenceRequest) []InferenceResult {
	results := make([]InferenceResult, len(requests))

	for i, req := range requests {
		// Parse state history from request
		history := StateHistory{ObjectID: -1}
		if states, ok := req.Inputs["states"].([]physics.State); ok {
			history.States = states
		} else if states, ok := req.Inputs["states"].([]interface{}); ok {
			// Convert interface slice
			history.States = make([]physics.State, len(states))
			for j, s := range states {
				if state, ok := s.(physics.State); ok {
					history.States[j] = state
				}
			}
		}

		if objID, ok := req.Inputs["object_id"].(int); ok {
			history.ObjectID = objID
		}

		output, err := m.Predict(history)
		results[i] = InferenceResult{
			RequestID:  req.ID,
			ModelID:    m.Name(),
			Version:    m.Version(),
			Confidence: output.Confidence,
			Outputs: map[string]interface{}{
				"maneuver_detected": output.ManeuverDetected,
				"maneuver_type":     output.ManeuverType.String(),
				"delta_v_magnitude": output.DeltaV.Magnitude(),
				"latex_explanation": output.LaTeXExplanation,
			},
			Error: err,
		}
	}

	return results
}
