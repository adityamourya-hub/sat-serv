package orbital

import (
	"fmt"
	"os"
	"sync"

	"github.com/yalue/onnxruntime_go"

	safemath "ssis/pkg/math"
)

type HybridTransformer struct {
	session *onnxruntime_go.DynamicAdvancedSession
	enabled bool
	history map[int][]State
	mu      sync.RWMutex
	runMu   sync.Mutex
}

func NewHybridTransformer(modelPath string) (*HybridTransformer, error) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return &HybridTransformer{enabled: false, history: make(map[int][]State)}, nil
	}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{"history_states"},
		[]string{"corrections"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load Hybrid Transformer: %w", err)
	}

	return &HybridTransformer{
		session: session,
		enabled: true,
		history: make(map[int][]State),
	}, nil
}

func (t *HybridTransformer) AddHistory(objectID int, state State) {
	if !t.enabled {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	h := t.history[objectID]
	h = append(h, state)
	if len(h) > 16 {
		h = h[1:]
	}
	t.history[objectID] = h
}

func (t *HybridTransformer) Refine(objectID int, currentState []float64, currentCov [][]float64) ([]float64, [][]float64, error) {
	if !t.enabled || t.session == nil || len(currentState) != 6 {
		return currentState, currentCov, nil
	}

	t.mu.RLock()
	h, ok := t.history[objectID]
	if ok {
		h = append([]State(nil), h...)
	}
	t.mu.RUnlock()
	if !ok || len(h) < 4 {
		return currentState, currentCov, nil // Need at least some history
	}

	// Prepare sequence input: [1, 16, 6]
	inputData := make([]float32, 16*6)
	// Fill with last available history, padding if necessary
	if len(h) < 16 {
		// Just repeat first element if history is short (or zero-pad)
	} else {
		h = h[len(h)-16:]
	}

	for i := 0; i < 16; i++ {
		srcIdx := i
		if i >= len(h) {
			srcIdx = len(h) - 1 // Clamp to last state for padding
		}
		s := h[srcIdx]
		base := i * 6
		inputData[base+0] = float32(s.X)
		inputData[base+1] = float32(s.Y)
		inputData[base+2] = float32(s.Z)
		inputData[base+3] = float32(s.VX)
		inputData[base+4] = float32(s.VY)
		inputData[base+5] = float32(s.VZ)
	}

	inputShape := onnxruntime_go.NewShape(1, 16, 6)
	inputTensor, err := onnxruntime_go.NewTensor(inputShape, inputData)
	if err != nil {
		return currentState, currentCov, err
	}
	defer inputTensor.Destroy()

	inputs := []onnxruntime_go.Value{inputTensor}
	outputs := []onnxruntime_go.Value{nil}

	t.runMu.Lock()
	if err := t.session.Run(inputs, outputs); err != nil {
		t.runMu.Unlock()
		return currentState, currentCov, err
	}
	t.runMu.Unlock()
	defer outputs[0].Destroy()

	outputTensor := outputs[0].(*onnxruntime_go.Tensor[float32])
	data := outputTensor.GetData() // [12]

	// Apply state corrections
	refinedState := make([]float64, 6)
	for i := 0; i < 6; i++ {
		refinedState[i] = currentState[i] + float64(data[i])*0.001 // Scale correction for stability
	}

	// Apply covariance tightening (indices 6-11 are diagonal scales)
	refinedCov := safemath.ZeroMatrix(6, 6)
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			refinedCov[i][j] = currentCov[i][j]
		}
		// Tighter covariance via scaling factor (0.8 + sigm(out))
		scale := 0.9 + 0.1*float64(data[6+i])
		if scale < 0.5 {
			scale = 0.5
		}
		if scale > 1.1 {
			scale = 1.1
		}
		refinedCov[i][i] *= scale
	}

	return refinedState, refinedCov, nil
}

func (t *HybridTransformer) Close() {
	if t.session != nil {
		t.session.Destroy()
	}
}
