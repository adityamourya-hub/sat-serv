package orbital

import (
	"fmt"
	"math"
	"os"

	"github.com/yalue/onnxruntime_go"
)

type PINNRefiner struct {
	session *onnxruntime_go.DynamicAdvancedSession
	enabled bool
}

type PINNRefineContext struct {
	LongArcThresholdSec float64
	HighDragAltitudeKm  float64
	CovTraceTrigger     float64
	MinCorrectionScale  float64
	MaxCorrectionScale  float64
}

func DefaultPINNRefineContext() PINNRefineContext {
	return PINNRefineContext{
		LongArcThresholdSec: 120.0,
		HighDragAltitudeKm:  550.0,
		CovTraceTrigger:     8.0,
		MinCorrectionScale:  0.2,
		MaxCorrectionScale:  1.0,
	}
}

func NewPINNRefiner(modelPath string) (*PINNRefiner, error) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return &PINNRefiner{enabled: false}, nil
	}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{"ukf_state_cov"},
		[]string{"accel_correction"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load PINN refiner: %w", err)
	}

	return &PINNRefiner{session: session, enabled: true}, nil
}

func (r *PINNRefiner) RefineState(state []float64, covariance [][]float64, dt float64) ([]float64, error) {
	ctx := DefaultPINNRefineContext()
	return r.RefineStateAdaptive(state, covariance, dt, ctx)
}

// RefineStateAdaptive applies PINN corrections only when dynamics indicate long-arc drift,
// high-drag regime, or covariance growth. This keeps short-arc updates stable while improving
// physically difficult segments.
func (r *PINNRefiner) RefineStateAdaptive(state []float64, covariance [][]float64, dt float64, ctx PINNRefineContext) ([]float64, error) {
	if !r.enabled || r.session == nil || len(state) != 6 {
		return state, nil
	}
	if !shouldApplyPINNRefinement(state, covariance, dt, ctx) {
		return state, nil
	}

	// Prepare input: 6 state elements + 6 diagonal cov elements
	inputData := make([]float32, 12)
	for i := 0; i < 6; i++ {
		inputData[i] = float32(state[i])
		if i < len(covariance) && i < len(covariance[i]) {
			inputData[6+i] = float32(covariance[i][i])
		}
	}

	inputShape := onnxruntime_go.NewShape(1, 12)
	inputTensor, err := onnxruntime_go.NewTensor(inputShape, inputData)
	if err != nil {
		return state, err
	}
	defer inputTensor.Destroy()

	inputs := []onnxruntime_go.Value{inputTensor}
	outputs := []onnxruntime_go.Value{nil}

	if err := r.session.Run(inputs, outputs); err != nil {
		return state, err
	}
	defer outputs[0].Destroy()

	outputTensor := outputs[0].(*onnxruntime_go.Tensor[float32])
	corrections := outputTensor.GetData()

	// Apply acceleration corrections [dVX, dVY, dVZ] with context-aware scaling.
	refined := make([]float64, 6)
	copy(refined, state)
	scale := refinementStrengthScale(state, covariance, dt, ctx)

	// Velocity is indices 3, 4, 5
	for i := 0; i < 3; i++ {
		refined[3+i] += float64(corrections[i]) * dt * scale
	}

	return refined, nil
}

func (r *PINNRefiner) Close() {
	if r.session != nil {
		r.session.Destroy()
	}
}

func shouldApplyPINNRefinement(state []float64, covariance [][]float64, dt float64, ctx PINNRefineContext) bool {
	if dt >= ctx.LongArcThresholdSec {
		return true
	}
	r := math.Sqrt(state[0]*state[0] + state[1]*state[1] + state[2]*state[2])
	if r > 1 {
		altKm := r - ER
		if altKm <= ctx.HighDragAltitudeKm {
			return true
		}
	}
	return covarianceTrace(covariance) >= ctx.CovTraceTrigger
}

func refinementStrengthScale(state []float64, covariance [][]float64, dt float64, ctx PINNRefineContext) float64 {
	s := 0.0

	// Long-arc component.
	s += pinnClamp01((dt - 30.0) / math.Max(ctx.LongArcThresholdSec, 1.0))

	// Drag component from altitude and density proxy.
	r := math.Sqrt(state[0]*state[0] + state[1]*state[1] + state[2]*state[2])
	if r > 1 {
		altKm := r - ER
		rho := AtmosphericDensity(altKm)
		s += pinnClamp01(math.Log10(1.0 + 1e9*rho))
	}

	// Uncertainty growth component.
	trace := covarianceTrace(covariance)
	s += pinnClamp01(trace / math.Max(ctx.CovTraceTrigger, 1e-6))

	// Normalize into configured bounds.
	s = s / 3.0
	return pinnClampRange(ctx.MinCorrectionScale+(ctx.MaxCorrectionScale-ctx.MinCorrectionScale)*pinnClamp01(s), ctx.MinCorrectionScale, ctx.MaxCorrectionScale)
}

func covarianceTrace(cov [][]float64) float64 {
	if len(cov) == 0 {
		return 0
	}
	n := len(cov)
	trace := 0.0
	for i := 0; i < n; i++ {
		if i < len(cov[i]) {
			trace += cov[i][i]
		}
	}
	return trace
}

func pinnClamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func pinnClampRange(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
