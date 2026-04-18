package orbital

import (
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"github.com/yalue/onnxruntime_go"

	safemath "ssis/pkg/math"
)

// DifferentiablePropagator implements high-fidelity orbit propagation using 
// differentiable physics (dSGP4) and ML corrections.
type DifferentiablePropagator struct {
	session *onnxruntime_go.DynamicAdvancedSession
	enabled bool
	runMu   sync.Mutex
}

// NewDifferentiablePropagator initializes the ONNX session for the hybrid model.
func NewDifferentiablePropagator(modelPath string) (*DifferentiablePropagator, error) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return &DifferentiablePropagator{enabled: false}, nil
	}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{"tle_params", "tsince", "solar_indices"},
		[]string{"state", "jacobian"}, // We assume Jacobian is exported as second output
		nil,
	)
	if err != nil {
		// Fallback: Model might only have 'state' output if Jacobian export was deferred
		session, err = onnxruntime_go.NewDynamicAdvancedSession(
			modelPath,
			[]string{"tle_params", "tsince", "solar_indices"},
			[]string{"state"},
			nil,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to load Differentiable Propagator: %w", err)
		}
	}

	return &DifferentiablePropagator{
		session: session,
		enabled: true,
	}, nil
}

// DifferentiableState represents a propagated state with its analytic Jacobian.
type DifferentiableState struct {
	State     State
	Jacobian  []float64 // 6x7 flattened matrix: d(State)/d(TLE_Params)
	Timestamp time.Time
	Signature []byte // Ed25519 signature for integrity
}

// Propagate performs a forward pass through the hybrid dSGP4 model.
// TLE parameters p = [inc, raan, ecc, argp, m_0, nm, bstar]
func (p *DifferentiablePropagator) Propagate(tle *TLE, epoch time.Time, solar Input) (DifferentiableState, error) {
	if !p.enabled || p.session == nil {
		return DifferentiableState{}, fmt.Errorf("differentiable propagator disabled or model missing")
	}

	// 1. Prepare Inputs
	minutes := epoch.Sub(tleToTime(tle)).Minutes()
	
	tleParams := []float32{
		float32(tle.Inclination),
		float32(tle.RAAN),
		float32(tle.Eccentricity),
		float32(tle.ArgOfPerigee),
		float32(tle.MeanAnomaly),
		float32(tle.MeanMotion),
		float32(tle.BStar),
	}
	
	tsince := []float32{float32(minutes)}
	
	solarIndices := []float32{
		float32(solar.F107),
		float32(solar.F107A),
		float32(solar.AP), // AP is a scalar in Input
	}

	// 2. Create Tensors
	tleTensor, _ := onnxruntime_go.NewTensor(onnxruntime_go.NewShape(1, 7), tleParams)
	tsinceTensor, _ := onnxruntime_go.NewTensor(onnxruntime_go.NewShape(1, 1), tsince)
	solarTensor, _ := onnxruntime_go.NewTensor(onnxruntime_go.NewShape(1, 3), solarIndices)
	
	defer tleTensor.Destroy()
	defer tsinceTensor.Destroy()
	defer solarTensor.Destroy()

	inputs := []onnxruntime_go.Value{tleTensor, tsinceTensor, solarTensor}
	outputs := make([]onnxruntime_go.Value, 2) // We expect [state, jacobian]

	// 3. Run Inference
	p.runMu.Lock()
	err := p.session.Run(inputs, outputs)
	p.runMu.Unlock()
	if err != nil {
		return DifferentiableState{}, fmt.Errorf("dSGP4 inference failed: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	// 4. Extract State
	stateData := outputs[0].(*onnxruntime_go.Tensor[float32]).GetData()
	result := DifferentiableState{
		State: State{
			X:  float64(stateData[0]),
			Y:  float64(stateData[1]),
			Z:  float64(stateData[2]),
			VX: float64(stateData[3]),
			VY: float64(stateData[4]),
			VZ: float64(stateData[5]),
		},
		Timestamp: epoch,
	}

	// 5. Extract Jacobian if available
	if len(outputs) > 1 && outputs[1] != nil {
		jacData := outputs[1].(*onnxruntime_go.Tensor[float32]).GetData()
		result.Jacobian = make([]float64, len(jacData))
		for i, v := range jacData {
			result.Jacobian[i] = float64(v)
		}
	}

	return result, nil
}

// DifferentialCorrectionBatch performs Batch Orbit Determination using Gauss-Newton.
// It iterates on TLE parameters to minimize the residual sum of squares vs observations.
func (p *DifferentiablePropagator) DifferentialCorrectionBatch(initialTLE *TLE, obs []Observation, solar Input) (*TLE, error) {
	if !p.enabled || len(obs) < 4 {
		return initialTLE, fmt.Errorf("insufficient data or propagator disabled")
	}

	currentTLE := *initialTLE
	maxIter := 10
	convergence := 1e-6

	for iter := 0; iter < maxIter; iter++ {
		// Normal equations: (J^T * J) * delta = J^T * residual
		ata := safemath.ZeroMatrix(7, 7)
		atr := make([]float64, 7)
		totalLoss := 0.0

		for _, o := range obs {
			diffState, err := p.Propagate(&currentTLE, o.Timestamp, solar)
			if err != nil {
				continue
			}

			// Residual [3x1] (positional only for simplicity, or 6x1 if velocity exists)
			res := []float64{
				o.Position[0] - diffState.State.X,
				o.Position[1] - diffState.State.Y,
				o.Position[2] - diffState.State.Z,
			}
			
			for i := 0; i < 3; i++ {
				totalLoss += res[i] * res[i]
			}

			// J is 6x7. We take the top 3x7 for position.
			for j := 0; j < 7; j++ {
				for i := 0; i < 3; i++ {
					grad := diffState.Jacobian[i*7+j]
					atr[j] += grad * res[i]
					for k := 0; k < 7; k++ {
						ata[j][k] += grad * diffState.Jacobian[i*7+k]
					}
				}
			}
		}

		// Regularization
		for i := 0; i < 7; i++ {
			ata[i][i] += 1e-8
		}

		invAta, err := safemath.MatrixInverse(ata, 1e-9)
		if err != nil {
			return &currentTLE, fmt.Errorf("OD matrix inversion failed: %w", err)
		}

		delta := safemath.MatrixVectorMultiply(invAta, atr)
		
		// Update TLE params: [inc, raan, ecc, argp, m0, nm, bstar]
		currentTLE.Inclination += delta[0]
		currentTLE.RAAN += delta[1]
		currentTLE.Eccentricity += delta[2]
		currentTLE.ArgOfPerigee += delta[3]
		currentTLE.MeanAnomaly += delta[4]
		currentTLE.MeanMotion += delta[5]
		currentTLE.BStar += delta[6]

		if matNorm(delta) < convergence {
			break
		}
	}

	return &currentTLE, nil
}

// Observation represents a ground-truth measurement of a satellite state.
type Observation struct {
	Timestamp time.Time
	Position  [3]float64
	Velocity  [3]float64 // Optional but recommended
}

func matNorm(v []float64) float64 {
	sum := 0.0
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

func tleToTime(tle *TLE) time.Time {
	// Simplified epoch conversion (Standard SGP4 epoch logic)
	// In production, uses the full TLE.EpochYear/Day logic
	year := tle.EpochYear
	days := tle.EpochDay
	
	startOfYear := time.Date(year, 1, 0, 0, 0, 0, 0, time.UTC)
	return startOfYear.Add(time.Duration(days * float64(24*time.Hour)))
}

func (p *DifferentiablePropagator) Close() {
	if p.session != nil {
		p.session.Destroy()
	}
}
