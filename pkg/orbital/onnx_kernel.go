package orbital

import (
	"ssis/pkg/ml/onnx"
)

/**
 * ONNXKernel implements the GHNN-based learned force potential using ONNX Runtime.
 * It provides the high-fidelity acceleration corrections predicted by Axiom-Zero.
 */
type ONNXKernel struct {
	session *onnx.Session
	enabled bool
}

/**
 * NewONNXKernel initializes the kernel with an active ONNX session.
 */
func NewONNXKernel(session *onnx.Session) *ONNXKernel {
	return &ONNXKernel{
		session: session,
		enabled: session != nil,
	}
}

const (
	RE_KM  = 6378.137
	MU_KM  = 398600.4418
	TU_SEC = 806.8109885 // sqrt(RE^3 / MU)
)

/**
 * GetAcceleration retrieves the learned dynamics from the GHNN vector field.
 * Inputs (q, v) are in km and km/s.
 * Internal GHNN works in Canonical Units (DU, DU/TU).
 * Output is converted back to km/s^2.
 */
func (k *ONNXKernel) GetAcceleration(q, v [3]float64, jd float64) [3]float64 {
	if !k.enabled {
		return [3]float64{0, 0, 0}
	}

	// 1. Normalize to Canonical Units (DU, TU)
	q_du := [3]float32{
		float32(q[0] / RE_KM),
		float32(q[1] / RE_KM),
		float32(q[2] / RE_KM),
	}
	v_tu := [3]float32{
		float32(v[0] * TU_SEC / RE_KM),
		float32(v[1] * TU_SEC / RE_KM),
		float32(v[2] * TU_SEC / RE_KM),
	}

	// Prepare input for GHNN [batch=1, state=6]
	input := map[string]interface{}{
		"state": []float32{
			q_du[0], q_du[1], q_du[2],
			v_tu[0], v_tu[1], v_tu[2],
		},
	}

	// 2. Run inference
	outputs, err := k.session.Infer(input)
	if err != nil {
		return [3]float64{0, 0, 0}
	}

	// 3. Extract and Denormalize
	// GHNN returns [dq/dt, dp/dt] in Canonical Units (DU/TU, DU/TU^2)
	dynamics, ok := outputs["dynamics"].([]float32)
	if !ok || len(dynamics) < 6 {
		return [3]float64{0, 0, 0}
	}

	// Acceleration is indices 3, 4, 5 in DU/TU^2
	// To convert DU/TU^2 back to km/s^2: multiply by RE_KM / TU_SEC^2
	scale := RE_KM / (TU_SEC * TU_SEC)
	return [3]float64{
		float64(dynamics[3]) * scale,
		float64(dynamics[4]) * scale,
		float64(dynamics[5]) * scale,
	}
}
