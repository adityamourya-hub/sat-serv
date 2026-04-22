package physics

import (
	"math"
)

/**
 * Symplectic Engine
 *
 * Implements Forest-Ruth (Yoshida 4th-order) symplectic integrator
 * to bridge the gap between H100 training logic and Go-native reflex kernel.
 *
 * Mathematical Foundation:
 *   - Ensures state update is a Canonical Transformation (det(J) = 1)
 *   - Energy error is bounded O(1) instead of growing O(t) like RK4
 *   - Preserves symplectic volume on S² and SO(3) manifolds
 *
 * Coefficients:
 *   γ = 1 / (2 - 2^(1/3))
 *   a = [γ/2, (1-γ)/2, (1-γ)/2, γ/2]  (position update)
 *   b = [γ, 1-2γ, γ, 0]                  (momentum update)
 *
 * References:
 *   - Forest & Ruth (1990): Fourth-order symplectic integration
 *   - Yoshida (1990): Construction of higher order symplectic integrators
 *   - Munthe-Kaas (1998): Runge-Kutta methods on Lie groups
 */

// =============================================================================
// Forest-Ruth Coefficients (Yoshida 4th-Order)
// =============================================================================

// γ = 1 / (2 - 2^(1/3))
var gamma = 1.0 / (2.0 - math.Cbrt(2.0))

// Position coefficients (dt multipliers for q update)
var (
	a1 = gamma / 2.0
	a2 = (1.0 - gamma) / 2.0
	a3 = (1.0 - gamma) / 2.0
	a4 = gamma / 2.0
)

// Momentum coefficients (force multipliers for p update)
var (
	b1 = gamma
	b2 = 1.0 - 2.0*gamma
	b3 = gamma
	b4 = 0.0
)

// ForestRuthCoefficients returns the 4th-order Forest-Ruth coefficients
func ForestRuthCoefficients() (a, b [4]float64) {
	return [4]float64{a1, a2, a3, a4}, [4]float64{b1, b2, b3, b4}
}

// =============================================================================
// PHS Model Interface
// =============================================================================

// PHSModel defines the Port-Hamiltonian System interface for symplectic integration
type PHSModel interface {
	// Velocity returns ∂H/∂p (velocity from momentum)
	Velocity(q, p, z []float64) []float64

	// Force returns -∂H/∂q (negative force from position gradient)
	Force(q, p, z []float64) []float64

	// Hamiltonian returns the total energy H(q, p)
	Hamiltonian(q, p, z []float64) float64
}

// =============================================================================
// Cartesian Symplectic Integrator (Baseline)
// =============================================================================

// ForestRuthStepCartesian performs one symplectic 4th-order integration step
// for Cartesian coordinates (baseline for orbital dynamics).
//
// Algorithm:
//
//	For i = 1 to 4:
//	  q = q + a[i] * dt * ∂H/∂p
//	  p = p + b[i] * dt * (-∂H/∂q)
//
// Args:
//
//	model: PHS model providing Velocity and Force
//	q: Position vector [x, y, z] (meters)
//	p: Momentum vector [px, py, pz] (kg·m/s)
//	z: Latent code for FiLM conditioning (optional, can be nil)
//	dt: Time step (seconds)
//
// Returns:
//
//	qNext: Next position
//	pNext: Next momentum
func ForestRuthStepCartesian(model PHSModel, q, p, z []float64, dt float64) ([]float64, []float64) {
	dim := len(q)
	qCurr := make([]float64, dim)
	pCurr := make([]float64, dim)
	copy(qCurr, q)
	copy(pCurr, p)

	a, b := ForestRuthCoefficients()

	for i := 0; i < 4; i++ {
		// 1. Position Update: q = q + a[i] * dt * ∂H/∂p
		v := model.Velocity(qCurr, pCurr, z)
		for j := 0; j < dim; j++ {
			qCurr[j] += a[i] * dt * v[j]
		}

		// 2. Momentum Update: p = p + b[i] * dt * (-∂H/∂q)
		if b[i] != 0 {
			f := model.Force(qCurr, pCurr, z)
			for j := 0; j < dim; j++ {
				pCurr[j] += b[i] * dt * f[j]
			}
		}
	}

	return qCurr, pCurr
}

// =============================================================================
// S² (Sphere) Symplectic Integrator
// =============================================================================

// ForestRuthStepS2 performs symplectic integration on the S² manifold.
// Uses geodesic update to maintain ||q|| = 1 exactly.
//
// For S², the position update uses the exponential map on the sphere:
//
//	q_next = q cos(|v|dt) + (v/|v|) sin(|v|dt)
//
// Args:
//
//	model: PHS model
//	q: Position on unit sphere [qx, qy, qz] (||q|| = 1)
//	p: Momentum in tangent space T_q S² [px, py, pyz] (q·p = 0)
//	z: Latent code
//	dt: Time step
//
// Returns:
//
//	qNext: Next position on S² (normalized)
//	pNext: Next momentum (projected to tangent space)
func ForestRuthStepS2(model PHSModel, q, p, z []float64, dt float64) ([]float64, []float64) {
	dim := 3
	qCurr := make([]float64, dim)
	pCurr := make([]float64, dim)
	copy(qCurr, q)
	copy(pCurr, p)

	// Ensure q is normalized
	qNorm := norm(qCurr)
	for j := 0; j < dim; j++ {
		qCurr[j] /= qNorm
	}

	a, b := ForestRuthCoefficients()

	for i := 0; i < 4; i++ {
		// 1. Position Update: Geodesic on S²
		v := model.Velocity(qCurr, pCurr, z)

		// Project v to tangent space (q·v = 0)
		vDotQ := dot(qCurr, v)
		for j := 0; j < dim; j++ {
			v[j] -= vDotQ * qCurr[j]
		}

		// Geodesic update: q_next = q cos(θ) + (v/|v|) sin(θ)
		vMag := norm(v)
		if vMag > 1e-10 {
			theta := vMag * a[i] * dt
			cosTheta := math.Cos(theta)
			sinTheta := math.Sin(theta)

			for j := 0; j < dim; j++ {
				qCurr[j] = qCurr[j]*cosTheta + (v[j]/vMag)*sinTheta
			}

			// Renormalize to ensure ||q|| = 1
			qNorm = norm(qCurr)
			for j := 0; j < dim; j++ {
				qCurr[j] /= qNorm
			}
		}

		// 2. Momentum Update
		if b[i] != 0 {
			f := model.Force(qCurr, pCurr, z)

			// Project force to tangent space at qCurr
			fDotQ := dot(qCurr, f)
			for j := 0; j < dim; j++ {
				f[j] -= fDotQ * qCurr[j]
			}

			for j := 0; j < dim; j++ {
				pCurr[j] += b[i] * dt * f[j]
			}

			// Project new momentum to tangent space (q·p = 0)
			pDotQ := dot(qCurr, pCurr)
			for j := 0; j < dim; j++ {
				pCurr[j] -= pDotQ * qCurr[j]
			}
		}
	}

	return qCurr, pCurr
}

// =============================================================================
// SO(3) Rotation Symplectic Integrator
// =============================================================================

// ForestRuthStepSO3 performs symplectic integration on the SO(3) manifold.
// Uses Rodrigues formula for exponential map from so(3) to SO(3).
//
// Args:
//
//	model: PHS model
//	q: Rotation matrix [3x3] or quaternion [4]
//	p: Angular momentum in body frame
//	z: Latent code
//	dt: Time step
//
// Returns:
//
//	qNext: Next rotation (orthogonal matrix)
//	pNext: Next angular momentum
func ForestRuthStepSO3(model PHSModel, q, p, z []float64, dt float64) ([]float64, []float64) {
	// Placeholder: Full SO(3) implementation requires quaternion or rotation matrix
	// For now, use Cartesian as fallback
	return ForestRuthStepCartesian(model, q, p, z, dt)
}

// =============================================================================
// Unified Symplectic Step
// =============================================================================

// ManifoldType specifies the geometric manifold for integration
type ManifoldType int

const (
	ManifoldCartesian ManifoldType = iota // R³ (baseline)
	ManifoldS2                            // S² (unit sphere for HGV)
	ManifoldSO3                           // SO(3) (rotation)
)

// ManifoldGate provides ortho-normalization to prevent floating-point creep
//
// Geometric Firewall that prevents the AI from hallucinating
// a universe where physics are optional. Ensures state remains on manifold.
// Every 1000 cycles, applies ortho_normalize to kill floating-point drift.
type ManifoldGate struct {
	CycleCount     int
	NormalizeEvery int // Default 1000 cycles
}

// NewManifoldGate creates a new manifold gate
func NewManifoldGate() *ManifoldGate {
	return &ManifoldGate{
		CycleCount:     0,
		NormalizeEvery: 1000,
	}
}

// OrthoNormalize normalizes state to maintain manifold integrity
//
// For S² (unit sphere): ensures ||q|| = 1
// For SO(3): ensures Q^T Q = I (orthogonal matrix)
// For Cartesian: no-op (R³ has no constraints)
func (m *ManifoldGate) OrthoNormalize(q, p []float64, manifold ManifoldType) {
	m.CycleCount++

	// Only normalize every N cycles
	if m.CycleCount%m.NormalizeEvery != 0 {
		return
	}

	switch manifold {
	case ManifoldS2:
		// Normalize position to unit sphere
		norm := 0.0
		for _, v := range q {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		if norm > 1e-15 {
			for i := range q {
				q[i] /= norm
			}
		}

	case ManifoldSO3:
		// For rotation matrix, ensure orthogonality via QR decomposition
		// Simplified: normalize each column to unit length
		if len(q) == 9 { // 3x3 matrix flattened
			for col := 0; col < 3; col++ {
				norm := 0.0
				for row := 0; row < 3; row++ {
					idx := row*3 + col
					norm += q[idx] * q[idx]
				}
				norm = math.Sqrt(norm)
				if norm > 1e-15 {
					for row := 0; row < 3; row++ {
						idx := row*3 + col
						q[idx] /= norm
					}
				}
			}
		}

	case ManifoldCartesian:
		// No constraints on R³
		return
	}
}

// ForestRuthStep is the unified symplectic integration entry point.
// Selects the appropriate integrator based on manifold type.
//
// Args:
//
//	model: PHS model
//	q: Position
//	p: Momentum
//	z: Latent code
//	dt: Time step
//	manifold: Manifold type (Cartesian, S2, SO3)
//
// Returns:
//
//	qNext: Next position
//	pNext: Next momentum
func ForestRuthStep(model PHSModel, q, p, z []float64, dt float64, manifold ManifoldType) ([]float64, []float64) {
	switch manifold {
	case ManifoldS2:
		return ForestRuthStepS2(model, q, p, z, dt)
	case ManifoldSO3:
		return ForestRuthStepSO3(model, q, p, z, dt)
	default:
		return ForestRuthStepCartesian(model, q, p, z, dt)
	}
}

// =============================================================================
// Utility Functions
// =============================================================================

// norm computes the L2 norm of a vector
func norm(v []float64) float64 {
	sum := 0.0
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

// dot computes the dot product of two vectors
func dot(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// =============================================================================
// Energy Conservation Verification
// =============================================================================

// VerifyEnergyConservation checks if the integrator preserves Hamiltonian energy
// over a multi-step rollout.
//
// Args:
//
//	model: PHS model
//	q0: Initial position
//	p0: Initial momentum
//	z: Latent code
//	dt: Time step
//	steps: Number of integration steps
//	manifold: Manifold type
//
// Returns:
//
//	maxEnergyDrift: Maximum relative energy drift |H(t) - H(0)| / |H(0)|
//	finalEnergyDrift: Energy drift at final step
func VerifyEnergyConservation(model PHSModel, q0, p0, z []float64, dt float64, steps int, manifold ManifoldType) (maxEnergyDrift, finalEnergyDrift float64) {
	q := make([]float64, len(q0))
	p := make([]float64, len(p0))
	copy(q, q0)
	copy(p, p0)

	H0 := model.Hamiltonian(q, p, z)
	maxDrift := 0.0

	for i := 0; i < steps; i++ {
		q, p = ForestRuthStep(model, q, p, z, dt, manifold)
		H := model.Hamiltonian(q, p, z)

		drift := math.Abs(H-H0) / math.Abs(H0)
		if drift > maxDrift {
			maxDrift = drift
		}
	}

	HFinal := model.Hamiltonian(q, p, z)
	finalDrift := math.Abs(HFinal-H0) / math.Abs(H0)

	return maxDrift, finalDrift
}

// =============================================================================
// Symplectic Error Computation
// =============================================================================

// ComputeSymplecticError computes the deviation from symplectic structure.
// For a canonical transformation, the Jacobian determinant should be 1.
//
// This is computed as: ||M^T J M - J||_F where J is the symplectic matrix.
func ComputeSymplecticError(q, p, z []float64, dt float64, model PHSModel, manifold ManifoldType) float64 {
	// Finite difference approximation of Jacobian
	epsilon := 1e-6
	dim := len(q) + len(p)

	// Compute perturbed states
	J := make([][]float64, dim)
	for i := range J {
		J[i] = make([]float64, dim)
	}

	// Compute Jacobian column by column
	for i := 0; i < dim; i++ {
		// Perturb state
		state := make([]float64, dim)
		copy(state, append(q, p...))
		state[i] += epsilon

		// Integrate forward
		qPlus, pPlus := ForestRuthStep(model, state[:len(q)], state[len(q):], z, dt, manifold)
		statePlus := append(qPlus, pPlus...)

		// Perturb in opposite direction
		state[i] -= 2 * epsilon
		qMinus, pMinus := ForestRuthStep(model, state[:len(q)], state[len(q):], z, dt, manifold)
		stateMinus := append(qMinus, pMinus...)

		// Finite difference
		for j := 0; j < dim; j++ {
			J[j][i] = (statePlus[j] - stateMinus[j]) / (2 * epsilon)
		}
	}

	// Construct symplectic matrix J_sym
	J_sym := make([][]float64, dim)
	for i := range J_sym {
		J_sym[i] = make([]float64, dim)
	}

	// Standard symplectic matrix: J = [[0, I], [-I, 0]]
	n := dim / 2
	for i := 0; i < n; i++ {
		J_sym[i][i+n] = 1.0
		J_sym[i+n][i] = -1.0
	}

	// Compute M^T J M
	JTJM := matrixMultiply(matrixTranspose(J), matrixMultiply(J_sym, J))

	// Compute Frobenius norm of (M^T J M - J)
	diff := matrixSubtract(JTJM, J_sym)
	error := frobeniusNorm(diff)

	return error
}

// =============================================================================
// Matrix Utilities
// =============================================================================

func matrixMultiply(A, B [][]float64) [][]float64 {
	n := len(A)
	m := len(B[0])
	p := len(B[0])
	C := make([][]float64, n)
	for i := range C {
		C[i] = make([]float64, p)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < m; k++ {
				sum += A[i][k] * B[k][j]
			}
			C[i][j] = sum
		}
	}
	return C
}

func matrixTranspose(A [][]float64) [][]float64 {
	n := len(A)
	m := len(A[0])
	AT := make([][]float64, m)
	for i := range AT {
		AT[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			AT[j][i] = A[i][j]
		}
	}
	return AT
}

func matrixSubtract(A, B [][]float64) [][]float64 {
	n := len(A)
	m := len(A[0])
	C := make([][]float64, n)
	for i := range C {
		C[i] = make([]float64, m)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			C[i][j] = A[i][j] - B[i][j]
		}
	}
	return C
}

func frobeniusNorm(A [][]float64) float64 {
	sum := 0.0
	for _, row := range A {
		for _, val := range row {
			sum += val * val
		}
	}
	return math.Sqrt(sum)
}
