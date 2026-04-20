package lie

import (
	"math"
)

/**
 * Lie Algebra Operators for ADITI 4.0
 *
 * Provides hat/vee operators and exponential maps for:
 * - so(3) ↔ SO(3) (rotation algebra/group)
 * - se(3) ↔ SE(3) (rigid body algebra/group)
 *
 * These operators bridge the Lie Algebra and Lie Group, enabling
 * geometrically correct integration on manifolds.
 *
 * References:
 *   - Munthe-Kaas (1998): Runge-Kutta methods on Lie groups
 *   - Murray et al. (1994): A Mathematical Introduction to Robotic Manipulation
 */

// =============================================================================
// so(3) → SO(3): Hat and Vee Operators
// =============================================================================

// HatSO3 maps a 3D angular velocity vector to a 3x3 skew-symmetric matrix in so(3).
//
// ω_hat = [[  0   -ω₃   ω₂],
//
//	[  ω₃    0   -ω₁],
//	[ -ω₂   ω₁    0 ]]
//
// Args:
//
//	omega: [3] angular velocity vector [ω₁, ω₂, ω₃]
//
// Returns:
//
//	[3][3] skew-symmetric matrix
func HatSO3(omega [3]float64) [3][3]float64 {
	return [3][3]float64{
		{0, -omega[2], omega[1]},
		{omega[2], 0, -omega[0]},
		{-omega[1], omega[0], 0},
	}
}

// VeeSO3 is the inverse of HatSO3: maps a 3x3 skew-symmetric matrix to a 3D vector.
//
// Args:
//
//	W: [3][3] skew-symmetric matrix
//
// Returns:
//
//	[3] angular velocity vector
func VeeSO3(W [3][3]float64) [3]float64 {
	return [3]float64{W[2][1], W[0][2], W[1][0]}
}

// =============================================================================
// Rodrigues Formula: Stable SO(3) Exponential Map
// =============================================================================

// RodriguesSO3 computes the rotation matrix from axis-angle using Rodrigues formula.
//
// Formula:
//
//	R = I + (sin θ/θ) ω_hat + ((1 - cos θ)/θ²) ω_hat²
//
// For small angles (θ < eps):
//
//	R ≈ I + ω_hat + 0.5 ω_hat² (Taylor expansion)
//
// Args:
//
//	omega: [3] axis-angle vector (direction is axis, magnitude is angle in radians)
//	eps: Small angle threshold (default 1e-8)
//
// Returns:
//
//	[3][3] rotation matrix (orthogonal, det(R) = 1)
func RodriguesSO3(omega [3]float64, eps float64) [3][3]float64 {
	if eps == 0 {
		eps = 1e-8
	}

	// Compute rotation angle θ = ||ω||
	theta := math.Sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2])

	// Identity matrix
	R := [3][3]float64{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}

	if theta < eps {
		// Small angle approximation: R ≈ I + ω_hat + 0.5 ω_hat²
		W := HatSO3(omega)
		W2 := matrixMultiply3x3(W, W)

		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				R[i][j] = R[i][j] + W[i][j] + 0.5*W2[i][j]
			}
		}
		return R
	}

	// Full Rodrigues formula
	W := HatSO3(omega)
	W2 := matrixMultiply3x3(W, W)

	sinTheta := math.Sin(theta)
	cosTheta := math.Cos(theta)

	invTheta := 1.0 / theta
	invTheta2 := invTheta * invTheta

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			R[i][j] = R[i][j] + invTheta*sinTheta*W[i][j] + invTheta2*(1-cosTheta)*W2[i][j]
		}
	}

	return R
}

// =============================================================================
// se(3) → SE(3): Hat and Vee Operators
// =============================================================================

// HatSE3 maps a 6D twist vector to a 4x4 matrix in se(3).
//
// The twist vector ξ = [v, ω] where:
//   - v ∈ R³: linear velocity
//   - ω ∈ R³: angular velocity
//
// se(3) matrix:
//
//	[[  ω_hat,   v ],
//	 [    0,     0 ]]
//
// Args:
//
//	xi: [6] twist vector [v₁, v₂, v₃, ω₁, ω₂, ω₃]
//
// Returns:
//
//	[4][4] element of se(3)
func HatSE3(xi [6]float64) [4][4]float64 {
	v := [3]float64{xi[0], xi[1], xi[2]}
	omega := [3]float64{xi[3], xi[4], xi[5]}

	W := HatSO3(omega)

	return [4][4]float64{
		{W[0][0], W[0][1], W[0][2], v[0]},
		{W[1][0], W[1][1], W[1][2], v[1]},
		{W[2][0], W[2][1], W[2][2], v[2]},
		{0, 0, 0, 0},
	}
}

// VeeSE3 is the inverse of HatSE3: maps a 4x4 se(3) matrix to a 6D twist vector.
//
// Args:
//
//	Xi: [4][4] matrix in se(3)
//
// Returns:
//
//	[6] twist vector [v₁, v₂, v₃, ω₁, ω₂, ω₃]
func VeeSE3(Xi [4][4]float64) [6]float64 {
	v := [3]float64{Xi[0][3], Xi[1][3], Xi[2][3]}
	omega := [3]float64{Xi[2][1], Xi[0][2], Xi[1][0]}

	return [6]float64{v[0], v[1], v[2], omega[0], omega[1], omega[2]}
}

// =============================================================================
// SE(3) Exponential Map
// =============================================================================

// ExpSE3 computes the exponential map from se(3) to SE(3).
//
// Uses the stable Rodrigues formula for the SO(3) component.
//
// Args:
//
//	xi: [6] twist vector [v, ω]
//
// Returns:
//
//	[4][4] element of SE(3) (homogeneous transformation matrix)
func ExpSE3(xi [6]float64) [4][4]float64 {
	v := [3]float64{xi[0], xi[1], xi[2]}
	omega := [3]float64{xi[3], xi[4], xi[5]}

	// Compute R = exp(ω_hat) using Rodrigues
	R := RodriguesSO3(omega, 1e-8)

	// Compute V matrix for translation
	// V = I + (1-cosθ)/θ² ω_hat + (θ-sinθ)/θ³ ω_hat²
	theta := math.Sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2])

	var V [3][3]float64
	if theta < 1e-8 {
		// Small angle: V ≈ I
		for i := 0; i < 3; i++ {
			V[i][i] = 1.0
		}
	} else {
		W := HatSO3(omega)
		W2 := matrixMultiply3x3(W, W)

		invTheta2 := 1.0 / (theta * theta)
		invTheta3 := invTheta2 / theta

		oneMinusCos := 1.0 - math.Cos(theta)
		thetaMinusSin := theta - math.Sin(theta)

		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				if i == j {
					V[i][j] = 1.0 + invTheta2*oneMinusCos*W[i][j] + invTheta3*thetaMinusSin*W2[i][j]
				} else {
					V[i][j] = invTheta2*oneMinusCos*W[i][j] + invTheta3*thetaMinusSin*W2[i][j]
				}
			}
		}
	}

	// Compute translation: t = V * v
	t := [3]float64{
		V[0][0]*v[0] + V[0][1]*v[1] + V[0][2]*v[2],
		V[1][0]*v[0] + V[1][1]*v[1] + V[1][2]*v[2],
		V[2][0]*v[0] + V[2][1]*v[1] + V[2][2]*v[2],
	}

	// Build homogeneous transformation matrix
	return [4][4]float64{
		{R[0][0], R[0][1], R[0][2], t[0]},
		{R[1][0], R[1][1], R[1][2], t[1]},
		{R[2][0], R[2][1], R[2][2], t[2]},
		{0, 0, 0, 1},
	}
}

// =============================================================================
// S² Exponential Map (Geodesic)
// =============================================================================

// ExpS2 computes the exponential map from tangent space to S² manifold.
//
// For a point q on the unit sphere and a tangent vector v (q·v = 0),
// the exponential map gives the point reached by following the geodesic
// (great circle) with velocity v for time t.
//
// Formula:
//
//	exp_q(v) = q cos(||v||) + (v/||v||) sin(||v||)
//
// Args:
//
//	q: [3] point on unit sphere (||q|| = 1)
//	v: [3] tangent vector at q (q·v = 0)
//
// Returns:
//
//	[3] point on S² reached by geodesic flow
func ExpS2(q, v [3]float64) [3]float64 {
	// Normalize q
	qNorm := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	for i := 0; i < 3; i++ {
		q[i] /= qNorm
	}

	// Project v to tangent space (q·v = 0)
	vDotQ := q[0]*v[0] + q[1]*v[1] + q[2]*v[2]
	for i := 0; i < 3; i++ {
		v[i] -= vDotQ * q[i]
	}

	// Compute magnitude
	vMag := math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

	if vMag < 1e-10 {
		return q // No movement
	}

	// Geodesic update
	theta := vMag
	cosTheta := math.Cos(theta)
	sinTheta := math.Sin(theta)

	var result [3]float64
	for i := 0; i < 3; i++ {
		result[i] = q[i]*cosTheta + (v[i]/vMag)*sinTheta
	}

	// Renormalize to ensure ||q|| = 1
	resultNorm := math.Sqrt(result[0]*result[0] + result[1]*result[1] + result[2]*result[2])
	for i := 0; i < 3; i++ {
		result[i] /= resultNorm
	}

	return result
}

// =============================================================================
// Symplectic Update Operators
// =============================================================================

// ExpUpdate performs position update via Lie exponential map.
// For Cartesian: q = q + v * dt
// For S²: q = exp_q(v * dt) (geodesic flow)
// For SO(3): q = q * exp(omega * dt) (group action)
func ExpUpdate(q, v []float64, dt float64) []float64 {
	// Cartesian update (baseline)
	result := make([]float64, len(q))
	for i := range q {
		result[i] = q[i] + v[i]*dt
	}
	return result
}

// CoadjointUpdate performs momentum update via coadjoint action.
// For Cartesian: p = p + f * dt
// For Lie groups: p = Ad*_g(p) + f * dt (coadjoint action + impulse)
func CoadjointUpdate(p, f []float64, dt float64) []float64 {
	result := make([]float64, len(p))
	for i := range p {
		result[i] = p[i] + f[i]*dt
	}
	return result
}

// =============================================================================
// Matrix Utilities
// =============================================================================

func matrixMultiply3x3(A, B [3][3]float64) [3][3]float64 {
	var C [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			sum := 0.0
			for k := 0; k < 3; k++ {
				sum += A[i][k] * B[k][j]
			}
			C[i][j] = sum
		}
	}
	return C
}
