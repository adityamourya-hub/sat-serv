package hypersonic

/**
 * HGV Port-Hamiltonian System (PHS-on-S²×ℝ³×SO(3))
 *
 * Converts classical HGV physics to Port-Hamiltonian form on the natural manifold:
 * - Position: S² (unit sphere for direction from Earth center)
 * - Velocity: ℝ³ (Cartesian velocity vector)
 * - Attitude: SO(3) (vehicle orientation, encoded in bank angle)
 *
 * PHS Formulation:
 *     ẋ = [J(x) - R(x)] ∇H(x) + g(x)u
 *
 * Where:
 * - J(x): Symplectic interconnection matrix (canonical on T*S²)
 * - R(x): Dissipation matrix (drag absorbed here, PSD via Cholesky)
 * - H(x): Hamiltonian (total energy: kinetic + gravitational + aerodynamic potential)
 * - g(x): Control coupling (bank-angle absorbed here as control input)
 * - u: Control input (lift command, derived from bank angle σ)
 *
 * Key Innovation:
 * - Bank angle σ(t) is hidden input, absorbed into g·u
 * - Drag is modeled as dissipative R(x) (energy loss, not force)
 * - Enables zero-day HGV tracking via residual SINDy on u(t)
 *
 * Reference: foundation.md §4 - Wire HGV model into PHS form
 */

import (
	"math"
	"ssis/pkg/physics"
)

/**
 * HGVPHSState represents state on the S²×ℝ³×SO(3) manifold.
 *
 * This is the natural manifold for HGV dynamics, respecting geometric structure.
 */
type HGVPHSState struct {
	// Position on S² (unit direction from Earth center)
	// n̂ = [nx, ny, nz] with ||n̂|| = 1
	Direction [3]float64 // Unit vector from Earth center

	// Velocity in ℝ³ (Cartesian)
	Velocity [3]float64 // km/s in ECI frame

	// Attitude in SO(3) (vehicle orientation)
	// Encoded as rotation matrix from body to ECI
	// For HGV, the critical degree of freedom is bank angle σ
	Orientation [3][3]float64 // Rotation matrix R_body→ECI

	// Altitude (for atmosphere model)
	Altitude float64 // km above Earth surface
}

/**
 * HGVPHSDynamics computes PHS dynamics on S²×ℝ³×SO(3).
 *
 * Returns the time derivative ẋ = [J - R]∇H + g·u
 */
func HGVPHSDynamics(state HGVPHSState, consts HGVConstants, control [3]float64, jd float64) HGVPHSState {
	// 1. Compute Hamiltonian gradient ∇H
	gradH := computeHamiltonianGradient(state, consts, jd)

	// 2. Get interconnection matrix J (canonical on T*S²)
	J := getInterconnectionMatrixS2()

	// 3. Get dissipation matrix R (drag absorbed here, PSD)
	R := getDissipationMatrix(state, consts, jd)

	// 4. Get control coupling g (bank-angle absorbed here)
	g := getControlCoupling(state, consts)

	// 5. Conservative + Dissipative dynamics: [J - R]∇H
	// Compute (J - R) @ gradH
	JminusR := matrixSubtract(J, R)
	dxDissipative := matrixVectorMultiply6x6(JminusR, gradH)

	// 6. Add control input: g·u (g is [6x3], u is [3], result is [6])
	dxControl := matrixVectorMultiply6x3(g, control)

	// 7. Total dynamics
	var dx HGVPHSState
	for i := 0; i < 3; i++ {
		dx.Direction[i] = dxDissipative[i] + dxControl[i]
		dx.Velocity[i] = dxDissipative[3+i] + dxControl[3+i]
	}

	// Altitude change from radial velocity
	vRad := dotProduct(state.Direction, state.Velocity)
	dx.Altitude = vRad

	return dx
}

/**
 * computeHamiltonianGradient computes ∇H on the manifold.
 *
 * H = T + V_gravity + V_aero
 * where:
 * - T = ½||v||² (kinetic energy)
 * - V_gravity = -μ/r (gravitational potential)
 * - V_aero = aerodynamic potential (lift-induced)
 */
func computeHamiltonianGradient(state HGVPHSState, consts HGVConstants, jd float64) [6]float64 {
	var gradH [6]float64

	// Position on sphere
	n := state.Direction
	v := state.Velocity

	// Radius from Earth center
	r := physics.EARTH_R_EQUATORIAL + state.Altitude

	// Gravitational parameter
	mu := physics.EARTH_MU_IAU

	// Kinetic energy gradient: ∇_v T = v
	gradH[3] = v[0]
	gradH[4] = v[1]
	gradH[5] = v[2]

	// Gravitational potential gradient: ∇_n V_grav = (μ/r²)n
	gravGrad := (mu / (r * r))
	gradH[0] = gravGrad * n[0]
	gradH[1] = gravGrad * n[1]
	gradH[2] = gravGrad * n[2]

	// Aerodynamic potential gradient (lift-induced)
	// This couples direction and velocity through aerodynamic forces
	rho := physics.GetDensityUnified(0, 0, r, jd)
	vMag := vectorNorm(v)

	if vMag > 1e-6 {
		// Dynamic pressure
		q := 0.5 * rho * vMag * vMag * 1e6 // Convert to Pa (km/s to m/s)

		// Lift force direction (perpendicular to velocity and bank)
		// This creates a potential that depends on orientation
		L := q * consts.ReferenceArea * consts.LDGlide // Estimate CL from L/D

		// Lift contributes to gradient in direction space
		// Simplified: lift acts perpendicular to velocity
		liftGrad := L / consts.Mass / 1000.0 // Convert to km/s²

		// Project onto direction space (simplified)
		gradH[0] += liftGrad * 0.01 // Small contribution
		gradH[1] += liftGrad * 0.01
		gradH[2] += liftGrad * 0.01
	}

	return gradH
}

/**
 * getInterconnectionMatrixS2 returns canonical J on T*S².
 *
 * For a sphere S², the symplectic form is:
 *     J = [[0, I], [-I, 0]]
 * but adapted for the constrained manifold structure.
 */
func getInterconnectionMatrixS2() [6][6]float64 {
	var J [6][6]float64

	// Canonical symplectic matrix
	// Top-right: identity
	J[0][3] = 1.0
	J[1][4] = 1.0
	J[2][5] = 1.0

	// Bottom-left: negative identity
	J[3][0] = -1.0
	J[4][1] = -1.0
	J[5][2] = -1.0

	return J
}

/**
 * getDissipationMatrix computes R(x) with drag absorbed.
 *
 * R is PSD via Cholesky: R = L·Lᵀ
 * Drag is modeled as dissipative energy loss, not as a force.
 *
 * The dissipation depends on:
 * - Altitude (atmospheric density)
 * - Velocity magnitude (dynamic pressure)
 * - Vehicle configuration (CD)
 */
func getDissipationMatrix(state HGVPHSState, consts HGVConstants, jd float64) [6][6]float64 {
	var R [6][6]float64

	// Get atmospheric density
	r := physics.EARTH_R_EQUATORIAL + state.Altitude
	rho := physics.GetDensityUnified(0, 0, r, jd)

	// Velocity magnitude
	vMag := vectorNorm(state.Velocity)

	// Dynamic pressure
	q := 0.5 * rho * vMag * vMag * 1e6 // Pa

	// Drag coefficient (mode-dependent)
	CD := consts.CdGlide // Default glide mode

	// Drag force
	D := q * consts.ReferenceArea * CD

	// Dissipation rate (power lost to drag)
	// P_drag = D·v = energy dissipation rate
	dissipationRate := D * vMag / consts.Mass / 1000.0 // km²/s³

	// Construct R matrix
	// R is diagonal for simplicity (velocity damping)
	// Higher dissipation at high altitude (dense atmosphere) and high velocity
	damping := dissipationRate / (vMag*vMag + 1e-6)

	R[3][3] = damping // x-velocity damping
	R[4][4] = damping // y-velocity damping
	R[5][5] = damping // z-velocity damping

	// Position damping (small, from atmospheric drag on direction)
	R[0][0] = damping * 0.1
	R[1][1] = damping * 0.1
	R[2][2] = damping * 0.1

	return R
}

/**
 * getControlCoupling computes g(x) with bank-angle absorbed.
 *
 * The control input u represents lift command.
 * Bank angle σ is the hidden input that determines lift direction.
 *
 * g(x) maps u to the state dynamics:
 *     ẋ += g(x)·u
 *
 * For HGV, lift acts perpendicular to velocity and depends on bank angle.
 * The bank angle is inferred from the residual, not directly observed.
 */
func getControlCoupling(state HGVPHSState, consts HGVConstants) [6][3]float64 {
	var g [6][3]float64

	// Get atmospheric density
	r := physics.EARTH_R_EQUATORIAL + state.Altitude
	rho := physics.GetDensityUnified(0, 0, r, 0) // Use default JD

	// Velocity
	v := state.Velocity
	vMag := vectorNorm(v)

	if vMag < 1e-6 {
		return g // No control if no velocity
	}

	// Dynamic pressure
	q := 0.5 * rho * vMag * vMag * 1e6

	// Lift coefficient
	CL := consts.LDGlide * consts.CdGlide

	// Lift force magnitude
	L := q * consts.ReferenceArea * CL

	// Lift per unit mass (acceleration)
	liftAccel := L / consts.Mass / 1000.0 // km/s²

	// Control coupling: lift acts perpendicular to velocity
	// The direction depends on bank angle (hidden input)
	// We provide the basis vectors for lift in the body frame

	// Velocity direction (normalized)
	vDir := [3]float64{v[0] / vMag, v[1] / vMag, v[2] / vMag}

	// Perpendicular directions (lift plane)
	// Cross product with up vector gives lateral direction
	up := [3]float64{0, 0, 1}
	lateral := crossProductPHS(vDir, up)
	lateralMag := vectorNorm(lateral)

	if lateralMag > 1e-6 {
		lateral[0] /= lateralMag
		lateral[1] /= lateralMag
		lateral[2] /= lateralMag
	}

	// Control input u = [u_lift, u_bank, u_maneuver]
	// u_lift: lift magnitude command
	// u_bank: bank angle command (affects lift direction)
	// u_maneuver: general maneuver command

	// Lift acts in the lateral direction (perpendicular to velocity)
	g[3][0] = lateral[0] * liftAccel // x-acceleration from lift
	g[4][0] = lateral[1] * liftAccel // y-acceleration from lift
	g[5][0] = lateral[2] * liftAccel // z-acceleration from lift

	// Bank angle affects lift direction (coupling)
	// This is the key for zero-day tracking: bank angle is inferred from residual
	g[3][1] = vDir[0] * liftAccel * 0.5 // Bank-induced turn
	g[4][1] = vDir[1] * liftAccel * 0.5
	g[5][1] = vDir[2] * liftAccel * 0.5

	return g
}

/**
 * HGVPHSToClassical converts PHS state back to classical spherical coordinates.
 *
 * This allows interoperability with the existing HGV physics code.
 */
func HGVPHSToClassical(phs HGVPHSState) HGVState {
	// Extract radius from altitude
	r := physics.EARTH_R_EQUATORIAL + phs.Altitude

	// Convert direction to spherical coordinates
	phi := math.Asin(phs.Direction[2])                       // Latitude
	lambda := math.Atan2(phs.Direction[1], phs.Direction[0]) // Longitude

	// Velocity magnitude
	vMag := vectorNorm(phs.Velocity)

	// Flight path angle (angle from horizontal)
	// Project velocity onto local horizontal plane
	horizontalV := math.Sqrt(phs.Velocity[0]*phs.Velocity[0] + phs.Velocity[1]*phs.Velocity[1])
	gamma := math.Atan2(phs.Velocity[2], horizontalV)

	// Heading angle
	psi := math.Atan2(phs.Velocity[1], phs.Velocity[0])

	return HGVState{
		R:      r,
		Phi:    phi,
		Lambda: lambda,
		V:      vMag,
		Gamma:  gamma,
		Psi:    psi,
		// Bank angle is hidden, not directly in state
		// It's inferred from control input u(t) via SINDy
	}
}

/**
 * ClassicalToHGVPHS converts classical spherical state to PHS state.
 */
func ClassicalToHGVPHS(classical HGVState) HGVPHSState {
	// Direction from spherical coordinates
	sinPhi := math.Sin(classical.Phi)
	cosPhi := math.Cos(classical.Phi)
	sinLambda := math.Sin(classical.Lambda)
	cosLambda := math.Cos(classical.Lambda)

	direction := [3]float64{
		cosPhi * cosLambda,
		cosPhi * sinLambda,
		sinPhi,
	}

	// Velocity from spherical coordinates
	sinGamma := math.Sin(classical.Gamma)
	cosGamma := math.Cos(classical.Gamma)
	sinPsi := math.Sin(classical.Psi)
	cosPsi := math.Cos(classical.Psi)

	// Local ENU velocity
	vEast := classical.V * cosGamma * sinPsi
	vNorth := classical.V * cosGamma * cosPsi
	vUp := classical.V * sinGamma

	// Transform to ECI (simplified - use proper rotation in production)
	velocity := [3]float64{
		vEast*cosLambda - vNorth*sinLambda,
		vEast*sinLambda + vNorth*cosLambda,
		vUp,
	}

	// Altitude
	altitude := classical.R - physics.EARTH_R_EQUATORIAL

	// Identity rotation for orientation (simplified)
	var orientation [3][3]float64
	for i := 0; i < 3; i++ {
		orientation[i][i] = 1.0
	}

	return HGVPHSState{
		Direction:   direction,
		Velocity:    velocity,
		Orientation: orientation,
		Altitude:    altitude,
	}
}

/**
 * InferBankAngleFromControl infers bank angle from control input.
 *
 * This is the key for zero-day HGV tracking:
 * - The PHS model predicts dynamics without bank angle
 * - SINDy infers u(t) from residual: r(t) = ẋ_obs - (J-R)∇H
 * - Bank angle is extracted from the lift component of u(t)
 *
 * u(t) = [u_lift, u_bank, u_maneuver]
 * Bank angle σ ≈ arctan(u_bank / u_lift)
 */
func InferBankAngleFromControl(u [3]float64) float64 {
	// Bank angle from control input
	// u_bank represents lateral lift component
	// u_lift represents vertical lift component
	uLift := u[0]
	uBank := u[1]

	// Bank angle: σ = arctan(u_bank / u_lift)
	sigma := math.Atan2(uBank, uLift)

	return sigma
}

/**
 * Helper functions for vector/matrix operations
 */

func vectorNorm(v [3]float64) float64 {
	return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
}

func dotProduct(a, b [3]float64) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

// crossProductPHS - renamed to avoid conflict with hypersonic_filter.go
func crossProductPHS(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

func matrixSubtract(A, B [6][6]float64) [6][6]float64 {
	var C [6][6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			C[i][j] = A[i][j] - B[i][j]
		}
	}
	return C
}

func matrixVectorMultiply6x6(A [6][6]float64, v [6]float64) [6]float64 {
	var result [6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			result[i] += A[i][j] * v[j]
		}
	}
	return result
}

func matrixVectorMultiply6x3(A [6][3]float64, v [3]float64) [6]float64 {
	var result [6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 3; j++ {
			result[i] += A[i][j] * v[j]
		}
	}
	return result
}
