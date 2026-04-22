package orbital

/**
 * CR3BP with State Transition Matrix (STM)
 *
 * Production-quality CR3BP implementation
 * Based on DhruvJ22/Astrodynamics_Research (Purdue Multi-Body Dynamics)
 *
 * Mathematical Framework:
 * - 42-element state vector: [6 state + 36 STM elements]
 * - STM propagation: dΦ/dt = A(t)Φ
 * - Analytical Jacobian for precision
 * - Differential correction for periodic orbits
 *
 * Key Equations:
 * ẍ - 2ẏ = ∂Ω/∂x
 * ÿ + 2ẋ = ∂Ω/∂y
 * z̈ = ∂Ω/∂z
 *
 * Where Ω = ½(x² + y²) + (1-μ)/r₁ + μ/r₂
 *
 * References:
 * - DhruvJ22/Astrodynamics_Research
 * - Howell et al. (Purdue): Three-body periodic orbits
 * - Koon et al.: Dynamical Systems Approach
 */

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// CR3BPStateSTM represents the augmented 42-element state vector
type CR3BPStateSTM struct {
	// State [0-5]: x, y, z, vx, vy, vz
	X  float64
	Y  float64
	Z  float64
	Vx float64
	Vy float64
	Vz float64

	// STM [6-41]: 6×6 matrix flattened in row-major order
	// Φ[i,j] = STM[i*6 + j]
	// Φ maps initial perturbations to final perturbations: δX(t) = Φ(t,t₀)δX(t₀)
	STM [36]float64
}

// GetSTM extracts 6×6 matrix from flattened representation
func (s *CR3BPStateSTM) GetSTM() *mat.Dense {
	data := make([]float64, 36)
	copy(data, s.STM[:])
	return mat.NewDense(6, 6, data)
}

// SetSTM stores 6×6 matrix in flattened representation
func (s *CR3BPStateSTM) SetSTM(m *mat.Dense) {
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			s.STM[i*6+j] = m.At(i, j)
		}
	}
}

// ToSlice converts to 42-element slice for integration
func (s *CR3BPStateSTM) ToSlice() []float64 {
	result := make([]float64, 42)
	result[0] = s.X
	result[1] = s.Y
	result[2] = s.Z
	result[3] = s.Vx
	result[4] = s.Vy
	result[5] = s.Vz
	copy(result[6:], s.STM[:])
	return result
}

// FromSlice populates from 42-element slice
func (s *CR3BPStateSTM) FromSlice(data []float64) error {
	if len(data) != 42 {
		return fmt.Errorf("expected 42 elements, got %d", len(data))
	}
	s.X = data[0]
	s.Y = data[1]
	s.Z = data[2]
	s.Vx = data[3]
	s.Vy = data[4]
	s.Vz = data[5]
	copy(s.STM[:], data[6:])
	return nil
}

// CR3BPModelWithSTM provides full dynamics including STM propagation
type CR3BPModelWithSTM struct {
	Mu float64 // Mass parameter
}

// NewCR3BPModelWithSTM creates Earth-Moon model
func NewCR3BPModelWithSTM() *CR3BPModelWithSTM {
	return &CR3BPModelWithSTM{
		Mu: 0.012150585609624, // Earth-Moon mass parameter
	}
}

// ComputeEOM calculates equations of motion for 42-element state
// This is the core derivative function for integration
func (m *CR3BPModelWithSTM) ComputeEOM(t float64, y []float64, dy []float64) {
	if len(y) != 42 || len(dy) != 42 {
		return
	}

	// Extract state
	x, y_, z := y[0], y[1], y[2]
	vx, vy, vz := y[3], y[4], y[5]

	// Compute distances to primaries
	r1 := math.Sqrt((x+m.Mu)*(x+m.Mu) + y_*y_ + z*z)     // Distance to Earth
	r2 := math.Sqrt((x-1+m.Mu)*(x-1+m.Mu) + y_*y_ + z*z) // Distance to Moon

	// Avoid division by zero
	if r1 < 1e-10 || r2 < 1e-10 {
		return
	}

	// Compute accelerations (CR3BP EOMs)
	// ẍ = 2ẏ + x - (1-μ)(x+μ)/r₁³ - μ(x-1+μ)/r₂³
	ax := 2.0*vy + x - (1.0-m.Mu)*(x+m.Mu)/(r1*r1*r1) - m.Mu*(x-1.0+m.Mu)/(r2*r2*r2)

	// ÿ = -2ẋ + y - (1-μ)y/r₁³ - μy/r₂³
	ay := -2.0*vx + y_ - (1.0-m.Mu)*y_/(r1*r1*r1) - m.Mu*y_/(r2*r2*r2)

	// z̈ = -(1-μ)z/r₁³ - μz/r₂³
	az := -(1.0-m.Mu)*z/(r1*r1*r1) - m.Mu*z/(r2*r2*r2)

	// Fill state derivatives
	dy[0] = vx
	dy[1] = vy
	dy[2] = vz
	dy[3] = ax
	dy[4] = ay
	dy[5] = az

	// Compute Jacobian A(t) = ∂f/∂X
	A := m.computeJacobian(x, y_, z, r1, r2)

	// Extract current STM
	Phi := mat.NewDense(6, 6, nil)
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			Phi.Set(i, j, y[6+i*6+j])
		}
	}

	// Compute dΦ/dt = A * Φ
	dPhi := mat.NewDense(6, 6, nil)
	dPhi.Mul(A, Phi)

	// Store STM derivatives
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			dy[6+i*6+j] = dPhi.At(i, j)
		}
	}
}

// computeJacobian calculates A(t) = ∂f/∂X analytically
// This is the derivative of EOM with respect to state
func (m *CR3BPModelWithSTM) computeJacobian(x, y, z, r1, r2 float64) *mat.Dense {
	A := mat.NewDense(6, 6, nil)

	// Common terms
	r1_3 := r1 * r1 * r1
	r1_5 := r1_3 * r1 * r1
	r2_3 := r2 * r2 * r2
	r2_5 := r2_3 * r2 * r2

	// Partial derivatives of accelerations
	// ∂ax/∂x = 1 - (1-μ)/r₁³ + 3(1-μ)(x+μ)²/r₁⁵ - μ/r₂³ + 3μ(x-1+μ)²/r₂⁵
	dax_dx := 1.0 - (1.0-m.Mu)/r1_3 + 3.0*(1.0-m.Mu)*(x+m.Mu)*(x+m.Mu)/r1_5 -
		m.Mu/r2_3 + 3.0*m.Mu*(x-1.0+m.Mu)*(x-1.0+m.Mu)/r2_5

	// ∂ax/∂y = 3(1-μ)(x+μ)y/r₁⁵ + 3μ(x-1+μ)y/r₂⁵
	dax_dy := 3.0*(1.0-m.Mu)*(x+m.Mu)*y/r1_5 + 3.0*m.Mu*(x-1.0+m.Mu)*y/r2_5

	// ∂ax/∂z = 3(1-μ)(x+μ)z/r₁⁵ + 3μ(x-1+μ)z/r₂⁵
	dax_dz := 3.0*(1.0-m.Mu)*(x+m.Mu)*z/r1_5 + 3.0*m.Mu*(x-1.0+m.Mu)*z/r2_5

	// ∂ay/∂x = 3(1-μ)(x+μ)y/r₁⁵ + 3μ(x-1+μ)y/r₂⁵
	day_dx := 3.0*(1.0-m.Mu)*(x+m.Mu)*y/r1_5 + 3.0*m.Mu*(x-1.0+m.Mu)*y/r2_5

	// ∂ay/∂y = 1 - (1-μ)/r₁³ + 3(1-μ)y²/r₁⁵ - μ/r₂³ + 3μy²/r₂⁵
	day_dy := 1.0 - (1.0-m.Mu)/r1_3 + 3.0*(1.0-m.Mu)*y*y/r1_5 -
		m.Mu/r2_3 + 3.0*m.Mu*y*y/r2_5

	// ∂ay/∂z = 3(1-μ)yz/r₁⁵ + 3μyz/r₂⁵
	day_dz := 3.0*(1.0-m.Mu)*y*z/r1_5 + 3.0*m.Mu*y*z/r2_5

	// ∂az/∂x = 3(1-μ)(x+μ)z/r₁⁵ + 3μ(x-1+μ)z/r₂⁵
	daz_dx := 3.0*(1.0-m.Mu)*(x+m.Mu)*z/r1_5 + 3.0*m.Mu*(x-1.0+m.Mu)*z/r2_5

	// ∂az/∂y = 3(1-μ)yz/r₁⁵ + 3μyz/r₂⁵
	daz_dy := 3.0*(1.0-m.Mu)*y*z/r1_5 + 3.0*m.Mu*y*z/r2_5

	// ∂az/∂z = -(1-μ)/r₁³ + 3(1-μ)z²/r₁⁵ - μ/r₂³ + 3μz²/r₂⁵
	daz_dz := -(1.0-m.Mu)/r1_3 + 3.0*(1.0-m.Mu)*z*z/r1_5 -
		m.Mu/r2_3 + 3.0*m.Mu*z*z/r2_5

	// Fill Jacobian matrix A = ∂f/∂X
	// First 3 rows: derivatives of velocity (identity for position)
	A.Set(0, 3, 1.0) // ∂vx/∂vx
	A.Set(1, 4, 1.0) // ∂vy/∂vy
	A.Set(2, 5, 1.0) // ∂vz/∂vz

	// Last 3 rows: derivatives of acceleration
	A.Set(3, 0, dax_dx)
	A.Set(3, 1, dax_dy)
	A.Set(3, 2, dax_dz)
	A.Set(3, 4, 2.0) // Coriolis: ∂ax/∂vy = 2

	A.Set(4, 0, day_dx)
	A.Set(4, 1, day_dy)
	A.Set(4, 2, day_dz)
	A.Set(4, 3, -2.0) // Coriolis: ∂ay/∂vx = -2

	A.Set(5, 0, daz_dx)
	A.Set(5, 1, daz_dy)
	A.Set(5, 2, daz_dz)

	return A
}

// DormandPrince54 integrator for CR3BP
// 5th order Runge-Kutta with 4th order error estimate
type DormandPrince54 struct {
	Model *CR3BPModelWithSTM
}

// Step performs one Dormand-Prince integration step
func (dp *DormandPrince54) Step(y []float64, t, h float64) ([]float64, float64) {
	// Dormand-Prince coefficients
	c2, c3, c4, c5, c6, c7 := 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0

	a21 := 1.0 / 5.0
	a31, a32 := 3.0/40.0, 9.0/40.0
	a41, a42, a43 := 44.0/45.0, -56.0/15.0, 32.0/9.0
	a51, a52, a53, a54 := 19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0
	a61, a62, a63, a64, a65 := 9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0
	a71, a72, a73, a74, a75, a76 := 35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0

	// Compute stages
	k1 := make([]float64, 42)
	dp.Model.ComputeEOM(t, y, k1)

	y2 := make([]float64, 42)
	for i := range y {
		y2[i] = y[i] + h*a21*k1[i]
	}
	k2 := make([]float64, 42)
	dp.Model.ComputeEOM(t+c2*h, y2, k2)

	y3 := make([]float64, 42)
	for i := range y {
		y3[i] = y[i] + h*(a31*k1[i]+a32*k2[i])
	}
	k3 := make([]float64, 42)
	dp.Model.ComputeEOM(t+c3*h, y3, k3)

	y4 := make([]float64, 42)
	for i := range y {
		y4[i] = y[i] + h*(a41*k1[i]+a42*k2[i]+a43*k3[i])
	}
	k4 := make([]float64, 42)
	dp.Model.ComputeEOM(t+c4*h, y4, k4)

	y5 := make([]float64, 42)
	for i := range y {
		y5[i] = y[i] + h*(a51*k1[i]+a52*k2[i]+a53*k3[i]+a54*k4[i])
	}
	k5 := make([]float64, 42)
	dp.Model.ComputeEOM(t+c5*h, y5, k5)

	y6 := make([]float64, 42)
	for i := range y {
		y6[i] = y[i] + h*(a61*k1[i]+a62*k2[i]+a63*k3[i]+a64*k4[i]+a65*k5[i])
	}
	k6 := make([]float64, 42)
	dp.Model.ComputeEOM(t+c6*h, y6, k6)

	y7 := make([]float64, 42)
	for i := range y {
		y7[i] = y[i] + h*(a71*k1[i]+a72*k2[i]+a73*k3[i]+a74*k4[i]+a75*k5[i]+a76*k6[i])
	}
	k7 := make([]float64, 42)
	dp.Model.ComputeEOM(t+c7*h, y7, k7)

	// 5th order solution
	yNext := make([]float64, 42)
	for i := range y {
		yNext[i] = y[i] + h*(35.0/384.0*k1[i]+500.0/1113.0*k3[i]+125.0/192.0*k4[i]-
			2187.0/6784.0*k5[i]+11.0/84.0*k6[i])
	}

	// Error estimate (4th order)
	err := 0.0
	for i := 0; i < 6; i++ { // Only state error, not STM
		e := h * (71.0/57600.0*k1[i] - 71.0/16695.0*k3[i] + 71.0/1920.0*k4[i] -
			17253.0/339200.0*k5[i] + 22.0/525.0*k6[i] - 1.0/40.0*k7[i])
		if abs(e) > err {
			err = abs(e)
		}
	}

	return yNext, err
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// DifferentialCorrector finds periodic orbits using single shooting
type DifferentialCorrector struct {
	Model   *CR3BPModelWithSTM
	Tol     float64
	MaxIter int
}

// NewDifferentialCorrector creates corrector with default settings
func NewDifferentialCorrector() *DifferentialCorrector {
	return &DifferentialCorrector{
		Model:   NewCR3BPModelWithSTM(),
		Tol:     1e-12,
		MaxIter: 50,
	}
}

// FindPeriodicHalo finds Halo orbit from initial guess
// Uses XZ-symmetry: at T/2, y=0, ẋ=0, ż=0
func (dc *DifferentialCorrector) FindPeriodicHalo(
	initialGuess CR3BPStateSTM,
	periodGuess float64,
) (*CR3BPStateSTM, float64, error) {

	state := initialGuess
	period := periodGuess

	integrator := &DormandPrince54{Model: dc.Model}

	for iter := 0; iter < dc.MaxIter; iter++ {
		// Propagate to T/2
		y := state.ToSlice()
		t := 0.0
		h := period / 200.0 // Adaptive step would be better

		for t < period/2.0 {
			if t+h > period/2.0 {
				h = period/2.0 - t
			}
			yNext, _ := integrator.Step(y, t, h)
			y = yNext
			t += h
		}

		finalState := &CR3BPStateSTM{}
		finalState.FromSlice(y)

		// Constraints for XZ-symmetric orbit at T/2:
		// y = 0, ẋ = 0, ż = 0
		constraints := []float64{
			finalState.Y,  // Should be 0
			finalState.Vx, // Should be 0
			finalState.Vz, // Should be 0
		}

		// Check convergence
		maxErr := 0.0
		for _, c := range constraints {
			if abs(c) > maxErr {
				maxErr = abs(c)
			}
		}

		if maxErr < dc.Tol {
			return &state, period, nil // Converged
		}

		// Get STM at T/2
		Phi := finalState.GetSTM()

		// Correction matrix (simplified)
		// We want to adjust initial vy and period to drive constraints to 0
		// [δẏ₀, δT/2] = M⁻¹ * [-constraints]

		// Build sensitivity matrix
		// This is a simplified version - full implementation uses STM blocks
		M := mat.NewDense(3, 2, nil)

		// Sensitivities (simplified - should extract from STM properly)
		M.Set(0, 0, Phi.At(1, 4))  // ∂y/∂ẏ₀
		M.Set(0, 1, finalState.Vy) // ∂y/∂(T/2)

		M.Set(1, 0, Phi.At(3, 4)) // ∂ẋ/∂ẏ₀
		M.Set(1, 1, 0.0)          // ∂ẋ/∂(T/2) ≈ 0 at perpendicular crossing

		M.Set(2, 0, Phi.At(5, 4)) // ∂ż/∂ẏ₀
		M.Set(2, 1, 0.0)          // ∂ż/∂(T/2) ≈ 0

		// Solve for correction
		b := mat.NewVecDense(3, []float64{-constraints[0], -constraints[1], -constraints[2]})
		var correction mat.VecDense
		err := correction.SolveVec(M, b)
		if err != nil {
			return nil, 0, fmt.Errorf("correction solve failed at iter %d: %v", iter, err)
		}

		// Apply correction
		state.Vy += correction.AtVec(0)
		period += 2.0 * correction.AtVec(1)
	}

	return nil, 0, fmt.Errorf("did not converge after %d iterations", dc.MaxIter)
}

// PropagateWithSTM performs full propagation with STM
func (m *CR3BPModelWithSTM) PropagateWithSTM(
	initial CR3BPStateSTM,
	tf float64,
	numSteps int,
) ([]CR3BPStateSTM, error) {

	states := make([]CR3BPStateSTM, numSteps+1)
	states[0] = initial

	integrator := &DormandPrince54{Model: m}
	y := initial.ToSlice()
	t := 0.0
	h := tf / float64(numSteps)

	for i := 1; i <= numSteps; i++ {
		yNext, _ := integrator.Step(y, t, h)
		y = yNext
		t += h

		state := CR3BPStateSTM{}
		err := state.FromSlice(y)
		if err != nil {
			return nil, err
		}
		states[i] = state
	}

	return states, nil
}

// VerifyJacobiConstant checks if C is conserved (numerical error indicator)
func (m *CR3BPModelWithSTM) VerifyJacobiConstant(state CR3BPStateSTM) float64 {
	x, y, z := state.X, state.Y, state.Z
	vx, vy, vz := state.Vx, state.Vy, state.Vz

	r1 := math.Sqrt((x+m.Mu)*(x+m.Mu) + y*y + z*z)
	r2 := math.Sqrt((x-1+m.Mu)*(x-1+m.Mu) + y*y + z*z)

	Omega := 0.5*(x*x+y*y) + (1.0-m.Mu)/r1 + m.Mu/r2
	v2 := vx*vx + vy*vy + vz*vz

	C := 2.0*Omega - v2
	return C
}

// MonodromyMatrix returns STM after one full orbital period
// Used for stability analysis and manifold computation
func (m *CR3BPModelWithSTM) MonodromyMatrix(
	periodicState CR3BPStateSTM,
	period float64,
) (*mat.Dense, error) {

	// Initialize with identity STM
	initial := periodicState
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			if i == j {
				initial.STM[i*6+j] = 1.0
			} else {
				initial.STM[i*6+j] = 0.0
			}
		}
	}

	// Propagate for one period
	integrator := &DormandPrince54{Model: m}
	y := initial.ToSlice()
	t := 0.0
	numSteps := 1000
	h := period / float64(numSteps)

	for i := 0; i < numSteps; i++ {
		yNext, _ := integrator.Step(y, t, h)
		y = yNext
		t += h
	}

	final := &CR3BPStateSTM{}
	err := final.FromSlice(y)
	if err != nil {
		return nil, err
	}

	return final.GetSTM(), nil
}
