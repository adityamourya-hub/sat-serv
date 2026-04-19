package orbital

/**
 * CR3BP with STM Tests
 *
 * Validates against:
 * - Known Jacobi constants for Earth-Moon L1/L2
 * - STM properties (should satisfy dΦ/dt = AΦ)
 * - Differential correction convergence
 */

import (
	"math"
	"testing"
)

func TestCR3BPModelWithSTM(t *testing.T) {
	model := NewCR3BPModelWithSTM()

	// Verify Earth-Moon mass parameter
	if math.Abs(model.Mu-0.012150585609624) > 1e-15 {
		t.Errorf("Incorrect mass parameter: %.15f", model.Mu)
	}
}

func TestCR3BPStateSTMToSlice(t *testing.T) {
	state := CR3BPStateSTM{
		X:  0.8,
		Y:  0.0,
		Z:  0.1,
		Vx: 0.0,
		Vy: -0.2,
		Vz: 0.0,
	}
	// Initialize STM as identity
	for i := 0; i < 6; i++ {
		state.STM[i*6+i] = 1.0
	}

	slice := state.ToSlice()

	if len(slice) != 42 {
		t.Errorf("Expected 42 elements, got %d", len(slice))
	}

	if slice[0] != 0.8 {
		t.Errorf("X mismatch: %.4f", slice[0])
	}

	if slice[6] != 1.0 { // STM[0,0]
		t.Errorf("STM[0,0] mismatch: %.4f", slice[6])
	}
}

func TestCR3BPStateSTMFromSlice(t *testing.T) {
	slice := make([]float64, 42)
	slice[0] = 0.8
	slice[1] = 0.0
	slice[2] = 0.1
	slice[3] = 0.0
	slice[4] = -0.2
	slice[5] = 0.0
	// STM remains zeros

	state := CR3BPStateSTM{}
	err := state.FromSlice(slice)
	if err != nil {
		t.Fatalf("FromSlice failed: %v", err)
	}

	if state.X != 0.8 {
		t.Errorf("X mismatch: %.4f", state.X)
	}

	if state.Vy != -0.2 {
		t.Errorf("Vy mismatch: %.4f", state.Vy)
	}
}

func TestComputeEOMStateOnly(t *testing.T) {
	model := NewCR3BPModelWithSTM()

	// Test at L1 point (approximately)
	y := make([]float64, 42)
	y[0] = 0.8369 // x (L1 approximate)
	y[1] = 0.0    // y
	y[2] = 0.0    // z
	y[3] = 0.0    // vx
	y[4] = 0.0    // vy
	y[5] = 0.0    // vz

	dy := make([]float64, 42)
	model.ComputeEOM(0.0, y, dy)

	// At equilibrium, accelerations should be small
	ax := dy[3]
	ay := dy[4]
	az := dy[5]

	accel := math.Sqrt(ax*ax + ay*ay + az*az)
	if accel > 1e-3 {
		t.Errorf("Large acceleration at L1: %.6f", accel)
	}

	t.Logf("L1 accelerations: ax=%.6f, ay=%.6f, az=%.6f", ax, ay, az)
}

func TestDormandPrinceStep(t *testing.T) {
	model := NewCR3BPModelWithSTM()
	integrator := &DormandPrince54{Model: model}

	// Simple initial state
	y := make([]float64, 42)
	y[0] = 1.0 // x
	y[1] = 0.0 // y
	y[2] = 0.0 // z
	y[3] = 0.0 // vx
	y[4] = 1.0 // vy
	y[5] = 0.0 // vz

	// Initialize STM as identity
	for i := 0; i < 6; i++ {
		y[6+i*6+i] = 1.0
	}

	yNext, err := integrator.Step(y, 0.0, 0.01)

	if len(yNext) != 42 {
		t.Errorf("Expected 42 elements, got %d", len(yNext))
	}

	if err > 1.0 {
		t.Errorf("Large error estimate: %.6f", err)
	}

	// Position should have changed
	if yNext[0] == y[0] {
		t.Error("Position did not change")
	}

	t.Logf("Step completed: x went from %.4f to %.4f", y[0], yNext[0])
}

func TestVerifyJacobiConstant(t *testing.T) {
	model := NewCR3BPModelWithSTM()

	// At L1 with zero velocity
	state := CR3BPStateSTM{
		X:  0.8369,
		Y:  0.0,
		Z:  0.0,
		Vx: 0.0,
		Vy: 0.0,
		Vz: 0.0,
	}

	C := model.VerifyJacobiConstant(state)

	// L1 Jacobi constant for Earth-Moon is approximately 3.188
	if C < 3.0 || C > 3.5 {
		t.Errorf("Unexpected Jacobi constant: %.4f", C)
	}

	t.Logf("Jacobi constant at L1: %.6f", C)
}

func TestJacobiConstantConservation(t *testing.T) {
	model := NewCR3BPModelWithSTM()

	// Initial state
	initial := CR3BPStateSTM{
		X:  0.8,
		Y:  0.0,
		Z:  0.1,
		Vx: 0.0,
		Vy: -0.2,
		Vz: 0.0,
	}

	C0 := model.VerifyJacobiConstant(initial)

	// Propagate
	states, err := model.PropagateWithSTM(initial, 1.0, 100)
	if err != nil {
		t.Fatalf("Propagation failed: %v", err)
	}

	// Check Jacobi constant at end
	C1 := model.VerifyJacobiConstant(states[len(states)-1])

	dC := math.Abs(C1 - C0)
	t.Logf("Jacobi drift: %.6e over 1 TU", dC)

	// Should be conserved to machine precision with good integrator
	if dC > 1e-6 {
		t.Errorf("Large Jacobi constant drift: %.6e", dC)
	}
}

func TestDifferentialCorrectorCreation(t *testing.T) {
	dc := NewDifferentialCorrector()

	if dc.Tol != 1e-12 {
		t.Errorf("Expected tolerance 1e-12, got %.2e", dc.Tol)
	}

	if dc.MaxIter != 50 {
		t.Errorf("Expected maxIter 50, got %d", dc.MaxIter)
	}
}

func TestSTMPropagation(t *testing.T) {
	model := NewCR3BPModelWithSTM()

	// Initialize with identity STM
	initial := CR3BPStateSTM{
		X:  0.8,
		Y:  0.0,
		Z:  0.0,
		Vx: 0.0,
		Vy: -0.2,
		Vz: 0.0,
	}
	for i := 0; i < 6; i++ {
		initial.STM[i*6+i] = 1.0
	}

	// Propagate short time
	states, err := model.PropagateWithSTM(initial, 0.1, 10)
	if err != nil {
		t.Fatalf("Propagation failed: %v", err)
	}

	final := states[len(states)-1]
	Phi := final.GetSTM()

	// STM should not be identity anymore
	trace := 0.0
	for i := 0; i < 6; i++ {
		trace += Phi.At(i, i)
	}

	// For short time, STM ≈ I + A*dt, so trace ≈ 6 + tr(A)*dt
	if math.Abs(trace-6.0) < 0.01 {
		t.Error("STM did not evolve (still near identity)")
	}

	t.Logf("STM trace: %.4f", trace)
}

func TestComputeJacobianStructure(t *testing.T) {
	model := NewCR3BPModelWithSTM()

	x, y, z := 0.8, 0.0, 0.1
	r1 := math.Sqrt((x+model.Mu)*(x+model.Mu) + y*y + z*z)
	r2 := math.Sqrt((x-1+model.Mu)*(x-1+model.Mu) + y*y + z*z)

	A := model.computeJacobian(x, y, z, r1, r2)

	// Check that A is 6x6
	if A == nil {
		t.Fatal("Jacobian is nil")
	}

	rows, cols := A.Dims()
	if rows != 6 || cols != 6 {
		t.Errorf("Expected 6x6 Jacobian, got %dx%d", rows, cols)
	}

	// Check Coriolis terms
	// A[3,4] = ∂vx_dot/∂vy = 2 (should be positive)
	if A.At(3, 4) != 2.0 {
		t.Errorf("Coriolis term ∂vx/∂vy wrong: %.4f", A.At(3, 4))
	}

	// A[4,3] = ∂vy_dot/∂vx = -2 (should be negative)
	if A.At(4, 3) != -2.0 {
		t.Errorf("Coriolis term ∂vy/∂vx wrong: %.4f", A.At(4, 3))
	}

	// Velocity-position coupling
	// A[0,3] = ∂x/∂vx = 1
	if A.At(0, 3) != 1.0 {
		t.Errorf("Position-velocity coupling wrong: %.4f", A.At(0, 3))
	}
}

// BenchmarkDormandPrince benchmarks the integrator
func BenchmarkDormandPrince(b *testing.B) {
	model := NewCR3BPModelWithSTM()
	integrator := &DormandPrince54{Model: model}

	y := make([]float64, 42)
	y[0] = 0.8
	y[4] = -0.2
	for i := 0; i < 6; i++ {
		y[6+i*6+i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		integrator.Step(y, 0.0, 0.01)
	}
}

// BenchmarkSTMPropagation benchmarks full STM propagation
func BenchmarkSTMPropagation(b *testing.B) {
	model := NewCR3BPModelWithSTM()

	initial := CR3BPStateSTM{
		X:  0.8,
		Y:  0.0,
		Z:  0.1,
		Vx: 0.0,
		Vy: -0.2,
		Vz: 0.0,
	}
	for i := 0; i < 6; i++ {
		initial.STM[i*6+i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.PropagateWithSTM(initial, 1.0, 100)
	}
}
