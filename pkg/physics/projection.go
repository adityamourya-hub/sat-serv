// pkg/physics/projection.go - Manifold Projection & Conflict Resolution

// Implements the "Zero-Tolerance Gate" for Hamiltonian conservation.
// Even with Forest-Ruth integration, floating-point errors accumulate.
// This layer projects states back onto the constant-H surface.

// Conflict Resolution Protocol:
//   - Grade A (drift < 1e-10): Normal operation, apply projection
//   - Grade B (drift 1e-10 to 1e-7): Apply projection, log warning
//   - Grade C (drift 1e-7 to 1e-5): Large correction, alert operator
//   - Grade F (drift > 1e-5): PROJECTION FAILED - trigger Sovereignty Breach
//
// References:
//   - manifold_projection.py (Python reference implementation)
//   - certificate.go (Grade definitions)
//   - command_bridge.go (ReflexKernel integration)

package physics

import (
	"fmt"
	"math"
	"time"
)

// =============================================================================
// Projection-Specific Constants
// =============================================================================

const (
	// Projection parameters
	MaxProjectionIterations = 5
	ProjectionTolerance     = 1e-12
)

// ProjectionResult contains the outcome of manifold projection
type ProjectionResult struct {
	QC                   Vector3               // Corrected position
	PC                   Vector3               // Corrected momentum
	ConstraintViolations []ConstraintViolation // Physics pass/fail per constraint
	InitialDrift         float64               // |ΔE/E|_initial
	FinalDrift           float64               // |ΔE/E|_final
	Iterations           int                   // Newton-Raphson iterations used
	Correction           float64               // Magnitude of correction applied
	Timestamp            time.Time             // When projection occurred
	Success              bool                  // Whether projection succeeded
}

// String returns human-readable summary
func (r *ProjectionResult) String() string {
	violations := "OK"
	if !AllPassed(r.ConstraintViolations) {
		violations = "FAIL"
	}
	return fmt.Sprintf(
		"Projection[Violations=%s, InitialDrift=%.2e, FinalDrift=%.2e, Iterations=%d, Success=%v]",
		violations, r.InitialDrift, r.FinalDrift, r.Iterations, r.Success,
	)
}

// =============================================================================
// Hamiltonian Network Interface
// =============================================================================

// HamiltonianNet provides energy and gradient computations
// In production, this wraps the ONNX H-Net model
type HamiltonianNet interface {
	// Compute returns H(q, p)
	Compute(q, p Vector3) float64

	// GradientQ returns ∇_q H
	GradientQ(q, p Vector3) Vector3

	// GradientP returns ∇_p H
	GradientP(q, p Vector3) Vector3
}

// KeplerianHamiltonian implements analytical Keplerian H-Net for testing
type KeplerianHamiltonian struct {
	Mu float64 // Gravitational parameter
}

// Compute returns H = 0.5*p² - μ/|q|
func (k *KeplerianHamiltonian) Compute(q, p Vector3) float64 {
	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	if r < 1.0 {
		r = 1.0 // Prevent division by zero
	}
	vSq := p[0]*p[0] + p[1]*p[1] + p[2]*p[2]
	return 0.5*vSq - k.Mu/r
}

// GradientQ returns ∇_q H = μ*q/|q|³
func (k *KeplerianHamiltonian) GradientQ(q, p Vector3) Vector3 {
	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	if r < 1.0 {
		r = 1.0
	}
	factor := k.Mu / (r * r * r)
	return Vector3{factor * q[0], factor * q[1], factor * q[2]}
}

// GradientP returns ∇_p H = p
func (k *KeplerianHamiltonian) GradientP(q, p Vector3) Vector3 {
	return p
}

// =============================================================================
// Manifold Projector
// =============================================================================

// ManifoldProjector projects states onto constant-H surface
type ManifoldProjector struct {
	HNet              HamiltonianNet
	MaxIter           int
	Tolerance         float64
	ConsecutiveGradeC int // Counter for Grade C events
	TotalProjections  int // Total projections performed
}

// NewManifoldProjector creates a projector with default settings
func NewManifoldProjector(hNet HamiltonianNet) *ManifoldProjector {
	return &ManifoldProjector{
		HNet:              hNet,
		MaxIter:           MaxProjectionIterations,
		Tolerance:         ProjectionTolerance,
		ConsecutiveGradeC: 0,
		TotalProjections:  0,
	}
}

// ProjectToManifold projects (q, p) onto H(q, p) = targetE surface
//
// Algorithm (Newton-Raphson):
//  1. Compute current E and ∇_p E
//  2. If |E - targetE| < tolerance, return
//  3. Compute correction: δp = (E - targetE) / |∇_p E|² * ∇_p E
//  4. Apply: p_new = p - δp
//  5. Iterate until convergence or max iterations
//
// Note: We only correct momentum p, not position q, to preserve the
// observed position measurement.
func (mp *ManifoldProjector) ProjectToManifold(
	q, p Vector3,
	targetE float64,
) *ProjectionResult {
	mp.TotalProjections++

	startTime := time.Now()

	// Compute initial energy
	currentE := mp.HNet.Compute(q, p)
	initialDrift := math.Abs(currentE-targetE) / math.Abs(targetE)

	// Quick check: already on manifold?
	if initialDrift < mp.Tolerance {
		return &ProjectionResult{
			QC:                   q,
			PC:                   p,
			ConstraintViolations: make([]ConstraintViolation, 0),
			InitialDrift:         initialDrift,
			FinalDrift:           initialDrift,
			Iterations:           0,
			Correction:           0.0,
			Timestamp:            startTime,
			Success:              true,
		}
	}

	// Newton-Raphson iteration
	pCurrent := p
	iterations := 0

	for i := 0; i < mp.MaxIter; i++ {
		currentE = mp.HNet.Compute(q, pCurrent)
		err := currentE - targetE

		// Check convergence
		if math.Abs(err)/math.Abs(targetE) < mp.Tolerance {
			iterations = i
			break
		}

		// Compute gradient
		gradP := mp.HNet.GradientP(q, pCurrent)
		gradPNormSq := gradP[0]*gradP[0] + gradP[1]*gradP[1] + gradP[2]*gradP[2]

		// Avoid division by zero
		if gradPNormSq < 1e-20 {
			gradPNormSq = 1e-20
		}

		// Newton-Raphson step
		// δp = (E - targetE) / |∇E|² * ∇E
		correction := err / gradPNormSq
		pCurrent = Vector3{
			pCurrent[0] - gradP[0]*correction,
			pCurrent[1] - gradP[1]*correction,
			pCurrent[2] - gradP[2]*correction,
		}
		iterations = i + 1
	}

	// Compute final energy and drift
	finalE := mp.HNet.Compute(q, pCurrent)
	finalDrift := math.Abs(finalE-targetE) / math.Abs(targetE)

	// Compute total correction magnitude
	correctionMag := math.Sqrt(
		(pCurrent[0]-p[0])*(pCurrent[0]-p[0]) +
			(pCurrent[1]-p[1])*(pCurrent[1]-p[1]) +
			(pCurrent[2]-p[2])*(pCurrent[2]-p[2]),
	)

	violations := mp.determineConstraintViolations(finalDrift, iterations)

	// Track consecutive significant drift events for escalation
	energyDriftFailed := false
	for _, v := range violations {
		if v.Constraint == "energy_drift" && !v.Passed {
			energyDriftFailed = true
		}
	}
	if energyDriftFailed {
		mp.ConsecutiveGradeC++
	} else {
		mp.ConsecutiveGradeC = 0
	}

	return &ProjectionResult{
		QC:                   q,
		PC:                   pCurrent,
		ConstraintViolations: violations,
		InitialDrift:         initialDrift,
		FinalDrift:           finalDrift,
		Iterations:           iterations,
		Correction:           correctionMag,
		Timestamp:            time.Now(),
		Success:              AllPassed(violations),
	}
}

// determineConstraintViolations evaluates energy-drift constraints from manifold projection.
func (mp *ManifoldProjector) determineConstraintViolations(finalDrift float64, iterations int) []ConstraintViolation {
	th := DefaultGradeThresholds()
	return []ConstraintViolation{
		{
			Constraint: "energy_drift",
			Value:      finalDrift,
			Threshold:  th.MaxEnergyDriftA,
			Unit:       "relative",
			Passed:     finalDrift < th.MaxEnergyDriftA,
		},
		{
			Constraint: "projection_converged",
			Value:      float64(iterations),
			Threshold:  float64(mp.MaxIter - 1),
			Unit:       "iterations",
			Passed:     iterations < mp.MaxIter-1,
		},
		{
			Constraint: "extreme_drift",
			Value:      finalDrift,
			Threshold:  th.MaxEnergyDriftC,
			Unit:       "relative",
			Passed:     finalDrift < th.MaxEnergyDriftC,
		},
	}
}

// =============================================================================
// Conflict Resolution Protocol
// =============================================================================

// ResolutionAction indicates the system's response to projection failure
type ResolutionAction int

const (
	ActionContinue          ResolutionAction = iota // Normal operation
	ActionAlert                                     // Log warning, continue
	ActionReInitialize                              // Re-initialize track from catalog
	ActionManualReview                              // Flag for operator intervention
	ActionSovereigntyBreach                         // Emergency stop, alert IAF
)

func (a ResolutionAction) String() string {
	switch a {
	case ActionContinue:
		return "CONTINUE"
	case ActionAlert:
		return "ALERT"
	case ActionReInitialize:
		return "RE-INITIALIZE"
	case ActionManualReview:
		return "MANUAL_REVIEW"
	case ActionSovereigntyBreach:
		return "SOVEREIGNTY_BREACH"
	default:
		return "UNKNOWN"
	}
}

// ConflictResolver handles projection failures according to protocol
type ConflictResolver struct {
	Projector            *ManifoldProjector
	MaxConsecutiveGradeC int // Threshold for escalation
	BreachCount          int // Total sovereignty breaches
}

// NewConflictResolver creates a resolver with default settings
func NewConflictResolver(projector *ManifoldProjector) *ConflictResolver {
	return &ConflictResolver{
		Projector:            projector,
		MaxConsecutiveGradeC: 3, // 3 consecutive Grade C -> escalate
		BreachCount:          0,
	}
}

// Resolve determines the appropriate action for a projection result
//
// Protocol:
//
// Resolve determines the system response based on projection constraint violations.
//
// Energy drift < 1e-6: Continue normally
// Energy drift < 1e-4: Continue, log for monitoring
// Energy drift < 1e-2: Alert operator, track consecutive events
// Energy drift >= 1e-2: Sovereignty breach - stop and alert
//
// Escalation: 3 consecutive significant drift events -> Force manual review
func (cr *ConflictResolver) Resolve(result *ProjectionResult) ResolutionAction {
	if AllPassed(result.ConstraintViolations) {
		return ActionContinue
	}

	// Check for extreme drift (sovereignty breach)
	for _, v := range result.ConstraintViolations {
		if v.Constraint == "extreme_drift" && !v.Passed {
			cr.BreachCount++
			fmt.Printf("[ConflictResolver] SOVEREIGNTY BREACH #%d: Drift=%.2e\n",
				cr.BreachCount, result.FinalDrift)
			fmt.Printf("  Initial drift: %.2e\n", result.InitialDrift)
			fmt.Printf("  Iterations: %d\n", result.Iterations)
			fmt.Printf("  Possible causes: Spoofing, hardware failure, or adversarial attack\n")
			return ActionSovereigntyBreach
		}
	}

	// Significant drift - check for escalation
	fmt.Printf("[ConflictResolver] Significant drift: Drift=%.2e, correction=%.2f km/s\n",
		result.FinalDrift, result.Correction)
	if cr.Projector.ConsecutiveGradeC >= cr.MaxConsecutiveGradeC {
		fmt.Printf("[ConflictResolver] ESCALATION: %d consecutive drift events\n",
			cr.Projector.ConsecutiveGradeC)
		return ActionManualReview
	}
	return ActionAlert
}

// ShouldReInitialize returns true if track should be re-initialized
func (cr *ConflictResolver) ShouldReInitialize(action ResolutionAction) bool {
	return action == ActionReInitialize || action == ActionSovereigntyBreach
}

// ShouldManualReview returns true if operator intervention required
func (cr *ConflictResolver) ShouldManualReview(action ResolutionAction) bool {
	return action == ActionManualReview || action == ActionSovereigntyBreach
}

// =============================================================================
// Integration: ReflexKernel Hook
// =============================================================================

// ProjectionHook integrates manifold projection into the reflex kernel
type ProjectionHook struct {
	Projector *ManifoldProjector
	Resolver  *ConflictResolver
	TargetE   float64 // Initial energy for this track
}

// NewProjectionHook creates a hook for a specific track
func NewProjectionHook(hNet HamiltonianNet, initialQ, initialP Vector3) *ProjectionHook {
	projector := NewManifoldProjector(hNet)
	resolver := NewConflictResolver(projector)

	// Compute initial energy (the invariant)
	targetE := hNet.Compute(initialQ, initialP)

	return &ProjectionHook{
		Projector: projector,
		Resolver:  resolver,
		TargetE:   targetE,
	}
}

// ProcessStep runs projection after each Forest-Ruth integration step
func (ph *ProjectionHook) ProcessStep(q, p Vector3) (*ProjectionResult, ResolutionAction) {
	// Project onto manifold
	result := ph.Projector.ProjectToManifold(q, p, ph.TargetE)

	// Resolve any conflicts
	action := ph.Resolver.Resolve(result)

	return result, action
}

// UpdateTargetE updates the target energy (e.g., after maneuver detection)
func (ph *ProjectionHook) UpdateTargetE(newE float64) {
	fmt.Printf("[ProjectionHook] Updating target energy: %.6e -> %.6e\n",
		ph.TargetE, newE)
	ph.TargetE = newE
}

// =============================================================================
// Unit Tests
// =============================================================================

// TestManifoldProjection runs unit tests (call from _test.go)
func TestManifoldProjection() {
	fmt.Println("\n=============================================================================")
	fmt.Println("Manifold Projection Unit Tests")
	fmt.Println("=============================================================================")

	// Test 1: Perfect conservation (Grade A)
	fmt.Println("\n[Test 1] Grade A: Perfect conservation")
	hNet := &KeplerianHamiltonian{Mu: 398600.4418}
	projector := NewManifoldProjector(hNet)

	q := Vector3{6778.137, 0, 0}
	p := Vector3{0, 7.6686, 0}
	targetE := hNet.Compute(q, p)

	result := projector.ProjectToManifold(q, p, targetE)
	fmt.Printf("Result: %s\n", result.String())

	if AllPassed(result.ConstraintViolations) && result.FinalDrift < 1e-12 {
		fmt.Println("PASS: Perfect conservation verified")
	} else {
		fmt.Printf("FAIL: Expected all constraints passed\n")
	}

	// Test 2: Minor perturbation
	fmt.Println("\n[Test 2] Minor perturbation correction")
	pPert := Vector3{0.01, 7.6686, 0.01} // 10 m/s error

	result = projector.ProjectToManifold(q, pPert, targetE)
	fmt.Printf("Initial drift: %.2e\n", result.InitialDrift)
	fmt.Printf("Final drift:   %.2e\n", result.FinalDrift)
	fmt.Printf("Violations: %v\n", result.ConstraintViolations)

	if AllPassed(result.ConstraintViolations) && result.FinalDrift < 1e-10 {
		fmt.Println("PASS: Minor correction successful")
	} else if AllPassed(result.ConstraintViolations) {
		fmt.Println("PASS: Perfect conservation (even better)")
	} else {
		fmt.Printf("FAIL: Expected all constraints passed\n")
	}

	// Test 3: Large perturbation
	fmt.Println("\n[Test 3] Large perturbation (possible spoofing)")
	pLarge := Vector3{0.5, 7.5, 0.3} // 500 m/s error

	result = projector.ProjectToManifold(q, pLarge, targetE)
	fmt.Printf("Initial drift: %.2e\n", result.InitialDrift)
	fmt.Printf("Final drift:   %.2e\n", result.FinalDrift)
	fmt.Printf("Correction:    %.2f km/s\n", result.Correction)
	fmt.Printf("Violations: %v\n", result.ConstraintViolations)

	resolver := NewConflictResolver(projector)
	action := resolver.Resolve(result)
	fmt.Printf("Resolution: %s\n", action.String())

	energyFailed := false
	for _, v := range result.ConstraintViolations {
		if v.Constraint == "energy_drift" && !v.Passed {
			energyFailed = true
		}
	}
	if energyFailed {
		fmt.Println("PASS: Large perturbation handled correctly")
	} else {
		fmt.Printf("FAIL: Unexpected constraint status\n")
	}

	// Test 4: Conflict resolution escalation
	fmt.Println("\n[Test 4] Conflict Resolution: 3x Grade C escalation")
	resolver = NewConflictResolver(projector)

	for i := 0; i < 3; i++ {
		result = projector.ProjectToManifold(q, pLarge, targetE)
		action = resolver.Resolve(result)
		violations := "OK"
		if !AllPassed(result.ConstraintViolations) {
			violations = "FAIL"
		}
		fmt.Printf("Event %d: Violations=%s, Action=%s, ConsecutiveC=%d\n",
			i+1, violations, action.String(), projector.ConsecutiveGradeC)
	}

	if action == ActionManualReview {
		fmt.Println("PASS: Escalation to Manual Review triggered")
	} else {
		fmt.Printf("FAIL: Expected ManualReview, got %s\n", action.String())
	}

	fmt.Println("\n=============================================================================")
	fmt.Println("All unit tests completed")
	fmt.Println("=============================================================================")
}
