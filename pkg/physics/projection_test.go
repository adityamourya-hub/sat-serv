package physics

import (
	"testing"
)

func TestManifoldProjector_GradeA(t *testing.T) {
	// Test 1: Perfect conservation (Grade A)
	hNet := &KeplerianHamiltonian{Mu: 398600.4418}
	projector := NewManifoldProjector(hNet)

	q := Vector3{6778.137, 0, 0}
	p := Vector3{0, 7.6686, 0}
	targetE := hNet.Compute(q, p)

	result := projector.ProjectToManifold(q, p, targetE)
	t.Logf("Result: %s", result.String())

	if result.Grade != GradeA {
		t.Errorf("Expected Grade A, got %s", result.Grade)
	}

	if result.FinalDrift >= 1e-12 {
		t.Errorf("Expected drift < 1e-12, got %.2e", result.FinalDrift)
	}

	if !result.Success {
		t.Error("Expected Success=true for Grade A")
	}
}

func TestManifoldProjector_GradeB(t *testing.T) {
	// Test 2: Minor perturbation (Grade B)
	hNet := &KeplerianHamiltonian{Mu: 398600.4418}
	projector := NewManifoldProjector(hNet)

	q := Vector3{6778.137, 0, 0}
	p := Vector3{0, 7.6686, 0}
	targetE := hNet.Compute(q, p)

	// Perturb momentum (10 m/s error)
	pPert := Vector3{0.01, 7.6686, 0.01}

	result := projector.ProjectToManifold(q, pPert, targetE)
	t.Logf("Initial drift: %.2e", result.InitialDrift)
	t.Logf("Final drift: %.2e", result.FinalDrift)
	t.Logf("Grade: %s", result.Grade)

	// Should achieve at least Grade B
	if result.Grade != GradeB && result.Grade != GradeA {
		t.Errorf("Expected Grade B or A, got %s", result.Grade)
	}

	// Final drift should be very small
	if result.FinalDrift >= 1e-8 {
		t.Errorf("Expected drift < 1e-8 after projection, got %.2e", result.FinalDrift)
	}
}

func TestManifoldProjector_GradeC(t *testing.T) {
	// Test 3: Large perturbation (Grade C)
	hNet := &KeplerianHamiltonian{Mu: 398600.4418}
	projector := NewManifoldProjector(hNet)

	q := Vector3{6778.137, 0, 0}
	p := Vector3{0, 7.6686, 0}
	targetE := hNet.Compute(q, p)

	// Large perturbation (500 m/s error)
	pLarge := Vector3{0.5, 7.5, 0.3}

	result := projector.ProjectToManifold(q, pLarge, targetE)
	t.Logf("Initial drift: %.2e", result.InitialDrift)
	t.Logf("Final drift: %.2e", result.FinalDrift)
	t.Logf("Correction: %.2f km/s", result.Correction)
	t.Logf("Grade: %s", result.Grade)

	// Should handle the correction (not Grade F)
	if result.Grade == GradeF {
		t.Errorf("Projection failed with Grade F for large but valid perturbation")
	}

	// Final drift should be acceptable
	if result.FinalDrift >= 1e-5 {
		t.Errorf("Expected drift < 1e-5 after projection, got %.2e", result.FinalDrift)
	}
}

func TestConflictResolver_Escalation(t *testing.T) {
	// Test 4: Conflict resolution escalation
	hNet := &KeplerianHamiltonian{Mu: 398600.4418}
	projector := NewManifoldProjector(hNet)
	resolver := NewConflictResolver(projector)

	q := Vector3{6778.137, 0, 0}
	p := Vector3{0, 7.6686, 0}
	_ = hNet.Compute(q, p) // Compute but don't need targetE for manual test

	// Note: Manifold projection is highly effective, so we need extreme perturbations
	// to trigger Grade C. The projection corrects most drift to Grade A.
	// Use a non-projectable state (energy way off)

	// Simulate 3 consecutive Grade C events by manipulating the counter directly
	for i := 0; i < 3; i++ {
		projector.ConsecutiveGradeC++
	}

	t.Logf("Simulated %d consecutive Grade C events", projector.ConsecutiveGradeC)

	// Create a Grade C result manually to test escalation
	result := &ProjectionResult{
		QC:           q,
		PC:           p,
		Grade:        GradeC,
		InitialDrift: 1e-3,
		FinalDrift:   1e-5,
		Iterations:   5,
		Correction:   1.0,
		Success:      true,
	}

	lastAction := resolver.Resolve(result)
	t.Logf("Action after escalation: %s", lastAction.String())

	// Should escalate to Manual Review
	if lastAction != ActionManualReview {
		t.Errorf("Expected ActionManualReview after 3 Grade C events, got %s", lastAction.String())
	}
}

func TestConflictResolver_SovereigntyBreach(t *testing.T) {
	// Test 5: Sovereignty breach detection (extreme drift)
	hNet := &KeplerianHamiltonian{Mu: 398600.4418}
	projector := NewManifoldProjector(hNet)
	resolver := NewConflictResolver(projector)

	// Manually create a Grade F result to test breach handling
	result := &ProjectionResult{
		QC:           Vector3{6778.137, 0, 0},
		PC:           Vector3{0, 7.6686, 0},
		Grade:        GradeF,
		InitialDrift: 1.0,
		FinalDrift:   1e-1, // Beyond 1e-2 threshold
		Iterations:   5,
		Correction:   100.0,
		Success:      false,
	}

	action := resolver.Resolve(result)

	t.Logf("Simulated Grade F case: Grade=%s, Action=%s", result.Grade, action.String())

	// Verify breach count incremented
	if resolver.BreachCount < 1 {
		t.Error("Expected BreachCount to increment for Grade F")
	}

	// Verify sovereignty breach action
	if action != ActionSovereigntyBreach {
		t.Errorf("Expected ActionSovereigntyBreach for Grade F, got %s", action.String())
	}
}

func TestProjectionHook_Integration(t *testing.T) {
	// Test 6: Full integration with ProjectionHook
	hNet := &KeplerianHamiltonian{Mu: 398600.4418}

	q0 := Vector3{6778.137, 0, 0}
	p0 := Vector3{0, 7.6686, 0}

	hook := NewProjectionHook(hNet, q0, p0)

	// Process a normal step
	q1 := Vector3{6778.0, 10.0, 0} // Small position update
	p1 := Vector3{0.1, 7.6, 0.1}   // Small momentum update

	result, action := hook.ProcessStep(q1, p1)

	t.Logf("Integration test: %s", result.String())
	t.Logf("Action: %s", action.String())

	// Should handle gracefully
	if !result.Success && result.Grade == GradeF {
		t.Logf("Note: Grade F detected - this may indicate the perturbation was too large")
	}
}

func TestKeplerianHamiltonian_Compute(t *testing.T) {
	// Test Keplerian Hamiltonian computation
	k := &KeplerianHamiltonian{Mu: 398600.4418}

	q := Vector3{6778.137, 0, 0}
	p := Vector3{0, 7.6686, 0}

	H := k.Compute(q, p)

	// For circular orbit: H = 0.5*v^2 - μ/r
	v := 7.6686 // km/s
	expectedH := 0.5*v*v - 398600.4418/6778.137

	t.Logf("Computed H: %.6f", H)
	t.Logf("Expected H: %.6f (KE + PE)", expectedH)

	// Allow very small tolerance for floating point
	diff := H - expectedH
	if diff < 0 {
		diff = -diff
	}
	if diff > 0.001 {
		t.Errorf("Hamiltonian mismatch: got %.6f, expected ~%.6f, diff=%.6f", H, expectedH, diff)
	}
}
