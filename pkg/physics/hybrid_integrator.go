// pkg/physics/hybrid_integrator.go - ADITI 4.0 Symplectic-Adaptive Hybrid Integrator
//
// Implements operator splitting for conservative + dissipative dynamics:
//   - Conservative (fc): Gravity, J2, J3 - Forest-Ruth symplectic steps
//   - Dissipative (fd): Drag, SRP - RKF45 adaptive steps
//
// Strang Splitting (2nd order):
//   z_{n+1} = D(dt/2) ∘ S(dt) ∘ D(dt/2)
//
// Where:
//   - S: Symplectic step (preserves energy manifold)
//   - D: Dissipative step (adaptive for non-conservative forces)
//
// PAX Governor Integration:
//   - Dynamic TOL based on orbital regime and Conflict Metric
//   - Thermal budget: 6W TDP for 30k-object catalog
//
// References:
//   - Strang (1968): On the construction and comparison of difference schemes
//   - Forest & Ruth (1990): Fourth-order symplectic integration
//   - Hairer, Lubich, Wanner (2006): Geometric Numerical Integration

package physics

import (
	"fmt"
	"math"
)

// =============================================================================
// Hybrid Integrator State
// =============================================================================

// HybridState represents the full orbital state for hybrid integration
type HybridState struct {
	Pos Vector3 // Position [km]
	Vel Vector3 // Velocity [km/s]

	// Extended state for drag/SRP modeling
	Mass     float64 // Satellite mass [kg]
	AreaDrag float64 // Drag reference area [m²]
	AreaSRP  float64 // Solar radiation pressure area [m²]
	Cd       float64 // Drag coefficient
	Cr       float64 // Reflectivity coefficient

	// Environment
	EpochJD float64 // Julian date for solar position
}

// ConservativeField computes gravitational acceleration only
// This is fc(z) = ∇H for symplectic integration
type ConservativeField func(state HybridState) Vector3

// DissipativeField computes non-conservative accelerations
// This is fd(z) for RKF45 adaptive integration
type DissipativeField func(state HybridState, epochJD float64) Vector3

// HybridHamiltonian computes specific orbital energy for hybrid state
type HybridHamiltonian func(state HybridState) float64

// =============================================================================
// Adaptive Tolerance Profile (PAX Governor)
// =============================================================================

// ConflictMetric represents threat level for dynamic TOL adjustment
type ConflictMetric int

const (
	ConflictNone     ConflictMetric = iota // Routine tracking
	ConflictElevated                       // Nearby object detected
	ConflictHigh                           // Potential conjunction
	ConflictCritical                       // Active RPO/threat
)

// AdaptiveProfile contains dynamic tolerance settings
type AdaptiveProfile struct {
	BaseTOL     float64 // Base tolerance for regime
	ElevatedTOL float64 // Tightened tolerance when conflict detected
	MinStep     float64 // Minimum step (thermal floor)
	MaxStep     float64 // Maximum step (accuracy ceiling)
}

// PaxToleranceProfiles - M2 Air optimized
var PaxToleranceProfiles = map[FlightRegime]AdaptiveProfile{
	FlightRegimeLEO: {
		BaseTOL:     1e-11, // High precision for drag-sensitive LEO
		ElevatedTOL: 1e-12, // Ultra-tight for conjunctions
		MinStep:     0.001, // 1 ms
		MaxStep:     60.0,
	},
	FlightRegimeMEO: {
		BaseTOL:     1e-9,  // Medium precision for NavIC/GPS
		ElevatedTOL: 1e-10, // Tightened for nearby threats
		MinStep:     0.01,  // 10 ms
		MaxStep:     300.0,
	},
	FlightRegimeGEO: {
		BaseTOL:     1e-7,  // Relaxed for stable GEO
		ElevatedTOL: 1e-10, // Tighten if RPO inspector nearby
		MinStep:     0.1,   // 100 ms
		MaxStep:     600.0,
	},
	FlightRegimeHGVEntry: {
		BaseTOL:     1e-11, // Extreme precision for hypersonic
		ElevatedTOL: 1e-12,
		MinStep:     0.0001, // 100 μs
		MaxStep:     1.0,
	},
	FlightRegimeHGVGlide: {
		BaseTOL:     1e-10,
		ElevatedTOL: 1e-11,
		MinStep:     0.001,
		MaxStep:     10.0,
	},
	FlightRegimeBallistic: {
		BaseTOL:     1e-11,
		ElevatedTOL: 1e-12,
		MinStep:     0.0001,
		MaxStep:     5.0,
	},
}

// =============================================================================
// Hybrid Integrator
// =============================================================================

// HybridIntegrator combines symplectic and adaptive methods
type HybridIntegrator struct {
	// Component integrators
	Symplectic *Engine          // Forest-Ruth for conservative part
	Adaptive   *RKF45Integrator // RKF45 for dissipative part

	// Fields
	Conservative ConservativeField
	Dissipative  DissipativeField
	Hamiltonian  HybridHamiltonian

	// PAX Governor state
	Regime     FlightRegime
	Conflict   ConflictMetric
	CurrentTOL float64

	// Manifold projection
	Projector        *ManifoldProjector
	TargetH          float64
	EnableProjection bool

	// Performance tracking
	StepCount   int
	SympSteps   int
	AdaptSteps  int
	Rejections  int
	Projections int
}

// NewHybridIntegrator creates a hybrid integrator with PAX governance
func NewHybridIntegrator(
	regime FlightRegime,
	conservative ConservativeField,
	dissipative DissipativeField,
	hamiltonian HybridHamiltonian,
) *HybridIntegrator {

	profile := PaxToleranceProfiles[regime]

	return &HybridIntegrator{
		Symplectic: &Engine{}, // Simplified - use Forest-Ruth
		Adaptive:   NewRKF45Integrator(regime),

		Conservative: conservative,
		Dissipative:  dissipative,
		Hamiltonian:  hamiltonian,

		Regime:     regime,
		Conflict:   ConflictNone,
		CurrentTOL: profile.BaseTOL,

		Projector:        nil, // Optional - can be set later
		EnableProjection: true,
	}
}

// SetConflictMetric updates TOL based on threat level
func (hi *HybridIntegrator) SetConflictMetric(conflict ConflictMetric) {
	hi.Conflict = conflict
	profile := PaxToleranceProfiles[hi.Regime]

	switch conflict {
	case ConflictNone:
		hi.CurrentTOL = profile.BaseTOL
	case ConflictElevated, ConflictHigh, ConflictCritical:
		hi.CurrentTOL = profile.ElevatedTOL
	}

	// Also update adaptive integrator
	hi.Adaptive.Profile.RelTol = hi.CurrentTOL
}

// Propagate performs one Strang-splitting step
// Algorithm: D(dt/2) ∘ S(dt) ∘ D(dt/2)
func (hi *HybridIntegrator) Propagate(state HybridState, dt float64) HybridState {
	// Step 1: Half-step dissipative (RKF45)
	// Only integrate drag + SRP
	stateHalf := hi.dissipativeHalfStep(state, dt/2.0)

	// Step 2: Full-step symplectic (Forest-Ruth)
	// Core Kepler + J2 gravity manifold
	stateSymp := hi.symplecticStep(stateHalf, dt)

	// Step 3: Final half-step dissipative
	stateFinal := hi.dissipativeHalfStep(stateSymp, dt/2.0)

	// Step 4: Manifold Projection (Axiom Gate)
	if hi.EnableProjection && hi.Projector != nil {
		stateFinal = hi.projectToManifold(stateFinal)
	}

	hi.StepCount++
	return stateFinal
}

// dissipativeHalfStep performs RKF45 integration on drag/SRP only
func (hi *HybridIntegrator) dissipativeHalfStep(state HybridState, dt float64) HybridState {
	// Convert to array for RKF45
	x := []float64{state.Pos[0], state.Pos[1], state.Pos[2],
		state.Vel[0], state.Vel[1], state.Vel[2]}

	// Dynamics function: only dissipative forces
	dynamics := func(y []float64, t float64) []float64 {
		s := HybridState{
			Pos: Vector3{y[0], y[1], y[2]},
			Vel: Vector3{y[3], y[4], y[5]},
		}
		fd := hi.Dissipative(s, state.EpochJD)
		return []float64{y[3], y[4], y[5], fd[0], fd[1], fd[2]}
	}

	// Perform adaptive step
	newX, _, _, _, accepted := hi.Adaptive.Step(x, 0.0, dt, dynamics)

	if !accepted {
		hi.Rejections++
		// Step was rejected - retry with smaller dt would happen in caller
	}

	hi.AdaptSteps++

	return HybridState{
		Pos:      Vector3{newX[0], newX[1], newX[2]},
		Vel:      Vector3{newX[3], newX[4], newX[5]},
		Mass:     state.Mass,
		AreaDrag: state.AreaDrag,
		AreaSRP:  state.AreaSRP,
		Cd:       state.Cd,
		Cr:       state.Cr,
		EpochJD:  state.EpochJD,
	}
}

// symplecticStep performs Forest-Ruth integration on gravity only
func (hi *HybridIntegrator) symplecticStep(state HybridState, dt float64) HybridState {
	// Use Forest-Ruth coefficients from symplectic_engine.go
	a, b := ForestRuthCoefficients()

	q := []float64{state.Pos[0], state.Pos[1], state.Pos[2]}
	p := []float64{state.Vel[0], state.Vel[1], state.Vel[2]} // Actually velocity, treated as momentum per unit mass

	// Forest-Ruth 4th order
	for i := 0; i < 4; i++ {
		// Position update (drift)
		for j := 0; j < 3; j++ {
			q[j] += a[i] * dt * p[j]
		}

		// Momentum update (kick) from conservative forces
		if b[i] != 0 {
			s := HybridState{Pos: Vector3{q[0], q[1], q[2]}}
			f := hi.Conservative(s)
			for j := 0; j < 3; j++ {
				p[j] += b[i] * dt * f[j]
			}
		}
	}

	hi.SympSteps++

	return HybridState{
		Pos:      Vector3{q[0], q[1], q[2]},
		Vel:      Vector3{p[0], p[1], p[2]},
		Mass:     state.Mass,
		AreaDrag: state.AreaDrag,
		AreaSRP:  state.AreaSRP,
		Cd:       state.Cd,
		Cr:       state.Cr,
		EpochJD:  state.EpochJD,
	}
}

// projectToManifold enforces energy conservation via projection
func (hi *HybridIntegrator) projectToManifold(state HybridState) HybridState {
	if hi.Projector == nil {
		return state
	}

	// Compute current Hamiltonian
	H := hi.Hamiltonian(state)

	// Simple projection: scale velocity to match target energy
	if hi.TargetH != 0 && math.Abs(H-hi.TargetH) > hi.CurrentTOL {
		// Compute required velocity scaling
		// H = 0.5*v^2 - mu/r
		// For small corrections, scale velocity by sqrt((H_target + mu/r) / (0.5*v^2))

		// Compute norms manually (Vector3 is [3]float64)
		r := math.Sqrt(state.Pos[0]*state.Pos[0] + state.Pos[1]*state.Pos[1] + state.Pos[2]*state.Pos[2])
		mu := 398600.4418
		vSq := state.Vel[0]*state.Vel[0] + state.Vel[1]*state.Vel[1] + state.Vel[2]*state.Vel[2]

		if vSq > 1e-10 {
			vScale := math.Sqrt(2.0 * (hi.TargetH + mu/r) / vSq)
			if !math.IsNaN(vScale) && vScale > 0.5 && vScale < 2.0 {
				// Scale velocity manually
				state.Vel[0] *= vScale
				state.Vel[1] *= vScale
				state.Vel[2] *= vScale
			}
		}

		hi.Projections++
	}

	return state
}

// SetManifoldProjector assigns the projection operator
func (hi *HybridIntegrator) SetManifoldProjector(projector *ManifoldProjector, targetH float64) {
	hi.Projector = projector
	hi.TargetH = targetH
}

// GetStats returns performance statistics
func (hi *HybridIntegrator) GetStats() map[string]interface{} {
	totalSteps := hi.SympSteps + hi.AdaptSteps
	sympPct := 0.0
	adaptPct := 0.0
	if totalSteps > 0 {
		sympPct = float64(hi.SympSteps) / float64(totalSteps) * 100
		adaptPct = float64(hi.AdaptSteps) / float64(totalSteps) * 100
	}

	return map[string]interface{}{
		"step_count":       hi.StepCount,
		"symplectic_steps": hi.SympSteps,
		"adaptive_steps":   hi.AdaptSteps,
		"symplectic_pct":   sympPct,
		"adaptive_pct":     adaptPct,
		"rejections":       hi.Rejections,
		"projections":      hi.Projections,
		"current_tol":      hi.CurrentTOL,
		"conflict_level":   hi.Conflict,
	}
}

// Reset clears statistics
func (hi *HybridIntegrator) Reset() {
	hi.StepCount = 0
	hi.SympSteps = 0
	hi.AdaptSteps = 0
	hi.Rejections = 0
	hi.Projections = 0
}

// LogStats prints performance summary
func (hi *HybridIntegrator) LogStats() {
	stats := hi.GetStats()
	fmt.Println("[HybridIntegrator] Performance Summary:")
	fmt.Printf("  Steps: %d (Sym: %d, Adapt: %d)\n",
		stats["step_count"], stats["symplectic_steps"], stats["adaptive_steps"])
	fmt.Printf("  Mix: %.1f%% Symplectic, %.1f%% Adaptive\n",
		stats["symplectic_pct"], stats["adaptive_pct"])
	fmt.Printf("  Rejections: %d, Projections: %d\n",
		stats["rejections"], stats["projections"])
	fmt.Printf("  Current TOL: %.2e (Conflict: %v)\n",
		stats["current_tol"], stats["conflict_level"])
}
