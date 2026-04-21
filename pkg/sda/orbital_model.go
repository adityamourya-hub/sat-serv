package sda

import (
	"math"

	"ssis/pkg/physics"
)

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * Orbital PHS Model for Symplectic Integration
 *
 * Implements the Port-Hamiltonian System interface for orbital dynamics,
 * enabling energy-conserving propagation via the Forest-Ruth integrator.
 */

// OrbitalPHSModel implements the PHSModel interface for Keplerian orbital dynamics
// with drag and J2 perturbations.
type OrbitalPHSModel struct {
	Mu   float64 // Gravitational parameter (km³/s²)
	J2   float64 // J2 perturbation coefficient
	Re   float64 // Earth radius (km)
	Drag bool    // Include drag
	SRP  bool    // Include solar radiation pressure
}

// NewOrbitalPHSModel creates a new orbital PHS model with Earth parameters
func NewOrbitalPHSModel() *OrbitalPHSModel {
	return &OrbitalPHSModel{
		Mu:   398600.4418, // Earth's gravitational parameter (km³/s²)
		J2:   1.08263e-3,  // Earth's J2 coefficient
		Re:   6378.137,    // Earth's equatorial radius (km)
		Drag: false,       // Drag disabled for Tier 1 reflex
		SRP:  false,       // SRP disabled for Tier 1 reflex
	}
}

// Velocity returns ∂H/∂p (velocity from momentum)
// For orbital mechanics: v = p/m (assuming unit mass)
func (m *OrbitalPHSModel) Velocity(q, p, z []float64) []float64 {
	// q = position [x, y, z], p = momentum [px, py, pz]
	// For unit mass, velocity = momentum
	v := make([]float64, len(p))
	copy(v, p)
	return v
}

// Force returns -∂H/∂q (negative force from position gradient)
// Implements central gravity + J2 perturbation
func (m *OrbitalPHSModel) Force(q, p, z []float64) []float64 {
	x, y, zCoord := q[0], q[1], q[2]
	r := math.Sqrt(x*x + y*y + zCoord*zCoord)

	if r < 1e-6 {
		return make([]float64, 3)
	}

	// Central gravity: F = -μ/r³ * r_vec
	fGravity := -m.Mu / (r * r * r)
	fx := fGravity * x
	fy := fGravity * y
	fz := fGravity * zCoord

	// J2 perturbation (zonal harmonic)
	// J2 acceleration in ECI frame
	if m.J2 != 0 {
		z2 := zCoord * zCoord
		r2 := r * r
		factor := 1.5 * m.J2 * m.Mu / (r2 * r2 * r) * (m.Re * m.Re)

		// J2 corrections
		fx += factor * x * (5*z2/r2 - 1)
		fy += factor * y * (5*z2/r2 - 1)
		fz += factor * zCoord * (5*z2/r2 - 3)
	}

	return []float64{fx, fy, fz}
}

// Hamiltonian returns the total energy H(q, p)
// H = T + V = ½|v|² - μ/r
func (m *OrbitalPHSModel) Hamiltonian(q, p, z []float64) float64 {
	// Kinetic energy: T = ½|v|² (assuming unit mass)
	vx, vy, vz := p[0], p[1], p[2]
	kinetic := 0.5 * (vx*vx + vy*vy + vz*vz)

	// Potential energy: V = -μ/r
	x, y, zCoord := q[0], q[1], q[2]
	r := math.Sqrt(x*x + y*y + zCoord*zCoord)
	potential := -m.Mu / r

	return kinetic + potential
}

// SymplecticPropagator wraps the Forest-Ruth integrator for orbital propagation
type SymplecticPropagator struct {
	Model    *OrbitalPHSModel
	Manifold physics.ManifoldType
	Dt       float64    // Time step (seconds)
	NavIC    *NavICSync // Sovereign time synchronization
}

// NewSymplecticPropagator creates a new symplectic propagator
func NewSymplecticPropagator() *SymplecticPropagator {
	return &SymplecticPropagator{
		Model:    NewOrbitalPHSModel(),
		Manifold: physics.ManifoldCartesian, // Use Cartesian for LEO
		Dt:       10.0,                      // 10-second time step
		NavIC:    NewNavICSync(),
	}
}

// Propagate propagates a state vector from t0 to t1 using symplectic integration
// Returns [x, y, z, vx, vy, vz]
func (sp *SymplecticPropagator) Propagate(state [6]float64, t0, t1 float64) [6]float64 {
	q := []float64{state[0], state[1], state[2]} // Position
	p := []float64{state[3], state[4], state[5]} // Momentum (velocity for unit mass)

	duration := t1 - t0
	nSteps := int(duration / sp.Dt)
	if nSteps < 1 {
		nSteps = 1
	}

	// Manifold gate for ortho-normalization
	gate := physics.NewManifoldGate()

	// Symplectic integration with NavIC PPS gating
	for i := 0; i < nSteps; i++ {
		// Use NavIC PPS to gate integration step (sovereign time sync)
		if sp.NavIC != nil && sp.NavIC.ValidateSovereignLock() {
			// Synchronize with NavIC PPS edge
			sp.NavIC.OnPPS()
		}

		q, p = physics.ForestRuthStep(sp.Model, q, p, nil, sp.Dt, sp.Manifold)
		gate.OrthoNormalize(q, p, sp.Manifold)
	}

	// Handle remaining time
	remaining := duration - float64(nSteps)*sp.Dt
	if remaining > 0 {
		q, p = physics.ForestRuthStep(sp.Model, q, p, nil, remaining, sp.Manifold)
	}

	return [6]float64{q[0], q[1], q[2], p[0], p[1], p[2]}
}

// VerifyEnergyConservation checks if the propagator preserves energy over a trajectory
func (sp *SymplecticPropagator) VerifyEnergyConservation(initialState [6]float64, duration float64) (maxDrift, finalDrift float64) {
	q := []float64{initialState[0], initialState[1], initialState[2]}
	p := []float64{initialState[3], initialState[4], initialState[5]}

	H0 := sp.Model.Hamiltonian(q, p, nil)
	maxDrift = 0.0

	nSteps := int(duration / sp.Dt)
	gate := physics.NewManifoldGate()

	for i := 0; i < nSteps; i++ {
		q, p = physics.ForestRuthStep(sp.Model, q, p, nil, sp.Dt, sp.Manifold)
		gate.OrthoNormalize(q, p, sp.Manifold)

		H := sp.Model.Hamiltonian(q, p, nil)
		drift := math.Abs(H-H0) / math.Abs(H0)
		if drift > maxDrift {
			maxDrift = drift
		}
	}

	HFinal := sp.Model.Hamiltonian(q, p, nil)
	finalDrift = math.Abs(HFinal-H0) / math.Abs(H0)

	return maxDrift, finalDrift
}
