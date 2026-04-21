package sda

import (
	"math"
)

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * Keplerian PHS Model for Symplectic Integration
 *
 * Implements the physics.PHSModel interface for pure Keplerian orbital dynamics.
 * Used by the HamiltonianValidator to propagate states and verify energy conservation.
 *
 * Physics:
 *   H = T + V = ½|v|² - μ/r
 *   ∂H/∂p = v = p/m (velocity, assuming unit mass)
 *   -∂H/∂q = F = -μ/r³ · r (gravitational force)
 */

// KeplerianPHSModel implements physics.PHSModel for Keplerian orbital dynamics
type KeplerianPHSModel struct {
	Mu float64 // Gravitational parameter (km³/s²)
}

// NewKeplerianPHSModel creates a new Keplerian PHS model
// Default μ = 398600.4418 km³/s² for Earth
func NewKeplerianPHSModel() *KeplerianPHSModel {
	return &KeplerianPHSModel{
		Mu: 398600.4418, // Earth standard gravitational parameter
	}
}

// NewKeplerianPHSModelWithMu creates a model with custom gravitational parameter
func NewKeplerianPHSModelWithMu(mu float64) *KeplerianPHSModel {
	return &KeplerianPHSModel{
		Mu: mu,
	}
}

// Velocity returns ∂H/∂p (velocity from momentum)
// For Keplerian with unit mass: v = p
func (m *KeplerianPHSModel) Velocity(q, p, z []float64) []float64 {
	// With unit mass, momentum equals velocity
	// Return copy to avoid modifying input
	v := make([]float64, len(p))
	copy(v, p)
	return v
}

// Force returns -∂H/∂q (negative force from position gradient)
// For Keplerian: F = -μ/r³ · r
func (m *KeplerianPHSModel) Force(q, p, z []float64) []float64 {
	// Compute distance from center
	r2 := q[0]*q[0] + q[1]*q[1] + q[2]*q[2]
	r := math.Sqrt(r2)
	
	// Avoid singularity at origin
	if r < 1e-10 {
		return []float64{0, 0, 0}
	}
	
	// Gravitational force: F = -μ/r³ · r = -μ/r² · (r/|r|)
	forceMag := -m.Mu / (r * r * r)
	
	return []float64{
		forceMag * q[0],
		forceMag * q[1],
		forceMag * q[2],
	}
}

// Hamiltonian returns the total specific orbital energy H(q, p)
// For Keplerian: H = ½|v|² - μ/r (specific mechanical energy)
func (m *KeplerianPHSModel) Hamiltonian(q, p, z []float64) float64 {
	// Distance from center
	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	if r < 1e-10 {
		return 0
	}
	
	// Kinetic energy per unit mass: T = ½|v|² = ½|p|² (unit mass)
	kinetic := 0.5 * (p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
	
	// Potential energy per unit mass: V = -μ/r
	potential := -m.Mu / r
	
	// Total specific energy
	return kinetic + potential
}

// ComputeSpecificEnergy computes specific orbital energy from state vectors
// Helper function for direct energy calculations without full PHS model
func ComputeSpecificEnergy(r, v [3]float64, mu float64) float64 {
	rMag := math.Sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])
	vMagSq := v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
	
	if rMag < 1e-10 {
		return 0
	}
	
	return 0.5*vMagSq - mu/rMag
}

// ComputeEnergyDrift computes relative energy change between two states
// Returns |ΔE/E| for energy conservation verification
func ComputeEnergyDrift(r1, v1, r2, v2 [3]float64, mu float64) float64 {
	E1 := ComputeSpecificEnergy(r1, v1, mu)
	E2 := ComputeSpecificEnergy(r2, v2, mu)
	
	if math.Abs(E1) < 1e-15 {
		return math.Abs(E2)
	}
	
	return math.Abs((E2 - E1) / E1)
}
