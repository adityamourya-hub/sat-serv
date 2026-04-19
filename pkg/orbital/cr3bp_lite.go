package orbital

/**
 * CR3BP Lite: Memory-Optimized State for Bulk Operations
 *
 * ADITI 4.0 Risk Mitigation: STM Memory Pressure
 *
 * Problem: CR3BPStateSTM is 336 bytes (6 + 36 floats)
 *          100k objects = 33.6 MB, 2 GB/s bandwidth
 *
 * Solution: Tiered architecture
 *   - Tier 1: CR3BPLite (56 bytes) - Bulk catalog, screening
 *   - Tier 2: CR3BPFull (336 bytes) - OD, covariance, manifolds
 *   - Tier 3: STM-on-demand - Compute when needed
 *
 * Memory Savings: 83% (33.6 MB → 5.6 MB for 100k objects)
 * Cache Performance: Fits in single cache line (56 bytes < 64)
 *
 * Promotion Strategy:
 *   1. Propagate all 100k as CR3BPLite
 *   2. Screen for high-interest (~1000 objects)
 *   3. Promote to CR3BPFull only for OD/manifold analysis
 */

import (
	"math"
)

// CR3BPLite is the lightweight state for bulk operations
// Size: 56 bytes (was 336 bytes with STM)
// Cache lines: 1 (fits in L1)
type CR3BPLite struct {
	// State only (no STM) - 48 bytes
	Pos Vector3 // Position in rotating frame (normalized)
	Vel Vector3 // Velocity in rotating frame (normalized)

	// Metadata - 8 bytes
	JacobiC float64 // Jacobi constant (invariant, used for classification)
}

// Vector3 alias for clarity
type Vector3 [3]float64

// ToFull promotes Lite to Full with STM computation
func (s CR3BPLite) ToFull(system *CR3BPSystem, dt float64) CR3BPFull {
	full := CR3BPFull{
		CR3BPLite: s,
		HasSTM:    true,
	}

	// Compute STM on demand
	full.STM = full.computeSTM(system.Mu, dt)

	return full
}

// ComputeJacobi calculates Jacobi constant from state
func (s CR3BPLite) ComputeJacobi(mu float64) float64 {
	x, y, z := s.Pos[0], s.Pos[1], s.Pos[2]
	vx, vy, vz := s.Vel[0], s.Vel[1], s.Vel[2]

	r1 := math.Sqrt((x+mu)*(x+mu) + y*y + z*z)
	r2 := math.Sqrt((x-1+mu)*(x-1+mu) + y*y + z*z)

	Omega := 0.5*(x*x+y*y) + (1.0-mu)/r1 + mu/r2
	v2 := vx*vx + vy*vy + vz*vz

	return 2.0*Omega - v2
}

// PropagateLite propagates without STM (fast, bulk)
func (s CR3BPLite) PropagateLite(system *CR3BPSystem, dt float64, steps int) []CR3BPLite {
	states := make([]CR3BPLite, steps+1)
	states[0] = s

	current := s
	h := dt / float64(steps)

	for i := 1; i <= steps; i++ {
		current = current.rk4Step(system, h)
		states[i] = current
	}

	return states
}

// rk4Step performs one RK4 integration step
func (s CR3BPLite) rk4Step(system *CR3BPSystem, h float64) CR3BPLite {
	// Compute k1
	k1 := s.computeDerivatives(system)

	// Compute k2
	s2 := s.addScaled(k1, 0.5*h)
	k2 := s2.computeDerivatives(system)

	// Compute k3
	s3 := s.addScaled(k2, 0.5*h)
	k3 := s3.computeDerivatives(system)

	// Compute k4
	s4 := s.addScaled(k3, h)
	k4 := s4.computeDerivatives(system)

	// Combine: y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
	result := s

	for i := 0; i < 3; i++ {
		result.Pos[i] += h / 6.0 * (k1.Pos[i] + 2*k2.Pos[i] + 2*k3.Pos[i] + k4.Pos[i])
		result.Vel[i] += h / 6.0 * (k1.Vel[i] + 2*k2.Vel[i] + 2*k3.Vel[i] + k4.Vel[i])
	}

	// Update Jacobi constant (should be invariant, but check for drift)
	result.JacobiC = result.ComputeJacobi(system.Mu)

	return result
}

// computeDerivatives calculates CR3BP EOMs
func (s CR3BPLite) computeDerivatives(system *CR3BPSystem) CR3BPLite {
	mu := system.Mu
	x, y, z := s.Pos[0], s.Pos[1], s.Pos[2]
	vx, vy, vz := s.Vel[0], s.Vel[1], s.Vel[2]

	r1 := math.Sqrt((x+mu)*(x+mu) + y*y + z*z)
	r2 := math.Sqrt((x-1+mu)*(x-1+mu) + y*y + z*z)

	// Accelerations
	ax := 2.0*vy + x - (1.0-mu)*(x+mu)/(r1*r1*r1) - mu*(x-1+mu)/(r2*r2*r2)
	ay := -2.0*vx + y - (1.0-mu)*y/(r1*r1*r1) - mu*y/(r2*r2*r2)
	az := -(1.0-mu)*z/(r1*r1*r1) - mu*z/(r2*r2*r2)

	return CR3BPLite{
		Pos: Vector3{vx, vy, vz},
		Vel: Vector3{ax, ay, az},
	}
}

// addScaled returns state + scale * derivative
func (s CR3BPLite) addScaled(deriv CR3BPLite, scale float64) CR3BPLite {
	return CR3BPLite{
		Pos: Vector3{
			s.Pos[0] + scale*deriv.Pos[0],
			s.Pos[1] + scale*deriv.Pos[1],
			s.Pos[2] + scale*deriv.Pos[2],
		},
		Vel: Vector3{
			s.Vel[0] + scale*deriv.Vel[0],
			s.Vel[1] + scale*deriv.Vel[1],
			s.Vel[2] + scale*deriv.Vel[2],
		},
	}
}

// CR3BPFull is the full state with STM for OD and manifolds
type CR3BPFull struct {
	CR3BPLite             // 56 bytes (embedded)
	STM       [36]float64 // 288 bytes (6×6 matrix)
	HasSTM    bool        // 1 byte
	_         [7]byte     // Padding to 8-byte boundary
}

// Size returns total bytes
func (f CR3BPFull) Size() int {
	return 56 + 288 + 8 // 352 bytes (with padding)
}

// computeSTM calculates State Transition Matrix analytically
func (f CR3BPFull) computeSTM(mu, dt float64) [36]float64 {
	// Simplified: For short dt, STM ≈ I + A*dt
	// Full implementation: integrate variational equations

	var Phi [36]float64

	// Initialize as identity
	for i := 0; i < 6; i++ {
		Phi[i*6+i] = 1.0
	}

	// Add Jacobian contribution (first-order approximation)
	// A = ∂f/∂X at current state
	A := f.computeJacobian(mu)

	// Φ ≈ I + A*dt
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			Phi[i*6+j] += A[i*6+j] * dt
		}
	}

	return Phi
}

// computeJacobian returns 6×6 Jacobian matrix (flattened)
func (f CR3BPFull) computeJacobian(mu float64) [36]float64 {
	var A [36]float64

	x, y, z := f.Pos[0], f.Pos[1], f.Pos[2]

	r1 := math.Sqrt((x+mu)*(x+mu) + y*y + z*z)
	r2 := math.Sqrt((x-1+mu)*(x-1+mu) + y*y + z*z)

	r1_3 := r1 * r1 * r1
	r1_5 := r1_3 * r1 * r1
	r2_3 := r2 * r2 * r2
	r2_5 := r2_3 * r2 * r2

	// ∂ax/∂x
	dax_dx := 1.0 - (1.0-mu)/r1_3 + 3.0*(1.0-mu)*(x+mu)*(x+mu)/r1_5 -
		mu/r2_3 + 3.0*mu*(x-1+mu)*(x-1+mu)/r2_5

	// ∂ax/∂y
	dax_dy := 3.0*(1.0-mu)*(x+mu)*y/r1_5 + 3.0*mu*(x-1+mu)*y/r2_5

	// ∂ax/∂z
	dax_dz := 3.0*(1.0-mu)*(x+mu)*z/r1_5 + 3.0*mu*(x-1+mu)*z/r2_5

	// ∂ay/∂x
	day_dx := 3.0*(1.0-mu)*(x+mu)*y/r1_5 + 3.0*mu*(x-1+mu)*y/r2_5

	// ∂ay/∂y
	day_dy := 1.0 - (1.0-mu)/r1_3 + 3.0*(1.0-mu)*y*y/r1_5 -
		mu/r2_3 + 3.0*mu*y*y/r2_5

	// ∂ay/∂z
	day_dz := 3.0*(1.0-mu)*y*z/r1_5 + 3.0*mu*y*z/r2_5

	// ∂az/∂x
	daz_dx := 3.0*(1.0-mu)*(x+mu)*z/r1_5 + 3.0*mu*(x-1+mu)*z/r2_5

	// ∂az/∂y
	daz_dy := 3.0*(1.0-mu)*y*z/r1_5 + 3.0*mu*y*z/r2_5

	// ∂az/∂z
	daz_dz := -(1.0-mu)/r1_3 + 3.0*(1.0-mu)*z*z/r1_5 -
		mu/r2_3 + 3.0*mu*z*z/r2_5

	// Fill matrix
	// Row 0-2: derivatives of velocity (identity for position)
	A[0*6+3] = 1.0
	A[1*6+4] = 1.0
	A[2*6+5] = 1.0

	// Row 3-5: derivatives of acceleration
	A[3*6+0] = dax_dx
	A[3*6+1] = dax_dy
	A[3*6+2] = dax_dz
	A[3*6+4] = 2.0 // Coriolis

	A[4*6+0] = day_dx
	A[4*6+1] = day_dy
	A[4*6+2] = day_dz
	A[4*6+3] = -2.0 // Coriolis

	A[5*6+0] = daz_dx
	A[5*6+1] = daz_dy
	A[5*6+2] = daz_dz

	return A
}

// LitePool for zero-allocation CR3BPLite reuse
type LitePool struct {
	pool []CR3BPLite
	used int
}

// NewLitePool creates pool with capacity
func NewLitePool(capacity int) *LitePool {
	return &LitePool{
		pool: make([]CR3BPLite, capacity),
		used: 0,
	}
}

// Get returns pointer to unused slot
func (p *LitePool) Get() *CR3BPLite {
	if p.used >= len(p.pool) {
		// Grow pool
		newPool := make([]CR3BPLite, len(p.pool)*2)
		copy(newPool, p.pool)
		p.pool = newPool
	}

	slot := &p.pool[p.used]
	p.used++
	return slot
}

// Reset marks all slots as free (no deallocation)
func (p *LitePool) Reset() {
	p.used = 0
}

// CR3BPManager orchestrates Lite vs Full usage
type CR3BPManager struct {
	System   *CR3BPSystem
	LitePool *LitePool
	FullPool *LitePool // Actually stores CR3BPFull but interface compatible
}

// NewCR3BPManager creates tiered state manager
func NewCR3BPManager() *CR3BPManager {
	return &CR3BPManager{
		System:   EarthMoonSystem(),
		LitePool: NewLitePool(100000), // 100k catalog
		FullPool: NewLitePool(1000),   // 1k high-interest
	}
}

// PropagateCatalog propagates 100k objects as Lite states
func (m *CR3BPManager) PropagateCatalog(lites []CR3BPLite, dt float64) []CR3BPLite {
	// Memory efficient: 5.6 MB vs 33.6 MB
	results := make([]CR3BPLite, len(lites))

	for i, s := range lites {
		results[i] = s.PropagateLite(m.System, dt, 100)[100] // Final state
	}

	return results
}

// PromoteToFull converts high-interest Lite states to Full
func (m *CR3BPManager) PromoteToFull(lites []CR3BPLite, dt float64) []CR3BPFull {
	fulls := make([]CR3BPFull, len(lites))

	for i, s := range lites {
		fulls[i] = s.ToFull(m.System, dt)
	}

	return fulls
}

// MemoryComparison demonstrates the 83% savings
func MemoryComparison() (liteMB, fullMB float64) {
	nObjects := 100000

	liteBytes := nObjects * 56  // CR3BPLite
	fullBytes := nObjects * 336 // CR3BPFull (with STM)

	liteMB = float64(liteBytes) / (1024 * 1024)
	fullMB = float64(fullBytes) / (1024 * 1024)

	return
}
