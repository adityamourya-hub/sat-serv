package orbital

/**
 * Circular Restricted Three-Body Problem (CR3BP) for Cislunar SDA
 *
 * ADITI 4.0: Tracks objects in Earth-Moon corridor using
 * Jacobi constant and rotating frame mechanics.
 *
 * Applications:
 * - Lunar transfer trajectory analysis
 * - Libration point (L1-L5) surveillance
 * - Stable/unstable manifold tracking for "dark" objects
 *
 * Mathematical Framework:
 * - Rotating frame: x-axis along Earth-Moon line
 * - Jacobi constant C = 2U - v² (invariant in rotating frame)
 * - Zero-velocity surfaces define forbidden regions
 *
 * References:
 * - DhruvJ22/Astrodynamics_Research: CR3BP implementation
 * - Koon et al.: Dynamical Systems Approach to Space Mission Design
 * - NASA JPL: Libration Point Missions
 */

import (
	"math"
)

// CR3BPSystem defines Earth-Moon system parameters
type CR3BPSystem struct {
	Mu       float64 // Mass ratio: m2/(m1+m2) = 0.01215 for Earth-Moon
	L        float64 // Distance between primaries (km) ~384,400
	Omega    float64 // Angular velocity of rotating frame (rad/s)
	EarthPos float64 // x-position of Earth in rotating frame (-mu)
	MoonPos  float64 // x-position of Moon in rotating frame (1-mu)
}

// EarthMoonSystem returns standard Earth-Moon CR3BP
func EarthMoonSystem() *CR3BPSystem {
	mu := 0.012150585609624  // Moon mass / (Earth + Moon mass)
	L := 384400.0            // km
	omega := 2.661699e-6     // rad/s (mean motion)
	
	return &CR3BPSystem{
		Mu:       mu,
		L:        L,
		Omega:    omega,
		EarthPos: -mu,
		MoonPos:  1.0 - mu,
	}
}

// CR3BPState represents state in rotating frame
type CR3BPState struct {
	Pos [3]float64 // Normalized position (distance in L units)
	Vel [3]float64 // Normalized velocity (velocity in L*Omega units)
}

// ToRotatingFrame converts inertial ECI state to rotating frame
func (sys *CR3BPSystem) ToRotatingFrame(
	eciPos, eciVel [3]float64,
	theta float64, // Angle of rotating frame (0 = x-axis aligned with Earth-Moon)
) CR3BPState {
	// Rotation matrix from inertial to rotating
	cosT := math.Cos(theta)
	sinT := math.Sin(theta)
	
	// Position rotation
	x := (eciPos[0]*cosT + eciPos[1]*sinT) / sys.L
	y := (-eciPos[0]*sinT + eciPos[1]*cosT) / sys.L
	z := eciPos[2] / sys.L
	
	// Velocity includes Coriolis term
	vx := (eciVel[0]*cosT + eciVel[1]*sinT) / (sys.L * sys.Omega)
	vy := (-eciVel[0]*sinT + eciVel[1]*cosT) / (sys.L * sys.Omega)
	vz := eciVel[2] / (sys.L * sys.Omega)
	
	// Add -omega × r term for rotating frame velocity
	vx -= -y  // -Omega × r_x = -(-y) = y (normalized)
	vy -= x   // -Omega × r_y = -x
	
	return CR3BPState{
		Pos: [3]float64{x, y, z},
		Vel: [3]float64{vx, vy, vz},
	}
}

// JacobiConstant computes the Jacobi constant C
// C = 2U - v² (invariant in rotating frame)
// Higher C = lower energy (trapped in potential well)
// Lower C = higher energy (can escape)
func (sys *CR3BPSystem) JacobiConstant(state CR3BPState) float64 {
	x, y, z := state.Pos[0], state.Pos[1], state.Pos[2]
	vx, vy, vz := state.Vel[0], state.Vel[1], state.Vel[2]
	
	// Distance to Earth (m1)
	r1 := math.Sqrt((x-sys.EarthPos)*(x-sys.EarthPos) + y*y + z*z)
	
	// Distance to Moon (m2)
	r2 := math.Sqrt((x-sys.MoonPos)*(x-sys.MoonPos) + y*y + z*z)
	
	// Pseudo-potential U
	U := 0.5*(x*x+y*y) + (1-sys.Mu)/r1 + sys.Mu/r2
	
	// Velocity squared
	v2 := vx*vx + vy*vy + vz*vz
	
	// Jacobi constant
	C := 2*U - v2
	
	return C
}

// ZeroVelocitySurface computes velocity required to reach point
// from given Jacobi constant. Returns 0 if point is forbidden.
func (sys *CR3BPSystem) ZeroVelocitySurface(C float64, pos [3]float64) float64 {
	x, y, z := pos[0], pos[1], pos[2]
	
	// Distance to Earth
	r1 := math.Sqrt((x-sys.EarthPos)*(x-sys.EarthPos) + y*y + z*z)
	
	// Distance to Moon
	r2 := math.Sqrt((x-sys.MoonPos)*(x-sys.MoonPos) + y*y + z*z)
	
	// Pseudo-potential at this point
	U := 0.5*(x*x+y*y) + (1-sys.Mu)/r1 + sys.Mu/r2
	
	// Maximum velocity from Jacobi: C = 2U - v² → v = sqrt(2U - C)
	v2max := 2*U - C
	
	if v2max < 0 {
		return 0 // Forbidden region
	}
	
	return math.Sqrt(v2max)
}

// IsForbiddenRegion checks if a point is in zero-velocity surface
func (sys *CR3BPSystem) IsForbiddenRegion(C float64, pos [3]float64) bool {
	return sys.ZeroVelocitySurface(C, pos) == 0
}

// LibrationPoint represents Earth-Moon Lagrange points L1-L5
type LibrationPoint struct {
	Name     string
	Position [3]float64 // Normalized coordinates
	Type     string     // "Collinear" (L1-L3) or "Triangular" (L4-L5)
}

// GetLibrationPoints returns all 5 Lagrange points
func (sys *CR3BPSystem) GetLibrationPoints() []LibrationPoint {
	mu := sys.Mu
	
	// L1: Between Earth and Moon (approximate)
	// Solve: (1-mu)/(x+mu)² + mu/(x-1+mu)² = (1-mu)(x+mu) + mu(x-1+mu)
	xL1 := 0.8369 // Approximate for Earth-Moon
	
	// L2: Beyond Moon
	xL2 := 1.1557
	
	// L3: Opposite side of Earth
	xL3 := -1.0051
	
	// L4: Leading triangle point
	xL4 := 0.5 - mu
	yL4 := math.Sqrt(3.0) / 2.0
	
	// L5: Trailing triangle point
	xL5 := 0.5 - mu
	yL5 := -math.Sqrt(3.0) / 2.0
	
	return []LibrationPoint{
		{"L1", [3]float64{xL1, 0, 0}, "Collinear"},
		{"L2", [3]float64{xL2, 0, 0}, "Collinear"},
		{"L3", [3]float64{xL3, 0, 0}, "Collinear"},
		{"L4", [3]float64{xL4, yL4, 0}, "Triangular"},
		{"L5", [3]float64{xL5, yL5, 0}, "Triangular"},
	}
}

// Manifold represents stable/unstable manifold from libration point
type Manifold struct {
	Type       string      // "Stable" or "Unstable"
	Point      string      // "L1", "L2", etc.
	Trajectory [][3]float64 // Normalized positions along manifold
	JacobianC  float64     // Jacobi constant for this manifold
}

// ComputeStableManifold approximates stable manifold from L1/L2
// Used to detect "dark" objects approaching GEO from lunar vicinity
func (sys *CR3BPSystem) ComputeStableManifold(
	point string,
	numPoints int,
	dt float64, // Time step
) Manifold {
	// Get libration point
	points := sys.GetLibrationPoints()
	var origin [3]float64
	for _, p := range points {
		if p.Name == point {
			origin = p.Position
			break
		}
	}
	
	// Approximate stable manifold by integrating backward from
	// small perturbation in stable direction
	// (Simplified - full implementation needs eigenvector analysis)
	
	trajectory := make([][3]float64, numPoints)
	trajectory[0] = origin
	
	// Small displacement toward Earth (stable direction for L1/L2)
	direction := [3]float64{-1, 0, 0} // Toward Earth
	if point == "L2" {
		direction = [3]float64{1, 0, 0} // Away from Earth
	}
	
	// Simple integration (simplified)
	scale := 0.001 // Small perturbation
	for i := 1; i < numPoints; i++ {
		trajectory[i] = [3]float64{
			trajectory[i-1][0] + direction[0]*scale,
			trajectory[i-1][1] + direction[1]*scale,
			trajectory[i-1][2] + direction[2]*scale,
		}
		scale *= 1.1 // Expand gradually
	}
	
	return Manifold{
		Type:       "Stable",
		Point:      point,
		Trajectory: trajectory,
		JacobianC:  sys.JacobiConstant(CR3BPState{Pos: origin, Vel: [3]float64{0, 0, 0}}),
	}
}

// CislunarTracker monitors objects in Earth-Moon corridor
type CislunarTracker struct {
	System *CR3BPSystem
	Objects map[string]CislunarObject
}

// CislunarObject represents tracked object in cislunar space
type CislunarObject struct {
	ID        string
	State     CR3BPState
	JacobiC   float64
	Region    string // "LEO", "GEO", "L1", "L2", "Lunar", "Escape"
}

// NewCislunarTracker creates Earth-Moon system tracker
func NewCislunarTracker() *CislunarTracker {
	return &CislunarTracker{
		System:  EarthMoonSystem(),
		Objects: make(map[string]CislunarObject),
	}
}

// AddObject tracks new cislunar object
func (ct *CislunarTracker) AddObject(id string, eciPos, eciVel [3]float64, theta float64) {
	// Convert to rotating frame
	state := ct.System.ToRotatingFrame(eciPos, eciVel, theta)
	
	// Compute Jacobi constant
	C := ct.System.JacobiConstant(state)
	
	// Classify region
	region := ct.classifyRegion(state, C)
	
	ct.Objects[id] = CislunarObject{
		ID:      id,
		State:   state,
		JacobiC: C,
		Region:  region,
	}
}

// classifyRegion determines cislunar region
func (ct *CislunarTracker) classifyRegion(state CR3BPState, C float64) string {
	x := state.Pos[0]
	// y := state.Pos[1]
	// r := math.Sqrt(x*x + y*y)
	
	// Check Jacobi constant for escape capability
	if C < 3.0 {
		return "Escape" // Can escape Earth-Moon system
	}
	
	// Position-based classification
	if x > 0.9 && x < 1.1 {
		return "L1" // Near L1
	}
	if x > 1.1 && x < 1.3 {
		return "L2" // Near L2
	}
	if x > 0.8 && x < 50.0 {
		return "Lunar" // Near Moon
	}
	if x > -1.5 && x < 0.8 {
		return "GEO" // GEO region
	}
	if x < -1.5 {
		return "LEO" // Earth vicinity
	}
	
	return "Unknown"
}

// DetectManifoldApproach checks if object is on stable manifold
// (indicator of potential "dark" object sneaking into GEO)
func (ct *CislunarTracker) DetectManifoldApproach(id string, tolerance float64) bool {
	obj, exists := ct.Objects[id]
	if !exists {
		return false
	}
	
	// Compute L1 stable manifold
	manifold := ct.System.ComputeStableManifold("L1", 100, 0.01)
	
	// Check if object position is close to manifold
	for _, point := range manifold.Trajectory {
		dx := obj.State.Pos[0] - point[0]
		dy := obj.State.Pos[1] - point[1]
		dz := obj.State.Pos[2] - point[2]
		dist := math.Sqrt(dx*dx + dy*dy + dz*dz)
		
		if dist < tolerance {
			return true // Object is on or near stable manifold
		}
	}
	
	return false
}

// StrategicImplications for IAF
func (ct *CislunarTracker) StrategicImplications() []string {
	implications := []string{
		"L1/L2 points: Gateway to GEO without ground detection",
		"Stable manifolds: Natural pathways for 'dark' approaches",
		"Low-energy transfers: Fuel-efficient paths from Moon to GEO",
		"Halo orbits: Persistent surveillance platforms",
	}
	return implications
}
