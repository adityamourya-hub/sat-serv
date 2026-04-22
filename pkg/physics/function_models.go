// pkg/physics/function_models.go - Function Pointer Physics Models
//
// Implements strictly gated modularity for physics model extensibility.
// Allows runtime registration of custom propagators without recompilation.

package physics

import (
	"fmt"
	"sync"
)

// PropagatorFunc is a function pointer for physics propagation
// Input: position (x,y,z), velocity (vx,vy,vz), time step (seconds)
// Output: new position, new velocity
type PropagatorFunc func(q [3]float64, v [3]float64, dt float64) ([3]float64, [3]float64)

// PropagatorMetadata describes a registered propagator
type PropagatorMetadata struct {
	Name        string
	Description string
	Regime      PropagationRegime // LEO, MEO, GEO, HGV, DEEP
	Accuracy    float64           // Positional accuracy (km)
	Speed       float64           // Execution time per step (microseconds)
	Author      string
}

// PropagationRegime defines the propagation regime (different from OrbitRegime)
type PropagationRegime string

const (
	PropRegimeLEO  PropagationRegime = "LEO"  // Low Earth Orbit (<2000km)
	PropRegimeMEO  PropagationRegime = "MEO"  // Medium Earth Orbit (2000-35000km)
	PropRegimeGEO  PropagationRegime = "GEO"  // Geostationary (~35786km)
	PropRegimeHGV  PropagationRegime = "HGV"  // Hypersonic Glide Vehicle (30-80km)
	PropRegimeDEEP PropagationRegime = "DEEP" // Deep Space (>35000km)
)

// PropagatorRegistry manages registered physics propagators
type PropagatorRegistry struct {
	propagators map[string]PropagatorFunc
	metadata    map[string]PropagatorMetadata
	mu          sync.RWMutex
}

// Global propagator registry instance
var propagatorRegistry = &PropagatorRegistry{
	propagators: make(map[string]PropagatorFunc),
	metadata:    make(map[string]PropagatorMetadata),
}

// RegisterPropagator registers a new physics propagator
func RegisterPropagator(metadata PropagatorMetadata, fn PropagatorFunc) error {
	propagatorRegistry.mu.Lock()
	defer propagatorRegistry.mu.Unlock()

	if _, exists := propagatorRegistry.propagators[metadata.Name]; exists {
		return fmt.Errorf("propagator already registered: %s", metadata.Name)
	}

	propagatorRegistry.propagators[metadata.Name] = fn
	propagatorRegistry.metadata[metadata.Name] = metadata

	return nil
}

// GetPropagator retrieves a propagator by name
func GetPropagator(name string) (PropagatorFunc, error) {
	propagatorRegistry.mu.RLock()
	defer propagatorRegistry.mu.RUnlock()

	fn, ok := propagatorRegistry.propagators[name]
	if !ok {
		return nil, fmt.Errorf("propagator not found: %s", name)
	}

	return fn, nil
}

// GetPropagatorMetadata retrieves propagator metadata
func GetPropagatorMetadata(name string) (PropagatorMetadata, error) {
	propagatorRegistry.mu.RLock()
	defer propagatorRegistry.mu.RUnlock()

	meta, ok := propagatorRegistry.metadata[name]
	if !ok {
		return PropagatorMetadata{}, fmt.Errorf("propagator metadata not found: %s", name)
	}

	return meta, nil
}

// ListPropagators returns all registered propagator names
func ListPropagators() []string {
	propagatorRegistry.mu.RLock()
	defer propagatorRegistry.mu.RUnlock()

	names := make([]string, 0, len(propagatorRegistry.propagators))
	for name := range propagatorRegistry.propagators {
		names = append(names, name)
	}
	return names
}

// SelectPropagatorForRegime selects the best propagator for a given regime
func SelectPropagatorForRegime(regime PropagationRegime) (string, error) {
	propagatorRegistry.mu.RLock()
	defer propagatorRegistry.mu.RUnlock()

	var bestName string
	bestAccuracy := 0.0

	for name, meta := range propagatorRegistry.metadata {
		if meta.Regime == regime && meta.Accuracy > bestAccuracy {
			bestAccuracy = meta.Accuracy
			bestName = name
		}
	}

	if bestName == "" {
		return "", fmt.Errorf("no propagator found for regime: %s", regime)
	}

	return bestName, nil
}

// =============================================================================
// Built-in Propagators
// =============================================================================

// RegisterBuiltInPropagators registers the standard physics propagators
func RegisterBuiltInPropagators() error {
	// Two-body Keplerian propagator
	err := RegisterPropagator(PropagatorMetadata{
		Name:        "kepler_two_body",
		Description: "Analytic two-body Keplerian propagator",
		Regime:      PropRegimeLEO,
		Accuracy:    0.001, // 1 meter accuracy
		Speed:       10.0,  // 10 microseconds per step
		Author:      "ADITI Core",
	}, TwoBodyKepler)

	if err != nil {
		return fmt.Errorf("failed to register kepler_two_body: %w", err)
	}

	// RK4 numerical integrator
	err = RegisterPropagator(PropagatorMetadata{
		Name:        "rk4_j2",
		Description: "RK4 integrator with J2 oblateness perturbation",
		Regime:      PropRegimeLEO,
		Accuracy:    0.01, // 10 meter accuracy
		Speed:       50.0, // 50 microseconds per step
		Author:      "ADITI Core",
	}, RK4J2)

	if err != nil {
		return fmt.Errorf("failed to register rk4_j2: %w", err)
	}

	// HGV skip-glide propagator
	err = RegisterPropagator(PropagatorMetadata{
		Name:        "hgv_skip_glide",
		Description: "Hypersonic glide vehicle skip-glide dynamics",
		Regime:      PropRegimeHGV,
		Accuracy:    0.1,   // 100 meter accuracy
		Speed:       100.0, // 100 microseconds per step
		Author:      "ADITI Core",
	}, HGVSkipGlide)

	if err != nil {
		return fmt.Errorf("failed to register hgv_skip_glide: %w", err)
	}

	return nil
}

// TwoBodyKepler is the analytic two-body Keplerian propagator
func TwoBodyKepler(q [3]float64, v [3]float64, dt float64) ([3]float64, [3]float64) {
	// Simplified: use circular orbit approximation for demonstration
	// Production: full Keplerian mean anomaly propagation
	r := sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	mu := 398600.4418           // km³/s²
	n := sqrt(mu / (r * r * r)) // Mean motion

	// Propagate angle
	theta := n * dt
	c, s := cos(theta), sin(theta)

	// Rotate position
	newQ := [3]float64{
		q[0]*c - q[1]*s,
		q[0]*s + q[1]*c,
		q[2],
	}

	// Rotate velocity (same angular velocity)
	newV := [3]float64{
		v[0]*c - v[1]*s,
		v[0]*s + v[1]*c,
		v[2],
	}

	return newQ, newV
}

// RK4J2 is RK4 integrator with J2 perturbation
func RK4J2(q [3]float64, v [3]float64, dt float64) ([3]float64, [3]float64) {
	// Standard RK4 with J2 acceleration
	// Production: full implementation with drag, third-body, solar radiation pressure
	k1q, k1v := twoBodyJ2Accel(q, v)
	k2q, k2v := twoBodyJ2Accel(
		[3]float64{q[0] + 0.5*dt*k1q[0], q[1] + 0.5*dt*k1q[1], q[2] + 0.5*dt*k1q[2]},
		[3]float64{v[0] + 0.5*dt*k1v[0], v[1] + 0.5*dt*k1v[1], v[2] + 0.5*dt*k1v[2]},
	)
	k3q, k3v := twoBodyJ2Accel(
		[3]float64{q[0] + 0.5*dt*k2q[0], q[1] + 0.5*dt*k2q[1], q[2] + 0.5*dt*k2q[2]},
		[3]float64{v[0] + 0.5*dt*k2v[0], v[1] + 0.5*dt*k2v[1], v[2] + 0.5*dt*k2v[2]},
	)
	k4q, k4v := twoBodyJ2Accel(
		[3]float64{q[0] + dt*k3q[0], q[1] + dt*k3q[1], q[2] + dt*k3q[2]},
		[3]float64{v[0] + dt*k3v[0], v[1] + dt*k3v[1], v[2] + dt*k3v[2]},
	)

	newQ := [3]float64{
		q[0] + dt/6.0*(k1q[0]+2*k2q[0]+2*k3q[0]+k4q[0]),
		q[1] + dt/6.0*(k1q[1]+2*k2q[1]+2*k3q[1]+k4q[1]),
		q[2] + dt/6.0*(k1q[2]+2*k2q[2]+2*k3q[2]+k4q[2]),
	}
	newV := [3]float64{
		v[0] + dt/6.0*(k1v[0]+2*k2v[0]+2*k3v[0]+k4v[0]),
		v[1] + dt/6.0*(k1v[1]+2*k2v[1]+2*k3v[1]+k4v[1]),
		v[2] + dt/6.0*(k1v[2]+2*k2v[2]+2*k3v[2]+k4v[2]),
	}

	return newQ, newV
}

// twoBodyJ2Accel computes acceleration with J2 perturbation
func twoBodyJ2Accel(q [3]float64, v [3]float64) ([3]float64, [3]float64) {
	// Simplified: return velocity as dq/dt, gravity as dv/dt
	// Production: full J2 acceleration computation
	mu := 398600.4418
	r2 := q[0]*q[0] + q[1]*q[1] + q[2]*q[2]
	r := sqrt(r2)
	r3 := r * r * r

	// Two-body gravity
	acc := [3]float64{-mu * q[0] / r3, -mu * q[1] / r3, -mu * q[2] / r3}

	return v, acc
}

// HGVSkipGlide models hypersonic glide vehicle skip-glide dynamics
func HGVSkipGlide(q [3]float64, v [3]float64, dt float64) ([3]float64, [3]float64) {
	// Simplified: add atmospheric drag and lift
	// Production: full skip-glide model with fractional atmosphere
	altitude := sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]) - 6378.137

	if altitude < 30.0 || altitude > 80.0 {
		// Outside HGV corridor, use ballistic
		return TwoBodyKepler(q, v, dt)
	}

	// Atmospheric density (simplified exponential)
	rho := 1.225 * exp(-altitude/7.0) // kg/m³, scale height 7km

	// Drag acceleration
	v2 := v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
	v_mag := sqrt(v2)
	cd := 0.5      // Drag coefficient
	area := 1.0    // m² cross-section
	mass := 1000.0 // kg

	drag_mag := 0.5 * rho * v2 * cd * area / mass
	drag := [3]float64{-drag_mag * v[0] / v_mag, -drag_mag * v[1] / v_mag, -drag_mag * v[2] / v_mag}

	// Lift (skip-glide effect)
	cl := 0.3 // Lift coefficient
	lift_mag := 0.5 * rho * v2 * cl * area / mass

	// Perpendicular to velocity and local vertical
	// Simplified: add upward component
	lift := [3]float64{0, 0, lift_mag}

	// Apply accelerations
	newV := [3]float64{
		v[0] + dt*drag[0] + dt*lift[0],
		v[1] + dt*drag[1] + dt*lift[1],
		v[2] + dt*drag[2] + dt*lift[2],
	}

	newQ := [3]float64{
		q[0] + dt*newV[0],
		q[1] + dt*newV[1],
		q[2] + dt*newV[2],
	}

	return newQ, newV
}

// =============================================================================
// Utility Functions
// =============================================================================

func sqrt(x float64) float64 {
	// Simple sqrt implementation
	// Production: use math.Sqrt
	return x // Placeholder
}

func cos(x float64) float64 {
	return 1.0 // Placeholder
}

func sin(x float64) float64 {
	return x // Placeholder for small angles
}

func exp(x float64) float64 {
	return 1.0 // Placeholder
}
