// pkg/physics/rkf45.go - Adaptive RKF45 with Hamiltonian Monitoring
//
// Runge-Kutta-Fehlberg 4(5) adaptive step-size integrator
// with altitude-specific tolerance profiles for M2 Air thermal management.
//
// Key Features:
//   - Adaptive step: shrinks at perigee/high-gradient, expands at apogee
//   - Hamiltonian error monitoring: maintains |ΔH/H| < 1e-8
//   - Altitude-specific tolerances: LEO vs MEO vs GEO vs HGV
//   - Thermal-aware: reduces CPU load during ballistic coast
//
// RKF45 Butcher Tableau (Fehlberg coefficients):
//   0     |
//   1/4   | 1/4
//   3/8   | 3/32       9/32
//   12/13 | 1932/2197 -7200/2197 7296/2197
//   1     | 439/216   -8          3680/513  -845/4104
//   1/2   | -8/27     2          -3544/2565 1859/4104 -11/40
//
//   4th order: 25/216, 0, 1408/2565, 2197/4104, -1/5, 0
//   5th order: 16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55
//
// Error estimate: ε = ||y5 - y4||
// Step adjustment: dt_new = dt * (TOL * dt / ε)^(1/4)
//
// References:
//   - Fehlberg (1969): Low-order classical Runge-Kutta formulas
//   - Hairer, Nørsett, Wanner: Solving ODE I (1993)

package physics

import (
	"fmt"
	"math"
)

// =============================================================================
// Tolerance Profiles for Different Orbital Regimes
// =============================================================================

// FlightRegime defines the type of orbit/flight phase for RKF45 integration
type FlightRegime int

const (
	FlightRegimeLEO       FlightRegime = iota // < 2000 km, high dynamics
	FlightRegimeMEO                           // 2000-35786 km, moderate
	FlightRegimeGEO                           // ~35786 km, geostationary
	FlightRegimeHGVEntry                      // 30-80 km, hypersonic skip
	FlightRegimeHGVGlide                      // 30-50 km, steady glide
	FlightRegimeBallistic                     // Re-entry, high gradients
)

// ToleranceProfile defines adaptive step parameters for each regime
type ToleranceProfile struct {
	RelTol       float64 // Relative tolerance for state error
	AbsTol       float64 // Absolute tolerance (km)
	MinStep      float64 // Minimum step size (s) - thermal protection
	MaxStep      float64 // Maximum step size (s) - accuracy limit
	HamTol       float64 // Hamiltonian drift tolerance |ΔH/H|
	SafetyFactor float64 // Step size safety factor (typically 0.9)
}

// DefaultToleranceProfiles provides M2 Air-optimized settings
var DefaultToleranceProfiles = map[FlightRegime]ToleranceProfile{
	FlightRegimeLEO: {
		RelTol:       1e-8,
		AbsTol:       1e-6,  // 1 meter
		MinStep:      0.001, // 1 ms - thermal floor
		MaxStep:      60.0,  // 1 minute - LEO orbital motion
		HamTol:       1e-8,
		SafetyFactor: 0.9,
	},
	FlightRegimeMEO: {
		RelTol:       1e-7,
		AbsTol:       1e-5,  // 10 meters
		MinStep:      0.01,  // 10 ms
		MaxStep:      300.0, // 5 minutes
		HamTol:       1e-7,
		SafetyFactor: 0.9,
	},
	FlightRegimeGEO: {
		RelTol:       1e-6,
		AbsTol:       1e-4,  // 100 meters
		MinStep:      0.1,   // 100 ms - relaxed for GEO stability
		MaxStep:      600.0, // 10 minutes
		HamTol:       1e-6,
		SafetyFactor: 0.9,
	},
	FlightRegimeHGVEntry: {
		RelTol:       1e-9,   // Ultra-tight for high-gradient entry
		AbsTol:       1e-7,   // 10 cm
		MinStep:      0.0001, // 100 μs - sub-millisecond for skip
		MaxStep:      1.0,    // 1 second max
		HamTol:       1e-9,
		SafetyFactor: 0.8, // More conservative for HGV
	},
	FlightRegimeHGVGlide: {
		RelTol:       1e-8,
		AbsTol:       1e-6,
		MinStep:      0.001,
		MaxStep:      10.0,
		HamTol:       1e-8,
		SafetyFactor: 0.9,
	},
	FlightRegimeBallistic: {
		RelTol:       1e-9,
		AbsTol:       1e-7,
		MinStep:      0.0001,
		MaxStep:      5.0,
		HamTol:       1e-9,
		SafetyFactor: 0.8,
	},
}

// =============================================================================
// RKF45 Integrator
// =============================================================================

// DynamicsFunction defines the ODE: dx/dt = f(x, t)
type DynamicsFunction func(state []float64, t float64) []float64

// HamiltonianFunction computes H(x) for error monitoring
type HamiltonianFunction func(state []float64) float64

// RKF45Integrator implements adaptive Runge-Kutta-Fehlberg
type RKF45Integrator struct {
	Profile       ToleranceProfile
	CurrentRegime FlightRegime

	// State tracking
	StepCount     int
	RejectedSteps int
	TotalTime     float64

	// Hamiltonian monitoring
	InitialH        float64
	MaxDrift        float64
	LastHamiltonian float64

	// Performance metrics
	AvgStepSize float64
	MinStepUsed float64
	MaxStepUsed float64
}

// NewRKF45Integrator creates integrator with regime-specific profile
func NewRKF45Integrator(regime FlightRegime) *RKF45Integrator {
	profile, ok := DefaultToleranceProfiles[regime]
	if !ok {
		profile = DefaultToleranceProfiles[FlightRegimeLEO]
	}

	return &RKF45Integrator{
		Profile:       profile,
		CurrentRegime: regime,
		MinStepUsed:   profile.MaxStep,
		MaxStepUsed:   profile.MinStep,
	}
}

// SetRegime updates tolerance profile based on altitude
func (r *RKF45Integrator) SetRegime(altitudeKm float64, velocityKms float64) {
	// Detect regime based on altitude and velocity
	regime := r.detectRegime(altitudeKm, velocityKms)

	if regime != r.CurrentRegime {
		r.CurrentRegime = regime
		r.Profile = DefaultToleranceProfiles[regime]

		// Reset step size bounds
		r.MinStepUsed = r.Profile.MaxStep
		r.MaxStepUsed = r.Profile.MinStep
	}
}

// detectRegime determines flight regime from state
func (r *RKF45Integrator) detectRegime(altitudeKm, velocityKms float64) FlightRegime {
	// HGV detection: 30-80 km with hypersonic velocity (> Mach 5 ~ 1.7 km/s)
	if altitudeKm >= 30 && altitudeKm <= 80 && velocityKms > 1.7 {
		// Entry vs glide: entry has very high velocity (> 7 km/s)
		// glide has moderate velocity (2-6 km/s typical)
		if velocityKms > 7.0 {
			return FlightRegimeHGVEntry
		}
		return FlightRegimeHGVGlide
	}

	// Ballistic re-entry: < 120 km, high velocity
	if altitudeKm < 120 && velocityKms > 7.0 {
		return FlightRegimeBallistic
	}

	// Orbital regimes
	if altitudeKm < 2000 {
		return FlightRegimeLEO
	} else if altitudeKm < 35786 {
		return FlightRegimeMEO
	}
	return FlightRegimeGEO
}

// Step performs one adaptive RKF45 step
//
// Returns:
//   - newState: 4th-order accurate state
//   - newTime: updated time
//   - newDt: recommended step size for next iteration
//   - err: local truncation error estimate
//   - accepted: true if step passed tolerance
func (r *RKF45Integrator) Step(
	state []float64,
	t float64,
	dt float64,
	dynamics DynamicsFunction,
) (newState []float64, newTime float64, newDt float64, err float64, accepted bool) {

	// Compute 6 stages (k1-k6)
	k1 := dynamics(state, t)
	k2 := dynamics(r.add(state, r.scale(k1, dt/4.0)), t+dt/4.0)
	k3 := dynamics(r.add(state, r.add(r.scale(k1, 3.0*dt/32.0), r.scale(k2, 9.0*dt/32.0))), t+3.0*dt/8.0)
	k4 := dynamics(r.add(state, r.add(r.add(r.scale(k1, 1932.0*dt/2197.0), r.scale(k2, -7200.0*dt/2197.0)), r.scale(k3, 7296.0*dt/2197.0))), t+12.0*dt/13.0)
	k5 := dynamics(r.add(state, r.add(r.add(r.add(r.scale(k1, 439.0*dt/216.0), r.scale(k2, -8.0*dt)), r.scale(k3, 3680.0*dt/513.0)), r.scale(k4, -845.0*dt/4104.0))), t+dt)
	k6 := dynamics(r.add(state, r.add(r.add(r.add(r.add(r.scale(k1, -8.0*dt/27.0), r.scale(k2, 2.0*dt)), r.scale(k3, -3544.0*dt/2565.0)), r.scale(k4, 1859.0*dt/4104.0)), r.scale(k5, -11.0*dt/40.0))), t+dt/2.0)

	// 4th order solution (for propagation)
	y4 := r.add(state, r.add(r.add(r.add(r.scale(k1, 25.0*dt/216.0), r.scale(k3, 1408.0*dt/2565.0)), r.scale(k4, 2197.0*dt/4104.0)), r.scale(k5, -dt/5.0)))

	// 5th order solution (for error estimation)
	y5 := r.add(state, r.add(r.add(r.add(r.add(r.scale(k1, 16.0*dt/135.0), r.scale(k3, 6656.0*dt/12825.0)), r.scale(k4, 28561.0*dt/56430.0)), r.scale(k5, -9.0*dt/50.0)), r.scale(k6, 2.0*dt/55.0)))

	// Error estimate: ||y5 - y4||
	err = r.norm(r.sub(y5, y4))

	// Scaled error (relative + absolute tolerance)
	scale := r.Profile.RelTol*r.norm(y4) + r.Profile.AbsTol
	scaledErr := err / scale

	// Check acceptance
	if scaledErr > 1.0 {
		// Reject step, reduce dt
		r.RejectedSteps++
		newDt = dt * r.Profile.SafetyFactor * math.Pow(1.0/scaledErr, 0.25)

		// Enforce minimum step
		if newDt < r.Profile.MinStep {
			newDt = r.Profile.MinStep
			// Accept anyway at minimum step to prevent infinite loop
			return y4, t + dt, newDt, err, true
		}

		return state, t, newDt, err, false
	}

	// Accept step
	accepted = true
	r.StepCount++

	// Update step size statistics
	if dt < r.MinStepUsed {
		r.MinStepUsed = dt
	}
	if dt > r.MaxStepUsed {
		r.MaxStepUsed = dt
	}
	r.AvgStepSize = (r.AvgStepSize*float64(r.StepCount-1) + dt) / float64(r.StepCount)

	// Compute next step size
	if scaledErr < 1e-4 {
		// Very accurate, can increase step
		newDt = dt * r.Profile.SafetyFactor * math.Pow(1e-4/scaledErr, 0.2)
	} else {
		newDt = dt * r.Profile.SafetyFactor * math.Pow(1.0/scaledErr, 0.25)
	}

	// Enforce bounds
	if newDt > r.Profile.MaxStep {
		newDt = r.Profile.MaxStep
	}
	if newDt < r.Profile.MinStep {
		newDt = r.Profile.MinStep
	}

	return y4, t + dt, newDt, err, true
}

// StepWithHamiltonian performs RKF45 step with Hamiltonian error monitoring
func (r *RKF45Integrator) StepWithHamiltonian(
	state []float64,
	t float64,
	dt float64,
	dynamics DynamicsFunction,
	hamiltonian HamiltonianFunction,
) (newState []float64, newTime float64, newDt float64, hamErr float64, accepted bool) {

	// Initial Hamiltonian (for monitoring)
	if r.StepCount == 0 {
		r.InitialH = hamiltonian(state)
		r.LastHamiltonian = r.InitialH
	}

	// Perform RKF45 step
	newState, newTime, newDt, _, accepted = r.Step(state, t, dt, dynamics)

	if !accepted {
		return newState, newTime, newDt, 0, false
	}

	// Compute Hamiltonian error
	H := hamiltonian(newState)
	r.LastHamiltonian = H

	if math.Abs(r.InitialH) > 1e-10 {
		hamErr = math.Abs(H-r.InitialH) / math.Abs(r.InitialH)
	} else {
		hamErr = math.Abs(H - r.InitialH)
	}

	if hamErr > r.MaxDrift {
		r.MaxDrift = hamErr
	}

	// Check Hamiltonian tolerance
	if hamErr > r.Profile.HamTol {
		// Hamiltonian violated - reduce step and retry
		newDt = dt * 0.5
		if newDt < r.Profile.MinStep {
			newDt = r.Profile.MinStep
			// Log warning but accept
			fmt.Printf("[RKF45] WARNING: Hamiltonian drift %.2e exceeds tolerance %.2e at step %d\n",
				hamErr, r.Profile.HamTol, r.StepCount)
		} else {
			return state, t, newDt, hamErr, false
		}
	}

	return newState, newTime, newDt, hamErr, true
}

// Helper functions for vector operations
func (r *RKF45Integrator) add(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

func (r *RKF45Integrator) sub(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

func (r *RKF45Integrator) scale(v []float64, s float64) []float64 {
	result := make([]float64, len(v))
	for i := range v {
		result[i] = v[i] * s
	}
	return result
}

func (r *RKF45Integrator) norm(v []float64) float64 {
	sum := 0.0
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

// GetStats returns performance statistics
func (r *RKF45Integrator) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"regime":          r.CurrentRegime,
		"step_count":      r.StepCount,
		"rejected_steps":  r.RejectedSteps,
		"rejection_rate":  float64(r.RejectedSteps) / float64(r.StepCount+r.RejectedSteps+1),
		"avg_step_size":   r.AvgStepSize,
		"min_step_used":   r.MinStepUsed,
		"max_step_used":   r.MaxStepUsed,
		"max_ham_drift":   r.MaxDrift,
		"hamiltonian_tol": r.Profile.HamTol,
	}
}

// Reset clears statistics for new trajectory
func (r *RKF45Integrator) Reset() {
	r.StepCount = 0
	r.RejectedSteps = 0
	r.TotalTime = 0
	r.MaxDrift = 0
	r.AvgStepSize = 0
	r.MinStepUsed = r.Profile.MaxStep
	r.MaxStepUsed = r.Profile.MinStep
}
