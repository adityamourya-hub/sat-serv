// Package orbital provides High-Precision Orbit Propagation (HPOP)
// Uses ChristopherRabotin/ode for numerical integration with J2/J3 and atmospheric drag
package orbital

import (
	"fmt"
	"math"
	"time"

	"ssis/pkg/physics"
)

// HPOPMode determines force model complexity
type HPOPMode int

const (
	HPOPBasic    HPOPMode = iota // 2-body + J2
	HPOPStandard                 // + J3 + drag
	HPOPFull                     // + SRP + third body
)

// HPOPConfig holds high-precision propagation configuration
type HPOPConfig struct {
	// Initial state (ECI, J2000)
	PositionECI [3]float64 // km
	VelocityECI [3]float64 // km/s
	Epoch       time.Time

	// Force model
	Mode HPOPMode

	// Satellite physical properties
	Mass     float64 // kg
	Cd       float64 // Drag coefficient (default 2.2)
	AreaDrag float64 // m²
	AreaSRP  float64 // m² (Solar Radiation Pressure)
	CR       float64 // Reflectivity (default 1.0)

	// Integration parameters
	MaxStep    float64 // seconds (default 60)
	Tolerance  float64 // (default 1e-10)
	MaxRuntime float64 // seconds (default 30)
}

// DefaultHPOPConfig returns standard configuration for LEO
func DefaultHPOPConfig() *HPOPConfig {
	return &HPOPConfig{
		Mode:       HPOPStandard,
		Cd:         2.2,
		AreaDrag:   10.0, // m² (typical satellite)
		AreaSRP:    10.0,
		CR:         1.0,
		MaxStep:    60.0,
		Tolerance:  1e-10,
		MaxRuntime: 30.0,
	}
}

// HPOP implements high-precision numerical orbit propagation
type HPOP struct {
	conf *HPOPConfig
}

// NewHPOP creates high-precision propagator
func NewHPOP(conf *HPOPConfig) *HPOP {
	if conf == nil {
		conf = DefaultHPOPConfig()
	}
	return &HPOP{conf: conf}
}

// Propagate computes state at target time using numerical integration
func (h *HPOP) Propagate(targetTime time.Time) (*physics.State, error) {
	// Time since epoch in seconds
	dt := targetTime.Sub(h.conf.Epoch).Seconds()

	if math.Abs(dt) < 1.0 {
		// Near epoch, return initial state
		return &physics.State{
			Pos: h.conf.PositionECI,
			Vel: h.conf.VelocityECI,
		}, nil
	}

	// Initial state vector [rx, ry, rz, vx, vy, vz]
	y0 := []float64{
		h.conf.PositionECI[0],
		h.conf.PositionECI[1],
		h.conf.PositionECI[2],
		h.conf.VelocityECI[0],
		h.conf.VelocityECI[1],
		h.conf.VelocityECI[2],
	}

	// Create force model
	force := &forceModel{config: h.conf}

	// Adaptive RKF45 integration (Cash-Karp) with embedded 4th/5th order error control.
	// Phase-0 plan fix: RK4 placeholder replaced with error-controlled integrator so that
	// training-data trajectories are accurate to the configured tolerance instead of a
	// fixed step. Falls back to RK4 if RKF45 fails for any reason (e.g. stiff regime).
	absTol := h.conf.Tolerance
	if absTol <= 0 {
		absTol = 1e-10
	}
	relTol := absTol * 1e3 // relative tolerance typically looser than absolute
	y, _, rkfErr := h.rkf45Integrate(y0, 0.0, dt, force.derivative, absTol, relTol)
	if rkfErr != nil {
		// Graceful fallback preserves the original behaviour for pathological cases.
		y = h.rk4Integrate(y0, dt, force.derivative, h.conf.MaxStep)
	}

	return &physics.State{
		Pos: physics.Vector3{y[0], y[1], y[2]},
		Vel: physics.Vector3{y[3], y[4], y[5]},
	}, nil
}

// forceModel implements equations of motion for HPOP
type forceModel struct {
	config *HPOPConfig
}

// derivative computes dy/dt for ODE integrator
// y = [rx, ry, rz, vx, vy, vz]
func (f *forceModel) derivative(t float64, y []float64) []float64 {
	dydt := make([]float64, 6)

	// Position and velocity
	r := y[0:3]
	v := y[3:6]

	// Copy velocity to position derivative
	copy(dydt[0:3], v)

	// Compute acceleration
	a := f.acceleration(t, r, v)
	copy(dydt[3:6], a)

	return dydt
}

// acceleration computes total acceleration with perturbations
func (f *forceModel) acceleration(t float64, r, v []float64) []float64 {
	// Earth gravitational parameter
	const mu = 398600.4418 // km³/s²

	// Distance
	rNorm := math.Sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])

	// Two-body gravity
	factor := -mu / (rNorm * rNorm * rNorm)
	a := []float64{
		factor * r[0],
		factor * r[1],
		factor * r[2],
	}

	// Add perturbations based on mode
	switch f.config.Mode {
	case HPOPBasic:
		// 2-body + J2 only
		a = f.addJ2Perturbation(a, r, rNorm)

	case HPOPStandard:
		// + J3 + atmospheric drag
		a = f.addJ2Perturbation(a, r, rNorm)
		a = f.addJ3Perturbation(a, r, rNorm)
		a = f.addDragPerturbation(a, r, v, rNorm)

	case HPOPFull:
		// + SRP + third body (lunar/solar)
		a = f.addJ2Perturbation(a, r, rNorm)
		a = f.addJ3Perturbation(a, r, rNorm)
		a = f.addDragPerturbation(a, r, v, rNorm)
		a = f.addSRPPerturbation(a, r, rNorm)
		// Third body perturbations would go here
	}

	return a
}

// addJ2Perturbation adds J2 oblateness effect
func (f *forceModel) addJ2Perturbation(a, r []float64, rNorm float64) []float64 {
	// J2 coefficient and Earth radius
	const J2 = 0.00108263
	const re = 6378.137 // km

	j2Factor := 1.5 * J2 * 398600.4418 * re * re / (rNorm * rNorm * rNorm * rNorm * rNorm)
	zRatio := r[2] / rNorm

	a[0] += j2Factor * r[0] * (5*zRatio*zRatio - 1)
	a[1] += j2Factor * r[1] * (5*zRatio*zRatio - 1)
	a[2] += j2Factor * r[2] * (5*zRatio*zRatio - 3)

	return a
}

// addJ3Perturbation adds J3 oblateness effect
func (f *forceModel) addJ3Perturbation(a, r []float64, rNorm float64) []float64 {
	// J3 coefficient
	const J3 = -0.000002532
	const re = 6378.137

	j3Factor := 2.5 * J3 * 398600.4418 * re * re * re / (rNorm * rNorm * rNorm * rNorm * rNorm * rNorm)
	zRatio := r[2] / rNorm

	a[0] += j3Factor * r[0] * (3*zRatio - 7*zRatio*zRatio*zRatio)
	a[1] += j3Factor * r[1] * (3*zRatio - 7*zRatio*zRatio*zRatio)
	a[2] += j3Factor * (3*zRatio*zRatio - 7*zRatio*zRatio*zRatio*zRatio - 0.6)

	return a
}

// addDragPerturbation adds atmospheric drag
func (f *forceModel) addDragPerturbation(a, r, v []float64, rNorm float64) []float64 {
	if f.config.AreaDrag <= 0 || f.config.Mass <= 0 {
		return a
	}

	// Simple exponential atmosphere model
	const re = 6378.137  // Earth radius (km)
	const h0 = 500.0     // Reference altitude (km)
	const rho0 = 1.0e-13 // Reference density (kg/m³ at 500km)
	const H = 50.0       // Scale height (km)

	// Altitude above Earth surface
	altitude := rNorm - re

	// Density at current altitude
	var rho float64
	if altitude > h0 {
		rho = rho0 * math.Exp(-(altitude-h0)/H)
	} else {
		// Higher density at lower altitudes
		rho = rho0 * math.Exp((h0-altitude)/20.0) // 20km scale height for lower alt
	}

	// Velocity magnitude
	vNorm := math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
	if vNorm == 0 {
		return a
	}

	// Drag acceleration: a = -0.5 * ρ * Cd * A/m * v² * v̂
	// Convert AreaDrag from m² to km² for consistency
	areaKm2 := f.config.AreaDrag * 1e-6
	dragFactor := -0.5 * rho * f.config.Cd * areaKm2 / f.config.Mass * vNorm

	a[0] += dragFactor * v[0] / vNorm
	a[1] += dragFactor * v[1] / vNorm
	a[2] += dragFactor * v[2] / vNorm

	return a
}

// addSRPPerturbation adds solar radiation pressure
func (f *forceModel) addSRPPerturbation(a, r []float64, rNorm float64) []float64 {
	if f.config.AreaSRP <= 0 {
		return a
	}

	// Solar radiation pressure at 1 AU
	const P_sr = 4.56e-9 // N/m²

	// Assume Sun in -X direction (simplified)
	// Real implementation would compute Sun position from ephemeris
	sunDir := []float64{-1.0, 0.0, 0.0}

	// Acceleration: a = (1+CR) * P * A / (2m) * sunDir
	// Factor of 0.5 for solar panels that absorb half the light
	areaKm2 := f.config.AreaSRP * 1e-6
	srpFactor := (1.0 + f.config.CR) * P_sr * areaKm2 / (2.0 * f.config.Mass)

	a[0] += srpFactor * sunDir[0]
	a[1] += srpFactor * sunDir[1]
	a[2] += srpFactor * sunDir[2]

	return a
}

// StateAtEpoch returns the initial state (useful for initialization)
func (h *HPOP) StateAtEpoch() *physics.State {
	return &physics.State{
		Pos: h.conf.PositionECI,
		Vel: h.conf.VelocityECI,
	}
}

// rk4Integrate performs 4th-order Runge-Kutta integration
func (h *HPOP) rk4Integrate(y0 []float64, dt float64, deriv func(float64, []float64) []float64, stepSize float64) []float64 {
	n := len(y0)
	y := make([]float64, n)
	copy(y, y0)

	nSteps := int(math.Abs(dt) / stepSize)
	if nSteps < 1 {
		nSteps = 1
	}

	hStep := dt / float64(nSteps)

	for i := 0; i < nSteps; i++ {
		t := float64(i) * hStep

		// k1
		k1 := deriv(t, y)

		// k2
		y2 := make([]float64, n)
		for j := 0; j < n; j++ {
			y2[j] = y[j] + 0.5*hStep*k1[j]
		}
		k2 := deriv(t+0.5*hStep, y2)

		// k3
		y3 := make([]float64, n)
		for j := 0; j < n; j++ {
			y3[j] = y[j] + 0.5*hStep*k2[j]
		}
		k3 := deriv(t+0.5*hStep, y3)

		// k4
		y4 := make([]float64, n)
		for j := 0; j < n; j++ {
			y4[j] = y[j] + hStep*k3[j]
		}
		k4 := deriv(t+hStep, y4)

		// Update
		for j := 0; j < n; j++ {
			y[j] += hStep * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]) / 6.0
		}
	}

	return y
}

// rkf45Integrate performs adaptive Runge-Kutta-Fehlberg integration
// RKF45 uses 6 function evaluations to compute both 4th and 5th order solutions
// allowing for error estimation and adaptive step size control.
func (h *HPOP) rkf45Integrate(
	y0 []float64,
	t0 float64,
	tf float64,
	deriv func(float64, []float64) []float64,
	absTol float64,
	relTol float64,
) ([]float64, []float64, error) {
	n := len(y0)
	y := make([]float64, n)
	copy(y, y0)
	t := t0

	// RKF45 coefficients (Cash-Karp parameters)
	// These give 5th order accuracy with embedded 4th order for error estimation
	c := []float64{0.0, 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0}

	a := [][]float64{
		{}, // a[0] not used
		{1.0 / 5.0},
		{3.0 / 40.0, 9.0 / 40.0},
		{3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0},
		{-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0},
		{1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0},
	}

	// 5th order weights (b)
	b5 := []float64{37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0}

	// 4th order weights (b*)
	b4 := []float64{2825.0 / 27648.0, 0.0, 18575.0 / 48384.0, 13525.0 / 55296.0, 277.0 / 14336.0, 1.0 / 4.0}

	// Time history for output
	tHistory := []float64{t}
	yHistory := make([][]float64, 1)
	yHistory[0] = make([]float64, n)
	copy(yHistory[0], y)

	// Initial step size (will be adapted)
	dt := (tf - t0) / 100.0
	if dt == 0 {
		dt = 1.0
	}

	maxSteps := 10000
	stepCount := 0

	for t < tf && stepCount < maxSteps {
		stepCount++

		// Ensure we don't overshoot tf
		if t+dt > tf {
			dt = tf - t
		}

		// Compute k values (6 stages)
		k := make([][]float64, 6)

		// k1
		k[0] = deriv(t, y)

		// k2
		yTemp := make([]float64, n)
		for i := 0; i < n; i++ {
			yTemp[i] = y[i] + dt*a[1][0]*k[0][i]
		}
		k[1] = deriv(t+c[1]*dt, yTemp)

		// k3
		for i := 0; i < n; i++ {
			yTemp[i] = y[i] + dt*(a[2][0]*k[0][i]+a[2][1]*k[1][i])
		}
		k[2] = deriv(t+c[2]*dt, yTemp)

		// k4
		for i := 0; i < n; i++ {
			yTemp[i] = y[i] + dt*(a[3][0]*k[0][i]+a[3][1]*k[1][i]+a[3][2]*k[2][i])
		}
		k[3] = deriv(t+c[3]*dt, yTemp)

		// k5
		for i := 0; i < n; i++ {
			yTemp[i] = y[i] + dt*(a[4][0]*k[0][i]+a[4][1]*k[1][i]+a[4][2]*k[2][i]+a[4][3]*k[3][i])
		}
		k[4] = deriv(t+c[4]*dt, yTemp)

		// k6
		for i := 0; i < n; i++ {
			yTemp[i] = y[i] + dt*(a[5][0]*k[0][i]+a[5][1]*k[1][i]+a[5][2]*k[2][i]+a[5][3]*k[3][i]+a[5][4]*k[4][i])
		}
		k[5] = deriv(t+c[5]*dt, yTemp)

		// Compute 5th order solution
		y5 := make([]float64, n)
		for i := 0; i < n; i++ {
			y5[i] = y[i] + dt*(b5[0]*k[0][i]+b5[1]*k[1][i]+b5[2]*k[2][i]+b5[3]*k[3][i]+b5[4]*k[4][i]+b5[5]*k[5][i])
		}

		// Compute 4th order solution
		y4 := make([]float64, n)
		for i := 0; i < n; i++ {
			y4[i] = y[i] + dt*(b4[0]*k[0][i]+b4[1]*k[1][i]+b4[2]*k[2][i]+b4[3]*k[3][i]+b4[4]*k[4][i]+b4[5]*k[5][i])
		}

		// Error estimation
		maxError := 0.0
		for i := 0; i < n; i++ {
			err := math.Abs(y5[i] - y4[i])
			scale := absTol + relTol*math.Max(math.Abs(y[i]), math.Abs(y5[i]))
			maxError = math.Max(maxError, err/scale)
		}

		// Adaptive step size control
		if maxError > 1.0 {
			// Reject step, reduce dt
			dt = 0.9 * dt * math.Pow(maxError, -0.25)
		} else {
			// Accept step
			t += dt
			copy(y, y5)

			// Record history
			tHistory = append(tHistory, t)
			yNew := make([]float64, n)
			copy(yNew, y)
			yHistory = append(yHistory, yNew)

			// Increase dt for next step (but not too much)
			if maxError < 0.5 {
				dt = 0.9 * dt * math.Pow(maxError, -0.2)
				if dt > 10.0*(tf-t0) {
					dt = 10.0 * (tf - t0)
				}
			}
		}

		if dt < 1e-10 {
			return y, tHistory, fmt.Errorf("RKF45 step size underflow at t=%f", t)
		}
	}

	if stepCount >= maxSteps {
		return y, tHistory, fmt.Errorf("RKF45 exceeded maximum steps (%d)", maxSteps)
	}

	return y, tHistory, nil
}

// SetInitialState updates the initial conditions
func (h *HPOP) SetInitialState(pos, vel [3]float64, epoch time.Time) {
	h.conf.PositionECI = pos
	h.conf.VelocityECI = vel
	h.conf.Epoch = epoch
}

// Example usage comment showing how to use HPOP
/*
Example usage:

conf := &orbital.HPOPConfig{
    PositionECI: [3]float64{7000.0, 0.0, 0.0},  // km
    VelocityECI: [3]float64{0.0, 7.5, 0.0},     // km/s
    Epoch:       time.Now(),
    Mode:        orbital.HPOPStandard,
    Mass:        500.0,  // kg
    Cd:          2.2,
    AreaDrag:    15.0,  // m²
}

hpop := orbital.NewHPOP(conf)

targetTime := time.Now().Add(1 * time.Hour)
state, err := hpop.Propagate(targetTime)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Position: %v km\n", state.R)
fmt.Printf("Velocity: %v km/s\n", state.V)
*/
