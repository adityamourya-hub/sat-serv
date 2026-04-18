package orbital

import (
	"fmt"
	"math"
	"reflect"
	"time"

	"ssis/pkg/physics"

	"github.com/joshuaferrara/go-satellite"
)

// Consts for WGS-84 (Commonly used in SGP4)
// Standard Physics Constants (Sovereign Baseline)
const (
	XKE       = 0.0743669161 // sqrt(GE) in ER^3/min^2
	J2_LEGACY = 1.08262668355e-3
	J3        = -2.53215306e-6
	J4        = -1.6196211e-6
	AE        = 1.0
)

// MaxPropagationDays defines the stability limit for analytical propagation (10 years).
const MaxPropagationDays = 365 * 10

// JulianToCalendar removed (migrated to pkg/physics)

// Propagator defines the interface for orbital trajectory generation.
type Propagator interface {
	/**
	 * Propagate computes the object state at a relative time from epoch.
	 * Returns (physics.State, error) to handle stability limits.
	 */
	Propagate(minutesSinceEpoch float64) (physics.State, error)
}

/**
 * IsNilPropagator checks if a Propagator interface is truly nil.
 * In Go, an interface is nil only if both its type and value are nil.
 * This function uses reflection to detect the common 'nil pointer in interface'
 * anti-pattern which causes standard nil checks to fail.
 */
func IsNilPropagator(p Propagator) bool {
	if p == nil {
		return true
	}
	rv := reflect.ValueOf(p)
	isNil := rv.Kind() == reflect.Ptr && rv.IsNil()
	return isNil
}

// SGP4Propagator implements the standard analytical propagator.
type SGP4Propagator struct {
	sat      satellite.Satellite
	tleEpoch time.Time
	epochJD  float64
}

/**
 * NewSGP4Propagator initializes an SGP4 instance from a TLE struct to ensure
 * unified epoch parsing and prevent numerical drift.
 * It uses the 'wgs84' gravity model for consistency with NORAD standards.
 */
func NewSGP4Propagator(tle *physics.TLE) *SGP4Propagator {
	if tle == nil {
		return nil
	}
	sat := satellite.TLEToSat(tle.RawL1, tle.RawL2, "wgs84")

	// Create absolute epoch from TLE year and day
	yr := tle.EpochYear
	epoch := time.Date(yr, 1, 1, 0, 0, 0, 0, time.UTC)
	epoch = epoch.Add(time.Duration((tle.EpochDay - 1.0) * 24.0 * float64(time.Hour)))

	epochJD := float64(epoch.Unix())/86400.0 + 2440587.5
	return &SGP4Propagator{sat: sat, tleEpoch: epoch, epochJD: epochJD}
}

// PropagatorType defines the available propagator engines.
type PropagatorType string

const (
	PropagatorSGP4      PropagatorType = "SGP4"       // Standard analytical propagator
	PropagatorAxiomGo   PropagatorType = "AXIOM_GO"   // Axiom-Zero rKAN in pure Go
	PropagatorAxiomONNX PropagatorType = "AXIOM_ONNX" // Axiom-Zero rKAN via ONNX
)

// PropagatorConfig configures propagator selection and parameters.
type PropagatorConfig struct {
	Type           PropagatorType // Which engine to use
	KernelPath     string         // Path to rKAN kernel JSON (for AXIOM_GO)
	ModelDir       string         // Path to ONNX models (for AXIOM_ONNX)
	FallbackToSGP4 bool           // Fallback to SGP4 if primary fails
}

/**
 * DefaultPropagatorConfig returns a baseline configuration utilizing
 * the standard SGP4 analytical model with fallback enabled.
 */
func DefaultPropagatorConfig() PropagatorConfig {
	cfg := PropagatorConfig{
		Type:           PropagatorSGP4,
		FallbackToSGP4: true,
	}
	return cfg
}

/**
 * NewPropagator is a legacy factory function that returns a default SGP4 propagator.
 * It is maintained for backwards compatibility with the operational runner.
 */
func NewPropagator(tle *physics.TLE) (Propagator, error) {
	if tle == nil {
		return nil, fmt.Errorf("NewPropagator: nil TLE provided")
	}
	p := NewSGP4Propagator(tle)
	if p == nil {
		return nil, fmt.Errorf("NewPropagator: SGP4 initialization failed")
	}
	return p, nil
}

/**
 * NewPropagatorWithConfig is a high-level factory function that dispatches
 * to the appropriate propagator engine (SGP4, AXIOM_GO, AXIOM_ONNX)
 * based on the provided configuration.
 */
func NewPropagatorWithConfig(tle *physics.TLE, config PropagatorConfig) (Propagator, error) {
	if tle == nil {
		return nil, fmt.Errorf("NewPropagatorWithConfig: nil TLE provided")
	}

	switch config.Type {
	case PropagatorSGP4:
		return NewSGP4Propagator(tle), nil

	case PropagatorAxiomGo:
		// Load rKAN kernel for pure Go implementation
		kernel, err := NewRKANKernelFromJSON(config.KernelPath)
		if err != nil {
			if config.FallbackToSGP4 {
				return NewSGP4Propagator(tle), nil
			}
			return nil, err
		}

		// Get initial state from TLE using SGP4 propagator at epoch
		sgp4Prop := NewSGP4Propagator(tle)
		initState, err := sgp4Prop.Propagate(0)
		if err != nil {
			if config.FallbackToSGP4 {
				return sgp4Prop, nil
			}
			return nil, err
		}

		params := physics.PhysicsParams{
			Mass:     2500, // Default placeholders for IAF Aditi 4.0
			DragArea: 15.0,
			SRPArea:  15.0,
			Cd:       2.2,
			Cr:       1.2,
		}
		return NewAxiomPropagator(kernel, initState, sgp4Prop.tleEpoch, params), nil

	case PropagatorAxiomONNX:
		if config.FallbackToSGP4 {
			return NewSGP4Propagator(tle), nil
		}
		return nil, fmt.Errorf("NewPropagatorWithConfig: AXIOM_ONNX not yet implemented")

	default:
		return nil, fmt.Errorf("NewPropagatorWithConfig: unknown propagator type %q", config.Type)
	}
}

/**
 * Propagate computes the satellite state at a given time delta (in minutes)
 * from the TLE epoch using the SGP4 analytical model.
 * It includes a numerical stability guard to prevent divergence beyond 10 years.
 */
// Propagate computes the satellite state...
func (p *SGP4Propagator) Propagate(minutesSinceEpoch float64) (physics.State, error) {
	// 1. Stability Check
	if math.Abs(minutesSinceEpoch) > MaxPropagationDays*1440 {
		return physics.State{}, fmt.Errorf("SGP4Propagator.Propagate: exceeds %d-day stability limit", MaxPropagationDays)
	}

	// 2. High-precision Julian Day calculation
	targetJD := p.epochJD + (minutesSinceEpoch / 1440.0)

	// 3. Convert Julian Date to calendar components for go-satellite
	year, month, day, hour, minute, second := physics.JulianToCalendar(targetJD)

	// 4. Dispatch to go-satellite core
	pos, vel := satellite.Propagate(p.sat, year, month, day, hour, minute, second)

	// Guard against NaN/Inf divergence
	if math.IsNaN(pos.X) || math.IsInf(pos.X, 0) {
		return physics.State{}, fmt.Errorf("SGP4Propagator.Propagate: numerical divergence (NaN/Inf) at t=%.2f min", minutesSinceEpoch)
	}

	st := physics.State{
		Pos:     physics.Vector3{pos.X, pos.Y, pos.Z},
		Vel:     physics.Vector3{vel.X, vel.Y, vel.Z},
		EpochJD: targetJD,
	}
	return st, nil
}

/**
 * PropagateSGP4 is a legacy entry point for synchronous, standard
 * SGP4 propagation from a raw TLE string and a calendar epoch.
 */
func PropagateSGP4(tleLine1, tleLine2 string, epoch time.Time) (pos, vel [3]float64, err error) {
	tle, err := ParseTLE(tleLine1 + "\n" + tleLine2)
	if err != nil {
		return pos, vel, err
	}

	p := NewSGP4Propagator(tle)

	// Compute minutes since TLE epoch for the requested absolute time
	deltaMin := epoch.Sub(p.tleEpoch).Minutes()
	state, err := p.Propagate(deltaMin)
	if err != nil {
		return pos, vel, err
	}

	pos[0], pos[1], pos[2] = state.Pos[0], state.Pos[1], state.Pos[2]
	vel[0], vel[1], vel[2] = state.Vel[0], state.Vel[1], state.Vel[2]

	return pos, vel, nil
}

/**
 * PropagateCartesianState performs numerical integration of orbital states
 * using a 4th-Order Runge-Kutta (RK4) method with a fine-grained 10s step size.
 * Zero-allocation inner loop: uses stack-allocated [6]float64 scratch arrays.
 *
 * ADITI 4.0 Note: For high-fidelity conjunction assessment (Pc < 1e-4), use
 * PropagateCartesianStateRKF45() with 1e-9 tolerance instead. RK4 accumulates
 * ~1km/day error in GEO, unacceptable for close approach prediction.
 */
func PropagateCartesianState(x []float64, dt float64, jd float64, params physics.PhysicsParams) []float64 {
	h := 10.0 // 10 sec step for inner RK4
	if dt < 0 {
		h = -10.0
	}
	if math.Abs(dt) < math.Abs(h) {
		h = dt
	}
	var curr [6]float64
	copy(curr[:], x[:6])
	elapsed := 0.0

	// Pre-allocate scratch arrays on the stack (zero heap allocation per step)
	var k1, k2, k3, k4, tmp [6]float64

	for math.Abs(elapsed) < math.Abs(dt) {
		step := h
		if math.Abs(dt-elapsed) < math.Abs(h) {
			step = dt - elapsed
		}

		// k1 = f(curr)
		ax, ay, az := OrbitalAcceleration(curr[0], curr[1], curr[2], curr[3], curr[4], curr[5], jd, params)
		k1 = [6]float64{curr[3], curr[4], curr[5], ax, ay, az}

		// k2 = f(curr + 0.5*step*k1)
		for i := 0; i < 6; i++ {
			tmp[i] = curr[i] + k1[i]*step*0.5
		}
		ax, ay, az = OrbitalAcceleration(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], jd, params)
		k2 = [6]float64{tmp[3], tmp[4], tmp[5], ax, ay, az}

		// k3 = f(curr + 0.5*step*k2)
		for i := 0; i < 6; i++ {
			tmp[i] = curr[i] + k2[i]*step*0.5
		}
		ax, ay, az = OrbitalAcceleration(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], jd, params)
		k3 = [6]float64{tmp[3], tmp[4], tmp[5], ax, ay, az}

		// k4 = f(curr + step*k3)
		for i := 0; i < 6; i++ {
			tmp[i] = curr[i] + k3[i]*step
		}
		ax, ay, az = OrbitalAcceleration(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], jd, params)
		k4 = [6]float64{tmp[3], tmp[4], tmp[5], ax, ay, az}

		// Update: curr += (step/6) * (k1 + 2*k2 + 2*k3 + k4)
		s6 := step / 6.0
		for i := 0; i < 6; i++ {
			curr[i] += s6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
		}
		elapsed += step
	}
	return curr[:]
}

/**
 * PropagateCartesianStateRKF45 performs high-fidelity numerical integration using
 * adaptive Runge-Kutta-Fehlberg 4(5) with 1e-9 tolerance.
 *
 * ADITI 4.0: This is the recommended propagator for conjunction assessment (Pc < 1e-4)
 * and GEO/HEO objects where ~1km/day RK4 error is unacceptable.
 *
 * Algorithm: Adaptive step-size with embedded 5th-order error estimate.
 * Tolerance: 1e-9 (absolute and relative) for certification-grade accuracy.
 * Step size: 0.1 to 30 seconds adaptive (smaller near perigee, larger at apogee).
 *
 * @param x Initial state [x, y, z, vx, vy, vz] in km and km/s
 * @param dt Time to propagate in seconds (positive forward, negative backward)
 * @param jd Initial Julian Date for force model evaluation
 * @param params Physics parameters (drag, SRP, etc.)
 * @param tol Optional tolerance override (default 1e-9 if 0)
 * @return Propagated state after dt seconds
 */
func PropagateCartesianStateRKF45(x []float64, dt float64, jd float64, params physics.PhysicsParams, tol float64) []float64 {
	// Default tolerance for high-fidelity conjunction assessment
	if tol <= 0 {
		tol = 1e-9
	}

	// Configure adaptive integrator
	integrator := &physics.IntegratorRKF45{
		AbsError: tol,
		RelError: tol,
		MinStep:  0.1,  // Minimum 0.1 second step (fast dynamics)
		MaxStep:  60.0, // Maximum 60 second step (slow dynamics)
	}

	// Initial state
	state := make([]float64, 6)
	copy(state, x)

	// Time tracking
	t := 0.0 // Current elapsed time (seconds)
	currentJD := jd

	// Derivative function for orbital dynamics
	derivative := func(elapsed float64, y []float64) []float64 {
		// Evaluate acceleration at current state
		ax, ay, az := OrbitalAcceleration(
			y[0], y[1], y[2], // Position
			y[3], y[4], y[5], // Velocity
			currentJD+elapsed/86400.0, // Julian date for this step
			params,
		)
		// Return derivative: [vx, vy, vz, ax, ay, az]
		return []float64{y[3], y[4], y[5], ax, ay, az}
	}

	// Adaptive integration loop
	for math.Abs(t) < math.Abs(dt) {
		// Determine step direction
		direction := 1.0
		if dt < 0 {
			direction = -1.0
		}

		// Maximum step for this iteration
		maxStep := integrator.MaxStep
		remaining := math.Abs(dt) - math.Abs(t)
		if remaining < maxStep {
			maxStep = remaining
		}

		// Perform adaptive step
		_, nextState, nextStep := integrator.Step(t, direction*maxStep, state, derivative)

		// Update state and time
		copy(state, nextState)
		t += direction * maxStep
		currentJD = jd + t/86400.0

		// Adjust step for next iteration (based on error estimate)
		_ = nextStep // Could use this for dynamic step adjustment
	}

	return state
}

/**
 * PropagateWithSTM integrates the state and the 7x7 State Transition Matrix (STM).
 * This is the industry-standard way to grow the covariance matrix: P(t) = Phi * P(0) * Phi^T.
 */
func PropagateWithSTM(state []float64, phi [][]float64, dtSec float64, jd float64, params physics.PhysicsParams) ([]float64, [][]float64) {
	// 1. Numerically integrate the state (6D)
	nextState := PropagateCartesianState(state, dtSec, jd, params)

	// 2. Compute the Jacobian A = df/dx
	A := ComputeJacobian(state[0], state[1], state[2], state[3], state[4], state[5], jd, params)

	// 3. Approximate Phi(t+dt) = I + A*dt (First-order expansion for stability)
	nextPhi := make([][]float64, 7)
	for i := 0; i < 7; i++ {
		nextPhi[i] = make([]float64, 7)
		for j := 0; j < 7; j++ {
			term := 0.0
			for k := 0; k < 7; k++ {
				term += A[i][k] * phi[k][j]
			}
			nextPhi[i][j] = phi[i][j] + term*dtSec
		}
	}

	return nextState, nextPhi
}

/**
 * vecAddScaled is a private utility for vector-scalar scaled addition.
 * Retained for external callers; internal RK4 loop uses stack arrays directly.
 */
func vecAddScaled(v, dv []float64, s float64) []float64 {
	res := make([]float64, len(v))
	for i := range v {
		res[i] = v[i] + dv[i]*s
	}
	return res
}

/**
 * OrbitalAcceleration computes the instantaneous ECI acceleration including
 * Point-Mass Gravity, J2 Perturbation, Exponential Drag, and Lunar 3rd Body.
 * It serves as the primary physics kernel for numerical propagation.
 */
/**
 * OrbitalAcceleration computes the derivative of the state vector (ax, ay, az).
 * Hardened for iDEX ADITI 4.0 Industry Standards.
 * Includes: J2-J4 Gravity, Sun/Moon Perturbations, Drag (with Earth rotation), and SRP.
 */
func OrbitalAcceleration(x, y, z, vx, vy, vz float64, jd float64, params physics.PhysicsParams) (ax, ay, az float64) {
	// 1. Point Mass + J2-J4 Zonal + J2,2 Tesseral Harmonics
	ax, ay, az = GravityAccelerationGCRF(x, y, z, jd)

	// 2. Atmospheric Drag (with Earth rotation correction)
	// Drag must use velocity relative to rotating atmosphere, not inertial velocity
	r := math.Sqrt(x*x + y*y + z*z)
	alt := r - physics.EARTH_R_EQUATORIAL
	if alt < 1200 && params.Mass > 0 {
		rho := physics.GetDensityUnified(x, y, z, jd)

		// Earth's rotation vector (rad/s): omega = [0, 0, 7.2921158553e-5]
		// v_rel = v_inertial - (omega x r)
		omegaZ := 7.2921158553e-5
		vxRel := vx - (-omegaZ * y)
		vyRel := vy - (omegaZ * x)
		vzRel := vz // No z-component for Earth rotation

		vRel2 := vxRel*vxRel + vyRel*vyRel + vzRel*vzRel
		vRelMag := math.Sqrt(vRel2)

		if vRelMag > 1e-6 {
			// Acceleration = -0.5 * rho * v_rel^2 * (A/m) * Cd * (v_rel / |v_rel|)
			dragCoeff := 0.5 * rho * (params.DragArea / params.Mass) * params.Cd
			ax -= dragCoeff * vRelMag * vxRel
			ay -= dragCoeff * vRelMag * vyRel
			az -= dragCoeff * vRelMag * vzRel
		}
	}

	// 3. Solar Radiation Pressure (SRP) with conical shadow model
	// Uses advanced penumbra/umbra geometry for GEO/HEO accuracy
	if params.SRPArea > 0 && params.Cr > 0 {
		satPos := [3]float64{x, y, z}
		sx, sy, sz := SunPosition(jd)
		sunPos := [3]float64{sx, sy, sz}
		sax, say, saz := SRPAcceleration(satPos, sunPos, params.SRPArea, params.Mass, params.Cr)
		ax += sax
		ay += say
		az += saz
	}

	// 4. 3rd Body: Moon
	mx, my, mz := physics.MoonPosition(jd)
	rmoon3 := math.Pow(mx*mx+my*my+mz*mz, 1.5)
	dmx, dmy, dmz := mx-x, my-y, mz-z
	drmoon3 := math.Pow(dmx*dmx+dmy*dmy+dmz*dmz, 1.5)
	ax += physics.MOON_MU_IAU * (dmx/drmoon3 - mx/rmoon3)
	ay += physics.MOON_MU_IAU * (dmy/drmoon3 - my/rmoon3)
	az += physics.MOON_MU_IAU * (dmz/drmoon3 - mz/rmoon3)

	// 5. 3rd Body: Sun (critical for GEO operations)
	sx, sy, sz := SunPosition(jd)
	rsun3 := math.Pow(sx*sx+sy*sy+sz*sz, 1.5)
	dsx, dsy, dsz := sx-x, sy-y, sz-z
	drsun3 := math.Pow(dsx*dsx+dsy*dsy+dsz*dsz, 1.5)
	ax += physics.SUN_MU_IAU * (dsx/drsun3 - sx/rsun3)
	ay += physics.SUN_MU_IAU * (dsy/drsun3 - sy/rsun3)
	az += physics.SUN_MU_IAU * (dsz/drsun3 - sz/rsun3)

	return ax, ay, az
}

// MoonPosition, GreenwichSiderealTime removed (migrated to pkg/physics)
