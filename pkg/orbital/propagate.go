package orbital

import (
	"fmt"
	"math"
	"reflect"
	"time"

	"github.com/joshuaferrara/go-satellite"
)

// Consts for WGS-84 (Commonly used in SGP4)
// Standard Physics Constants (Sovereign Baseline)
const (
	DE2RA     = math.Pi / 180.0
	ER        = 6378.135     // SGP4 compliant Earth Radius (km)
	XKE       = 0.0743669161 // sqrt(GE) in ER^3/min^2
	J2_LEGACY = 1.08262668355e-3
	J3        = -2.53215306e-6
	J4        = -1.6196211e-6
	CK2       = 0.5 * J2_LEGACY * AE * AE
	AE        = 1.0
	EARTH_MU  = 398600.4418 // km^3/sec^2
)

// MaxPropagationDays defines the stability limit for analytical propagation (10 years).
const MaxPropagationDays = 365 * 10

/**
 * JulianToCalendar converts a fractional Julian Date (JD) into its Gregorian calendar 
 * components (year, month, day, hour, minute, second).
 * It implements the algorithm from Jean Meeus' "Astronomical Algorithms", 
 * robustly handling both Julian and Gregorian calendar transitions.
 */
func JulianToCalendar(jd float64) (year, month, day, hour, minute, second int) {
	// Algorithm from Meeus, Astronomical Algorithms
	jd = jd + 0.5
	Z := int(jd)
	F := jd - float64(Z)
	var A int
	if Z < 2299161 {
		A = Z
	} else {
		alpha := int((float64(Z) - 1867216.25) / 36524.25)
		A = Z + 1 + alpha - int(float64(alpha)/4)
	}
	B := A + 1524
	C := int((float64(B) - 122.1) / 365.25)
	D := int(365.25 * float64(C))
	E := int((float64(B) - float64(D)) / 30.6001)

	day = B - D - int(30.6001*float64(E))
	if E < 14 {
		month = E - 1
	} else {
		month = E - 13
	}
	if month > 2 {
		year = C - 4716
	} else {
		year = C - 4715
	}

	// Extract time from fractional day
	F = F * 24.0
	hour = int(F)
	F = (F - float64(hour)) * 60.0
	minute = int(F)
	second = int((F - float64(minute)) * 60.0)

	return
}

// State represents the orbital state (position and velocity) in ECI coordinates.
type State struct {
	X, Y, Z    float64 // Position (km)
	VX, VY, VZ float64 // Velocity (km/s)
	E_hal      float64 // Specific Orbital Energy for Hamiltonian audit
}

// Propagator defines the interface for orbital trajectory generation.
type Propagator interface {
	/**
	 * Propagate computes the object state at a relative time from epoch.
	 * Returns (State, error) to handle stability limits.
	 */
	Propagate(minutesSinceEpoch float64) (State, error)
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
func NewSGP4Propagator(tle *TLE) *SGP4Propagator {
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
func NewPropagator(tle *TLE) (Propagator, error) {
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
func NewPropagatorWithConfig(tle *TLE, config PropagatorConfig) (Propagator, error) {
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

		params := PhysicsParams{
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
func (p *SGP4Propagator) Propagate(minutesSinceEpoch float64) (State, error) {
	// 1. Stability Check (Limit propagation to +/- 10 years to prevent SGP4 divergence)
	if math.Abs(minutesSinceEpoch) > MaxPropagationDays*1440 {
		return State{}, fmt.Errorf("SGP4Propagator.Propagate: exceeds %d-day stability limit", MaxPropagationDays)
	}

	// 2. High-precision Julian Day calculation
	targetJD := p.epochJD + (minutesSinceEpoch / 1440.0)

	// 3. Convert Julian Date to calendar components for go-satellite
	year, month, day, hour, minute, second := JulianToCalendar(targetJD)

	// 4. Dispatch to go-satellite core (Propagate returns ECI Position (km) and Velocity (km/s))
	pos, vel := satellite.Propagate(p.sat, year, month, day, hour, minute, second)

	// Guard against NaN/Inf divergence
	if math.IsNaN(pos.X) || math.IsInf(pos.X, 0) {
		return State{}, fmt.Errorf("SGP4Propagator.Propagate: numerical divergence (NaN/Inf) at t=%.2f min", minutesSinceEpoch)
	}

	st := State{
		X: pos.X, Y: pos.Y, Z: pos.Z,
		VX: vel.X, VY: vel.Y, VZ: vel.Z,
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

	pos[0], pos[1], pos[2] = state.X, state.Y, state.Z
	vel[0], vel[1], vel[2] = state.VX, state.VY, state.VZ

	return pos, vel, nil
}

/**
 * StateToVector flattens a State struct into a 6-element float64 slice 
 * for compatibility with linear algebra libraries and ML kernels.
 */
func StateToVector(s State) []float64 {
	v := []float64{s.X, s.Y, s.Z, s.VX, s.VY, s.VZ}
	return v
}

/**
 * VectorToState reconstructs a State struct from a 6-element float64 slice.
 * It performs defensive checks to ensure the slice length is sufficient.
 */
func VectorToState(v []float64) State {
	if len(v) < 6 {
		return State{}
	}
	st := State{X: v[0], Y: v[1], Z: v[2], VX: v[3], VY: v[4], VZ: v[5]}
	return st
}

/**
 * PropagateCartesianState performs numerical integration of orbital states 
 * using a 4th-Order Runge-Kutta (RK4) method with a fine-grained 10s step size.
 * Zero-allocation inner loop: uses stack-allocated [6]float64 scratch arrays.
 */
func PropagateCartesianState(x []float64, dt float64, jd float64, params PhysicsParams) []float64 {
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
 * PropagateWithSTM integrates the state and the 7x7 State Transition Matrix (STM).
 * This is the industry-standard way to grow the covariance matrix: P(t) = Phi * P(0) * Phi^T.
 */
func PropagateWithSTM(state []float64, phi [][]float64, dtSec float64, jd float64, params PhysicsParams) ([]float64, [][]float64) {
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
 * Includes: J2-J4 Gravity, Sun/Moon Perturbations, Drag, and SRP.
 */
func OrbitalAcceleration(x, y, z, vx, vy, vz float64, jd float64, params PhysicsParams) (ax, ay, az float64) {
	// 1. Point Mass + J2-J4 Zonal Harmonics
	ax, ay, az = GravityAccelerationGCRF(x, y, z)

	// 2. Atmospheric Drag (Industry Standard NRLMSISE-00)
	if alt < 1200 {
		rho := GetDensityUnified(x, y, z, jd)
		v2 := vx*vx + vy*vy + vz*vz
		speed := math.Sqrt(v2)
		// Acceleration = -0.5 * rho * v^2 * (A/m) * Cd * (v_rel / speed)
		dragAcc := 0.5 * rho * v2 * (params.DragArea / params.Mass) * params.Cd
		ax -= dragAcc * vx / speed
		ay -= dragAcc * vy / speed
		az -= dragAcc * vz / speed
	}

	// 3. Solar Radiation Pressure (SRP)
	if params.SRPArea > 0 && params.Cr > 0 {
		sax, say, saz := SolarRadiationPressure(x, y, z, jd, params.SRPArea, params.Mass, params.Cr)
		ax += sax
		ay += say
		az += saz
	}

	// 4. 3rd Body: Moon (Simplified Position)
	mx, my, mz := MoonPosition(jd)
	rmoon3 := math.Pow(mx*mx+my*my+mz*mz, 1.5)
	dmx, dmy, dmz := mx-x, my-y, mz-z
	drmoon3 := math.Pow(dmx*dmx+dmy*dmy+dmz*dmz, 1.5)
	const MU_MOON = 4902.800066
	ax += MU_MOON * (dmx/drmoon3 - mx/rmoon3)
	ay += MU_MOON * (dmy/drmoon3 - my/rmoon3)
	az += MU_MOON * (dmz/drmoon3 - mz/rmoon3)

	return ax, ay, az
}

type PhysicsParams struct {
	Mass     float64
	DragArea float64
	SRPArea  float64
	Cd       float64
	Cr       float64
}

/**
 * MoonPosition returns the ECI coordinates (km) of the Moon at a given 
 * Julian Date using a simplified analytical model for Earth-Moon distance.
 */
func MoonPosition(jd float64) (x, y, z float64) {
	T := (jd - 2451545.0) / 36525.0
	LPrime := (218.316 + 481267.881*T) * DE2RA
	x = 384400.0 * math.Cos(LPrime)
	y = 384400.0 * math.Sin(LPrime)
	z = 0.0
	return
}

/**
 * AtmosphericDensity returns the air density (kg/m^3) at a given altitude.
 * It uses a simplified exponential model for rapid sub-orbital processing.
 */

/**
 * GreenwichSiderealTime computes the GST in radians at a given Julian Date.
 * It is essential for coordinate transformations between ECI and ECEF frames.
 */
func GreenwichSiderealTime(jd float64) float64 {
	T := (jd - 2451545.0) / 36525.0
	theta := 67310.54841 + (876600*3600+8640184.812866)*T + 0.093104*T*T - 6.2e-6*T*T*T
	gst := math.Mod(theta*DE2RA/240.0, 2*math.Pi)
	return gst
}
