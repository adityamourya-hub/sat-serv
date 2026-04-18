package hypersonic

import (
	"math"
	"ssis/pkg/physics"
)

/**
 * ADITI 4.0 Hypersonic Glide Vehicle (HGV) Physics
 *
 * Skip-glide trajectory with lift generation (L/D = 2-4).
 * Equations of motion in spherical, rotating Earth frame.
 *
 * Unlike ballistic missiles, HGVs generate lift and can maneuver
 * via bank angle (σ), creating a "footprint" rather than fixed impact point.
 *
 * Reference: Regan & Anandakrishnan, "Dynamics of Atmospheric Re-Entry"
 */

/**
 * HGVState represents the 6-DOF state in spherical coordinates.
 */
type HGVState struct {
	// Position
	R      float64 // Radius from Earth center (km)
	Phi    float64 // Latitude (rad)
	Lambda float64 // Longitude (rad)

	// Velocity
	V     float64 // Velocity magnitude (km/s)
	Gamma float64 // Flight path angle (rad), negative for re-entry
	Psi   float64 // Heading angle/azimuth (rad), clockwise from North

	// Aerodynamic (estimated)
	CL    float64 // Lift coefficient (estimated)
	CD    float64 // Drag coefficient (estimated)
	Sigma float64 // Bank angle (rad), surveillance nightmare variable

	// Mass (constant for HGV, but included for completeness)
	Mass float64 // kg

	// Time
	Epoch float64 // Julian Date
}

/**
 * HGVConstants for typical hypersonic glider.
 * Based on DF-17 / similar HGV class.
 */
type HGVConstants struct {
	ReferenceArea float64 // m^2
	Mass          float64 // kg
	MaxLD         float64 // Max L/D ratio (typically 2-4)
	MinLD         float64 // Min L/D ratio
}

/**
 * DefaultHGV returns typical HGV parameters.
 * DF-17 class: ~1000 kg warhead, 0.5 m² reference area.
 */
func DefaultHGV() HGVConstants {
	return HGVConstants{
		ReferenceArea: 0.5,    // m^2
		Mass:          1000.0, // kg
		MaxLD:         4.0,    // Maximum L/D
		MinLD:         2.0,    // Minimum L/D
	}
}

/**
 * HGVEquationsOfMotion computes state derivatives.
 *
 * Equations (spherical, rotating Earth):
 * V̇ = -D/m - g·sin(γ)
 * γ̇ = (1/V)·[L·cos(σ)/m - (g - V²/r)·cos(γ)]
 * ψ̇ = (1/V·cos(γ))·[L·sin(σ)/m + (V²/r)·cos²(γ)·sin(ψ)·tan(φ)]
 * ṙ = V·sin(γ)
 * φ̇ = (V·cos(γ)·cos(ψ))/r
 * λ̇ = (V·cos(γ)·sin(ψ))/(r·cos(φ))
 *
 * Where:
 * - D = 0.5·ρ·V²·A·CD (Drag force)
 * - L = 0.5·ρ·V²·A·CL (Lift force)
 * - g = μ/r² (gravity)
 * - σ = bank angle (turn control)
 */
func HGVEquationsOfMotion(state HGVState, consts HGVConstants, jd float64) HGVDerivatives {
	// Earth rotation effects
	omegaEarth := 7.2921158553e-5 // rad/s

	// Atmospheric density from space weather model
	rho := physics.GetDensityUnified(0, 0, state.R, jd) // Simplified - should use lat/lon

	// Gravity
	g := physics.EARTH_MU_IAU / (state.R * state.R)

	// Velocity in m/s for aerodynamics
	vMs := state.V * 1000.0 // km/s to m/s

	// Dynamic pressure
	q := 0.5 * rho * vMs * vMs // Pa

	// Drag and Lift forces (Newtons)
	A := consts.ReferenceArea
	D := q * A * state.CD
	L := q * A * state.CL

	// Bank angle components
	cosSigma := math.Cos(state.Sigma)
	sinSigma := math.Sin(state.Sigma)

	// Vertical (V̇) - Drag slows vehicle, gravity component
	dV := -(D/consts.Mass)/1000.0 - g*math.Sin(state.Gamma) // Convert to km/s^2

	// Flight path angle (γ̇) - Lift changes trajectory angle
	// Includes centrifugal term (V²/r)
	centrifugal := (state.V * state.V) / state.R
	term1 := (L*cosSigma/consts.Mass)/1000.0 - (g-centrifugal)*math.Cos(state.Gamma)
	dGamma := term1 / state.V

	// Heading angle (ψ̇) - Bank angle turns the vehicle
	// Coriolis and centripetal terms from Earth's rotation
	coriolisTerm := (L * sinSigma / consts.Mass) / 1000.0
	centripetalTerm := (state.V * state.V / state.R) *
		math.Cos(state.Gamma) * math.Cos(state.Gamma) *
		math.Sin(state.Psi) * math.Tan(state.Phi)

	// Earth rotation correction
	rotationCorrection := 2 * omegaEarth * state.V * math.Cos(state.Gamma) * math.Sin(state.Psi)
	dPsi := (coriolisTerm + centripetalTerm + rotationCorrection) / (state.V * math.Cos(state.Gamma))

	// Position derivatives
	dR := state.V * math.Sin(state.Gamma)
	dPhi := (state.V * math.Cos(state.Gamma) * math.Cos(state.Psi)) / state.R
	dLambda := (state.V * math.Cos(state.Gamma) * math.Sin(state.Psi)) / (state.R * math.Cos(state.Phi))

	return HGVDerivatives{
		DV:      dV,
		DGamma:  dGamma,
		DPsi:    dPsi,
		DR:      dR,
		DPhi:    dPhi,
		DLambda: dLambda,
	}
}

/**
 * HGVDerivatives holds state rates of change.
 */
type HGVDerivatives struct {
	DV      float64 // km/s^2
	DGamma  float64 // rad/s
	DPsi    float64 // rad/s
	DR      float64 // km/s
	DPhi    float64 // rad/s
	DLambda float64 // rad/s
}

/**
 * HGVAccelerationECI converts HGV aerodynamic acceleration to ECI frame.
 *
 * Used to integrate HGV into standard orbital propagators.
 *
 * @param state: HGV state
 * @param consts: Vehicle constants
 * @param jd: Julian Date
 * @return: (ax, ay, az) acceleration in ECI km/s^2
 */
func HGVAccelerationECI(state HGVState, consts HGVConstants, jd float64) (ax, ay, az float64) {
	// Get derivatives
	dot := HGVEquationsOfMotion(state, consts, jd)

	// Convert spherical velocity to ECI
	// First, get local ENU (East-North-Up) frame
	sinPhi := math.Sin(state.Phi)
	cosPhi := math.Cos(state.Phi)
	sinPsi := math.Sin(state.Psi)
	cosPsi := math.Cos(state.Psi)
	sinGamma := math.Sin(state.Gamma)
	cosGamma := math.Cos(state.Gamma)

	// Velocity in local ENU frame
	// V_East = V * cos(γ) * sin(ψ)
	// V_North = V * cos(γ) * cos(ψ)
	// V_Up = V * sin(γ)
	vEast := state.V * cosGamma * sinPsi
	vNorth := state.V * cosGamma * cosPsi
	vUp := state.V * sinGamma

	// Transform to ECI
	// This is a simplified transformation
	// Real implementation would use proper rotation matrices

	// Local up direction in ECI
	upX := cosPhi * math.Cos(state.Lambda)
	upY := cosPhi * math.Sin(state.Lambda)
	upZ := sinPhi

	// Local north direction
	northX := -sinPhi * math.Cos(state.Lambda)
	northY := -sinPhi * math.Sin(state.Lambda)
	northZ := cosPhi

	// Local east direction (cross product: east = north × up)
	eastX := northY*upZ - northZ*upY
	eastY := northZ*upX - northX*upZ
	eastZ := northX*upY - northY*upX

	// Velocity in ECI
	vx := vEast*eastX + vNorth*northX + vUp*upX
	vy := vEast*eastY + vNorth*northY + vUp*upY
	vz := vEast*eastZ + vNorth*northZ + vUp*upZ

	// Acceleration from derivatives
	dotV := dot.DV

	// Approximate acceleration in ECI
	// This is the rate of change of velocity
	vMag := math.Sqrt(vx*vx + vy*vy + vz*vz)
	if vMag > 1e-9 {
		// Direction of velocity
		ux, uy, uz := vx/vMag, vy/vMag, vz/vMag

		// Acceleration from velocity change
		ax = dotV * ux
		ay = dotV * uy
		az = dotV * uz

		// Add lift-induced acceleration perpendicular to velocity
		// Simplified: perpendicular to velocity and in the plane of motion
	}

	return ax, ay, az
}

/**
 * EstimateAeroCoefficients estimates CL, CD from observed trajectory.
 *
 * Uses inverse dynamics: given observed acceleration, solve for L, D.
 *
 * @param state: Current HGV state
 * @param observedAcc: Observed acceleration (from radar/IR)
 * @param consts: Vehicle constants
 * @return: Estimated CL, CD
 */
func EstimateAeroCoefficients(
	state HGVState,
	observedAcc [3]float64,
	consts HGVConstants,
	jd float64,
) (CL, CD float64) {
	// Get atmospheric density
	rho := physics.GetDensityUnified(0, 0, state.R, jd)

	// Velocity in m/s
	vMs := state.V * 1000.0

	// Dynamic pressure
	q := 0.5 * rho * vMs * vMs
	if q < 1.0 {
		// Too high or too low for meaningful estimate
		return 0, 0
	}

	// Observed acceleration magnitude
	accMag := math.Sqrt(observedAcc[0]*observedAcc[0] +
		observedAcc[1]*observedAcc[1] +
		observedAcc[2]*observedAcc[2])

	// Decompose into drag and lift components
	// Drag opposes velocity, lift is perpendicular

	// Simplified estimate based on deceleration
	// Drag causes deceleration along velocity vector
	// Lift causes turn (change in gamma and psi)

	// Estimate CD from deceleration
	// D = m * a_drag / q / A
	// Assuming most deceleration is drag
	CD = (consts.Mass * accMag * 1000.0) / (q * consts.ReferenceArea)

	// Clamp to realistic values
	if CD < 0.1 {
		CD = 0.1
	}
	if CD > 2.0 {
		CD = 2.0
	}

	// Estimate CL from L/D ratio
	// Typical HGV: L/D = 2-4
	// Use state history to estimate turn rate
	// Simplified: assume moderate L/D
	LD := 2.5 + 1.5*math.Sin(state.Gamma*10) // Variable based on flight phase
	CL = CD * LD

	return CL, CD
}

/**
 * ComputeLDFromTurnRate estimates L/D from observed turn.
 *
 * L = m * V * γ̇ / cos(σ)
 * L/D = (m * V * γ̇) / (D * cos(σ))
 *
 * If bank angle unknown, assume σ = 0 for minimum L/D estimate.
 */
func ComputeLDFromTurnRate(
	state HGVState,
	turnRate float64, // γ̇ in rad/s
	consts HGVConstants,
	jd float64,
) float64 {
	// Required lift for this turn rate (with σ = 0)
	L := consts.Mass * state.V * 1000.0 * turnRate // N

	// Get drag (from deceleration or estimate)
	rho := physics.GetDensityUnified(0, 0, state.R, jd)
	vMs := state.V * 1000.0
	q := 0.5 * rho * vMs * vMs

	// Estimate CD from typical HGV (0.3-0.5)
	CD := 0.4
	D := q * consts.ReferenceArea * CD

	if D < 1.0 {
		return 0
	}

	return L / D
}

/**
 * PredictHGVImpactFootprint computes reachable area for HGV.
 *
 * Unlike ballistic impact point, HGV has a "footprint" due to bank angle.
 *
 * @param state: Current HGV state
 * @param consts: Vehicle constants
 * @param jd: Julian Date
 * @return: Center impact point and footprint radius (km)
 */
func PredictHGVImpactFootprint(
	state HGVState,
	consts HGVConstants,
	jd float64,
) (centerLat, centerLon, radiusKm float64) {
	// Ballistic trajectory (no bank, no lift)
	balLat, balLon, _ := predictBallisticImpact(state, jd)

	// Maximum cross-range due to bank angle
	// HGV can turn within L/D constraints

	// Estimate remaining range
	remainingRange := state.V * state.V / (2 * 9.81 * 0.5) // Simplified

	// Cross-range capability: proportional to L/D and remaining energy
	// LD = 4 gives ~2000 km cross-range for typical HGV
	maxCrossRange := consts.MaxLD * remainingRange * 0.1

	// Footprint radius
	radiusKm = maxCrossRange

	// Center is biased by current heading
	centerLat = balLat + state.V*math.Cos(state.Gamma)*math.Cos(state.Psi)*0.1
	centerLon = balLon + state.V*math.Cos(state.Gamma)*math.Sin(state.Psi)*0.1

	return centerLat, centerLon, radiusKm
}

/**
 * predictBallisticImpact computes ballistic-only impact point.
 * Used as baseline for footprint calculation.
 */
func predictBallisticImpact(state HGVState, jd float64) (lat, lon, timeToImpact float64) {
	// Simplified: use orbital mechanics
	// Real: numerical integration with drag

	// Energy equation
	energy := state.V*state.V/2 - physics.EARTH_MU_IAU/state.R

	// If energy > 0, it's on an escape trajectory
	if energy >= 0 {
		return state.Phi, state.Lambda, -1 // No impact
	}

	// Estimate time to impact from current altitude and velocity
	alt := state.R - physics.EARTH_R_EQUATORIAL
	timeToImpact = alt / (-state.V * math.Sin(state.Gamma))

	// Estimate impact position (simplified)
	dist := state.V * timeToImpact * math.Cos(state.Gamma)

	lat = state.Phi + dist*math.Cos(state.Psi)/physics.EARTH_R_EQUATORIAL
	lon = state.Lambda + dist*math.Sin(state.Psi)/(physics.EARTH_R_EQUATORIAL*math.Cos(state.Phi))

	return lat, lon, timeToImpact
}

/**
 * HGVToECIState converts HGV spherical state to ECI Cartesian.
 */
func HGVToECIState(hgv HGVState) physics.State {
	// Position
	cosPhi := math.Cos(hgv.Phi)
	sinPhi := math.Sin(hgv.Phi)
	cosLambda := math.Cos(hgv.Lambda)
	sinLambda := math.Sin(hgv.Lambda)

	x := hgv.R * cosPhi * cosLambda
	y := hgv.R * cosPhi * sinLambda
	z := hgv.R * sinPhi

	// Velocity
	cosGamma := math.Cos(hgv.Gamma)
	sinGamma := math.Sin(hgv.Gamma)
	cosPsi := math.Cos(hgv.Psi)
	sinPsi := math.Sin(hgv.Psi)

	// Local ENU frame
	vEast := hgv.V * cosGamma * sinPsi
	vNorth := hgv.V * cosGamma * cosPsi
	vUp := hgv.V * sinGamma

	// Velocity in ECI
	vx := -sinLambda * vEast
	vy := cosLambda * vEast
	vz := 0.0

	// North
	vx += -sinPhi * cosLambda * vNorth
	vy += -sinPhi * sinLambda * vNorth
	vz += cosPhi * vNorth

	// Up
	vx += cosPhi * cosLambda * vUp
	vy += cosPhi * sinLambda * vUp
	vz += sinPhi * vUp

	return physics.State{
		Pos:     [3]float64{x, y, z},
		Vel:     [3]float64{vx, vy, vz},
		EpochJD: hgv.Epoch,
	}
}

/**
 * ECIToHGVState converts ECI Cartesian to HGV spherical state.
 */
func ECIToHGVState(eci physics.State) HGVState {
	x, y, z := eci.Pos[0], eci.Pos[1], eci.Pos[2]
	vx, vy, vz := eci.Vel[0], eci.Vel[1], eci.Vel[2]

	// Radius
	r := math.Sqrt(x*x + y*y + z*z)

	// Latitude
	phi := math.Asin(z / r)

	// Longitude
	lambda := math.Atan2(y, x)

	// Velocity magnitude
	v := math.Sqrt(vx*vx + vy*vy + vz*vz)

	// Local ENU frame vectors
	sinPhi := math.Sin(phi)
	cosPhi := math.Cos(phi)
	sinLambda := math.Sin(lambda)
	cosLambda := math.Cos(lambda)

	// Up
	upX := cosPhi * cosLambda
	upY := cosPhi * sinLambda
	upZ := sinPhi

	// North
	northX := -sinPhi * cosLambda
	northY := -sinPhi * sinLambda
	northZ := cosPhi

	// East
	eastX := northY*upZ - northZ*upY
	eastY := northZ*upX - northX*upZ
	eastZ := northX*upY - northY*upX

	// Velocity components
	vUp := vx*upX + vy*upY + vz*upZ
	vNorth := vx*northX + vy*northY + vz*northZ
	vEast := vx*eastX + vy*eastY + vz*eastZ

	// Flight path angle
	horizontalV := math.Sqrt(vEast*vEast + vNorth*vNorth)
	gamma := math.Atan2(vUp, horizontalV)

	// Heading
	psi := math.Atan2(vEast, vNorth)
	if psi < 0 {
		psi += 2 * math.Pi
	}

	return HGVState{
		R:      r,
		Phi:    phi,
		Lambda: lambda,
		V:      v,
		Gamma:  gamma,
		Psi:    psi,
		Epoch:  eci.EpochJD,
	}
}
