// Package sim provides synthetic data generation for sovereign training
// Implements Go-native "Truth Engine" for Siliguri Corridor digital twin
package sim

import (
	"math"
	"math/rand"
	"time"
)

const (
	// Siliguri Corridor coordinates
	SiliguriLat = 26.5 // degrees N
	SiliguriLon = 88.4 // degrees E

	// Earth parameters (WGS-84)
	EarthRadiusKm = 6371.0
	MuM3S2        = 3.986004418e14 // m³/s²

	// Simulation parameters
	LEOAltitudeKm = 400.0
	SimulationDT  = 1.0 // seconds
)

// Intent represents adversary behavioral classification
type Intent int

const (
	IntentPassive           Intent = 0 // Pure Keplerian, minimal Δv
	IntentRPOInspection     Intent = 1 // HCW-locked, circumnavigation
	IntentElectronicWarfare Intent = 2 // Nadir-pointing, thermal spikes
	IntentKineticThreat     Intent = 3 // High-thrust, non-Keplerian
)

// IntentNames maps intents to human-readable strings
var IntentNames = map[Intent]string{
	IntentPassive:           "PASSIVE",
	IntentRPOInspection:     "RPO_INSPECTION",
	IntentElectronicWarfare: "ELECTRONIC_WARFARE",
	IntentKineticThreat:     "KINETIC_THREAT",
}

// SimState represents the 6-DOF state of a simulated object
type SimState struct {
	SatID     int
	Position  [3]float64 // ECI coordinates (km)
	Velocity  [3]float64 // ECI velocity (km/s)
	Magnitude float64    // Visual brightness (apparent mag)
	Type      string     // TARGET, DEBRIS, STAR
	Intent    Intent     // Behavioral classification
	Thrust    [3]float64 // Non-Keplerian acceleration (m/s²)
	DeltaV    float64    // Recent maneuver magnitude (m/s)
}

// SiliguriSimulator manages the digital twin simulation
type SiliguriSimulator struct {
	rng                     *rand.Rand
	states                  map[int]*SimState
	observers               []StateObserver
	simTime                 float64
	coordinateRandomization bool // Enable ECI frame rotation for training
}

// StateObserver receives simulation updates
type StateObserver func(states []SimState)

// NewSiliguriSimulator creates a new Siliguri Corridor simulator
func NewSiliguriSimulator() *SiliguriSimulator {
	return &SiliguriSimulator{
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
		states:    make(map[int]*SimState),
		observers: make([]StateObserver, 0),
		simTime:   0.0,
	}
}

// AddObserver registers a state observer
func (ss *SiliguriSimulator) AddObserver(observer StateObserver) {
	ss.observers = append(ss.observers, observer)
}

// GenerateSiliguriPass produces a batch of truth states for a 60-second window
func (ss *SiliguriSimulator) GenerateSiliguriPass(numDebris int, includeTarget bool) []SimState {
	states := make([]SimState, 0)

	if includeTarget {
		// 1. Primary Target (e.g., An Adversary Stalker)
		// Calculate ECI position for Siliguri at current time
		latRad := SiliguriLat * math.Pi / 180.0
		lonRad := SiliguriLon * math.Pi / 180.0

		// Simplified ECI (assuming equinox)
		r := EarthRadiusKm + LEOAltitudeKm
		targetPos := [3]float64{
			r * math.Cos(latRad) * math.Cos(lonRad),
			r * math.Cos(latRad) * math.Sin(lonRad),
			r * math.Sin(latRad),
		}

		// Velocity for circular orbit (~7.5 km/s at 400km)
		v := math.Sqrt(MuM3S2/(r*1000)) / 1000 // Convert to km/s
		targetVel := [3]float64{
			-v * math.Sin(lonRad),
			v * math.Cos(lonRad),
			0.0,
		}

		states = append(states, SimState{
			SatID:     99999,
			Position:  targetPos,
			Velocity:  targetVel,
			Magnitude: 3.5,
			Type:      "TARGET",
		})

		ss.states[99999] = &states[0]
	}

	// 2. Background Clutter (Debris)
	for i := 1; i <= numDebris; i++ {
		// Random position around Siliguri
		r := EarthRadiusKm + LEOAltitudeKm + ss.rng.NormFloat64()*10.0
		lat := SiliguriLat + ss.rng.NormFloat64()*5.0
		lon := SiliguriLon + ss.rng.NormFloat64()*5.0

		latRad := lat * math.Pi / 180.0
		lonRad := lon * math.Pi / 180.0

		debrisPos := [3]float64{
			r * math.Cos(latRad) * math.Cos(lonRad),
			r * math.Cos(latRad) * math.Sin(lonRad),
			r * math.Sin(latRad),
		}

		// Random velocity with orbital speed
		v := math.Sqrt(MuM3S2/(r*1000)) / 1000
		v += ss.rng.NormFloat64() * 0.05 // Small perturbation

		debrisVel := [3]float64{
			-v*math.Sin(lonRad) + ss.rng.NormFloat64()*0.01,
			v*math.Cos(lonRad) + ss.rng.NormFloat64()*0.01,
			ss.rng.NormFloat64() * 0.01,
		}

		// Dimmer debris (higher magnitude = dimmer)
		mag := 8.0 + ss.rng.Float64()*4.0

		states = append(states, SimState{
			SatID:     i,
			Position:  debrisPos,
			Velocity:  debrisVel,
			Magnitude: mag,
			Type:      "DEBRIS",
		})

		ss.states[i] = &states[len(states)-1]
	}

	return states
}

// Propagate advances simulation by dt seconds using Keplerian dynamics
func (ss *SiliguriSimulator) Propagate(dt float64) {
	ss.simTime += dt

	for id, state := range ss.states {
		// Convert to meters for calculation
		x := state.Position[0] * 1000
		y := state.Position[1] * 1000
		z := state.Position[2] * 1000
		vx := state.Velocity[0] * 1000
		vy := state.Velocity[1] * 1000
		vz := state.Velocity[2] * 1000

		// Gravitational acceleration
		r := math.Sqrt(x*x + y*y + z*z)
		ax := -MuM3S2 * x / (r * r * r)
		ay := -MuM3S2 * y / (r * r * r)
		az := -MuM3S2 * z / (r * r * r)

		// Simple Euler integration
		vx += ax * dt
		vy += ay * dt
		vz += az * dt
		x += vx * dt
		y += vy * dt
		z += vz * dt

		// Update state
		state.Position = [3]float64{x / 1000, y / 1000, z / 1000}
		state.Velocity = [3]float64{vx / 1000, vy / 1000, vz / 1000}

		ss.states[id] = state
	}

	// Notify observers
	stateSlice := make([]SimState, 0, len(ss.states))
	for _, s := range ss.states {
		stateSlice = append(stateSlice, *s)
	}

	for _, observer := range ss.observers {
		observer(stateSlice)
	}
}

// GetState retrieves current state for a satellite
func (ss *SiliguriSimulator) GetState(satID int) (SimState, bool) {
	state, exists := ss.states[satID]
	if !exists {
		return SimState{}, false
	}
	return *state, true
}

// GetAllStates returns all current states
func (ss *SiliguriSimulator) GetAllStates() []SimState {
	states := make([]SimState, 0, len(ss.states))
	for _, s := range ss.states {
		states = append(states, *s)
	}
	return states
}

// Reset clears all simulation states
func (ss *SiliguriSimulator) Reset() {
	ss.states = make(map[int]*SimState)
	ss.simTime = 0.0
}

// BehavioralProfile defines adversary behavior patterns
type BehavioralProfile struct {
	Intent            Intent
	OrbitType         string  // CIRCUMNAVIGATION, NADIR_LOCK, INTERCEPT
	ThrustMagnitude   float64 // m/s²
	ManeuverFrequency float64 // Hz (maneuvers per second)
	ThermalVariance   float64 // IR signature variance
	PowerOutput       float64 // dBm (for EW emitters)
}

// CurriculumLevel defines difficulty for curriculum learning
type CurriculumLevel int

const (
	LevelEasy   CurriculumLevel = 0 // Clean, bright, no debris
	LevelMedium CurriculumLevel = 1 // Star fields + noise
	LevelHard   CurriculumLevel = 2 // Dense debris, dim sleepers
	LevelTactic CurriculumLevel = 3 // Intent-labeled adversaries
)

// GenerateCurriculumPass creates training data for specific curriculum level
func (ss *SiliguriSimulator) GenerateCurriculumPass(level CurriculumLevel) []SimState {
	switch level {
	case LevelEasy:
		// Clean pass: just the target, no debris, bright
		return ss.GenerateSiliguriPass(0, true)

	case LevelMedium:
		// Add some debris and star background
		return ss.GenerateSiliguriPass(20, true)

	case LevelHard:
		// Dense debris, dim "sleeper" satellites
		return ss.GenerateSiliguriPass(100, true)

	case LevelTactic:
		// Extreme conditions: radiation hits, many debris
		return ss.GenerateSiliguriPass(200, true)

	default:
		return ss.GenerateSiliguriPass(0, true)
	}
}

// ToJSON exports simulation states to JSON-serializable format
func (ss *SiliguriSimulator) ToJSON() []map[string]interface{} {
	result := make([]map[string]interface{}, 0, len(ss.states))

	for _, state := range ss.states {
		result = append(result, map[string]interface{}{
			"sat_id":   state.SatID,
			"x":        state.Position[0],
			"y":        state.Position[1],
			"z":        state.Position[2],
			"vx":       state.Velocity[0],
			"vy":       state.Velocity[1],
			"vz":       state.Velocity[2],
			"mag":      state.Magnitude,
			"type":     state.Type,
			"intent":   state.Intent,
			"thrust":   state.Thrust,
			"delta_v":  state.DeltaV,
			"sim_time": ss.simTime,
		})
	}

	return result
}

// GenerateIntentScenario creates adversaries with specific intent profiles
func (ss *SiliguriSimulator) GenerateIntentScenario() []SimState {
	states := make([]SimState, 0)

	// 1. RPO Inspector (Circumnavigation behavior)
	rpoState := ss.generateRPOState(99998)
	states = append(states, rpoState)
	ss.states[99998] = &states[len(states)-1]

	// 2. Electronic Warfare (Nadir-pointing, high power)
	ewState := ss.generateEWState(99997)
	states = append(states, ewState)
	ss.states[99997] = &states[len(states)-1]

	// 3. Kinetic Threat (High-thrust intercept arc)
	kineticState := ss.generateKineticState(99996)
	states = append(states, kineticState)
	ss.states[99996] = &states[len(states)-1]

	// 4. Passive (Standard satellite)
	passiveState := ss.generatePassiveState(99995)
	states = append(states, passiveState)
	ss.states[99995] = &states[len(states)-1]

	return states
}

// generateRPOState creates RPO Inspection (HCW-locked circumnavigation) satellite
func (ss *SiliguriSimulator) generateRPOState(satID int) SimState {
	// Hill-Clohessy-Wiltshire relative orbit around Siliguri
	latRad := SiliguriLat * math.Pi / 180.0
	lonRad := SiliguriLon * math.Pi / 180.0

	r := EarthRadiusKm + LEOAltitudeKm

	// HCW orbit parameters (relative motion)
	x0 := 0.5                                             // km (radial offset)
	z0 := 1.0                                             // km (cross-track)
	meanMotion := math.Sqrt(MuM3S2 / math.Pow(r*1000, 3)) // rad/s

	// Phase for elliptical circumnavigation
	phase := meanMotion * ss.simTime

	// HCW equations (simplified)
	x := x0 * math.Cos(phase)
	y := 2.0 * x0 * math.Sin(phase) // Along-track
	z := z0 * math.Sin(phase)       // Cross-track

	// Absolute position (Siliguri reference + relative)
	refPos := [3]float64{
		r * math.Cos(latRad) * math.Cos(lonRad),
		r * math.Cos(latRad) * math.Sin(lonRad),
		r * math.Sin(latRad),
	}

	position := [3]float64{
		refPos[0] + x,
		refPos[1] + y,
		refPos[2] + z,
	}

	// Velocity from HCW equations
	vx := -x0 * meanMotion * 1000 * math.Sin(phase)
	vy := 2.0 * x0 * meanMotion * 1000 * math.Cos(phase)
	vz := z0 * meanMotion * 1000 * math.Cos(phase)

	velocity := [3]float64{
		refPos[0]/r*7.5 + vx/1000,
		refPos[1]/r*7.5 + vy/1000,
		refPos[2]/r*7.5 + vz/1000,
	}

	// Low thrust for station-keeping (characteristic of RPO)
	thrust := [3]float64{0.001, 0.001, 0.001} // m/s²

	return SimState{
		SatID:     satID,
		Position:  position,
		Velocity:  velocity,
		Magnitude: 4.0,
		Type:      "TARGET",
		Intent:    IntentRPOInspection,
		Thrust:    thrust,
		DeltaV:    0.1, // Low delta-V (station-keeping)
	}
}

// generateEWState creates Electronic Warfare (Nadir-pointing) satellite
func (ss *SiliguriSimulator) generateEWState(satID int) SimState {
	latRad := SiliguriLat * math.Pi / 180.0
	lonRad := SiliguriLon * math.Pi / 180.0

	r := EarthRadiusKm + LEOAltitudeKm - 50.0 // Slightly lower orbit

	// Position directly above Siliguri (Nadir-pointing)
	position := [3]float64{
		r * math.Cos(latRad) * math.Cos(lonRad),
		r * math.Cos(latRad) * math.Sin(lonRad),
		r * math.Sin(latRad),
	}

	// Orbital velocity
	v := math.Sqrt(MuM3S2/(r*1000)) / 1000
	velocity := [3]float64{
		-v * math.Sin(lonRad),
		v * math.Cos(lonRad),
		0.0,
	}

	// Thermal jitter signature (random thrust perturbations)
	thermalJitter := [3]float64{
		ss.rng.NormFloat64() * 0.01,
		ss.rng.NormFloat64() * 0.01,
		ss.rng.NormFloat64() * 0.005,
	}

	return SimState{
		SatID:     satID,
		Position:  position,
		Velocity:  velocity,
		Magnitude: 5.0 + ss.rng.Float64()*2.0, // Variable brightness (thermal)
		Type:      "TARGET",
		Intent:    IntentElectronicWarfare,
		Thrust:    thermalJitter,
		DeltaV:    5.0, // Periodic station-keeping burns
	}
}

// generateKineticState creates Kinetic Threat (High-thrust intercept)
func (ss *SiliguriSimulator) generateKineticState(satID int) SimState {
	latRad := SiliguriLat * math.Pi / 180.0
	lonRad := SiliguriLon * math.Pi / 180.0

	// Starting from slightly offset orbit
	offsetLat := SiliguriLat + 5.0
	offsetLon := SiliguriLon - 10.0
	latRadOffset := offsetLat * math.Pi / 180.0
	lonRadOffset := offsetLon * math.Pi / 180.0

	r := EarthRadiusKm + LEOAltitudeKm

	position := [3]float64{
		r * math.Cos(latRadOffset) * math.Cos(lonRadOffset),
		r * math.Cos(latRadOffset) * math.Sin(lonRadOffset),
		r * math.Sin(latRadOffset),
	}

	// High-thrust intercept trajectory (not Keplerian)
	// Directed toward Siliguri
	targetPos := [3]float64{
		r * math.Cos(latRad) * math.Cos(lonRad),
		r * math.Cos(latRad) * math.Sin(lonRad),
		r * math.Sin(latRad),
	}

	// Velocity vector toward target (intercept arc)
	dx := targetPos[0] - position[0]
	dy := targetPos[1] - position[1]
	dz := targetPos[2] - position[2]
	dist := math.Sqrt(dx*dx + dy*dy + dz*dz)

	// High velocity toward target (10 km/s, faster than LEO circular)
	interceptSpeed := 10.0
	velocity := [3]float64{
		dx / dist * interceptSpeed,
		dy / dist * interceptSpeed,
		dz / dist * interceptSpeed,
	}

	// High thrust for intercept
	thrust := [3]float64{
		dx / dist * 0.5, // 0.5 m/s² thrust toward target
		dy / dist * 0.5,
		dz / dist * 0.5,
	}

	return SimState{
		SatID:     satID,
		Position:  position,
		Velocity:  velocity,
		Magnitude: 3.0, // Bright (high albedo from solar panels)
		Type:      "TARGET",
		Intent:    IntentKineticThreat,
		Thrust:    thrust,
		DeltaV:    500.0, // Large recent delta-V (intercept burn)
	}
}

// generatePassiveState creates Passive (standard) satellite
func (ss *SiliguriSimulator) generatePassiveState(satID int) SimState {
	latRad := SiliguriLat * math.Pi / 180.0
	lonRad := SiliguriLon * math.Pi / 180.0

	r := EarthRadiusKm + LEOAltitudeKm + ss.rng.NormFloat64()*20.0

	position := [3]float64{
		r * math.Cos(latRad) * math.Cos(lonRad),
		r * math.Cos(latRad) * math.Sin(lonRad),
		r * math.Sin(latRad),
	}

	v := math.Sqrt(MuM3S2/(r*1000)) / 1000
	velocity := [3]float64{
		-v * math.Sin(lonRad),
		v * math.Cos(lonRad),
		ss.rng.NormFloat64() * 0.01,
	}

	return SimState{
		SatID:     satID,
		Position:  position,
		Velocity:  velocity,
		Magnitude: 6.0 + ss.rng.Float64()*2.0,
		Type:      "TARGET",
		Intent:    IntentPassive,
		Thrust:    [3]float64{0, 0, 0},
		DeltaV:    0.0,
	}
}

// PropagateWithThrust advances simulation with non-Keplerian thrust
func (ss *SiliguriSimulator) PropagateWithThrust(dt float64) {
	ss.simTime += dt

	for _, state := range ss.states {
		// Convert to meters
		x := state.Position[0] * 1000
		y := state.Position[1] * 1000
		z := state.Position[2] * 1000
		vx := state.Velocity[0] * 1000
		vy := state.Velocity[1] * 1000
		vz := state.Velocity[2] * 1000

		// Gravitational acceleration
		r := math.Sqrt(x*x + y*y + z*z)
		ax := -MuM3S2 * x / (r * r * r)
		ay := -MuM3S2 * y / (r * r * r)
		az := -MuM3S2 * z / (r * r * r)

		// Add non-Keplerian thrust (intent-based behavior)
		ax += state.Thrust[0]
		ay += state.Thrust[1]
		az += state.Thrust[2]

		// Update velocity
		vx += ax * dt
		vy += ay * dt
		vz += az * dt

		// Update position
		x += vx * dt
		y += vy * dt
		z += vz * dt

		state.Position = [3]float64{x / 1000, y / 1000, z / 1000}
		state.Velocity = [3]float64{vx / 1000, vy / 1000, vz / 1000}
	}

	// Notify observers
	stateSlice := make([]SimState, 0, len(ss.states))
	for _, s := range ss.states {
		stateSlice = append(stateSlice, *s)
	}

	for _, observer := range ss.observers {
		observer(stateSlice)
	}
}

// GetIntentProfileSummary returns summary of simulated intents
func (ss *SiliguriSimulator) GetIntentProfileSummary() map[string]int {
	summary := make(map[string]int)

	for _, state := range ss.states {
		intentName := IntentNames[state.Intent]
		summary[intentName]++
	}

	return summary
}

// EnableCoordinateRandomization enables ECI frame rotation for training augmentation
func (ss *SiliguriSimulator) EnableCoordinateRandomization(enabled bool) {
	ss.coordinateRandomization = enabled
}

// ApplyCoordinateRandomization rotates all states by random ECI rotation
// Prevents sun-synchronous overfitting by varying lighting conditions
func (ss *SiliguriSimulator) ApplyCoordinateRandomization() {
	if !ss.coordinateRandomization {
		return
	}

	// Generate random rotation axis and angle
	phi := ss.rng.Float64() * 2 * math.Pi // Random angle 0-2π
	cosPhi := math.Cos(phi)
	sinPhi := math.Sin(phi)

	// Random axis (normalize random vector)
	ux := ss.rng.NormFloat64()
	uy := ss.rng.NormFloat64()
	uz := ss.rng.NormFloat64()
	norm := math.Sqrt(ux*ux + uy*uy + uz*uz)
	ux /= norm
	uy /= norm
	uz /= norm

	// Rotation matrix (Rodrigues' rotation formula)
	// R = I + sin(φ)K + (1-cos(φ))K² where K is skew-symmetric matrix
	oneMinusCos := 1.0 - cosPhi
	kx := ux
	ky := uy
	kz := uz

	// Apply rotation to all states
	for _, state := range ss.states {
		x, y, z := state.Position[0], state.Position[1], state.Position[2]
		vx, vy, vz := state.Velocity[0], state.Velocity[1], state.Velocity[2]

		// Rotate position
		state.Position[0] = (cosPhi+kx*kx*oneMinusCos)*x + (kx*ky*oneMinusCos-kz*sinPhi)*y + (kx*kz*oneMinusCos+ky*sinPhi)*z
		state.Position[1] = (ky*kx*oneMinusCos+kz*sinPhi)*x + (cosPhi+ky*ky*oneMinusCos)*y + (ky*kz*oneMinusCos-kx*sinPhi)*z
		state.Position[2] = (kz*kx*oneMinusCos-ky*sinPhi)*x + (kz*ky*oneMinusCos+kx*sinPhi)*y + (cosPhi+kz*kz*oneMinusCos)*z

		// Rotate velocity
		state.Velocity[0] = (cosPhi+kx*kx*oneMinusCos)*vx + (kx*ky*oneMinusCos-kz*sinPhi)*vy + (kx*kz*oneMinusCos+ky*sinPhi)*vz
		state.Velocity[1] = (ky*kx*oneMinusCos+kz*sinPhi)*vx + (cosPhi+ky*ky*oneMinusCos)*vy + (ky*kz*oneMinusCos-kx*sinPhi)*vz
		state.Velocity[2] = (kz*kx*oneMinusCos-ky*sinPhi)*vx + (kz*ky*oneMinusCos+kx*sinPhi)*vy + (cosPhi+kz*kz*oneMinusCos)*vz
	}
}

// ApplyFeatureMasking randomly masks altitude feature to force reliance on relative metrics
// Prevents sun-synchronous overfitting by forcing agent to use HCW residuals and relative velocity
func (ss *SiliguriSimulator) ApplyFeatureMasking(states []SimState, maskAltitude bool, maskIntent bool) []map[string]interface{} {
	result := make([]map[string]interface{}, 0, len(states))

	for _, state := range states {
		stateData := map[string]interface{}{
			"sat_id":   state.SatID,
			"x":        state.Position[0],
			"y":        state.Position[1],
			"z":        state.Position[2],
			"vx":       state.Velocity[0],
			"vy":       state.Velocity[1],
			"vz":       state.Velocity[2],
			"mag":      state.Magnitude,
			"type":     state.Type,
			"intent":   state.Intent,
			"thrust":   state.Thrust,
			"delta_v":  state.DeltaV,
			"sim_time": ss.simTime,
		}

		// Mask altitude (derived from position magnitude) if requested
		if maskAltitude {
			// Zero out radial component to prevent altitude inference
			stateData["z"] = 0.0
			stateData["altitude_masked"] = true
		}

		// Mask intent to force reliance on physical signatures
		if maskIntent {
			stateData["intent"] = -1 // Unknown intent
			stateData["intent_masked"] = true
		}

		result = append(result, stateData)
	}

	return result
}
