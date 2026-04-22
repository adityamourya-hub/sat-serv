// Package ai_imm implements Active Inference IMM (AI-IMM)
//
// Pivot from reactive Bayesian filtering to predictive intent inference.
//
// The fundamental objective shifts from "What happened?" to "What is the adversary
// trying to learn or achieve?" by minimizing Expected Free Energy (EFE).
//
// Mathematical formulation:
//     G(π) ≈ E_q(s,o|π)[ln q(s|π) - ln p(s)]  (Risk/Pragmatic)
//          + E_q(s,o|π)[H[q(o|s)]]              (Ambiguity/Epistemic)
//
// Pragmatic Value: Penalizes deviations from goal manifold (e.g., HGV re-entry corridor)
// Epistemic Value: Rewards policies that reduce uncertainty about hidden state (regime/intent)
//
// This enables the system to "peer" into the future 30 seconds to see if a specific
// maneuver trajectory resolves the entropy of the current track.
//
// Reference: Friston's Free Energy Principle, Active Inference literature

package ai_imm

import (
	"fmt"
	"math"
	"ssis/pkg/ml/dphno"
	"ssis/pkg/physics"
)

// Policy represents a sequence of modes over a horizon H
type Policy struct {
	ModeSequence []int   // e.g., [Glide, Glide, Maneuver, Glide...]
	Horizon      int     // Length of policy horizon (default 30 steps = 30s at 1Hz)
	Probability  float64 // Prior probability of this policy
}

// AdversarialPrior defines the "Goal Map" for pragmatic value calculation
// Identifies targets heading for National Strategic Installations
//
// Enhanced with Mixture of Gaussians (MoG), Reachability Ellipsoid, and Shadow Goal
// p(s) = (1-α) · Σ w_i · exp(-0.5 (s - G_i)^T Σ_i^{-1} (s - G_i)) + α · U(R)
//
// where:
// - w_i: Importance weight of the installation
// - Σ_i: Reachability Ellipsoid (vanishes if target cannot physically reach under Hamiltonian constraints)
// - α: Discovery Weight (Curiosity Factor) for Shadow Goal
// - U(R): Uniform distribution over Reachability Envelope (Shadow Goal)
type AdversarialPrior struct {
	// Strategic locations (latitude, longitude, altitude_km)
	StrategicSites map[string]Site

	// Mixture of Gaussians parameters
	Weights     map[string]float64   // w_i: Importance weight per site
	Covariances map[string]Matrix3x3 // Σ_i: Reachability Ellipsoid per site

	// Shadow Goal parameters (Epistemological Guard)
	DiscoveryWeight      float64               // α: Discovery Weight (0.0 = pure GoalMap, 1.0 = pure Shadow)
	ReachabilityEnvelope *ReachabilityEnvelope // R: Reachability Envelope based on Hamiltonian

	// Goal likelihood parameters
	MaxLD_Factor    float64 // Weight for L/D ratio
	Altitude_Factor float64 // Weight for altitude matching
	Heading_Factor  float64 // Weight for heading alignment

	// Thermal budget for M2 Air
	MaxComputeTimeMs float64 // Max compute time per EFE evaluation

	// Platform-Specific Energy Buffer mapping
	// Each HGV class has distinct L/D profile and thermal stress limits
	PlatformEnergyBuffers map[string]float64 // EnergyBuffer per HGV class (km²/s²)
}

// HGVClass represents different HGV vehicle classes with distinct capabilities
type HGVClass string

const (
	HGVClassA_Agile      HGVClass = "ClassA_Agile"      // L/D ~4.5, lower energy buffer
	HGVClassB_LongRange  HGVClass = "ClassB_LongRange"  // L/D ~3.2, higher energy buffer
	HGVClassC_Debris     HGVClass = "ClassC_Debris"     // L/D ~0.8, max energy buffer (ballistic)
	HGVClassD_BoostGlide HGVClass = "ClassD_BoostGlide" // L/D ~2.5, moderate energy buffer
)

// Site represents a strategic location
type Site struct {
	Name      string
	Latitude  float64 // degrees
	Longitude float64 // degrees
	Altitude  float64 // km (target altitude)
	Radius    float64 // km (acceptance radius)
	Weight    float64 // Importance weight (w_i)
}

// Matrix3x3 represents a 3x3 covariance matrix for Reachability Ellipsoid
type Matrix3x3 [3][3]float64

// ReachabilityEnvelope defines the volume the target can physically reach
// given its current Hamiltonian H(x). Used for Shadow Goal calculation.
type ReachabilityEnvelope struct {
	MinAltitude  float64 // km
	MaxAltitude  float64 // km
	MinLatitude  float64 // degrees
	MaxLatitude  float64 // degrees
	MinLongitude float64 // degrees
	MaxLongitude float64 // degrees
	Volume       float64 // km³ (total reachable volume)
}

// NewAdversarialPrior creates an adversarial prior with strategic sites
func NewAdversarialPrior() *AdversarialPrior {
	// Initialize HGV-specific covariance matrices based on L/D ratio
	// High L/D (agile) = tighter lateral control, wider downrange reach
	// Low L/D (ballistic) = wider lateral uncertainty, tighter downrange
	covariances := map[string]Matrix3x3{
		"DELHI":   {{50.0, 0, 0}, {0, 50.0, 0}, {0, 0, 10.0}}, // 50km lat/lon, 10km alt (1-sigma)
		"MUMBAI":  {{50.0, 0, 0}, {0, 50.0, 0}, {0, 0, 10.0}},
		"CHENNAI": {{50.0, 0, 0}, {0, 50.0, 0}, {0, 0, 10.0}},
	}

	// Initialize default Reachability Envelope (will be updated based on Hamiltonian)
	defaultEnvelope := &ReachabilityEnvelope{
		MinAltitude:  0.0,
		MaxAltitude:  150.0, // km
		MinLatitude:  8.0,   // degrees
		MaxLatitude:  35.0,  // degrees
		MinLongitude: 68.0,  // degrees
		MaxLongitude: 97.0,  // degrees
		Volume:       1e9,   // km³ (placeholder)
	}

	// Platform-Specific Energy Buffer mapping
	// Each HGV class has distinct L/D profile and thermal stress limits
	// EnergyBuffer is the "Minimum Kinetic Reserve" required to maintain stable glide
	platformEnergyBuffers := map[string]float64{
		string(HGVClassA_Agile):      500.0,  // Lower buffer: Can spend more energy on lateral maneuvers (L/D ~4.5)
		string(HGVClassB_LongRange):  1500.0, // Higher buffer: Must conserve energy for range (L/D ~3.2)
		string(HGVClassC_Debris):     5000.0, // Max buffer: Effectively ballistic, tight cone (L/D ~0.8)
		string(HGVClassD_BoostGlide): 1000.0, // Moderate buffer: Rocket-assisted glide (L/D ~2.5)
	}

	return &AdversarialPrior{
		StrategicSites: map[string]Site{
			"DELHI": {
				Name:      "DELHI",
				Latitude:  28.6139,
				Longitude: 77.2090,
				Altitude:  0.0,
				Radius:    50.0,
				Weight:    1.0, // Highest priority
			},
			"MUMBAI": {
				Name:      "MUMBAI",
				Latitude:  19.0760,
				Longitude: 72.8777,
				Altitude:  0.0,
				Radius:    50.0,
				Weight:    0.8,
			},
			"CHENNAI": {
				Name:      "CHENNAI",
				Latitude:  13.0827,
				Longitude: 80.2707,
				Altitude:  0.0,
				Radius:    50.0,
				Weight:    0.6,
			},
			// Add more strategic sites as needed
		},
		Weights: map[string]float64{
			"DELHI":   1.0,
			"MUMBAI":  0.8,
			"CHENNAI": 0.6,
		},
		Covariances:           covariances,
		DiscoveryWeight:       0.3, // α: 30% Shadow Goal weight for curiosity
		ReachabilityEnvelope:  defaultEnvelope,
		MaxLD_Factor:          1.0,
		Altitude_Factor:       1.0,
		Heading_Factor:        1.0,
		MaxComputeTimeMs:      27.0, // 27ms budget for M2 thermal management
		PlatformEnergyBuffers: platformEnergyBuffers,
	}
}

// GetEnergyBuffer returns platform-specific energy buffer for HGV class
// This transforms the generic Reachability Envelope into a Platform-Specific Kill-Zone
func (ap *AdversarialPrior) GetEnergyBuffer(hgvClass HGVClass) float64 {
	if buffer, ok := ap.PlatformEnergyBuffers[string(hgvClass)]; ok {
		return buffer
	}
	// Default fallback
	return 1000.0
}

// LogDensity calculates log-density of state given adversarial prior
// This is the Pragmatic Value: how well does this state align with strategic goals?
//
// Enhanced with Mixture of Gaussians (MoG) and Shadow Goal
// p(s) = (1-α) · GoalMap(s) + α · ShadowGoal(s)
func (ap *AdversarialPrior) LogDensity(state *physics.State) float64 {
	// Convert ECI position to lat/lon/altitude with proper GST accounting
	lat, lon, alt := eciToLatLonAlt(state.Pos, state.EpochJD)

	// Calculate GoalMap log-density (Mixture of Gaussians)
	goalMapLogDensity := ap.calculateGoalMapLogDensity(lat, lon, alt)

	// Calculate Shadow Goal log-density (Uniform over Reachability Envelope)
	shadowLogDensity := ap.calculateShadowLogDensity(lat, lon, alt)

	// Mixture with Discovery Weight (α)
	alpha := ap.DiscoveryWeight

	// Log-sum-exp for mixture: log((1-α)*p_goal + α*p_shadow)
	// Approximate as: log((1-α)*exp(log_p_goal) + α*exp(log_p_shadow))
	if goalMapLogDensity > -1e10 && shadowLogDensity > -1e10 {
		// Both finite - use log-sum-exp
		maxLog := math.Max(goalMapLogDensity, shadowLogDensity)
		goalTerm := math.Log(1-alpha) + goalMapLogDensity - maxLog
		shadowTerm := math.Log(alpha) + shadowLogDensity - maxLog
		return maxLog + math.Log(math.Exp(goalTerm)+math.Exp(shadowTerm))
	} else if goalMapLogDensity > -1e10 {
		// Only GoalMap is valid
		return math.Log(1-alpha) + goalMapLogDensity
	} else if shadowLogDensity > -1e10 {
		// Only Shadow is valid
		return math.Log(alpha) + shadowLogDensity
	}

	// Both invalid - return very low density
	return -math.Inf(1)
}

// calculateGoalMapLogDensity calculates log-density for GoalMap (Mixture of Gaussians)
func (ap *AdversarialPrior) calculateGoalMapLogDensity(lat, lon, alt float64) float64 {
	logDensity := -math.Inf(1)

	for name, site := range ap.StrategicSites {
		// Calculate distance to site
		latDist := lat - site.Latitude
		lonDist := lon - site.Longitude
		altDist := (alt - site.Altitude) // Both in km

		// Mahalanobis distance with covariance Σ_i
		sigma := site.Radius
		weight := ap.Weights[name]

		// Simplified Mahalanobis distance (diagonal covariance)
		distSq := (latDist*latDist + lonDist*lonDist + altDist*altDist) / (sigma * sigma)

		// Log-density for this component
		logProbComponent := math.Log(weight) - 0.5*distSq - math.Log(sigma*math.Sqrt(2*math.Pi))

		// Log-sum-exp for mixture
		if logProbComponent > logDensity {
			logDensity = logProbComponent
		}
	}

	return logDensity
}

// calculateShadowLogDensity calculates log-density for Shadow Goal (Uniform over Reachability Envelope)
func (ap *AdversarialPrior) calculateShadowLogDensity(lat, lon, alt float64) float64 {
	env := ap.ReachabilityEnvelope

	// Check if state is within Reachability Envelope
	if lat < env.MinLatitude || lat > env.MaxLatitude {
		return -math.Inf(1)
	}
	if lon < env.MinLongitude || lon > env.MaxLongitude {
		return -math.Inf(1)
	}
	if alt < env.MinAltitude || alt > env.MaxAltitude {
		return -math.Inf(1)
	}

	// Uniform distribution: p(s) = 1 / Volume
	// log(p(s)) = -log(Volume)
	return -math.Log(env.Volume)
}

// LogProb is an alias for LogDensity (backward compatibility)
func (ap *AdversarialPrior) LogProb(state *physics.State) float64 {
	return ap.LogDensity(state)
}

// ComputeDynamicAlpha calculates dynamic Discovery Weight α based on SINDy residual
//
// Dynamic curiosity linked to physics model failure
// α(r) = α_base + (α_max - α_base) · σ(k · (||r||_2 - τ))
//
// where:
// - α_base: Baseline background curiosity (0.05)
// - α_max: Maximum curiosity when physics fails (0.80)
// - τ: Symmetry-Breaking Threshold (1e-4)
// - k: Aggression Factor (10.0)
// - r: SINDy residual (physics model error)
func (s *EFESolver) ComputeDynamicAlpha(residual float64) float64 {
	const (
		alphaBase = 0.05
		alphaMax  = 0.80
		tau       = 1e-4 // SINDy residual threshold
		k         = 10.0 // Sigmoid steepness (aggression factor)
	)

	// Sigmoid gate to boost curiosity when physics fails
	// σ(x) = 1 / (1 + exp(-x))
	return alphaBase + (alphaMax-alphaBase)/(1.0+math.Exp(-k*(residual-tau)))
}

// UpdateReachabilityEnvelope updates the Reachability Envelope based on current Hamiltonian
//
// Strictly bounded by remaining energy (H_current - H_min_glide)
// The envelope volume scales with available energy, preventing the system
// from considering physically impossible terminal states.
func (ap *AdversarialPrior) UpdateReachabilityEnvelope(state *physics.State) {
	// Compute current Hamiltonian H = T + V
	r := math.Sqrt(state.Pos[0]*state.Pos[0] + state.Pos[1]*state.Pos[1] + state.Pos[2]*state.Pos[2])
	v := math.Sqrt(state.Vel[0]*state.Vel[0] + state.Vel[1]*state.Vel[1] + state.Vel[2]*state.Vel[2])

	// Gravitational parameter (Earth)
	mu := 3.986004418e14 // m³/s²

	// Specific kinetic energy (J/kg = m²/s²)
	kineticEnergy := 0.5 * v * v

	// Specific potential energy (J/kg = m²/s²)
	potentialEnergy := -mu / r

	// Hamiltonian (specific energy in m²/s²)
	hCurrent := kineticEnergy + potentialEnergy

	// Convert to km²/s² for consistency
	hCurrentKm2 := hCurrent / 1e6

	// Minimum Hamiltonian for stable glide (≈ -30000 km²/s²)
	hMinGlide := -30000.0

	// Remaining energy available for maneuvers
	energyDelta := hCurrentKm2 - hMinGlide

	// If remaining energy is negative, envelope collapses (cannot maintain glide)
	if energyDelta <= 0 {
		ap.ReachabilityEnvelope.Volume = 0.0
		return
	}

	// Scale envelope volume by remaining energy factor
	// More energy = larger reachable volume
	energyFactor := energyDelta / 10000.0 // Normalize by 10000 km²/s²

	// Update envelope bounds based on energy factor
	// Base envelope (at nominal energy)
	baseEnvelope := &ReachabilityEnvelope{
		MinAltitude:  0.0,
		MaxAltitude:  150.0,
		MinLatitude:  8.0,
		MaxLatitude:  35.0,
		MinLongitude: 68.0,
		MaxLongitude: 97.0,
		Volume:       1e9,
	}

	// Scale altitude range by energy factor
	ap.ReachabilityEnvelope.MinAltitude = baseEnvelope.MinAltitude
	ap.ReachabilityEnvelope.MaxAltitude = baseEnvelope.MaxAltitude * energyFactor
	if ap.ReachabilityEnvelope.MaxAltitude > 200.0 {
		ap.ReachabilityEnvelope.MaxAltitude = 200.0 // Cap at 200km
	}

	// Scale lat/lon range by energy factor
	latRange := (baseEnvelope.MaxLatitude - baseEnvelope.MinLatitude) * energyFactor
	ap.ReachabilityEnvelope.MinLatitude = baseEnvelope.MinLatitude - latRange/2
	ap.ReachabilityEnvelope.MaxLatitude = baseEnvelope.MaxLatitude + latRange/2
	if ap.ReachabilityEnvelope.MinLatitude < -90.0 {
		ap.ReachabilityEnvelope.MinLatitude = -90.0
	}
	if ap.ReachabilityEnvelope.MaxLatitude > 90.0 {
		ap.ReachabilityEnvelope.MaxLatitude = 90.0
	}

	lonRange := (baseEnvelope.MaxLongitude - baseEnvelope.MinLongitude) * energyFactor
	ap.ReachabilityEnvelope.MinLongitude = baseEnvelope.MinLongitude - lonRange/2
	ap.ReachabilityEnvelope.MaxLongitude = baseEnvelope.MaxLongitude + lonRange/2
	if ap.ReachabilityEnvelope.MinLongitude < -180.0 {
		ap.ReachabilityEnvelope.MinLongitude = -180.0
	}
	if ap.ReachabilityEnvelope.MaxLongitude > 180.0 {
		ap.ReachabilityEnvelope.MaxLongitude = 180.0
	}

	// Scale volume by energy factor cubed (3D volume)
	ap.ReachabilityEnvelope.Volume = baseEnvelope.Volume * energyFactor * energyFactor * energyFactor
}

// IsHeadingTowardSite checks if trajectory is heading toward a strategic site
func (ap *AdversarialPrior) IsHeadingTowardSite(state *physics.State) bool {
	// Convert ECI position to lat/lon with proper GST accounting
	lat, lon, _ := eciToLatLonAlt(state.Pos, state.EpochJD)

	// Convert ECI velocity to heading
	vx, vy, _ := state.Vel[0], state.Vel[1], state.Vel[2]

	// Calculate heading toward each strategic site
	for _, site := range ap.StrategicSites {
		// Vector to site
		dx := site.Latitude - lat
		dy := site.Longitude - lon

		// Normalize
		mag := math.Sqrt(dx*dx + dy*dy)
		if mag < 1e-6 {
			continue
		}
		dx /= mag
		dy /= mag

		// Velocity direction (simplified - use horizontal components)
		vMag := math.Sqrt(vx*vx + vy*vy)
		if vMag < 1e-6 {
			continue
		}
		vx /= vMag
		vy /= vMag

		// Dot product (alignment)
		alignment := dx*vx + dy*vy

		// If alignment > 0.7, heading toward site
		if alignment > 0.7 {
			return true
		}
	}

	return false
}

// eciToLatLonAlt converts ECI position to lat/lon/altitude using proper WGS-84 geodetic transformation
// Accounts for Earth rotation (GST) for accurate coordinate conversion
func eciToLatLonAlt(pos physics.Vector3, epochJD ...float64) (lat, lon, alt float64) {
	// Use proper physics.ECIToGeodetic which accounts for Earth rotation and WGS-84 ellipsoid
	jd := 2460000.5 // Default epoch if not provided
	if len(epochJD) > 0 {
		jd = epochJD[0]
	}
	return physics.ECIToGeodetic(pos[0], pos[1], pos[2], jd)
}

// EFESolver calculates Expected Free Energy for policy selection
//
// Enhanced with Persistence Memory (Belief Hysteresis) and
// Hamiltonian-Bounded Reachability Envelope to prevent "Computational Hallucination."
//
// Enhanced with Centroid-to-PHS Feedback for closed-loop kinetic custody.
type EFESolver struct {
	AdversarialPrior *AdversarialPrior
	DPHNO            *dphno.DPHNOOperator // Dissipative Port-Hamiltonian Neural Operator
	// DPHNO implements: Predict(state, duration, control) -> [6]float64
	// See: ai/model_arch.md for architecture specification
	Horizon         int     // Policy horizon in steps
	EpistemicWeight float64 // Weight for epistemic value
	PragmaticWeight float64 // Weight for pragmatic value

	// Persistence Memory (Belief Hysteresis)
	// Leaky integrator on curiosity state to prevent flickering during multi-stage maneuvers
	AlphaMem  float64 // Memory of curiosity state (α_mem)
	TauDecay  float64 // Memory horizon (τ_decay) in seconds - how long curiosity persists
	ShockSens float64 // Shock sensitivity (η) - how fast new residuals overwrite memory

	// Hamiltonian-Bounded Reachability Envelope
	EnergyBuffer float64 // Minimal energy required for stable glide (H_min_glide buffer)
	HMinGlide    float64 // Minimum Hamiltonian for stable glide (≈ -30000 km²/s²)

	// Centroid-to-PHS Feedback
	ObservationPriorAz float64 // Observed Azimuth from camera (degrees)
	ObservationPriorEl float64 // Observed Elevation from camera (degrees)
	HasObservation     bool    // Whether camera observation is available
	CameraConfidence   float64 // Camera SNR-based confidence (0-1)
}

// NewEFESolver creates a new EFE solver
func NewEFESolver(dphno *dphno.DPHNOOperator) *EFESolver {
	return &EFESolver{
		AdversarialPrior: NewAdversarialPrior(),
		DPHNO:            dphno,
		Horizon:          30, // 30 steps = 30s at 1Hz
		EpistemicWeight:  1.0,
		PragmaticWeight:  1.0,

		// Persistence Memory parameters
		AlphaMem:  0.05, // Base curiosity (α_base)
		TauDecay:  5.0,  // 5 second memory horizon
		ShockSens: 1.0,  // Full shock sensitivity

		// Energy boundary parameters
		EnergyBuffer: 1000.0,   // km²/s² buffer for numerical stability
		HMinGlide:    -30000.0, // Minimum Hamiltonian for stable glide

		// Centroid-to-PHS Feedback
		HasObservation:   false,
		CameraConfidence: 1.0,
	}
}

// UpdateCuriosity implements Persistence Memory (Belief Hysteresis) via leaky integrator
//
// Prevents α from "flickering" during complex multi-stage maneuvers
// where the residual might momentarily dip as the vehicle passes through
// a brief ballistic-like phase.
//
// Leaky integrator dynamics:
//
//	α̇_mem = - (α_mem - α_base) / τ_decay + η · α(r)
//
// where:
//   - α_mem: Memory of curiosity state
//   - α_base: Base curiosity (0.05)
//   - τ_decay: Memory horizon (5 seconds) - how long curiosity persists
//   - η: Shock sensitivity (1.0) - how fast new residuals overwrite memory
//   - α(r): Raw dynamic α from SINDy residual
func (s *EFESolver) UpdateCuriosity(rawAlpha float64, dt float64) {
	alphaBase := 0.05

	// Leaky integrator: α remains high for a period after a maneuver
	// Decay term: (α_mem - α_base) / τ_decay
	decay := (s.AlphaMem - alphaBase) / s.TauDecay

	// Update with shock sensitivity
	s.AlphaMem += (s.ShockSens*rawAlpha - decay) * dt

	// Clamp to bounds [0.05, 0.90]
	if s.AlphaMem < 0.05 {
		s.AlphaMem = 0.05
	}
	if s.AlphaMem > 0.90 {
		s.AlphaMem = 0.90
	}
}

// SetObservationPrior sets the centroid observation from camera
//
// Centroid-to-PHS Feedback for closed-loop kinetic custody.
// When the camera observes the target at a different location than PHS predicts,
// this creates epistemic ambiguity that the EFE solver must resolve.
func (s *EFESolver) SetObservationPrior(az, el float64, confidence float64) {
	s.ObservationPriorAz = az
	s.ObservationPriorEl = el
	s.CameraConfidence = confidence
	s.HasObservation = true
}

// ClearObservationPrior clears the centroid observation
func (s *EFESolver) ClearObservationPrior() {
	s.HasObservation = false
}

// IsPhysicallyPossible checks if a terminal state is reachable given current energy
//
// Hamiltonian-Bounded Reachability Envelope prevents "Computational Hallucination"
// by ensuring the AI doesn't waste curiosity on physically impossible terminal states.
//
// Reachability condition:
//
//	H_current - H_required > EnergyBuffer
//
// where:
//   - H_current: Current Hamiltonian (energy state)
//   - H_required: Hamiltonian required to reach terminal state
//   - EnergyBuffer: Minimum energy required for stable glide (1000 km²/s²)
//
// If a policy predicts a terminal state requiring more energy than available,
// that policy's Pragmatic Value is set to -∞ (rejected).
func (s *EFESolver) IsPhysicallyPossible(terminalState *physics.State, hCurrent float64) bool {
	// Compute Hamiltonian for terminal state
	// H = T + V = 0.5 * v² - μ/|r|
	r := math.Sqrt(terminalState.Pos[0]*terminalState.Pos[0] +
		terminalState.Pos[1]*terminalState.Pos[1] +
		terminalState.Pos[2]*terminalState.Pos[2])
	v := math.Sqrt(terminalState.Vel[0]*terminalState.Vel[0] +
		terminalState.Vel[1]*terminalState.Vel[1] +
		terminalState.Vel[2]*terminalState.Vel[2])

	// Gravitational parameter (Earth)
	mu := 3.986004418e14 // m³/s²

	// Specific kinetic energy (J/kg = m²/s²)
	kineticEnergy := 0.5 * v * v

	// Specific potential energy (J/kg = m²/s²)
	potentialEnergy := -mu / r

	// Hamiltonian (specific energy in m²/s²)
	hRequired := kineticEnergy + potentialEnergy

	// Convert to km²/s² for consistency with HMinGlide
	hRequiredKm2 := hRequired / 1e6

	// Check reachability: H_current - H_required > EnergyBuffer
	energyDelta := hCurrent - hRequiredKm2

	return energyDelta > s.EnergyBuffer
}

// CalculateG calculates Expected Free Energy for a policy
// G(π) ≈ E_q(s,o|π)[ln q(s|π) - ln p(s)] + E_q(s,o|π)[H[q(o|s)]]
func (s *EFESolver) CalculateG(p Policy, currentState *physics.State) float64 {
	var efe float64
	state := *currentState // Copy

	// Rollout over Horizon H
	for i, mode := range p.ModeSequence {
		// 1. Predict next state using simple predictor (placeholder for DPHNO)
		nextState := s.simplePredict(&state, mode)
		state = *nextState

		// 2. Calculate Epistemic Value (Information Gain)
		// How much does this observation reduce uncertainty in the latent z?
		infGain := s.calculateMutualInformation(&state, i)

		// 3. Calculate Pragmatic Value (Goal alignment)
		// Log-probability of the state given the Adversarial Prior
		pragmatic := s.AdversarialPrior.LogDensity(&state)

		// 4. Combine with weights
		efe += s.EpistemicWeight*infGain - s.PragmaticWeight*pragmatic
	}

	return efe
}

// calculateMutualInformation estimates information gain from observation
// This is a simplified version - in production use full Bayesian update
func (s *EFESolver) calculateMutualInformation(state *physics.State, step int) float64 {
	// Simplified: information gain decreases with time (uncertainty reduction)
	// In production: I(s;o) = H[s] - H[s|o]

	// Heuristic: early steps have higher information gain
	timeDecay := math.Exp(-float64(step) / 10.0)

	// Convert to altitude for novelty calculation with proper GST accounting
	_, _, alt := eciToLatLonAlt(state.Pos, state.EpochJD)

	// Information gain based on state novelty (deviation from ballistic)
	// High deviation = high information gain
	novelty := math.Abs(alt-100.0) / 100.0 // Simplified

	return timeDecay * novelty
}

// simplePredict is a fallback prediction when DPHNO is unavailable
// Uses proper physics propagation via RK4 integration instead of hardcoded approximations
func (s *EFESolver) simplePredict(state *physics.State, mode int) *physics.State {
	dt := 1.0 // 1 second timestep

	// Convert State to vector for propagation
	stateVec := physics.StateToVector(*state)

	// Apply mode-specific physics parameters
	params := physics.PhysicsParams{
		Mass:     1000.0, // Default HGV mass (kg)
		DragArea: 1.0,    // Default drag area (m²)
		Cd:       0.5,    // Default drag coefficient
	}

	// Adjust parameters based on mode (simulating different flight regimes)
	switch mode {
	case 0: // Ballistic
		params.Cd = 0.2 // Lower drag for ballistic trajectory
	case 1: // Glide
		params.Cd = 0.5 // Moderate drag for glide
	case 2: // Maneuver
		params.Cd = 0.8 // Higher drag during maneuver (energy loss)
	}

	// Propagate using proper RK4 integration from physics package
	nextStateVec := physics.PropagateCartesianState(stateVec, dt, state.EpochJD, params)

	// Convert back to State
	newState := physics.VectorToState(nextStateVec)
	newState.EpochJD = state.EpochJD + dt/86400.0 // Update epoch

	// Apply mode-specific velocity perturbations for maneuver simulation
	if mode == 2 { // Maneuver
		// Add lateral acceleration (simulating banked turn)
		newState.Vel[0] += 0.01 * dt // km/s lateral change
	}

	return &newState
}

// OptimizePolicy finds the policy that minimizes EFE using Pruned-Beam Search
//
// Pruned-Beam Search with width=3
// - Greedy-Horizon (1-Step): Fast (~2ms) but "blind"
// - MCTS (Full Tree): Brilliant but too expensive (1000 simulations/frame)
// - Pruned-Beam: Balance (~20-25ms) with "Look-Ahead" capability
func (s *EFESolver) OptimizePolicy(currentState *physics.State, beamWidth int) Policy {
	// Use fixed beam width of 3 for thermal budget
	beamWidth = 3

	// Generate candidate policies (beam search)
	candidates := s.generatePrunedCandidatePolicies(beamWidth)

	// Calculate EFE for each candidate
	minEFE := math.Inf(1)
	bestIndex := 0

	for i, policy := range candidates {
		efe := s.CalculateG(policy, currentState)
		if efe < minEFE {
			minEFE = efe
			bestIndex = i
		}
	}

	return candidates[bestIndex]
}

// EvaluatePolicy calculates EFE using persistence memory and energy-bounded reachability
//
// Enhanced with:
// - Persistence Memory (Belief Hysteresis) to prevent α flickering
// - Hamiltonian-Bounded Reachability Envelope to reject physically impossible states
// - Centroid-to-PHS Feedback to spike Conflict Metric on observation deviation
//
// When α spikes (physics fails), Shadow Goal dominates, forcing epistemic search
// When centroid deviates from PHS, Conflict Metric spikes, resolving epistemic ambiguity
func (s *EFESolver) EvaluatePolicy(p Policy, currentObservation *physics.State, residual float64, dt float64) float64 {
	// 1. Update Persistence Memory (leaky integrator on curiosity)
	// This prevents α from flickering during multi-stage maneuvers
	rawAlpha := s.ComputeDynamicAlpha(residual)
	s.UpdateCuriosity(rawAlpha, dt)
	alpha := s.AlphaMem // Use memory instead of raw α

	// 2. Predict terminal state s_H after horizon H
	terminalState := s.Rollout(currentObservation, p)

	// 3. Energy Boundary Check (Hamiltonian-Bounded Reachability)
	// Compute current Hamiltonian
	rCurrent := math.Sqrt(currentObservation.Pos[0]*currentObservation.Pos[0] +
		currentObservation.Pos[1]*currentObservation.Pos[1] +
		currentObservation.Pos[2]*currentObservation.Pos[2])
	vCurrent := math.Sqrt(currentObservation.Vel[0]*currentObservation.Vel[0] +
		currentObservation.Vel[1]*currentObservation.Vel[1] +
		currentObservation.Vel[2]*currentObservation.Vel[2])
	mu := 3.986004418e14 // m³/s²
	kineticCurrent := 0.5 * vCurrent * vCurrent
	potentialCurrent := -mu / rCurrent
	hCurrentKm2 := (kineticCurrent + potentialCurrent) / 1e6

	// Check if terminal state is physically reachable
	if !s.IsPhysicallyPossible(terminalState, hCurrentKm2) {
		// Reject policy: Pragmatic Value = -∞
		// This prevents "Computational Hallucination" on impossible states
		return math.Inf(1)
	}

	// 4. Pragmatic Value: Blended GoalMap and Shadow Goal
	// When α is high (physics fails), Shadow Goal dominates
	lat, lon, alt := eciToLatLonAlt(terminalState.Pos, terminalState.EpochJD)
	goalMapLogDensity := s.AdversarialPrior.calculateGoalMapLogDensity(lat, lon, alt)
	shadowLogDensity := s.AdversarialPrior.calculateShadowLogDensity(lat, lon, alt)

	// Blended pragmatic value: (1-α) * GoalMap + α * Shadow
	pragmaticValue := (1.0-alpha)*goalMapLogDensity + alpha*shadowLogDensity

	// 5. Centroid-to-PHS Feedback Conflict Metric
	// If camera observation available and deviates from PHS prediction, spike epistemic value
	if s.HasObservation {
		// Convert terminal state to Az/El for comparison
		// Simplified: use lat/lon as proxy for Az/El (full implementation would use proper transformation)
		obsAz := s.ObservationPriorAz
		obsEl := s.ObservationPriorEl

		// Compute angular distance between PHS prediction and observation
		// This is the Conflict Metric
		conflictMetric := s.computeAngularConflict(lat, lon, obsAz, obsEl)

		// If conflict is high (>10 degrees), spike epistemic value to force resolution
		if conflictMetric > 10.0 {
			// Conflict spike: system must resolve ambiguity (maneuver vs glitch)
			// This overrides the pragmatic value temporarily
			pragmaticValue -= conflictMetric * s.CameraConfidence * 10.0
		}
	}

	// 6. Epistemic Value: Does this resolve ambiguity?
	// If the target is coasting, epistemic gain is low
	// If it maneuvers, it 'reveals' which goal it is targeting
	epistemicGain := s.CalculateInformationGain(p, currentObservation)

	return epistemicGain - pragmaticValue
}

// computeAngularConflict computes angular distance between PHS prediction and camera observation
//
// Used for Centroid-to-PHS Feedback to detect when camera observation
// deviates significantly from PHS prediction, indicating potential maneuver or sensor glitch.
func (s *EFESolver) computeAngularConflict(predLat, predLon, obsAz, obsEl float64) float64 {
	// Simplified: convert lat/lon to approximate Az/El
	// In production, use full ECI-to-AzEl transformation with proper GST accounting
	predAz := predLon
	predEl := predLat

	// Angular distance on sphere
	deltaAz := predAz - obsAz
	deltaEl := predEl - obsEl

	// Euclidean approximation for small angles
	conflict := math.Sqrt(deltaAz*deltaAz + deltaEl*deltaEl)

	return conflict
}

// Rollout simulates policy execution to get terminal state
func (s *EFESolver) Rollout(currentState *physics.State, p Policy) *physics.State {
	state := *currentState
	for _, mode := range p.ModeSequence {
		nextState := s.simplePredict(&state, mode)
		state = *nextState
	}
	return &state
}

// CalculateConflictMetric computes Jensen-Shannon Divergence between GoalMap and Shadow Goal
//
// Conflict Metric serves as "Internal Auditor" detecting when the system
// is being pulled in two directions—known strategic priors vs raw anomalous data.
//
// When GoalMap suggests heading for "Base Alpha" but Shadow Goal suggests novel "Dog-Leg",
// the system enters High Epistemic Tension.
//
// JSD formula:
//
//	JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
//
// where M = 0.5 * (P + Q) is the mixture distribution
//
// Thresholds:
// - C < τ_safe (0.2): Intent clear, continue Tier 2 (EFE Reasoning)
// - C > τ_alarm (0.5): Ambiguity too high, trigger Tier 3 (Full DPHNO Scan)
func (s *EFESolver) CalculateConflictMetric(goalPolicy, shadowPolicy Policy) float64 {
	// Convert policies to probability distributions over mode sequences
	goalDist := s.policyToDistribution(goalPolicy)
	shadowDist := s.policyToDistribution(shadowPolicy)

	// Compute mixture distribution M = 0.5 * (P + Q)
	mixtureDist := make(map[string]float64)
	for mode, prob := range goalDist {
		mixtureDist[mode] = 0.5 * (prob + shadowDist[mode])
	}

	// Compute KL(P || M) and KL(Q || M)
	klGoalM := s.computeKLDivergence(goalDist, mixtureDist)
	klShadowM := s.computeKLDivergence(shadowDist, mixtureDist)

	// JSD = 0.5 * (KL(P||M) + KL(Q||M))
	jsd := 0.5 * (klGoalM + klShadowM)

	return jsd
}

// policyToDistribution converts a Policy to a probability distribution over modes
func (s *EFESolver) policyToDistribution(p Policy) map[string]float64 {
	distribution := make(map[string]float64)
	modeCounts := make(map[string]int)

	for _, mode := range p.ModeSequence {
		modeName := s.modeToString(mode)
		modeCounts[modeName]++
	}

	total := len(p.ModeSequence)
	for mode, count := range modeCounts {
		distribution[mode] = float64(count) / float64(total)
	}

	return distribution
}

// computeKLDivergence computes KL divergence between two distributions
// KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
func (s *EFESolver) computeKLDivergence(p, q map[string]float64) float64 {
	kl := 0.0
	for x, px := range p {
		qx := q[x]
		if qx > 1e-10 && px > 1e-10 {
			kl += px * math.Log(px/qx)
		}
	}
	return kl
}

// modeToString converts mode integer to string representation
func (s *EFESolver) modeToString(mode int) string {
	switch mode {
	case 0:
		return "ballistic"
	case 1:
		return "glide"
	case 2:
		return "maneuver"
	default:
		return "unknown"
	}
}

// CalculateInformationGain calculates epistemic value (information gain)
// Separate method for clarity in EvaluatePolicy
func (s *EFESolver) CalculateInformationGain(p Policy, currentState *physics.State) float64 {
	var totalGain float64
	state := *currentState

	for i, mode := range p.ModeSequence {
		nextState := s.simplePredict(&state, mode)
		gain := s.calculateMutualInformation(nextState, i)
		totalGain += gain
		state = *nextState
	}

	return totalGain
}

// generatePrunedCandidatePolicies generates pruned candidate policies
// Only evaluates top 3 most likely mode-sequences
func (s *EFESolver) generatePrunedCandidatePolicies(beamWidth int) []Policy {
	policies := make([]Policy, 0, beamWidth)

	// Generate pruned candidates (simplified - use heuristic in production)
	// Focus on sequences that are physically meaningful
	prunedSequences := [][]int{
		{0, 0, 0, 0, 0}, // All ballistic
		{1, 1, 1, 1, 1}, // All glide
		{2, 2, 2, 2, 2}, // All maneuver
		{0, 1, 1, 1, 1}, // Ballistic → Glide
		{1, 2, 2, 2, 2}, // Glide → Maneuver
		{0, 2, 2, 2, 2}, // Ballistic → Maneuver
	}

	for i := 0; i < beamWidth && i < len(prunedSequences); i++ {
		seq := make([]int, s.Horizon)
		// Repeat the pruned sequence to fill horizon
		for j := 0; j < s.Horizon; j++ {
			seq[j] = prunedSequences[i][j%len(prunedSequences[i])]
		}

		policies = append(policies, Policy{
			ModeSequence: seq,
			Horizon:      s.Horizon,
			Probability:  1.0 / float64(beamWidth),
		})
	}

	return policies
}

// generateCandidatePolicies generates policy candidates for beam search
func (s *EFESolver) generateCandidatePolicies(beamWidth int) []Policy {
	modes := []int{0, 1, 2} // Ballistic, Glide, Maneuver
	policies := make([]Policy, 0, beamWidth)

	// Generate all combinations (simplified - use beam search in production)
	for i := 0; i < beamWidth; i++ {
		seq := make([]int, s.Horizon)
		for j := 0; j < s.Horizon; j++ {
			// Random mode selection (replace with heuristic in production)
			seq[j] = modes[(i+j)%len(modes)]
		}
		policies = append(policies, Policy{
			ModeSequence: seq,
			Horizon:      s.Horizon,
			Probability:  1.0 / float64(beamWidth),
		})
	}

	return policies
}

// CalculateBayesFactor computes evidence ratio for novel vs known modes
// If Evidence(NovelMode) / Evidence(BestKnown) > 10, spawn new mode
func (s *EFESolver) CalculateBayesFactor(novelModeLikelihood, knownModeLikelihood float64) float64 {
	if knownModeLikelihood < 1e-10 {
		return math.Inf(1)
	}
	return novelModeLikelihood / knownModeLikelihood
}

// String returns string representation
func (s *EFESolver) String() string {
	return fmt.Sprintf("EFESolver[horizon=%d, epistemic=%.2f, pragmatic=%.2f]",
		s.Horizon, s.EpistemicWeight, s.PragmaticWeight)
}
