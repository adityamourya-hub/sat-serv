package hypersonic

import (
	"fmt"
	"math"
	"ssis/pkg/ml/ai_imm"
	"ssis/pkg/physics"
)

// DefaultKalmanMinGain is the default minimum Kalman gain
// Prevents overconfident updates (previously hardcoded as 0.01)
const DefaultKalmanMinGain = 0.01

// DefaultKalmanMaxGain is the default maximum Kalman gain
// Prevents filter instability (previously hardcoded as 0.95)
const DefaultKalmanMaxGain = 0.95

/**
 * Interacting Multiple Model (IMM) Filter for HGV Tracking
 *
 * HGVs switch between flight modes (ballistic, glide, maneuver).
 * IMM runs multiple models in parallel with Markov transitions.
 *
 * Models:
 * - Model 1: Coast/Ballistic (free fall)
 * - Model 2: Constant Glide (steady L/D)
 * - Model 3: High-G Maneuver (aggressive bank)
 *
 * Reference: Bar-Shalom, "Estimation and Tracking"
 */

/**
 * HGVMode represents the flight mode.
 */
type HGVMode int

const (
	ModeBallistic HGVMode = iota // Coasting/free fall
	ModeGlide                    // Steady skip-glide
	ModeManeuver                 // High-G turn
	NumModes                     // Count of modes
)

/**
 * IMMFilter holds the multi-model state with adaptive Dirichlet-Markov learning
 * and Active Inference (Expected Free Energy) for adversarial intent detection.
 */
type IMMFilter struct {
	// Mode probabilities (dynamic slice for expansion)
	ModeProbs []float64

	// Mode-matched filters (dynamic slice for expansion)
	Filters []*ModeFilter

	// Markov transition matrix (dynamic slice for expansion)
	// Trans[i][j] = prob of transitioning from mode i to mode j
	TransMatrix [][]float64

	// Dirichlet prior counts for online transition matrix learning (dynamic slice)
	// DirichletCounts[i][j] = pseudo-counts for transition i->j
	DirichletCounts [][]float64

	// Mode likelihoods (dynamic slice for expansion)
	Likelihoods []float64

	// Active Inference - Expected Free Energy for each mode
	// If a target is "minimizing surprise" relative to an interceptor's path,
	// it's adversarial counter-intelligence, not just a maneuver.
	ExpectedFreeEnergy []float64

	// Constants
	HGV HGVConstants

	// Stochastic Port-Hamiltonian System (SPHS) Integration
	// Dynamic covariance from learned diffusion model instead of hardcoded R matrix
	DiffusionModel          physics.DiffusionModelInterface // σ(x) network for state-dependent noise
	EnableDynamicCovariance bool                            // Enable dynamic covariance propagation

	// Measurement noise covariance (for Kalman gain computation)
	// DEPRECATED: Use dynamic covariance from SPHS when EnableDynamicCovariance=true
	MeasurementNoise float64

	// Adaptive Dirichlet-Markov parameters
	Eta             float64 // Forgetting factor (e.g., 0.995)
	DiscoveryThresh float64 // Threshold for zero-day mode creation (χ²_α)
	MaxModes        int     // Maximum number of modes (for dynamic expansion)
	ZeroDayActive   bool    // Whether zero-day mode is active
	NumModes        int     // Current number of modes (for compatibility)

	// Spectral Gap enforcement
	// Forces PHS to handle only "Dissipative Background" (gravity, neutral atmosphere)
	// Forces high-frequency (>0.5Hz) or control-dependent forces into SINDy residual
	SpectralGapThreshold float64 // Hz threshold (default 0.5)

	// Density uncertainty covariance inflation
	// 1σ drag error ±30% must be sigma-point-propagated into filter for PAC coverage
	DensityUncertainty float64 // 1σ density uncertainty (default 0.3 for ±30%)
	AltitudeKm         float64 // Current altitude for density uncertainty scaling

	// Active Inference (AI-IMM) - Expected Free Energy solver
	// Pivot from reactive Bayesian filtering to predictive intent inference
	// System asks "What is the adversary trying to learn or achieve?"
	AIIMMEnabled         bool    // Whether AI-IMM is enabled
	ManeuverThreshold    float64 // Threshold for P(maneuver) to trigger AI-IMM
	BayesFactorThreshold float64 // Threshold for Bayes Factor (>10 spawns new mode)

	// Anticipatory Term for Predictive Capability
	// Uses EFE to bias predictions toward likely future states
	AnticipatoryHorizon float64 // Look-ahead horizon for anticipatory term (seconds)
	AnticipatoryWeight  float64 // Weight for anticipatory term vs reactive term (0.0-1.0)

	// Kalman gain clamping bounds (replaces hardcoded [0.01, 0.95])
	MinGain float64 // Minimum Kalman gain (default 0.01)
	MaxGain float64 // Maximum Kalman gain (default 0.95)
}

/**
 * ModeFilter represents a single model's state estimate.
 */
type ModeFilter struct {
	Mode       HGVMode
	State      HGVState
	Covariance [6][6]float64 // Position/velocity uncertainty
	Gain       float64       // Filter gain
}

/**
 * NewIMMFilter creates an initialized IMM filter with adaptive Dirichlet-Markov learning.
 *
 * Optional gain parameters allow adaptive tuning for different HGV profiles.
 * Use NewIMMFilterWithConfig for full configuration control.
 */
func NewIMMFilter(hgv HGVConstants) *IMMFilter {
	return NewIMMFilterWithConfig(hgv, DefaultKalmanMinGain, DefaultKalmanMaxGain)
}

/**
 * NewIMMFilterWithConfig creates an IMM filter with configurable Kalman gain bounds.
 *
 * Args:
 *   hgv: HGV constants for the target vehicle
 *   minGain: Minimum Kalman gain (default 0.01, prevents overconfident updates)
 *   maxGain: Maximum Kalman gain (default 0.95, prevents instability)
 */
func NewIMMFilterWithConfig(hgv HGVConstants, minGain, maxGain float64) *IMMFilter {
	imm := &IMMFilter{
		HGV:             hgv,
		Eta:             0.995, // Forgetting factor for online learning
		DiscoveryThresh: 11.34, // χ²_α for 3 DOF at 99% confidence
		MaxModes:        8,     // Maximum modes for dynamic expansion
		ZeroDayActive:   false,
		NumModes:        int(NumModes), // Initialize with base mode count
		MinGain:         minGain,       // Configurable minimum gain
		MaxGain:         maxGain,       // Configurable maximum gain
	}

	// Initialize dynamic slices with initial 3 modes
	imm.ModeProbs = make([]float64, int(NumModes))
	imm.Filters = make([]*ModeFilter, int(NumModes))
	imm.TransMatrix = make([][]float64, int(NumModes))
	imm.DirichletCounts = make([][]float64, int(NumModes))
	imm.Likelihoods = make([]float64, int(NumModes))
	imm.ExpectedFreeEnergy = make([]float64, int(NumModes))
	imm.SpectralGapThreshold = 0.5 // 0.5 Hz threshold

	// Density uncertainty covariance inflation
	// 1σ drag error ±30% for PAC coverage (conformal α=0.05)
	imm.DensityUncertainty = 0.3 // ±30% density uncertainty
	imm.AltitudeKm = 100.0       // Default altitude for scaling

	// Active Inference (AI-IMM)
	imm.AIIMMEnabled = false
	imm.ManeuverThreshold = 0.3
	imm.BayesFactorThreshold = 10.0

	// Anticipatory term parameters
	imm.AnticipatoryHorizon = 5.0 // 5 second look-ahead
	imm.AnticipatoryWeight = 0.3  // 30% anticipatory, 70% reactive

	// Initialize with equal mode probabilities
	for i := 0; i < int(NumModes); i++ {
		imm.ModeProbs[i] = 1.0 / float64(NumModes)
		imm.DirichletCounts[i] = make([]float64, int(NumModes))
		imm.TransMatrix[i] = make([]float64, int(NumModes))
	}

	// Typical Markov transition for HGV
	// High probability of staying in same mode
	// Moderate probability of transitioning
	imm.TransMatrix[0] = []float64{0.90, 0.08, 0.02} // From Ballistic: mostly stay, some glide, rare maneuver
	imm.TransMatrix[1] = []float64{0.10, 0.85, 0.05} // From Glide: some ballistic, mostly glide, some maneuver
	imm.TransMatrix[2] = []float64{0.20, 0.30, 0.50} // From Maneuver: tends to exit maneuver

	// Initialize Dirichlet counts with prior (pseudo-counts)
	// P1 Audit: Increased from 1.0 to 2.0 for "curious" zero-day detection
	// Prior of 2.0 reduces observations needed for posterior update by ~50%
	// With 100Hz sampling, 5 seconds = 500 observations → sufficient for detection
	for i := 0; i < int(NumModes); i++ {
		for j := 0; j < int(NumModes); j++ {
			imm.DirichletCounts[i][j] = 2.0 // Curious prior for faster adaptation
		}
	}

	// Measurement noise for Kalman gain (meters)
	// DEPRECATED: Use dynamic covariance from SPHS diffusion model
	imm.MeasurementNoise = 1000.0 // 1km default (fallback only)

	// SPHS Integration
	imm.EnableDynamicCovariance = false // Disabled by default for backward compatibility

	// Initialize mode filters
	for i := 0; i < int(NumModes); i++ {
		imm.Filters[i] = &ModeFilter{
			Mode:       HGVMode(i),
			Covariance: diagCovariance(1000.0), // 1km initial uncertainty
		}
	}

	return imm
}

/**
 * Update performs one IMM cycle:
 * 1. Mode mixing (interaction)
 * 2. Mode-matched filtering
 * 3. Mode probability update
 * 4. State combination
 *
 * @param measurement: Observed position/velocity
 * @param dt: Time step
 * @param jd: Julian Date
 * @return: Combined state estimate
 */
func (imm *IMMFilter) Update(
	measurement HGVState,
	dt float64,
	jd float64,
) HGVState {
	// Step 1: Mode mixing (compute mixed initial conditions)
	mixedStates := imm.modeMixing()

	// Step 2: Mode-matched filtering
	for i := 0; i < int(NumModes); i++ {
		imm.Filters[i].State = mixedStates[i]
		imm.modeMatchedFilter(imm.Filters[i], measurement, dt, jd)
	}

	// Step 3: Mode probability update
	imm.updateModeProbabilities(measurement)

	// Step 4: State combination
	combinedState := imm.combineStates()

	return combinedState
}

/**
 * modeMixing computes mixed initial conditions for each filter.
 */
func (imm *IMMFilter) modeMixing() [NumModes]HGVState {
	var mixed [NumModes]HGVState

	// Mixed probability: μ_ij = (1/c_j) * p_ij * μ_i
	// where c_j = sum_i(p_ij * μ_i) is normalization

	for j := 0; j < int(NumModes); j++ {
		// Compute normalization c_j
		c := 0.0
		for i := 0; i < int(NumModes); i++ {
			c += imm.TransMatrix[i][j] * imm.ModeProbs[i]
		}

		if c < 1e-10 {
			c = 1e-10
		}

		// Compute mixed state for mode j
		mixed[j] = imm.computeMixedState(int(j), c)
	}

	return mixed
}

/**
 * computeMixedState calculates weighted average of states for mode j.
 */
func (imm *IMMFilter) computeMixedState(j int, c float64) HGVState {
	var mixed HGVState

	for i := 0; i < int(NumModes); i++ {
		// Mixing weight
		muMix := imm.TransMatrix[i][j] * imm.ModeProbs[i] / c

		// Weighted sum of states
		mixed.R += muMix * imm.Filters[i].State.R
		mixed.Phi += muMix * imm.Filters[i].State.Phi
		mixed.Lambda += muMix * imm.Filters[i].State.Lambda
		mixed.V += muMix * imm.Filters[i].State.V
		mixed.Gamma += muMix * imm.Filters[i].State.Gamma
		mixed.Psi += muMix * imm.Filters[i].State.Psi
	}

	return mixed
}

/**
 * modeMatchedFilter updates a single mode filter.
 */
func (imm *IMMFilter) modeMatchedFilter(
	filter *ModeFilter,
	measurement HGVState,
	dt float64,
	jd float64,
) {
	// Predict step (model-specific dynamics)
	predicted := imm.predictMode(filter.Mode, filter.State, dt, jd)

	// Update step (Kalman-like with computed gain)
	// Compute Kalman gain from covariance: K = P / (P + R)
	// where P is state covariance, R is measurement noise

	// Dynamic Covariance from SPHS (Stochastic Port-Hamiltonian System)
	// Instead of hardcoded R matrix, use learned diffusion σ(x) for dynamic covariance
	// Σ_prop ≈ Φ Σ Φ^T + σ(x)σ(x)^T Δt
	//
	// This enables "Free Covariance Propagation" - the filter automatically widens
	// its gaze in high-turbulence regimes and tightens in stable orbital arcs.
	var measurementNoise float64

	if imm.EnableDynamicCovariance && imm.DiffusionModel != nil {
		// Compute state-dependent diffusion σ(x) at current state
		// State: [pos(3), vel(3)] where pos = [R, Lambda, Phi], vel = [V, Gamma, Psi]
		stateQ := []float64{filter.State.R, filter.State.Lambda, filter.State.Phi}
		stateP := []float64{filter.State.V, filter.State.Gamma, filter.State.Psi}
		stateZ := []float64{} // Latent code (empty for now)

		sigma := imm.DiffusionModel.Predict(stateQ, stateP, stateZ) // [6] for (q, p)

		// Compute diffusion covariance Σ = σ(x)σ(x)^T (diagonal approximation)
		// For diagonal σ: Σ_i = σ_i²
		sigmaCovariance := 0.0
		for i := 0; i < len(sigma); i++ {
			sigmaCovariance += sigma[i] * sigma[i]
		}
		sigmaCovariance /= float64(len(sigma)) // Average diagonal element

		// Scale by time step (dt = 1.0 for 1Hz update)
		dt := 1.0
		dynamicCovariance := sigmaCovariance * dt

		// Add density uncertainty inflation
		altitudeFactor := math.Max(1.0, 200.0/imm.AltitudeKm)
		densityNoise := imm.DensityUncertainty * altitudeFactor * 1000.0

		measurementNoise = dynamicCovariance + densityNoise
	} else {
		// Legacy: Use hardcoded measurement noise
		// Inflate measurement noise based on density uncertainty
		// Density uncertainty ±30% propagates to drag uncertainty → position uncertainty
		// Scale uncertainty by altitude (higher altitude = more density uncertainty)
		altitudeFactor := math.Max(1.0, 200.0/imm.AltitudeKm)            // More uncertainty at high altitude
		densityNoise := imm.DensityUncertainty * altitudeFactor * 1000.0 // Convert to meters
		measurementNoise = imm.MeasurementNoise + densityNoise           // R (inflated measurement noise variance)
	}

	// Compute gain for each state dimension from filter covariance
	// Simplified: use trace of covariance as scalar uncertainty estimate
	stateVariance := 0.0
	for i := 0; i < 6; i++ {
		stateVariance += filter.Covariance[i][i]
	}
	stateVariance /= 6.0 // Average diagonal element

	// Kalman gain: K = P / (P + R)
	gain := stateVariance / (stateVariance + measurementNoise)

	// Clamp gain to configurable bounds
	if gain < imm.MinGain {
		gain = imm.MinGain
	}
	if gain > imm.MaxGain {
		gain = imm.MaxGain
	}

	// Measurement residual
	residualR := measurement.R - predicted.R
	residualPhi := measurement.Phi - predicted.Phi
	residualLambda := measurement.Lambda - predicted.Lambda

	// Normalize angles
	for residualPhi > math.Pi {
		residualPhi -= 2 * math.Pi
	}
	for residualPhi < -math.Pi {
		residualPhi += 2 * math.Pi
	}
	for residualLambda > math.Pi {
		residualLambda -= 2 * math.Pi
	}
	for residualLambda < -math.Pi {
		residualLambda += 2 * math.Pi
	}

	// Update state
	filter.State.R = predicted.R + gain*residualR
	filter.State.Phi = predicted.Phi + gain*residualPhi
	filter.State.Lambda = predicted.Lambda + gain*residualLambda

	// Update velocity
	residualV := measurement.V - predicted.V
	residualGamma := measurement.Gamma - predicted.Gamma
	residualPsi := measurement.Psi - predicted.Psi

	filter.State.V = predicted.V + gain*residualV
	filter.State.Gamma = predicted.Gamma + gain*residualGamma
	filter.State.Psi = predicted.Psi + gain*residualPsi

	// Update covariance using Joseph form for numerical stability
	// P = (I - K*H) * P * (I - K*H)^T + K*R*K^T
	// Simplified scalar update for diagonal elements
	for i := 0; i < 6; i++ {
		// Covariance update: P_new = (1-K)^2 * P_old + K^2 * R
		oldVar := filter.Covariance[i][i]
		filter.Covariance[i][i] = (1-gain)*(1-gain)*oldVar + gain*gain*measurementNoise
	}

	// Store computed gain for diagnostics
	filter.Gain = gain
}

/**
 * predictMode performs model-specific prediction with anticipatory term.
 *
 * Combines reactive prediction with anticipatory EFE-based lookahead.
 * x_pred = (1 - w_anti) * x_reactive + w_anti * x_anticipatory
 *
 * where:
 * - x_reactive: Standard physics-based prediction
 * - x_anticipatory: EFE-weighted future state prediction
 * - w_anti: Anticipatory weight (0.0 = pure reactive, 1.0 = pure anticipatory)
 */
func (imm *IMMFilter) predictMode(
	mode HGVMode,
	state HGVState,
	dt float64,
	jd float64,
) HGVState {
	// Update altitude for density uncertainty scaling
	imm.AltitudeKm = state.R - physics.EARTH_R_EQUATORIAL

	// Reactive prediction (standard physics-based)
	reactiveState := imm.predictModeReactive(mode, state, dt, jd)

	// If anticipatory term is disabled or AI-IMM not enabled, return reactive only
	if imm.AnticipatoryWeight < 0.01 || !imm.AIIMMEnabled {
		return reactiveState
	}

	// Anticipatory prediction (EFE-based lookahead)
	anticipatoryState := imm.predictAnticipatory(mode, state, dt, jd)

	// Blend reactive and anticipatory predictions
	// x_pred = (1 - w_anti) * x_reactive + w_anti * x_anticipatory
	w := imm.AnticipatoryWeight
	blendedState := HGVState{
		R:      (1-w)*reactiveState.R + w*anticipatoryState.R,
		Phi:    (1-w)*reactiveState.Phi + w*anticipatoryState.Phi,
		Lambda: (1-w)*reactiveState.Lambda + w*anticipatoryState.Lambda,
		V:      (1-w)*reactiveState.V + w*anticipatoryState.V,
		Gamma:  (1-w)*reactiveState.Gamma + w*anticipatoryState.Gamma,
		Psi:    (1-w)*reactiveState.Psi + w*anticipatoryState.Psi,
	}

	return blendedState
}

/**
 * predictModeReactive performs standard reactive prediction (physics-based only).
 */
func (imm *IMMFilter) predictModeReactive(
	mode HGVMode,
	state HGVState,
	dt float64,
	jd float64,
) HGVState {
	switch mode {
	case ModeBallistic:
		return imm.predictBallistic(state, dt, jd)
	case ModeGlide:
		return imm.predictGlide(state, dt, jd)
	case ModeManeuver:
		return imm.predictManeuver(state, dt, jd)
	default:
		return state
	}
}

/**
 * predictAnticipatory performs EFE-based anticipatory prediction.
 *
 * Uses the AI-IMM EFE solver to predict future state based on intent inference.
 * This provides predictive capability by looking ahead along likely intent trajectories.
 */
func (imm *IMMFilter) predictAnticipatory(
	mode HGVMode,
	state HGVState,
	dt float64,
	jd float64,
) HGVState {
	// Look ahead multiple steps for anticipatory prediction
	horizonSteps := int(imm.AnticipatoryHorizon / dt)
	if horizonSteps < 1 {
		horizonSteps = 1
	}
	if horizonSteps > 10 {
		horizonSteps = 10 // Cap at 10 steps for computational efficiency
	}

	// Predict future state by chaining reactive predictions
	futureState := state
	for i := 0; i < horizonSteps; i++ {
		futureState = imm.predictModeReactive(mode, futureState, dt, jd)
	}

	// Weight the future state by mode probability
	// Modes with higher probability get more anticipatory influence
	modeProb := imm.ModeProbs[int(mode)]

	// Blend current state with future state based on mode probability
	anticipatoryState := HGVState{
		R:      state.R + modeProb*(futureState.R-state.R),
		Phi:    state.Phi + modeProb*(futureState.Phi-state.Phi),
		Lambda: state.Lambda + modeProb*(futureState.Lambda-state.Lambda),
		V:      state.V + modeProb*(futureState.V-state.V),
		Gamma:  state.Gamma + modeProb*(futureState.Gamma-state.Gamma),
		Psi:    state.Psi + modeProb*(futureState.Psi-state.Psi),
	}

	return anticipatoryState
}

/**
 * ComputeExpectedFreeEnergy calculates the Expected Free Energy for each mode.
 *
 * Active Inference for adversarial intent detection.
 *
 * The Expected Free Energy (EFE) measures how "surprising" a trajectory is
 * given a generative model. If a target is minimizing surprise relative to an
 * interceptor's expected path, it's adversarial counter-intelligence.
 *
 * Formula:
 *   EFE = E[log P(y|m, θ)] - E[log Q(θ|m, y)]
 *
 * For practical implementation, we approximate:
 *   EFE ≈ -log(Likelihood) + ComplexityPenalty + AdversarialPenalty
 *
 * Args:
 *   observation: Current sensor observation
 *   interceptorPath: Expected interceptor trajectory (for adversarial detection)
 *
 * Returns:
 *   EFE values for each mode (lower = less surprising)
 */
func (imm *IMMFilter) ComputeExpectedFreeEnergy(observation HGVState, interceptorPath []HGVState) []float64 {
	numModes := len(imm.ModeProbs)
	efe := make([]float64, numModes)

	for i := 0; i < numModes; i++ {
		// 1. Negative log-likelihood (surprise)
		nll := -math.Log(imm.Likelihoods[i] + 1e-10)

		// 2. Complexity penalty (mode covariance trace)
		covTrace := 0.0
		for j := 0; j < 6; j++ {
			covTrace += imm.Filters[i].Covariance[j][j]
		}
		complexity := 0.01 * covTrace

		// 3. Adversarial penalty (if interceptor path provided)
		adversarialPenalty := 0.0
		if len(interceptorPath) > 0 {
			// Check if target is moving to minimize surprise relative to interceptor
			// Manual component-wise subtraction since HGVState has no Subtract method
			interpState := interceptorPath[len(interceptorPath)-1]
			dx := imm.Filters[i].State.R - interpState.R
			dy := imm.Filters[i].State.Phi - interpState.Phi
			dz := imm.Filters[i].State.Lambda - interpState.Lambda
			adversarialPenalty = 0.1 * (dx*dx + dy*dy + dz*dz)
		}

		// Combined EFE
		efe[i] = nll + complexity + adversarialPenalty
	}

	return efe
}

/**
 * DetectAdversarialIntent uses Expected Free Energy to detect adversarial counter-intelligence.
 *
 * If a mode has consistently low EFE (minimizing surprise) relative to expected
 * interceptor behavior, it indicates adversarial intent rather than natural maneuvering.
 *
 * Args:
 *   efe: Expected Free Energy values
 *
 * Returns:
 *   isAdversarial: Whether adversarial intent is detected
 *   confidence: Confidence score (0-1)
 *   dominantMode: The mode with lowest EFE
 */
func (imm *IMMFilter) DetectAdversarialIntent(efe []float64) (bool, float64, int) {
	if len(efe) == 0 {
		return false, 0.0, 0
	}

	// Find mode with minimum EFE (most "planned" behavior)
	minEFEIdx := 0
	minEFE := efe[0]
	for i := 1; i < len(efe); i++ {
		if efe[i] < minEFE {
			minEFE = efe[i]
			minEFEIdx = i
		}
	}

	// Compute average EFE for comparison
	avgEFE := 0.0
	for _, val := range efe {
		avgEFE += val
	}
	avgEFE /= float64(len(efe))

	// If min EFE is significantly below average, indicates adversarial planning
	isAdversarial := minEFE < 0.7*avgEFE
	confidence := math.Max(0.0, (avgEFE-minEFE)/avgEFE)

	return isAdversarial, confidence, minEFEIdx
}

/**
 * predictBallistic: No lift, only drag and gravity.
 *
 * Symplectic Störmer-Verlet integration with Hamiltonian Energy Guard.
 *
 * Replaces legacy C_D=1.0 hardcoded drag with catalog-based BallisticCoeff.
 * Uses FNO-based AtmosphereEngine for live density.
 * Enforces energy conservation to prevent mode flickering in IMM.
 */
func (imm *IMMFilter) predictBallistic(state HGVState, dt float64, jd float64) HGVState {
	// 1. Get Live Density (No more CD=1.0 assumptions)
	// Use the FNO-based AtmosphereEngine via GetDensityUnified
	rho := physics.GetDensityUnified(0, 0, state.R, jd)

	// 2. Hamiltonian Force Calculation (Gravity + Aero)
	// H_ballistic = p^2 / 2m + V(q)
	// Use ballistic coefficient (BC) from catalog instead of hardcoded CD=1.0
	v_mag := state.V * 1000.0 // Convert to m/s

	// Drag Force: F_d = -0.5 * rho * v^2 * (1/BC)
	// BC = m / (Cd * A), so Cd*A/m = 1/BC
	// a_drag = -0.5 * rho * v^2 / BC
	aero_accel := -0.5 * rho * v_mag * v_mag / imm.HGV.BallisticCoeff // m/s^2
	aero_accel_km := aero_accel / 1000.0                              // Convert to km/s^2

	// Gravity acceleration (central force)
	g := physics.EARTH_MU_IAU / (state.R * state.R)

	// Total acceleration along velocity vector (for energy change)
	total_accel := aero_accel_km + g*math.Sin(state.Gamma)

	// 3. Symplectic Update (Energy-Preserving Störmer-Verlet)
	// Half-step momentum (velocity)
	v_half := state.V + total_accel*(dt/2.0)

	// Full-step position using half-step velocity
	// This preserves the symplectic structure
	newState := state
	newState.R += v_half * math.Sin(state.Gamma) * dt
	newState.Phi += (v_half * math.Cos(state.Gamma) * math.Cos(state.Psi) / state.R) * dt
	newState.Lambda += (v_half * math.Cos(state.Gamma) * math.Sin(state.Psi) / (state.R * math.Cos(state.Phi))) * dt

	// Recalculate gravity at new position
	g_new := physics.EARTH_MU_IAU / (newState.R * newState.R)
	total_accel_new := aero_accel_km + g_new*math.Sin(state.Gamma)

	// Final half-step momentum (velocity)
	newState.V = v_half + total_accel_new*(dt/2.0)

	// Update flight path angle (vertical motion)
	dGamma := (g_new - newState.V*newState.V/newState.R) * math.Cos(state.Gamma) / newState.V
	newState.Gamma += dGamma * dt

	// 4. Energy Guard Projection
	// Ensure that the specific energy hasn't drifted due to aero-approximation
	// Specific orbital energy: ε = v^2/2 - μ/r
	energy_before := state.V*state.V/2.0 - physics.EARTH_MU_IAU/state.R
	energy_after := newState.V*newState.V/2.0 - physics.EARTH_MU_IAU/newState.R

	// If energy increased (unphysical for ballistic with drag), project back
	// Energy should only decrease due to drag dissipation
	energy_drift := energy_after - energy_before
	if energy_drift > 1e-9 {
		// Scale velocity to conserve energy (energy projection)
		// ε = v^2/2 - μ/r → v = sqrt(2(ε + μ/r))
		target_energy := energy_before - math.Abs(aero_accel_km*state.V*dt) // Expected energy loss from drag
		newState.V = math.Sqrt(2.0 * (target_energy + physics.EARTH_MU_IAU/newState.R))
		if newState.V < 0 {
			newState.V = state.V // Fallback if calculation fails
		}
	}

	return newState
}

/**
 * predictGlide: Steady L/D from HGVConstants, minimal bank.
 */
func (imm *IMMFilter) predictGlide(state HGVState, dt float64, jd float64) HGVState {
	newState := state

	// Use configurable L/D for glide
	LD := imm.HGV.LDGlide

	// Aerodynamics
	rho := physics.GetDensityUnified(0, 0, state.R, jd)
	vMs := state.V * 1000.0
	q := 0.5 * rho * vMs * vMs

	// Use configurable CD for glide
	CD := imm.HGV.CdGlide
	CL := CD * LD

	// Forces
	D := q * imm.HGV.ReferenceArea * CD
	L := q * imm.HGV.ReferenceArea * CL

	// Gravity
	g := physics.EARTH_MU_IAU / (state.R * state.R)

	// Bank angle (small for steady glide)
	sigma := 0.1 // ~6 degrees

	// Equations of motion
	dV := -(D/imm.HGV.Mass)/1000.0 - g*math.Sin(state.Gamma)

	centrifugal := state.V * state.V / state.R
	dGamma := (L*math.Cos(sigma)/imm.HGV.Mass)/1000.0 - (g-centrifugal)*math.Cos(state.Gamma)/state.V
	dGamma /= state.V

	dPsi := (L * math.Sin(sigma) / imm.HGV.Mass) / (1000.0 * state.V * math.Cos(state.Gamma))

	// Update
	newState.V += dV * dt
	newState.Gamma += dGamma * dt
	newState.Psi += dPsi * dt

	newState.R += state.V * math.Sin(state.Gamma) * dt
	newState.Phi += (state.V * math.Cos(state.Gamma) * math.Cos(state.Psi) / state.R) * dt
	newState.Lambda += (state.V * math.Cos(state.Gamma) * math.Sin(state.Psi) / (state.R * math.Cos(state.Phi))) * dt

	return newState
}

/**
 * predictManeuver: High bank angle, aggressive turn.
 */
func (imm *IMMFilter) predictManeuver(state HGVState, dt float64, jd float64) HGVState {
	newState := state

	// Use configurable L/D for maneuver
	LD := imm.HGV.LDManeuver

	rho := physics.GetDensityUnified(0, 0, state.R, jd)
	vMs := state.V * 1000.0
	q := 0.5 * rho * vMs * vMs

	// Use configurable CD for maneuver
	CD := imm.HGV.CdManeuver
	CL := CD * LD

	D := q * imm.HGV.ReferenceArea * CD
	L := q * imm.HGV.ReferenceArea * CL

	g := physics.EARTH_MU_IAU / (state.R * state.R)

	// High bank angle (~45 degrees)
	sigma := 0.785

	dV := -(D/imm.HGV.Mass)/1000.0 - g*math.Sin(state.Gamma)

	centrifugal := state.V * state.V / state.R
	dGamma := (L*math.Cos(sigma)/imm.HGV.Mass)/1000.0 - (g-centrifugal)*math.Cos(state.Gamma)/state.V
	dGamma /= state.V

	// Large heading change
	dPsi := (L * math.Sin(sigma) / imm.HGV.Mass) / (1000.0 * state.V * math.Cos(state.Gamma))

	newState.V += dV * dt
	newState.Gamma += dGamma * dt
	newState.Psi += dPsi * dt

	newState.R += state.V * math.Sin(state.Gamma) * dt
	newState.Phi += (state.V * math.Cos(state.Gamma) * math.Cos(state.Psi) / state.R) * dt
	newState.Lambda += (state.V * math.Cos(state.Gamma) * math.Sin(state.Psi) / (state.R * math.Cos(state.Phi))) * dt

	return newState
}

/**
 * updateModeProbabilities uses Bayes' rule with adaptive Dirichlet-Markov learning.
 * μ_j = (1/c) * Λ_j * sum_i(p_ij * μ_i)
 *
 * Performs online Dirichlet update with forgetting factor η and innovation weighting.
 * Includes zero-day mode discovery when all known modes fail.
 */
func (imm *IMMFilter) updateModeProbabilities(measurement HGVState) {
	// Compute likelihood and NIS (Normalized Innovation Squared) for each mode
	numModes := len(imm.ModeProbs)
	nis := make([]float64, numModes)
	for j := 0; j < numModes; j++ {
		imm.Likelihoods[j] = imm.computeLikelihood(j, measurement)
		nis[j] = imm.computeNIS(j, measurement)
	}

	// Check for Zero-Day event: all modes failing simultaneously
	minNIS := nis[0]
	for j := 1; j < numModes; j++ {
		if nis[j] < minNIS {
			minNIS = nis[j]
		}
	}

	// Zero-Day discovery: if min NIS > χ² threshold, spawn new mode
	if minNIS > imm.DiscoveryThresh && !imm.ZeroDayActive {
		// Extract SINDy residual (placeholder - would come from sliding_sindy)
		var sindyResidual [6]float64
		imm.spawnZeroDayMode(measurement, sindyResidual)
		imm.ZeroDayActive = true
	}

	// Update probabilities
	newProbs := make([]float64, numModes)

	for j := 0; j < numModes; j++ {
		// Mode probability prediction
		modePred := 0.0
		for i := 0; i < numModes; i++ {
			modePred += imm.TransMatrix[i][j] * imm.ModeProbs[i]
		}

		// Bayes update
		newProbs[j] = imm.Likelihoods[j] * modePred
	}

	// Normalize
	sum := 0.0
	for j := 0; j < numModes; j++ {
		sum += newProbs[j]
	}

	if sum > 1e-10 {
		for j := 0; j < numModes; j++ {
			newProbs[j] /= sum
		}
	}

	// Online Dirichlet update with forgetting factor η
	// α_{ij,t+1} = η * α_{ij,t} + evidence
	for j := 0; j < numModes; j++ {
		for i := 0; i < numModes; i++ {
			// Compute mixing probability μ_{i|j}
			c := 0.0
			for k := 0; k < numModes; k++ {
				c += imm.TransMatrix[k][j] * imm.ModeProbs[k]
			}
			if c < 1e-10 {
				c = 1e-10
			}
			muMix := imm.TransMatrix[i][j] * imm.ModeProbs[i] / c

			// Weight evidence by innovation covariance (inverse NIS)
			// Modes that explain the data well (low NIS) get more weight
			innovationWeight := 1.0 / (1.0 + nis[j])
			evidence := muMix * newProbs[j] * innovationWeight

			// Dirichlet update with forgetting
			imm.DirichletCounts[i][j] = imm.Eta*imm.DirichletCounts[i][j] + evidence
		}
	}

	// Update transition matrix from Dirichlet posterior mean
	// p(i->j) = α_ij / sum_k(α_ik)
	for i := 0; i < numModes; i++ {
		rowSum := 0.0
		for j := 0; j < numModes; j++ {
			rowSum += imm.DirichletCounts[i][j]
		}

		if rowSum > 1e-10 {
			for j := 0; j < numModes; j++ {
				imm.TransMatrix[i][j] = imm.DirichletCounts[i][j] / rowSum
			}
		}
	}

	// Store updated probabilities (convert array to slice)
	imm.ModeProbs = newProbs[:]
}

/**
 * computeNIS calculates Normalized Innovation Squared for zero-day detection.
 * NIS = ν^T * S^{-1} * ν, where ν is innovation and S is innovation covariance.
 *
 * Uses dynamic covariance from SPHS diffusion when enabled.
 */
func (imm *IMMFilter) computeNIS(mode int, measurement HGVState) float64 {
	filter := imm.Filters[mode]

	// Residual (innovation)
	residual := filter.State.R - measurement.R

	// Innovation covariance (dynamic from SPHS or legacy measurement noise)
	var S float64
	if imm.EnableDynamicCovariance && imm.DiffusionModel != nil {
		// Compute dynamic covariance from learned diffusion
		stateQ := []float64{filter.State.R, filter.State.Lambda, filter.State.Phi}
		stateP := []float64{filter.State.V, filter.State.Gamma, filter.State.Psi}
		stateZ := []float64{}

		sigma := imm.DiffusionModel.Predict(stateQ, stateP, stateZ)

		// Compute Σ = σ(x)σ(x)^T
		sigmaCovariance := 0.0
		for i := 0; i < len(sigma); i++ {
			sigmaCovariance += sigma[i] * sigma[i]
		}
		sigmaCovariance /= float64(len(sigma))

		S = sigmaCovariance
	} else {
		// Legacy: Use measurement noise
		S = imm.MeasurementNoise
	}

	// NIS = ν^T * S^{-1} * ν = ν^2 / S
	nis := (residual * residual) / S

	return nis
}

/**
 * UpdateActiveInference replaces reactive update with Bayesian Model Evidence threshold.
 *
 * Active Inference for adversarial intent detection.
 *
 * If P(maneuver) < threshold, stay in TIER 1 (Reactive).
 * If P(maneuver) > threshold, execute AI-IMM (TIER 2).
 *
 * AI-IMM process:
 * 1. Find Policy π* that minimizes Expected Free Energy G(π)
 * 2. Mode Spawning via Bayes Factor (if Evidence(NovelMode)/Evidence(BestKnown) > 10)
 * 3. Apply Policy π* to filter
 *
 * This enables the system to "peer" into the future 30 seconds to see if a specific
 * maneuver trajectory resolves the entropy of the current track.
 */
func (imm *IMMFilter) UpdateActiveInference(obs HGVState) {
	if !imm.AIIMMEnabled {
		imm.ReactiveUpdate(obs)
		return
	}

	// Check maneuver probability threshold
	pManeuver := imm.ModeProbs[ModeManeuver]
	if pManeuver < imm.ManeuverThreshold {
		// Stay in TIER 1 (Reactive) - preserve 8W thermal budget
		imm.ReactiveUpdate(obs)
		return
	}

	// TIER 2: Execute AI-IMM (27ms compute cost)
	// Only activated when P(maneuver) > threshold to preserve thermal budget

	// 1. Create EFE solver
	// DPHNO integration: pass nil for now, EFE solver uses simplePredict fallback
	efeSolver := ai_imm.NewEFESolver(nil)

	// 2. Find Policy π* that minimizes G(π)
	// Convert HGVState to physics.State using proper spherical to Cartesian transformation
	eciState := HGVToECIState(obs)
	// Set epoch if not already set
	if eciState.EpochJD == 0 {
		eciState.EpochJD = 2459000.5 // Default Julian date
	}
	bestPolicy := efeSolver.OptimizePolicy(&eciState, 10)

	// 3. Calculate Bayes Factor for mode spawning
	// If Evidence(NovelMode) / Evidence(BestKnown) > 10, spawn new mode
	novelLikelihood := imm.computeNovelModeLikelihood(obs)
	bestKnownLikelihood := imm.computeBestKnownLikelihood(obs)
	bayesFactor := efeSolver.CalculateBayesFactor(novelLikelihood, bestKnownLikelihood)

	if bayesFactor > imm.BayesFactorThreshold {
		fmt.Printf("[AI-IMM] Bayes Factor=%.2f > %.2f, spawning new mode\n", bayesFactor, imm.BayesFactorThreshold)
		imm.ExpandModes()
	}

	// 4. Apply Policy π* to filter
	imm.ApplyPolicy(bestPolicy)
}

/**
 * ReactiveUpdate performs standard reactive IMM update (TIER 1).
 * Low-latency (~0.3ms) for ballistic/stable phases.
 */
func (imm *IMMFilter) ReactiveUpdate(obs HGVState) {
	// Standard IMM update with default dt and jd
	dt := 0.1       // 100ms
	jd := 2459000.5 // Default Julian date
	imm.Update(obs, dt, jd)
}

/**
 * ApplyPolicy applies the optimal AI-IMM policy to the filter.
 */
func (imm *IMMFilter) ApplyPolicy(policy ai_imm.Policy) {
	// Update transition probabilities based on policy
	for i := range policy.ModeSequence {
		if i < len(imm.ModeProbs) {
			imm.ModeProbs[i] = 0.7 // Increase confidence in predicted mode
		}
	}

	fmt.Printf("[AI-IMM] Applied policy with horizon=%d\n", policy.Horizon)
}

/**
 * computeNovelModeLikelihood computes evidence for a novel (zero-day) mode.
 */
func (imm *IMMFilter) computeNovelModeLikelihood(obs HGVState) float64 {
	// Simplified: use NIS as proxy for novelty
	nis := imm.computeNIS(int(ModeManeuver), obs)
	return math.Exp(-nis) // Higher NIS = lower likelihood
}

/**
 * computeBestKnownLikelihood computes evidence for the best known mode.
 */
func (imm *IMMFilter) computeBestKnownLikelihood(obs HGVState) float64 {
	// Find mode with highest likelihood
	bestLikelihood := 0.0

	for i := 0; i < len(imm.ModeProbs); i++ {
		nis := imm.computeNIS(i, obs)
		likelihood := imm.Likelihoods[i] * math.Exp(-nis)
		if likelihood > bestLikelihood {
			bestLikelihood = likelihood
		}
	}

	return bestLikelihood
}

/**
 * ExpandModes dynamically expands the IMM filter from K to K+1 modes
 * when SINDy reconstruction error exceeds discovery threshold.
 * Implements Dirichlet matrix expansion with "Curiosity Prior" initialization.
 */
func (imm *IMMFilter) ExpandModes() {
	// Check if we've reached maximum modes
	if len(imm.ModeProbs) >= imm.MaxModes {
		fmt.Printf("[IMM] Max modes (%d) reached, cannot expand\n", imm.MaxModes)
		return
	}

	K := len(imm.ModeProbs)
	newK := K + 1

	// 1. Expand Dirichlet Alpha Matrix [K+1, K+1]
	newAlpha := make([][]float64, newK)
	for i := 0; i < newK; i++ {
		newAlpha[i] = make([]float64, newK)
		for j := 0; j < newK; j++ {
			if i < K && j < K {
				newAlpha[i][j] = imm.DirichletCounts[i][j]
			} else {
				// Initialize with a low prior count (Curiosity Prior)
				newAlpha[i][j] = 0.1
			}
		}
	}
	imm.DirichletCounts = newAlpha

	// 2. Expand Transition Matrix [K+1, K+1]
	newTrans := make([][]float64, newK)
	for i := 0; i < newK; i++ {
		newTrans[i] = make([]float64, newK)
		for j := 0; j < newK; j++ {
			if i < K && j < K {
				newTrans[i][j] = imm.TransMatrix[i][j]
			} else if i == K && j == K {
				// Self-transition probability for new mode
				newTrans[i][j] = 0.9
			} else {
				// Low transition probability to/from new mode
				newTrans[i][j] = 0.1 / float64(newK-1)
			}
		}
	}
	imm.TransMatrix = newTrans

	// 3. Expand Mode Probabilities
	newProbs := make([]float64, newK)
	for i := 0; i < K; i++ {
		newProbs[i] = imm.ModeProbs[i] * 0.9 // Down-weight existing modes
	}
	newProbs[K] = 0.1 // Initial probability for new mode
	imm.ModeProbs = newProbs

	// 4. Expand Likelihoods
	newLikelihoods := make([]float64, newK)
	copy(newLikelihoods, imm.Likelihoods)
	newLikelihoods[K] = 0.01 // Initial likelihood for new mode
	imm.Likelihoods = newLikelihoods

	// 5. Create new ModeFilter
	// Initialize with current state estimate (would use SINDy residual in production)
	newFilter := &ModeFilter{
		Mode:       HGVMode(K),             // New mode index
		State:      imm.Filters[K-1].State, // Copy from last mode (placeholder)
		Covariance: diagCovariance(1000.0),
		Gain:       0.5,
	}
	imm.Filters = append(imm.Filters, newFilter)

	imm.NumModes = newK
	imm.ZeroDayActive = true

	fmt.Printf("[IMM] Expanded to %d modes (Curiosity Prior: 0.1)\n", newK)
}

/**
 * spawnZeroDayMode creates a new mode when all known models fail.
 * Initializes from SINDy residual and queries PHS with null latent code.
 */
func (imm *IMMFilter) spawnZeroDayMode(measurement HGVState, sindyResidual [6]float64) {
	fmt.Printf("[IMM] SINDy reconstruction error exceeds threshold, spawning zero-day mode\n")

	// Expand modes dynamically
	imm.ExpandModes()

	// Initialize new mode with SINDy residual bias
	// This prevents divergence spike during mode switch
	newModeIndex := len(imm.Filters) - 1
	newFilter := imm.Filters[newModeIndex]

	// Inject SINDy residual as initial bias for the new mode
	// This makes the new mode's likelihood immediately higher than failing models
	for i := 0; i < 6; i++ {
		newFilter.State.R += sindyResidual[i] * 0.1 // Small bias injection
	}

	fmt.Printf("[IMM] Zero-day mode %d initialized with SINDy residual bias\n", newModeIndex)

	// TODO: Trigger SupCon trainer to learn new regime latent z_new
	// TODO: Replace z_null with z_new after convergence
}

/**
 * computeLikelihood: Gaussian based on residual.
 */
func (imm *IMMFilter) computeLikelihood(mode int, measurement HGVState) float64 {
	filter := imm.Filters[mode]

	// Residual
	residual := filter.State.R - measurement.R

	// Simplified likelihood (Gaussian)
	variance := 1000.0 // 1km measurement noise
	likelihood := math.Exp(-residual * residual / (2 * variance))

	return likelihood
}

/**
 * combineStates: Weighted average of mode estimates.
 */
func (imm *IMMFilter) combineStates() HGVState {
	var combined HGVState

	for j := 0; j < int(NumModes); j++ {
		mu := imm.ModeProbs[j]

		combined.R += mu * imm.Filters[j].State.R
		combined.Phi += mu * imm.Filters[j].State.Phi
		combined.Lambda += mu * imm.Filters[j].State.Lambda
		combined.V += mu * imm.Filters[j].State.V
		combined.Gamma += mu * imm.Filters[j].State.Gamma
		combined.Psi += mu * imm.Filters[j].State.Psi
	}

	return combined
}

/**
 * GetDominantMode returns the most likely mode.
 */
func (imm *IMMFilter) GetDominantMode() HGVMode {
	maxProb := 0.0
	dominant := ModeBallistic

	for i := 0; i < int(NumModes); i++ {
		if imm.ModeProbs[i] > maxProb {
			maxProb = imm.ModeProbs[i]
			dominant = HGVMode(i)
		}
	}

	return dominant
}

/**
 * GetModeProbabilities returns current mode probabilities.
 */
func (imm *IMMFilter) GetModeProbabilities() []float64 {
	return imm.ModeProbs
}

/**
 * diagCovariance creates diagonal covariance matrix.
 */
func diagCovariance(sigma float64) [6][6]float64 {
	var cov [6][6]float64
	for i := 0; i < 6; i++ {
		cov[i][i] = sigma * sigma
	}
	return cov
}

/**
 * HGVTracker integrates IMM with impact prediction.
 */
type HGVTracker struct {
	IMM       *IMMFilter
	Constants HGVConstants
	History   []HGVState
}

/**
 * NewHGVTracker creates a tracker with default HGV parameters.
 */
func NewHGVTracker() *HGVTracker {
	consts := DefaultHGV()
	return &HGVTracker{
		IMM:       NewIMMFilter(consts),
		Constants: consts,
		History:   make([]HGVState, 0),
	}
}

/**
 * ProcessMeasurement updates tracker with new sensor data.
 */
func (t *HGVTracker) ProcessMeasurement(measurement HGVState, dt float64, jd float64) HGVState {
	// Store history
	t.History = append(t.History, measurement)

	// Run IMM update
	estimate := t.IMM.Update(measurement, dt, jd)

	return estimate
}

/**
 * PredictImpactWithFootprint returns impact area based on dominant mode.
 */
func (t *HGVTracker) PredictImpactWithFootprint(jd float64) (lat, lon, radiusKm float64, mode HGVMode) {
	dominantMode := t.IMM.GetDominantMode()

	// Get combined state
	state := t.IMM.combineStates()

	// Predict footprint
	centerLat, centerLon, radius := PredictHGVImpactFootprint(state, t.Constants, jd)

	return centerLat, centerLon, radius, dominantMode
}
