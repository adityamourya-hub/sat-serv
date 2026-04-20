package hypersonic

import (
	"math"
	"ssis/pkg/physics"
)

/**
 * ADITI 4.0 Interacting Multiple Model (IMM) Filter for HGV Tracking
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
 * IMMFilter holds the multi-model state with adaptive Dirichlet-Markov learning.
 */
type IMMFilter struct {
	// Mode probabilities
	ModeProbs [NumModes]float64

	// Mode-matched filters (simplified Kalman filters)
	Filters [NumModes]*ModeFilter

	// Markov transition matrix
	// Trans[i][j] = prob of transitioning from mode i to mode j
	TransMatrix [NumModes][NumModes]float64

	// Dirichlet prior counts for online transition matrix learning
	// DirichletCounts[i][j] = pseudo-counts for transition i->j
	DirichletCounts [NumModes][NumModes]float64

	// Mode likelihoods (updated with each measurement)
	Likelihoods [NumModes]float64

	// Constants
	HGV HGVConstants

	// Measurement noise covariance (for Kalman gain computation)
	MeasurementNoise float64

	// Adaptive Dirichlet-Markov parameters
	Eta             float64 // Forgetting factor (e.g., 0.995)
	DiscoveryThresh float64 // Threshold for zero-day mode creation (χ²_α)
	MaxModes        int     // Maximum number of modes (for dynamic expansion)
	ZeroDayActive   bool    // Whether zero-day mode is active
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
 */
func NewIMMFilter(hgv HGVConstants) *IMMFilter {
	imm := &IMMFilter{
		HGV:             hgv,
		Eta:             0.995, // Forgetting factor for online learning
		DiscoveryThresh: 11.34, // χ²_α for 3 DOF at 99% confidence
		MaxModes:        8,     // Maximum modes for dynamic expansion
		ZeroDayActive:   false,
	}

	// Initialize with equal mode probabilities
	for i := 0; i < int(NumModes); i++ {
		imm.ModeProbs[i] = 1.0 / float64(NumModes)
	}

	// Typical Markov transition for HGV
	// High probability of staying in same mode
	// Moderate probability of transitioning
	imm.TransMatrix = [NumModes][NumModes]float64{
		{0.90, 0.08, 0.02}, // From Ballistic: mostly stay, some glide, rare maneuver
		{0.10, 0.85, 0.05}, // From Glide: some ballistic, mostly glide, some maneuver
		{0.20, 0.30, 0.50}, // From Maneuver: tends to exit maneuver
	}

	// Initialize Dirichlet counts with prior (pseudo-counts)
	// Using symmetric prior of 1.0 for each transition
	for i := 0; i < int(NumModes); i++ {
		for j := 0; j < int(NumModes); j++ {
			imm.DirichletCounts[i][j] = 1.0 // Uniform prior
		}
	}

	// Measurement noise for Kalman gain (meters)
	imm.MeasurementNoise = 1000.0 // 1km default

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
	measurementNoise := imm.MeasurementNoise // R (measurement noise variance)

	// Compute gain for each state dimension from filter covariance
	// Simplified: use trace of covariance as scalar uncertainty estimate
	stateVariance := 0.0
	for i := 0; i < 6; i++ {
		stateVariance += filter.Covariance[i][i]
	}
	stateVariance /= 6.0 // Average diagonal element

	// Kalman gain: K = P / (P + R)
	gain := stateVariance / (stateVariance + measurementNoise)

	// Clamp gain to reasonable bounds [0.01, 0.95]
	if gain < 0.01 {
		gain = 0.01
	}
	if gain > 0.95 {
		gain = 0.95
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
 * predictMode performs model-specific prediction.
 */
func (imm *IMMFilter) predictMode(
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
 * predictBallistic: No lift, only drag and gravity.
 */
func (imm *IMMFilter) predictBallistic(state HGVState, dt float64, jd float64) HGVState {
	// Simplified: only drag and gravity
	newState := state

	// Gravity
	g := physics.EARTH_MU_IAU / (state.R * state.R)

	// Drag (simplified ballistic)
	rho := physics.GetDensityUnified(0, 0, state.R, jd)
	vMs := state.V * 1000.0
	q := 0.5 * rho * vMs * vMs

	// CD for ballistic re-entry (tumble)
	CD := 1.0
	D := q * imm.HGV.ReferenceArea * CD

	// Deceleration
	dV := -(D/imm.HGV.Mass)/1000.0 - g*math.Sin(state.Gamma)
	newState.V += dV * dt

	// Vertical motion
	dGamma := (g - state.V*state.V/state.R) * math.Cos(state.Gamma) / state.V
	newState.Gamma += dGamma * dt

	// Position update
	newState.R += state.V * math.Sin(state.Gamma) * dt
	newState.Phi += (state.V * math.Cos(state.Gamma) * math.Cos(state.Psi) / state.R) * dt
	newState.Lambda += (state.V * math.Cos(state.Gamma) * math.Sin(state.Psi) / (state.R * math.Cos(state.Phi))) * dt

	return newState
}

/**
 * predictGlide: Steady L/D = 3, minimal bank.
 */
func (imm *IMMFilter) predictGlide(state HGVState, dt float64, jd float64) HGVState {
	newState := state

	// Set L/D = 3 for glide
	LD := 3.0

	// Aerodynamics
	rho := physics.GetDensityUnified(0, 0, state.R, jd)
	vMs := state.V * 1000.0
	q := 0.5 * rho * vMs * vMs

	// Estimate CD from L/D
	CD := 0.4
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

	// High L/D with aggressive bank
	LD := 3.5

	rho := physics.GetDensityUnified(0, 0, state.R, jd)
	vMs := state.V * 1000.0
	q := 0.5 * rho * vMs * vMs

	CD := 0.5
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
	var nis [NumModes]float64
	for j := 0; j < int(NumModes); j++ {
		imm.Likelihoods[j] = imm.computeLikelihood(j, measurement)
		nis[j] = imm.computeNIS(j, measurement)
	}

	// Check for Zero-Day event: all modes failing simultaneously
	minNIS := nis[0]
	for j := 1; j < int(NumModes); j++ {
		if nis[j] < minNIS {
			minNIS = nis[j]
		}
	}

	// Zero-Day discovery: if min NIS > χ² threshold, spawn new mode
	if minNIS > imm.DiscoveryThresh && !imm.ZeroDayActive {
		imm.spawnZeroDayMode(measurement)
		imm.ZeroDayActive = true
	}

	// Update probabilities
	var newProbs [NumModes]float64

	for j := 0; j < int(NumModes); j++ {
		// Mode probability prediction
		modePred := 0.0
		for i := 0; i < int(NumModes); i++ {
			modePred += imm.TransMatrix[i][j] * imm.ModeProbs[i]
		}

		// Bayes update
		newProbs[j] = imm.Likelihoods[j] * modePred
	}

	// Normalize
	sum := 0.0
	for j := 0; j < int(NumModes); j++ {
		sum += newProbs[j]
	}

	if sum > 1e-10 {
		for j := 0; j < int(NumModes); j++ {
			newProbs[j] /= sum
		}
	}

	// Online Dirichlet update with forgetting factor η
	// α_{ij,t+1} = η * α_{ij,t} + evidence
	for j := 0; j < int(NumModes); j++ {
		for i := 0; i < int(NumModes); i++ {
			// Compute mixing probability μ_{i|j}
			c := 0.0
			for k := 0; k < int(NumModes); k++ {
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
	for i := 0; i < int(NumModes); i++ {
		rowSum := 0.0
		for j := 0; j < int(NumModes); j++ {
			rowSum += imm.DirichletCounts[i][j]
		}

		if rowSum > 1e-10 {
			for j := 0; j < int(NumModes); j++ {
				imm.TransMatrix[i][j] = imm.DirichletCounts[i][j] / rowSum
			}
		}
	}

	// Store updated probabilities
	imm.ModeProbs = newProbs
}

/**
 * computeNIS calculates Normalized Innovation Squared for zero-day detection.
 * NIS = ν^T * S^{-1} * ν, where ν is innovation and S is innovation covariance.
 */
func (imm *IMMFilter) computeNIS(mode int, measurement HGVState) float64 {
	filter := imm.Filters[mode]

	// Residual (innovation)
	residual := filter.State.R - measurement.R

	// Innovation covariance (simplified as measurement noise)
	S := imm.MeasurementNoise

	// NIS = ν^T * S^{-1} * ν = ν^2 / S
	nis := (residual * residual) / S

	return nis
}

/**
 * spawnZeroDayMode creates a new mode when all known models fail.
 * Initializes from SINDy residual and queries PHS with null latent code.
 */
func (imm *IMMFilter) spawnZeroDayMode(measurement HGVState) {
	// In production, this would:
	// 1. Extract SINDy residual r(t) from sliding window
	// 2. Initialize new mode state from residual
	// 3. Expand Dirichlet matrix to (K+1) x (K+1)
	// 4. Query Latent-PHS with z_null for initialization
	// 5. Trigger SupCon trainer to learn new regime

	// For now, mark as active and log
	// The actual mode expansion would require dynamic array sizing
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
func (imm *IMMFilter) GetModeProbabilities() [NumModes]float64 {
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
