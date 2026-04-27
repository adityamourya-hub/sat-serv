package tracking

import (
	"math"

	"ssis/pkg/physics"
)

type IMMMode int

const (
	ModeBallistic IMMMode = iota
	ModeManeuvering
	ModeStationKeeping
	ModeCoasting
	NumModes
)

var ModeNames = map[IMMMode]string{
	ModeBallistic:      "BALLISTIC",
	ModeManeuvering:    "MANEUVERING",
	ModeStationKeeping: "STATION_KEEPING",
	ModeCoasting:       "COASTING",
}

type IMMFilter struct {
	// Mode probabilities
	ModeProb [NumModes]float64

	// State estimates for each mode
	States [NumModes]*ExtendedKalmanFilter

	// Mode transition probabilities (Markov chain)
	TransitionMatrix [NumModes][NumModes]float64

	// Mixing probabilities
	MixingProb [NumModes][NumModes]float64

	// Mode likelihoods
	ModeLikelihood [NumModes]float64

	// Current mode estimate
	CurrentMode IMMMode

	// Covariance mixing weight
	MixingWeight float64

	SatelliteID int
	LastUpdate  float64
}

func NewIMMFilter(satelliteID int, initialState physics.State) *IMMFilter {
	imm := &IMMFilter{
		SatelliteID:  satelliteID,
		LastUpdate:   initialState.EpochJD,
		MixingWeight: 0.1,
	}

	// Initialize mode probabilities (favor ballistic)
	imm.ModeProb[ModeBallistic] = 0.7
	imm.ModeProb[ModeManeuvering] = 0.1
	imm.ModeProb[ModeStationKeeping] = 0.15
	imm.ModeProb[ModeCoasting] = 0.05

	// Initialize transition matrix
	// Rows = from mode, Columns = to mode
	imm.TransitionMatrix = [NumModes][NumModes]float64{
		// Ballistic: high probability of staying ballistic
		{ModeBallistic: 0.95, ModeManeuvering: 0.02, ModeStationKeeping: 0.02, ModeCoasting: 0.01},
		// Maneuvering: likely to transition to station-keeping or ballistic
		{ModeBallistic: 0.3, ModeManeuvering: 0.6, ModeStationKeeping: 0.1, ModeCoasting: 0.0},
		// Station-keeping: tends to stay or go ballistic
		{ModeBallistic: 0.4, ModeManeuvering: 0.1, ModeStationKeeping: 0.45, ModeCoasting: 0.05},
		// Coasting: likely to go ballistic or station-keep
		{ModeBallistic: 0.5, ModeManeuvering: 0.05, ModeStationKeeping: 0.25, ModeCoasting: 0.2},
	}

	// Initialize EKF for each mode with different process noise
	imm.initModeFilters(initialState)

	return imm
}

func (imm *IMMFilter) initModeFilters(initialState physics.State) {
	// Ballistic: low process noise (natural orbital motion)
	ballistic := NewEKF(imm.SatelliteID, initialState, 0)
	ballistic.Q = imm.buildProcessNoise(0.001) // Very low noise
	imm.States[ModeBallistic] = ballistic

	// Maneuvering: high process noise (unknown thrust)
	maneuvering := NewEKF(imm.SatelliteID, initialState, 0)
	maneuvering.Q = imm.buildProcessNoise(10.0) // High noise for maneuvers
	imm.States[ModeManeuvering] = maneuvering

	// Station-keeping: moderate noise (controlled thrust)
	stationKeeping := NewEKF(imm.SatelliteID, initialState, 0)
	stationKeeping.Q = imm.buildProcessNoise(1.0) // Moderate noise
	imm.States[ModeStationKeeping] = stationKeeping

	// Coasting: low noise (no thrust)
	coasting := NewEKF(imm.SatelliteID, initialState, 0)
	coasting.Q = imm.buildProcessNoise(0.01)
	imm.States[ModeCoasting] = coasting
}

func (imm *IMMFilter) buildProcessNoise(scale float64) [28]float64 {
	return [28]float64{
		scale * 100, 0, 0, 0, 0, 0, 0,
		scale * 100, 0, 0, 0, 0, 0,
		scale * 100, 0, 0, 0, 0,
		scale * 1e-4, 0, 0, 0,
		scale * 1e-4, 0, 0,
		scale * 1e-4, 0,
		scale * 100,
	}
}

func (imm *IMMFilter) Predict(dt float64, propagator func(physics.State, float64) physics.State) {
	// Step 1: Mixing - compute mixing probabilities
	imm.computeMixingProbabilities()

	// Step 2: Mixed initialization for each mode
	imm.mixStates()

	// Step 3: Mode-matched prediction
	for m := IMMMode(0); m < NumModes; m++ {
		if imm.States[m] != nil {
			imm.States[m].Predict(dt, propagator)
		}
	}

	// Step 4: Update mode probabilities
	imm.updateModeProbabilities()

	// Step 5: Estimate current mode
	imm.estimateMode()

	imm.LastUpdate += dt / 86400.0
}

func (imm *IMMFilter) computeMixingProbabilities() {
	// Compute P(m_j | X_k-1, Z_k-1) for all mode pairs
	// Using Bayes' rule with transition matrix

	sum := 0.0
	for j := IMMMode(0); j < NumModes; j++ {
		rowSum := 0.0
		for i := IMMMode(0); i < NumModes; i++ {
			rowSum += imm.TransitionMatrix[i][j] * imm.ModeProb[i]
		}
		for i := IMMMode(0); i < NumModes; i++ {
			if rowSum > 0 {
				imm.MixingProb[i][j] = (imm.TransitionMatrix[i][j] * imm.ModeProb[i]) / rowSum
			} else {
				imm.MixingProb[i][j] = 1.0 / float64(NumModes)
			}
		}
		sum += rowSum
	}
}

func (imm *IMMFilter) mixStates() {
	// Mix state estimates based on mixing probabilities
	// This is a simplified version - full IMM would weight states by mixing probabilities

	for m := IMMMode(0); m < NumModes; m++ {
		if imm.States[m] == nil {
			continue
		}

		// Get weighted average of all mode states
		var weightedX [7]float64
		var totalWeight float64

		for i := IMMMode(0); i < NumModes; i++ {
			if imm.States[i] == nil {
				continue
			}
			weight := imm.MixingProb[i][m]
			for j := 0; j < 7; j++ {
				weightedX[j] += weight * imm.States[i].x[j]
			}
			totalWeight += weight
		}

		if totalWeight > 0 {
			for j := 0; j < 7; j++ {
				weightedX[j] /= totalWeight
			}
			// Blend with current estimate
			blendFactor := imm.MixingWeight
			for j := 0; j < 7; j++ {
				imm.States[m].x[j] = blendFactor*weightedX[j] + (1-blendFactor)*imm.States[m].x[j]
			}
		}
	}
}

func (imm *IMMFilter) updateModeProbabilities() {
	// Compute likelihood for each mode based on innovation
	sum := 0.0

	for m := IMMMode(0); m < NumModes; m++ {
		if imm.States[m] != nil {
			// Simplified: use position covariance trace as uncertainty
			cov := imm.States[m].GetPositionCovariance()
			uncertainty := (cov[0] + cov[4] + cov[8]) / 3.0

			// Higher uncertainty = lower likelihood for this mode
			if uncertainty > 0 {
				imm.ModeLikelihood[m] = 1.0 / math.Sqrt(uncertainty)
			} else {
				imm.ModeLikelihood[m] = 1.0
			}
		} else {
			imm.ModeLikelihood[m] = 0.0
		}
		sum += imm.ModeLikelihood[m] * imm.ModeProb[m]
	}

	// Update mode probabilities
	if sum > 0 {
		for m := IMMMode(0); m < NumModes; m++ {
			imm.ModeProb[m] = (imm.ModeLikelihood[m] * imm.ModeProb[m]) / sum
		}
	}
}

func (imm *IMMFilter) estimateMode() {
	// Select mode with highest probability
	maxProb := 0.0
	for m := IMMMode(0); m < NumModes; m++ {
		if imm.ModeProb[m] > maxProb {
			maxProb = imm.ModeProb[m]
			imm.CurrentMode = m
		}
	}
}

func (imm *IMMFilter) Update(measurement [3]float64) (innovation [3]float64, mahalanobis float64) {
	// Update each mode-matched filter
	for m := IMMMode(0); m < NumModes; m++ {
		if imm.States[m] != nil {
			inn, maha := imm.States[m].Update(measurement)
			if m == imm.CurrentMode {
				innovation = inn
				mahalanobis = maha
			}
		}
	}

	// Recompute mode probabilities after update
	imm.updateModeProbabilities()
	imm.estimateMode()

	return innovation, mahalanobis
}

func (imm *IMMFilter) GetState() physics.State {
	// Return state from most likely mode
	if imm.States[imm.CurrentMode] != nil {
		return imm.States[imm.CurrentMode].GetState()
	}
	return physics.State{}
}

func (imm *IMMFilter) GetPositionCovariance() [9]float64 {
	if imm.States[imm.CurrentMode] != nil {
		return imm.States[imm.CurrentMode].GetPositionCovariance()
	}
	return [9]float64{}
}

func (imm *IMMFilter) GetModeProbability(mode IMMMode) float64 {
	return imm.ModeProb[mode]
}

func (imm *IMMFilter) GetCurrentModeName() string {
	return ModeNames[imm.CurrentMode]
}

func (imm *IMMFilter) IsManeuvering() bool {
	return imm.CurrentMode == ModeManeuvering && imm.ModeProb[ModeManeuvering] > 0.5
}

func (imm *IMMFilter) GetManeuverProbability() float64 {
	return imm.ModeProb[ModeManeuvering]
}

type ReachabilityRiskIndex struct {
	imm *IMMFilter
}

func NewReachabilityRiskIndex(imm *IMMFilter) *ReachabilityRiskIndex {
	return &ReachabilityRiskIndex{imm: imm}
}

func (r *ReachabilityRiskIndex) Compute(primaryState, secondaryState physics.State, timeToTCASeconds float64) float64 {
	// RRI = P(collision) * severity * intent_probability

	// Compute miss distance
	dx := primaryState.Pos[0] - secondaryState.Pos[0]
	dy := primaryState.Pos[1] - secondaryState.Pos[1]
	dz := primaryState.Pos[2] - secondaryState.Pos[2]
	missDistance := math.Sqrt(dx*dx + dy*dy + dz*dz) // km

	// Probability of collision (simplifiedPc formula)
	// Pc = (1/2) * erfc(missDistance / (sqrt(2) * sigma_miss))
	sigmaMiss := 0.1 // 100m uncertainty (km)
	if timeToTCASeconds > 0 {
		sigmaMiss = 0.1 + 0.01*timeToTCASeconds/3600.0 // Grows with time
	}

	normalizedMiss := missDistance / (math.Sqrt(2) * sigmaMiss)
	Pc := 0.5 * math.Erfc(normalizedMiss)

	// Severity based on miss distance
	severity := 1.0
	if missDistance < 1.0 {
		severity = 10.0 // < 1km is critical
	} else if missDistance < 10.0 {
		severity = 5.0
	} else if missDistance < 100.0 {
		severity = 2.0
	}

	// Intent probability from IMM
	intentProb := r.imm.GetManeuverProbability()

	// Combined RRI
	RRI := Pc * severity * (1.0 + intentProb)

	return RRI
}

func (r *ReachabilityRiskIndex) GetProbabilityBubble() (rx, ry, rz float64) {
	cov := r.imm.GetPositionCovariance()
	// 3-sigma bounds
	rx = 3 * math.Sqrt(cov[0])
	ry = 3 * math.Sqrt(cov[4])
	rz = 3 * math.Sqrt(cov[8])
	return rx, ry, rz
}
