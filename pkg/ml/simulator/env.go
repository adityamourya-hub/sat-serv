package simulator

import "math"

type SimAction int

const (
	ActionMonitor SimAction = iota
	ActionSensorRetask
	ActionSmallBurn
	ActionLargeBurn
)

type EnvState struct {
	MissDistanceKm      float64
	RelativeVelocityKmS float64
	TimeToTCASeconds    float64
	FuelFraction        float64
	TrackConfidence     float64
	ThreatLevel         float64
	AltitudeKm          float64 // Altitude for orbital mechanics (km)
}

type EnvStepResult struct {
	State     EnvState
	Reward    float64
	Done      bool
	Truncated bool
	Info      map[string]float64
}

type TrainingEnv struct {
	Current     EnvState
	StepCount   int
	MaxSteps    int
	EpisodeRisk float64
}

func NewTrainingEnv(init EnvState, maxSteps int) *TrainingEnv {
	if maxSteps <= 0 {
		maxSteps = 30
	}
	return &TrainingEnv{Current: init, MaxSteps: maxSteps}
}

func (e *TrainingEnv) Reset(init EnvState) EnvState {
	e.Current = init
	e.StepCount = 0
	e.EpisodeRisk = stateRisk(init)
	return e.Current
}

func (e *TrainingEnv) Step(action SimAction) EnvStepResult {
	e.StepCount++
	prevRisk := stateRisk(e.Current)

	// Use physics-based dynamics instead of hardcoded linear transitions
	// Default altitude to LEO (500km) if not set
	altitudeKm := e.Current.AltitudeKm
	if altitudeKm <= 0 {
		altitudeKm = 500.0
	}

	next := ComputePhysicsStep(e.Current, action, altitudeKm)

	nextRisk := next.ThreatLevel

	// Fuel-aware reward scaling to prevent over-aggressive maneuvers
	// Action costs scale with remaining fuel to encourage conservation
	// When fuel is low, maneuver costs increase exponentially
	fuelFraction := next.FuelFraction
	fuelPenaltyMultiplier := 1.0
	if fuelFraction < 0.3 {
		// Exponential penalty when fuel < 30%
		fuelPenaltyMultiplier = 1.0 + math.Pow((0.3-fuelFraction)/0.3, 2.0)
	} else if fuelFraction < 0.5 {
		// Linear penalty when fuel < 50%
		fuelPenaltyMultiplier = 1.0 + (0.5 - fuelFraction)
	}

	// Base reward from risk reduction
	riskReward := (prevRisk - nextRisk) * 5.0 // Reduced from 10.0 to prevent overemphasis

	// Action costs (scaled by fuel penalty multiplier)
	var actionCost float64
	switch action {
	case ActionMonitor:
		actionCost = 0.05 // Small cost for passive monitoring
	case ActionSensorRetask:
		actionCost = 0.10 // Moderate cost for sensor retasking
	case ActionSmallBurn:
		actionCost = 1.0 // Significant cost for small burn (was 0.15)
	case ActionLargeBurn:
		actionCost = 3.0 // Large cost for large burn (was 0.30)
	default:
		actionCost = 0.0
	}

	// Apply fuel-aware scaling to action cost
	actionCost *= fuelPenaltyMultiplier

	reward := riskReward - actionCost

	// Bonus for safe resolution
	if next.MissDistanceKm >= 0.25 && nextRisk < 0.08 {
		reward += 2.0
	}

	// Penalty for missing TCA with high risk
	if next.TimeToTCASeconds <= 0 && nextRisk > 0.2 {
		reward -= 5.0 // Increased from 2.5
	}

	// Severe penalty for fuel exhaustion
	if next.FuelFraction <= 0.02 {
		reward -= 10.0 // Increased from 0.8
	}

	// Penalty for wasting fuel on low-risk scenarios
	if fuelFraction < 0.5 && nextRisk < 0.1 && (action == ActionSmallBurn || action == ActionLargeBurn) {
		reward -= 2.0 // Discourage unnecessary maneuvers when fuel is low
	}

	// Mission priority weighting
	// High-priority scenarios (ASAT, RapidThreat) have higher reward for successful resolution
	// This reflects real mission priorities where some threats are more critical
	missionPriorityBonus := 0.0
	if next.MissDistanceKm >= 0.25 && nextRisk < 0.08 {
		// Successful resolution bonus scales with scenario type priority
		// This is a simplified model - in practice would come from scenario metadata
		if e.Current.TimeToTCASeconds < 600 {
			missionPriorityBonus = 1.5 // Bonus for resolving high-urgency threats
		}
		reward += missionPriorityBonus
	}

	e.Current = next
	done := next.TimeToTCASeconds <= 0 || nextRisk < 0.02
	trunc := e.StepCount >= e.MaxSteps

	return EnvStepResult{
		State:     next,
		Reward:    reward,
		Done:      done,
		Truncated: trunc,
		Info: map[string]float64{
			"risk_prev": prevRisk,
			"risk_next": nextRisk,
			"fuel":      next.FuelFraction,
		},
	}
}

func stateRisk(s EnvState) float64 {
	// Use adaptive sigma based on miss distance to prevent near-zero Gaussian
	// for far-field conjunctions (e.g., reentry with 15km miss distance)
	// Default sigmaR, sigmaI, sigmaC to 0.03km if not available
	sigma := adaptiveSigmaFromCovariance(s.MissDistanceKm, 0.03, 0.03, 0.03)
	gaussian := math.Exp(-0.5 * math.Pow(s.MissDistanceKm/math.Max(sigma, 1e-6), 2.0))
	velocity := 1.0 / (1.0 + s.RelativeVelocityKmS*0.02)
	timePressure := 1.0 - math.Exp(-math.Max(s.TimeToTCASeconds, 0)/900.0)
	confidenceBias := 0.8 + 0.2*s.TrackConfidence
	fuelPenalty := 1.0 + (1.0-s.FuelFraction)*0.25

	risk := gaussian * velocity * (0.5 + 0.5*timePressure) * confidenceBias * fuelPenalty
	return bounded(risk, 0.0, 1.0)
}
