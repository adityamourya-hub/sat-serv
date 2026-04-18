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
	next := e.Current

	switch action {
	case ActionMonitor:
		next.TimeToTCASeconds = math.Max(0, next.TimeToTCASeconds-90)
		next.TrackConfidence = bounded(next.TrackConfidence+0.01, 0, 1)
	case ActionSensorRetask:
		next.TimeToTCASeconds = math.Max(0, next.TimeToTCASeconds-75)
		next.TrackConfidence = bounded(next.TrackConfidence+0.08, 0, 1)
		next.FuelFraction = math.Max(0, next.FuelFraction-0.005)
	case ActionSmallBurn:
		next.TimeToTCASeconds = math.Max(0, next.TimeToTCASeconds-60)
		next.MissDistanceKm = bounded(next.MissDistanceKm+0.03, 0.001, 2.0)
		next.RelativeVelocityKmS = math.Max(0.05, next.RelativeVelocityKmS-0.05)
		next.FuelFraction = math.Max(0, next.FuelFraction-0.03)
		next.TrackConfidence = bounded(next.TrackConfidence-0.01, 0, 1)
	case ActionLargeBurn:
		next.TimeToTCASeconds = math.Max(0, next.TimeToTCASeconds-55)
		next.MissDistanceKm = bounded(next.MissDistanceKm+0.08, 0.001, 3.5)
		next.RelativeVelocityKmS = math.Max(0.05, next.RelativeVelocityKmS-0.12)
		next.FuelFraction = math.Max(0, next.FuelFraction-0.08)
		next.TrackConfidence = bounded(next.TrackConfidence-0.03, 0, 1)
	}

	next.ThreatLevel = stateRisk(next)
	nextRisk := next.ThreatLevel

	reward := (prevRisk - nextRisk) * 10.0
	if action == ActionMonitor {
		reward -= 0.08
	}
	if action == ActionSensorRetask {
		reward -= 0.03
	}
	if action == ActionSmallBurn {
		reward -= 0.15
	}
	if action == ActionLargeBurn {
		reward -= 0.30
	}

	if next.MissDistanceKm >= 0.25 && nextRisk < 0.08 {
		reward += 1.5
	}
	if next.TimeToTCASeconds <= 0 && nextRisk > 0.2 {
		reward -= 2.5
	}
	if next.FuelFraction <= 0.02 {
		reward -= 0.8
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
	sigma := 0.03
	gaussian := math.Exp(-0.5 * math.Pow(s.MissDistanceKm/math.Max(sigma, 1e-6), 2.0))
	velocity := 1.0 / (1.0 + s.RelativeVelocityKmS*0.02)
	timePressure := 1.0 - math.Exp(-math.Max(s.TimeToTCASeconds, 0)/900.0)
	confidenceBias := 0.8 + 0.2*s.TrackConfidence
	fuelPenalty := 1.0 + (1.0-s.FuelFraction)*0.25

	risk := gaussian * velocity * (0.5 + 0.5*timePressure) * confidenceBias * fuelPenalty
	return bounded(risk, 0.0, 1.0)
}
