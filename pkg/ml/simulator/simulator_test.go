package simulator

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGenerateSyntheticScenarios(t *testing.T) {
	cfg := GeneratorConfig{Count: 1200, Seed: 9}
	scenarios := GenerateSyntheticScenarios(cfg)
	if len(scenarios) != cfg.Count {
		t.Fatalf("expected %d scenarios, got %d", cfg.Count, len(scenarios))
	}

	hasConjunction := false
	hasRapid := false
	hasASAT := false
	hasProximity := false
	hasReentry := false
	for _, s := range scenarios {
		if s.Type == ScenarioConjunction {
			hasConjunction = true
		}
		if s.Type == ScenarioRapidThreat {
			hasRapid = true
		}
		if s.Type == ScenarioASAT {
			hasASAT = true
		}
		if s.Type == ScenarioProximity {
			hasProximity = true
		}
		if s.Type == ScenarioReentry {
			hasReentry = true
		}
		if s.RiskLabel < 0 || s.RiskLabel > 1 {
			t.Fatalf("risk label out of range: %f", s.RiskLabel)
		}
		if s.PriorityLabel < 0 || s.PriorityLabel > 300 {
			t.Fatalf("priority label out of range: %f", s.PriorityLabel)
		}
		if len(s.ScenarioRationale) == 0 {
			t.Fatalf("expected scenario rationale for %s", s.ScenarioID)
		}
		if len(s.DecisionFactors) == 0 {
			t.Fatalf("expected decision factors for %s", s.ScenarioID)
		}
		if s.AdversarialIntent < 0 || s.AdversarialIntent > 1 {
			t.Fatalf("adversarial intent out of range: %f", s.AdversarialIntent)
		}
	}
	if !(hasConjunction && hasRapid && hasASAT && hasProximity && hasReentry) {
		t.Fatalf("expected all scenario families to be sampled")
	}
}

func TestWriteScenariosJSONL(t *testing.T) {
	scenarios := GenerateSyntheticScenarios(GeneratorConfig{Count: 25, Seed: 2})
	d := t.TempDir()
	p := filepath.Join(d, "sim.jsonl")

	if err := WriteScenariosJSONL(p, scenarios); err != nil {
		t.Fatalf("failed to write jsonl: %v", err)
	}
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("expected file to exist: %v", err)
	}
}

func TestTrainingEnvStep(t *testing.T) {
	init := EnvState{
		MissDistanceKm:      0.05,
		RelativeVelocityKmS: 2.5,
		TimeToTCASeconds:    1200,
		FuelFraction:        0.85,
		TrackConfidence:     0.7,
	}
	env := NewTrainingEnv(init, 20)
	env.Reset(init)

	res := env.Step(ActionSmallBurn)
	if res.State.MissDistanceKm <= init.MissDistanceKm {
		t.Fatalf("expected miss distance increase after small burn")
	}
	if res.State.FuelFraction >= init.FuelFraction {
		t.Fatalf("expected fuel usage after maneuver")
	}
	if res.Info["risk_next"] < 0 || res.Info["risk_next"] > 1 {
		t.Fatalf("expected normalized risk")
	}
}
