package rl

import "testing"

func TestRecommendTaskingFallbackSortsByUncertaintyAndStaleness(t *testing.T) {
	agent := &SteeringAgent{session: nil}
	features := []SDAObjectFeature{
		{ObjectID: 101, CovarianceTrace: 0.01, TimeSinceLastObs: 20, IsProtected: 0, PriorityClass: 1, SNREstimate: 18, Reserved: 0.1},
		{ObjectID: 202, CovarianceTrace: 0.05, TimeSinceLastObs: 200, IsProtected: 1, PriorityClass: 5, SNREstimate: 10, Reserved: 0.1},
		{ObjectID: 303, CovarianceTrace: 0.03, TimeSinceLastObs: 90, IsProtected: 0, PriorityClass: 3, SNREstimate: 12, Reserved: 0.2},
	}

	constraints := DefaultRadarConstraints()
	constraints.MaxTaskings = 3

	taskings, err := agent.RecommendTasking(features, constraints)
	if err != nil {
		t.Fatalf("RecommendTasking failed: %v", err)
	}
	if len(taskings) != 3 {
		t.Fatalf("expected 3 taskings, got %d", len(taskings))
	}
	if taskings[0].ObjectID != 202 {
		t.Fatalf("expected object 202 to lead ranking, got %d", taskings[0].ObjectID)
	}
}

func TestRecommendTaskingAppliesSNRConstraint(t *testing.T) {
	agent := &SteeringAgent{session: nil}
	features := []SDAObjectFeature{
		{ObjectID: 1, CovarianceTrace: 0.02, TimeSinceLastObs: 50, IsProtected: 0, PriorityClass: 2, SNREstimate: 3.0, Reserved: 0.1},
	}

	constraints := DefaultRadarConstraints()
	constraints.MinSNR = 6.0
	constraints.MaxTaskings = 1

	taskings, err := agent.RecommendTasking(features, constraints)
	if err != nil {
		t.Fatalf("RecommendTasking failed: %v", err)
	}
	if len(taskings) != 1 {
		t.Fatalf("expected 1 tasking, got %d", len(taskings))
	}
	if taskings[0].Tasking != "defer_low_snr" {
		t.Fatalf("expected low-SNR defer tasking, got %s", taskings[0].Tasking)
	}
	if !taskings[0].ConstraintFiltered {
		t.Fatalf("expected constraint filtered flag for low-SNR target")
	}
}

func TestRankObjectsCompatibility(t *testing.T) {
	agent := &SteeringAgent{session: nil}
	features := []SDAObjectFeature{
		{ObjectID: 2, CovarianceTrace: 0.03, TimeSinceLastObs: 100, IsProtected: 1, PriorityClass: 4, SNREstimate: 12},
		{ObjectID: 1, CovarianceTrace: 0.01, TimeSinceLastObs: 10, IsProtected: 0, PriorityClass: 1, SNREstimate: 20},
	}

	ids, err := agent.RankObjects(features)
	if err != nil {
		t.Fatalf("RankObjects failed: %v", err)
	}
	if len(ids) == 0 {
		t.Fatalf("expected non-empty ranked object list")
	}
	if ids[0] != 2 {
		t.Fatalf("expected highest priority object ID 2, got %d", ids[0])
	}
}
