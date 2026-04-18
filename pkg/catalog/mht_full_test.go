package catalog

import (
	"math"
	"testing"
	"time"

	"ssis/pkg/octree"
)

func TestHOEngine_BasicAssociation(t *testing.T) {
	e := NewHOEngine(nil)
	e.kBest = 3

	// Setup Store with 2 tracks
	store := NewStore(50)
	t1 := ObjectRecord{ID: 1, Name: "SAT_A", Pos: octree.Vector3{X: 7000, Y: 0, Z: 0}, PosSigmaKm: 1.0}
	t2 := ObjectRecord{ID: 2, Name: "SAT_B", Pos: octree.Vector3{X: 0, Y: 7000, Z: 0}, PosSigmaKm: 1.0}
	store.ApplyDeltaWithRebuild(Delta{Upserts: []ObjectRecord{t1, t2}}, true)

	// Observations: 1 near t1, 1 near t2, 1 outlier (False Alarm)
	obs := []TrackObservation{
		{TrackID: 1, Pos: octree.Vector3{X: 7000.1, Y: 0.1, Z: 0}, SigmaKm: 0.5, TimestampUnix: time.Now().Unix()},
		{TrackID: 2, Pos: octree.Vector3{X: 0.1, Y: 7000.1, Z: 0}, SigmaKm: 0.5, TimestampUnix: time.Now().Unix()},
		{TrackID: 0, Pos: octree.Vector3{X: 5000, Y: 5000, Z: 5000}, SigmaKm: 0.5, TimestampUnix: time.Now().Unix()},
	}

	// Manually inject a cluster for test
	e.mu.Lock()
	e.clusters[0] = &MHTCluster{
		TrackIDs: []int{1, 2},
	}
	e.mu.Unlock()
	e.initializeCluster(e.clusters[0])

	// Process Scan
	err := e.ProcessScan(obs, store)
	if err != nil {
		t.Fatalf("ProcessScan failed: %v", err)
	}

	winner := e.GetWinner(0)
	if winner == nil {
		t.Fatal("Winner hypothesis is nil")
	}

	t.Logf("Winner Hypothesis Probability: %.4f", winner.Probability)
	t.Logf("Winner Assignments: %+v", winner.Assignments)

	// Verify assignments: 1 should map to obs 0, 2 should map to obs 1
	if winner.Assignments[1] != 0 {
		t.Errorf("Expected track 1 -> obs 0, got %d", winner.Assignments[1])
	}
	if winner.Assignments[2] != 1 {
		t.Errorf("Expected track 2 -> obs 1, got %d", winner.Assignments[2])
	}
}

func TestHOEngine_ProbabilityNormalization(t *testing.T) {
	e := NewHOEngine(nil)
	hyps := []*GlobalHypothesis{
		{LogLikelihood: -1.0},
		{LogLikelihood: -2.0},
		{LogLikelihood: -5.0},
	}

	e.normalize(hyps)

	sum := 0.0
	for _, h := range hyps {
		sum += h.Probability
	}

	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("Normalized probabilities sum to %f, expected 1.0", sum)
	}

	if hyps[0].Probability <= hyps[1].Probability {
		t.Errorf("Probabilities not ranked correctly: %f vs %f", hyps[0].Probability, hyps[1].Probability)
	}
}
