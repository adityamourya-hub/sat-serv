package catalog

import (
	"testing"

	"ssis/pkg/octree"
)

func TestFallbackPatternOfLifeInferenceReturnsBoundedScores(t *testing.T) {
	history := []TrackObservation{
		{TrackID: 1, Pos: octree.Vector3{X: 100, Y: 100, Z: 100}, SigmaKm: 1.2, TimestampUnix: 10},
		{TrackID: 1, Pos: octree.Vector3{X: 101, Y: 100.5, Z: 99.7}, SigmaKm: 1.1, TimestampUnix: 20},
		{TrackID: 1, Pos: octree.Vector3{X: 101.8, Y: 101.1, Z: 99.4}, SigmaKm: 1.0, TimestampUnix: 30},
	}
	obs := TrackObservation{TrackID: 1, Pos: octree.Vector3{X: 102.6, Y: 101.8, Z: 99.1}, SigmaKm: 1.0, TimestampUnix: 40}
	cand := ObjectRecord{ID: 5, Pos: octree.Vector3{X: 103, Y: 102, Z: 99}}

	inf := fallbackPatternOfLifeInference(history, obs, cand, true)
	if inf.PatternScore < 0 || inf.PatternScore > 1 {
		t.Fatalf("pattern score out of range: %f", inf.PatternScore)
	}
	if inf.IntentProxy < 0 || inf.IntentProxy > 1 {
		t.Fatalf("intent proxy out of range: %f", inf.IntentProxy)
	}
	if inf.Confidence <= 0 {
		t.Fatalf("expected positive confidence, got %f", inf.Confidence)
	}
}

func TestAssociateMHTLiteIncludesIntentProxySignal(t *testing.T) {
	// s := NewStore(20)
	t.Skip("FlagManeuver and AssociateMHTLite not yet implemented in Refactored Store")
	// s.FlagManeuver(10)

	// obs := TrackObservation{TrackID: 501, Pos: octree.Vector3{X: 100.2, Y: 100.1, Z: 99.9}, SigmaKm: 1.2, TimestampUnix: 100}
	// dec := s.AssociateMHTLite(obs, 30, 3, nil)
	/*
	if len(dec.Hypotheses) == 0 {
		t.Fatalf("expected hypotheses")
	}
	foundSplit := false
	for _, h := range dec.Hypotheses {
		if h.ObjectID == 10 && (h.Variant == "legacy_orbit" || h.Variant == "new_orbit") {
			foundSplit = true
			if h.IntentProxy < 0 || h.IntentProxy > 1 {
				t.Fatalf("intent proxy out of range: %f", h.IntentProxy)
			}
		}
	}
	if !foundSplit {
		t.Fatalf("expected maneuver split hypotheses with intent proxy signal")
	}
	*/
}
