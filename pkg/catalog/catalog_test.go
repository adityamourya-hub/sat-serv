package catalog

import (
	"testing"

	"ssis/pkg/octree"
)

type mockAssociationModel struct{}

func (m mockAssociationModel) Name() string { return "mock_transformer_tracker" }

func (m mockAssociationModel) Associate(obs FusedTrackObservation, candidates []ObjectRecord) (TrackAssociationInference, error) {
	scores := make(map[int]float64, len(candidates))
	attn := make(map[int]float64, len(candidates))
	for _, c := range candidates {
		if c.ID == 2 {
			scores[c.ID] = 0.92
			attn[c.ID] = 0.81
		} else {
			scores[c.ID] = 0.40
			attn[c.ID] = 0.25
		}
	}
	return TrackAssociationInference{Scores: scores, Confidence: 0.88, Attention: attn}, nil
}

func TestStoreApplyAndCorrelate(t *testing.T) {
	s := NewStore(10)
	v := s.ApplyDeltaWithRebuild(Delta{
		Upserts: []ObjectRecord{
			{ID: 1, Name: "IRNSS-1I", Pos: octree.Vector3{X: 10, Y: 10, Z: 10}},
			{ID: 2, Name: "GSAT-30", Pos: octree.Vector3{X: 50, Y: 50, Z: 50}},
		},
	}, true)
	if v != 1 {
		t.Fatalf("Expected version 1, got %d", v)
	}

	cand := s.Correlate(TrackObservation{TrackID: 100, Pos: octree.Vector3{X: 12, Y: 10, Z: 9}}, 20, 2)
	if len(cand) == 0 {
		t.Fatalf("Expected correlation candidates")
	}
	if cand[0].Record.ID != 1 {
		t.Fatalf("Expected closest candidate ID 1, got %d", cand[0].Record.ID)
	}

	v = s.ApplyDeltaWithRebuild(Delta{Deletes: []int{1}}, true)
	if v != 2 {
		t.Fatalf("Expected version 2, got %d", v)
	}

	cand = s.Correlate(TrackObservation{TrackID: 101, Pos: octree.Vector3{X: 11, Y: 11, Z: 11}}, 20, 2)
	if len(cand) != 0 {
		t.Fatalf("Expected no candidates after delete, got %d", len(cand))
	}
}

func TestCorrelateAdvancedFusedObservation(t *testing.T) {
	s := NewStore(10)
	s.ApplyDeltaWithRebuild(Delta{
		Upserts: []ObjectRecord{
			{ID: 1, Name: "IRNSS", Pos: octree.Vector3{X: 100, Y: 100, Z: 100}},
			{ID: 2, Name: "GSAT", Pos: octree.Vector3{X: 110, Y: 108, Z: 105}},
		},
	}, true)

	obs := TrackObservation{
		TrackID: 555,
		Pos:     octree.Vector3{X: 111.5, Y: 106.8, Z: 104.7},
		SigmaKm: 2.2,
	}

	assoc := s.Correlate(obs, 30, 2)
	if len(assoc) == 0 {
		t.Fatalf("expected fused correlation candidates")
	}
	if assoc[0].Record.ID != 2 {
		t.Fatalf("expected closest fused candidate id=2, got %d", assoc[0].Record.ID)
	}
	if assoc[0].AssociationScore <= 0 {
		t.Fatalf("expected positive association score")
	}
}

func TestCorrelateAdvancedHybridModelRerank(t *testing.T) {
	t.Fatal("NewHybridAssociationEngine undefined in current catalog package — must pass before merge")
	s := NewStore(15)
	s.ApplyDeltaWithRebuild(Delta{
		Upserts: []ObjectRecord{
			{ID: 1, Name: "Obj-1", Pos: octree.Vector3{X: 5, Y: 5, Z: 5}},
			{ID: 2, Name: "Obj-2", Pos: octree.Vector3{X: 7, Y: 6, Z: 5}},
			{ID: 3, Name: "Obj-3", Pos: octree.Vector3{X: 25, Y: 25, Z: 25}},
		},
	}, true)

	obs := TrackObservation{TrackID: 99, Pos: octree.Vector3{X: 6, Y: 5.8, Z: 5.1}, SigmaKm: 1.5}
	_ = obs
	// engine := NewHybridAssociationEngine(mockAssociationModel{})
	// assoc := s.CorrelateAdvanced(obs, 30, 3, engine)
}

func TestAssociateMHTLiteManeuverSplitAndPromotion(t *testing.T) {
	t.Fatal("FlagManeuver and AssociateMHTLite missing from Store — must pass before merge")
}

func TestAssociateMHTLiteUCT(t *testing.T) {
	t.Fatal("AssociateMHTLite missing from Store — must pass before merge")
}

func TestRapidRecatalogAfterManeuver(t *testing.T) {
	t.Fatal("RapidRecatalogAfterManeuver missing from Store — must pass before merge")
}
