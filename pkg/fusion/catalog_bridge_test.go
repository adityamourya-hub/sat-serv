package fusion

import (
	"testing"

	"ssis/pkg/catalog"
	"ssis/pkg/physics"
)

func TestBuildCatalogRecordFromFusion(t *testing.T) {
	state := physics.State{Pos: physics.Vector3{10, 20, 30}}
	cov := [][]float64{
		{1, 0, 0, 0, 0, 0},
		{0, 4, 0, 0, 0, 0},
		{0, 0, 9, 0, 0, 0},
		{0, 0, 0, 0.01, 0, 0},
		{0, 0, 0, 0, 0.01, 0},
		{0, 0, 0, 0, 0, 0.01},
	}
	rec := BuildCatalogRecordFromFusion(100, "OBJ", state, cov)
	if rec.ID != 100 || rec.Name != "OBJ" {
		t.Fatalf("unexpected record identity: %+v", rec)
	}
	if rec.PosSigmaKm <= 0 {
		t.Fatalf("expected positive position sigma")
	}
}

func TestRapidRecatalogFromFusion(t *testing.T) {
	store := catalog.NewStore(10)
	store.ApplyDeltaWithRebuild(catalog.Delta{Upserts: []catalog.ObjectRecord{{ID: 7, Name: "A", PosSigmaKm: 1.2}}}, false)
	fusionRes := FusionResult{
		State: physics.State{Pos: physics.Vector3{11, 12, 13}},
		Covariance: [][]float64{
			{1, 0, 0, 0, 0, 0},
			{0, 1, 0, 0, 0, 0},
			{0, 0, 1, 0, 0, 0},
			{0, 0, 0, 0.01, 0, 0},
			{0, 0, 0, 0, 0.01, 0},
			{0, 0, 0, 0, 0, 0.01},
		},
	}
	v := RapidRecatalogFromFusion(store, 7, "A", fusionRes, 20)
	if v <= 1 {
		t.Fatalf("expected version bump from rapid recatalog, got %d", v)
	}
}
