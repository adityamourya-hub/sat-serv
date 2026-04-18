package persistence

import (
	"os"
	"testing"
	"time"

	"ssis/pkg/catalog"
	"ssis/pkg/octree"
)

func TestPersistentStore_CRUD(t *testing.T) {
	dbPath := "test_persist.db"
	defer os.Remove(dbPath)

	cfg := Config{
		Backend: "boltdb",
		Path:    dbPath,
	}

	// 1. Initial Store
	ps, err := NewPersistentStore(cfg, 50)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}

	r1 := catalog.ObjectRecord{
		ID:        1,
		Name:      "TEST_SAT_1",
		Pos:       octree.Vector3{X: 7000, Y: 0, Z: 0},
		Timestamp: float64(time.Now().Unix()),
	}

	// 2. Insert and verify memory/DB sync
	ps.ApplyDeltaWithRebuild(catalog.Delta{Upserts: []catalog.ObjectRecord{r1}}, true)
	
	if len(ps.Snapshot()) != 1 {
		t.Errorf("Expected 1 record in memory, got %d", len(ps.Snapshot()))
	}

	ps.Close()

	// 3. Re-open and verify reload
	ps2, err := NewPersistentStore(cfg, 50)
	if err != nil {
		t.Fatalf("Failed to restart store: %v", err)
	}
	defer ps2.Close()

	snapshot := ps2.Snapshot()
	if len(snapshot) != 1 {
		t.Errorf("Expected 1 record reloaded from DB, got %d", len(snapshot))
	} else if snapshot[0].Name != "TEST_SAT_1" {
		t.Errorf("Record name mismatch: %s", snapshot[0].Name)
	}
}

func TestPersistentStore_HistoricalQuery(t *testing.T) {
	dbPath := "test_history.db"
	defer os.Remove(dbPath)

	cfg := Config{
		Backend: "boltdb",
		Path:    dbPath,
	}

	ps, _ := NewPersistentStore(cfg, 50)
	
	now := time.Now().Unix()
	r1 := catalog.ObjectRecord{ID: 101, Name: "V1", Timestamp: float64(now - 100)}
	r2 := catalog.ObjectRecord{ID: 101, Name: "V2", Timestamp: float64(now)}
	
	ps.ApplyDeltaWithRebuild(catalog.Delta{Upserts: []catalog.ObjectRecord{r1}}, true)
	time.Sleep(10 * time.Millisecond)
	ps.ApplyDeltaWithRebuild(catalog.Delta{Upserts: []catalog.ObjectRecord{r2}}, true)
	
	// Query historical state
	past, err := ps.GetObjectAtTime(101, time.Unix(now-50, 0))
	if err != nil {
		t.Fatalf("Historical query failed: %v", err)
	}
	
	if past.Name != "V1" {
		t.Errorf("Expected historical state V1, got %s", past.Name)
	}
	
	ps.Close()
}
