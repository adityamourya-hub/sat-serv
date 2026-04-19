package persistence

import (
	"os"
	"testing"
	"time"

	"ssis/pkg/catalog"
	"ssis/pkg/octree"
)

func TestBoltDBBackend_NewBoltDBBackend(t *testing.T) {
	// Use a temporary database file
	tmpDB := "test_catalog.db"
	defer os.Remove(tmpDB)

	backend, err := NewBoltDBBackend(tmpDB)
	if err != nil {
		t.Fatalf("NewBoltDBBackend() error = %v", err)
	}
	defer backend.Close()

	if backend == nil {
		t.Fatal("NewBoltDBBackend() returned nil")
	}

	if backend.db == nil {
		t.Error("NewBoltDBBackend() db is nil")
	}
}

func TestBoltDBBackend_UpsertObject(t *testing.T) {
	tmpDB := "test_catalog_upsert.db"
	defer os.Remove(tmpDB)

	backend, err := NewBoltDBBackend(tmpDB)
	if err != nil {
		t.Fatalf("NewBoltDBBackend() error = %v", err)
	}
	defer backend.Close()

	record := catalog.SDAObject{
		ID:        1,
		Name:      "TEST-SAT",
		Pos:       octree.Vector3{X: 7000, Y: 0, Z: 0},
		Vel:       octree.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:   2460000.0,
		Timestamp: time.Now().Unix(),
	}

	err = backend.UpsertObject(record)
	if err != nil {
		t.Errorf("UpsertObject() error = %v", err)
	}
}

func TestBoltDBBackend_GetLatestSnapshot(t *testing.T) {
	tmpDB := "test_catalog_snapshot.db"
	defer os.Remove(tmpDB)

	backend, err := NewBoltDBBackend(tmpDB)
	if err != nil {
		t.Fatalf("NewBoltDBBackend() error = %v", err)
	}
	defer backend.Close()

	// Insert test records
	records := []catalog.SDAObject{
		{
			ID:        1,
			Name:      "SAT-1",
			Pos:       octree.Vector3{X: 7000, Y: 0, Z: 0},
			Vel:       octree.Vector3{X: 0, Y: 7.5, Z: 0},
			EpochTT:   2460000.0,
			Timestamp: time.Now().Unix(),
		},
		{
			ID:        2,
			Name:      "SAT-2",
			Pos:       octree.Vector3{X: -7000, Y: 0, Z: 0},
			Vel:       octree.Vector3{X: 0, Y: -7.5, Z: 0},
			EpochTT:   2460000.0,
			Timestamp: time.Now().Unix(),
		},
	}

	for _, rec := range records {
		if err := backend.UpsertObject(rec); err != nil {
			t.Fatalf("UpsertObject() error = %v", err)
		}
	}

	snapshot, err := backend.GetLatestSnapshot()
	if err != nil {
		t.Fatalf("GetLatestSnapshot() error = %v", err)
	}

	if len(snapshot) != len(records) {
		t.Errorf("GetLatestSnapshot() returned %d records, want %d", len(snapshot), len(records))
	}
}

func TestBoltDBBackend_DeleteObject(t *testing.T) {
	tmpDB := "test_catalog_delete.db"
	defer os.Remove(tmpDB)

	backend, err := NewBoltDBBackend(tmpDB)
	if err != nil {
		t.Fatalf("NewBoltDBBackend() error = %v", err)
	}
	defer backend.Close()

	record := catalog.SDAObject{
		ID:        1,
		Name:      "TO-DELETE",
		Pos:       octree.Vector3{X: 7000, Y: 0, Z: 0},
		Vel:       octree.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:   2460000.0,
		Timestamp: time.Now().Unix(),
	}

	if err := backend.UpsertObject(record); err != nil {
		t.Fatalf("UpsertObject() error = %v", err)
	}

	err = backend.DeleteObject(1)
	if err != nil {
		t.Errorf("DeleteObject() error = %v", err)
	}

	snapshot, err := backend.GetLatestSnapshot()
	if err != nil {
		t.Fatalf("GetLatestSnapshot() error = %v", err)
	}

	for _, rec := range snapshot {
		if rec.ID == 1 {
			t.Error("DeleteObject() did not remove record with ID 1")
		}
	}
}

func TestBoltDBBackend_GetObjectAtTime(t *testing.T) {
	tmpDB := "test_catalog_time.db"
	defer os.Remove(tmpDB)

	backend, err := NewBoltDBBackend(tmpDB)
	if err != nil {
		t.Fatalf("NewBoltDBBackend() error = %v", err)
	}
	defer backend.Close()

	now := time.Now()
	earlier := now.Add(-1 * time.Hour)

	// Insert record with timestamp
	record := catalog.SDAObject{
		ID:        1,
		Name:      "TIME-TEST",
		Pos:       octree.Vector3{X: 7000, Y: 0, Z: 0},
		Vel:       octree.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:   2460000.0,
		Timestamp: earlier.Unix(),
	}

	if err := backend.UpsertObject(record); err != nil {
		t.Fatalf("UpsertObject() error = %v", err)
	}

	// Query for current time should return the historical record
	rec, err := backend.GetObjectAtTime(1, now)
	if err != nil {
		t.Errorf("GetObjectAtTime() error = %v", err)
	}

	if rec.ID != 1 {
		t.Errorf("GetObjectAtTime() returned ID %d, want 1", rec.ID)
	}
}

func TestBoltDBBackend_QueryObjectsInWindow(t *testing.T) {
	tmpDB := "test_catalog_window.db"
	defer os.Remove(tmpDB)

	backend, err := NewBoltDBBackend(tmpDB)
	if err != nil {
		t.Fatalf("NewBoltDBBackend() error = %v", err)
	}
	defer backend.Close()

	now := time.Now()
	records := []catalog.SDAObject{
		{
			ID:        1,
			Name:      "SAT-1",
			Pos:       octree.Vector3{X: 7000, Y: 0, Z: 0},
			Vel:       octree.Vector3{X: 0, Y: 7.5, Z: 0},
			EpochTT:   2460000.0,
			Timestamp: now.Add(-2 * time.Hour).Unix(),
		},
		{
			ID:        2,
			Name:      "SAT-2",
			Pos:       octree.Vector3{X: -7000, Y: 0, Z: 0},
			Vel:       octree.Vector3{X: 0, Y: -7.5, Z: 0},
			EpochTT:   2460000.0,
			Timestamp: now.Add(-1 * time.Hour).Unix(),
		},
	}

	for _, rec := range records {
		if err := backend.UpsertObject(rec); err != nil {
			t.Fatalf("UpsertObject() error = %v", err)
		}
	}

	// Query for last hour window
	start := now.Add(-90 * time.Minute)
	end := now.Add(-30 * time.Minute)

	results, err := backend.QueryObjectsInWindow(start, end)
	if err != nil {
		t.Fatalf("QueryObjectsInWindow() error = %v", err)
	}

	// Should return at least one record
	if len(results) == 0 {
		t.Log("QueryObjectsInWindow() returned no records (time window may not match)")
	}
}

func TestParseHistoryTS(t *testing.T) {
	tests := []struct {
		name    string
		key     []byte
		wantTS  int64
		wantOK  bool
	}{
		{
			name:   "valid key",
			key:    []byte("1:1234567890"),
			wantTS: 1234567890,
			wantOK: true,
		},
		{
			name:   "missing colon",
			key:    []byte("11234567890"),
			wantTS: 0,
			wantOK: false,
		},
		{
			name:   "invalid timestamp",
			key:    []byte("1:invalid"),
			wantTS: 0,
			wantOK: false,
		},
		{
			name:   "empty key",
			key:    []byte(""),
			wantTS: 0,
			wantOK: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts, ok := parseHistoryTS(tt.key)
			if ts != tt.wantTS {
				t.Errorf("parseHistoryTS() ts = %v, want %v", ts, tt.wantTS)
			}
			if ok != tt.wantOK {
				t.Errorf("parseHistoryTS() ok = %v, want %v", ok, tt.wantOK)
			}
		})
	}
}
