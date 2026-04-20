package octree

import (
	"sync"
	"testing"
)

type MockObject struct {
	ID  int
	Pos Vector3
}

func (m MockObject) GetPosition() Vector3  { return m.Pos }
func (m MockObject) GetID() int            { return m.ID }
func (m MockObject) GetTimestamp() float64 { return float64(m.ID) } // Use ID as timestamp for testing

func TestOctree(t *testing.T) {
	oct := NewOctree(Vector3{-100, -100, -100}, Vector3{100, 100, 100})

	objs := []MockObject{
		{ID: 1, Pos: Vector3{10, 10, 10}},
		{ID: 2, Pos: Vector3{-10, -10, -10}},
		{ID: 3, Pos: Vector3{50, 50, 50}},
		{ID: 4, Pos: Vector3{90, 90, 90}},
	}

	for _, obj := range objs {
		oct.Insert(obj)
	}

	// Query range that should only find obj 1
	results := oct.QueryRange(Vector3{0, 0, 0}, Vector3{20, 20, 20}, 0, 1e18)
	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	} else if results[0].GetID() != 1 {
		t.Errorf("Expected ID 1, got %d", results[0].GetID())
	}

	// Query range for most objects
	results = oct.QueryRange(Vector3{-20, -20, -20}, Vector3{60, 60, 60}, 0, 1e18)
	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}
}

func TestOctreeNoDuplicationAfterSubdivision(t *testing.T) {
	oct := NewOctree(Vector3{-1000, -1000, -1000}, Vector3{1000, 1000, 1000})

	const count = 64
	for i := 0; i < count; i++ {
		oct.Insert(MockObject{
			ID: i + 1,
			Pos: Vector3{
				X: float64(i*3 - 90),
				Y: float64(i*2 - 80),
				Z: float64(i - 30),
			},
		})
	}

	results := oct.QueryRange(Vector3{-1000, -1000, -1000}, Vector3{1000, 1000, 1000}, 0, 1e18)
	if len(results) != count {
		t.Fatalf("Expected %d unique objects, got %d", count, len(results))
	}

	seen := make(map[int]struct{}, count)
	for _, obj := range results {
		if _, ok := seen[obj.GetID()]; ok {
			t.Fatalf("Duplicate object id %d returned from query", obj.GetID())
		}
		seen[obj.GetID()] = struct{}{}
	}
}

func TestOctreeRemove(t *testing.T) {
	oct := NewOctree(Vector3{-100, -100, -100}, Vector3{100, 100, 100})

	objs := []MockObject{
		{ID: 1, Pos: Vector3{10, 10, 10}},
		{ID: 2, Pos: Vector3{-10, -10, -10}},
		{ID: 3, Pos: Vector3{50, 50, 50}},
	}

	for _, obj := range objs {
		oct.Insert(obj)
	}

	// Remove middle object
	if !oct.Remove(objs[1]) {
		t.Fatal("Expected Remove to return true")
	}

	results := oct.QueryRange(Vector3{-100, -100, -100}, Vector3{100, 100, 100}, 0, 1e18)
	if len(results) != 2 {
		t.Errorf("Expected 2 objects after removal, got %d", len(results))
	}

	// Try to remove already removed object
	if oct.Remove(objs[1]) {
		t.Error("Expected Remove to return false for already-removed object")
	}

	// Remove object at exact boundary
	boundaryObj := MockObject{ID: 4, Pos: Vector3{0, 0, 0}}
	oct.Insert(boundaryObj)
	if !oct.Remove(boundaryObj) {
		t.Error("Expected Remove to handle boundary object")
	}
}

func TestOctreeConcurrentAccess(t *testing.T) {
	oct := NewOctree(Vector3{-1000, -1000, -1000}, Vector3{1000, 1000, 1000})
	const numGoroutines = 100
	const perGoroutine = 10

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < perGoroutine; j++ {
				obj := MockObject{
					ID:  id*perGoroutine + j + 1,
					Pos: Vector3{float64(id), float64(j), 0},
				}
				oct.Insert(obj)
			}
		}(i)
	}

	wg.Wait()

	results := oct.QueryRange(Vector3{-1000, -1000, -1000}, Vector3{1000, 1000, 1000}, 0, 1e18)
	if len(results) != numGoroutines*perGoroutine {
		t.Errorf("Expected %d objects, got %d", numGoroutines*perGoroutine, len(results))
	}
}

func TestOctreeTemporalQuery(t *testing.T) {
	oct := NewOctree(Vector3{-100, -100, -100}, Vector3{100, 100, 100})

	// Objects with different timestamps
	oct.Insert(MockObject{ID: 1, Pos: Vector3{10, 10, 10}})   // ts=1
	oct.Insert(MockObject{ID: 50, Pos: Vector3{10, 10, 10}})  // ts=50
	oct.Insert(MockObject{ID: 100, Pos: Vector3{10, 10, 10}}) // ts=100

	// Query only early timestamps
	results := oct.QueryRange(Vector3{0, 0, 0}, Vector3{20, 20, 20}, 0, 25)
	if len(results) != 1 || results[0].GetID() != 1 {
		t.Errorf("Expected 1 result with ID 1, got %d", len(results))
	}

	// Query middle range
	results = oct.QueryRange(Vector3{0, 0, 0}, Vector3{20, 20, 20}, 25, 75)
	if len(results) != 1 || results[0].GetID() != 50 {
		t.Errorf("Expected 1 result with ID 50, got %v", results)
	}
}

func TestOctreeIterator(t *testing.T) {
	oct := NewOctree(Vector3{-100, -100, -100}, Vector3{100, 100, 100})

	for i := 1; i <= 100; i++ {
		oct.Insert(MockObject{ID: i, Pos: Vector3{float64(i), 0, 0}})
	}

	// Count via iterator
	count := 0
	oct.QueryRangeIter(Vector3{0, 0, 0}, Vector3{50, 1, 1}, 0, 1e18, func(obj Object) bool {
		count++
		return true
	})
	if count != 50 {
		t.Errorf("Expected 50 objects via iterator, got %d", count)
	}

	// Early termination
	stopCount := 0
	oct.QueryRangeIter(Vector3{-100, -100, -100}, Vector3{100, 100, 100}, 0, 1e18, func(obj Object) bool {
		stopCount++
		return stopCount < 10 // Stop after 10
	})
	if stopCount != 10 {
		t.Errorf("Expected early stop at 10, got %d", stopCount)
	}
}

func TestOctreeStats(t *testing.T) {
	oct := NewOctree(Vector3{-1000, -1000, -1000}, Vector3{1000, 1000, 1000})

	// Empty tree
	stats := oct.Stats()
	if stats.Nodes != 1 || stats.Leafs != 1 || stats.Objects != 0 {
		t.Errorf("Empty tree stats wrong: %+v", stats)
	}

	// Add objects to trigger subdivision
	for i := 0; i < 100; i++ {
		oct.Insert(MockObject{ID: i + 1, Pos: Vector3{float64(i), 0, 0}})
	}

	stats = oct.Stats()
	if stats.Objects != 100 {
		t.Errorf("Expected 100 objects, got %d", stats.Objects)
	}
	if stats.Nodes < 1 {
		t.Errorf("Expected at least 1 node, got %d", stats.Nodes)
	}
	if stats.Leafs < 1 {
		t.Errorf("Expected at least 1 leaf, got %d", stats.Leafs)
	}
	if stats.MemoryBytes <= 0 {
		t.Errorf("Expected positive memory, got %d", stats.MemoryBytes)
	}
}

func TestOctreeNearest(t *testing.T) {
	oct := NewOctree(Vector3{-100, -100, -100}, Vector3{100, 100, 100})

	// Insert objects at known positions
	objects := []MockObject{
		{ID: 1, Pos: Vector3{0, 0, 0}},
		{ID: 2, Pos: Vector3{10, 0, 0}},
		{ID: 3, Pos: Vector3{20, 0, 0}},
		{ID: 4, Pos: Vector3{30, 0, 0}},
		{ID: 5, Pos: Vector3{100, 100, 100}}, // Far away
	}
	for _, obj := range objects {
		oct.Insert(obj)
	}

	// Find 3 nearest to origin
	nearest := oct.Nearest(Vector3{0, 0, 0}, 3, -1)
	if len(nearest) != 3 {
		t.Fatalf("Expected 3 nearest, got %d", len(nearest))
	}

	// Should get IDs 1, 2, 3 (closest to origin)
	expected := map[int]bool{1: true, 2: true, 3: true}
	for _, obj := range nearest {
		if !expected[obj.GetID()] {
			t.Errorf("Unexpected ID %d in nearest", obj.GetID())
		}
	}

	// Exclude ID 1, should get 2, 3, 4
	nearest = oct.Nearest(Vector3{0, 0, 0}, 3, 1)
	if len(nearest) != 3 {
		t.Fatalf("Expected 3 nearest, got %d", len(nearest))
	}
	for _, obj := range nearest {
		if obj.GetID() == 1 {
			t.Error("Excluded ID 1 should not appear")
		}
	}
}

func TestOctreeInsertBatch(t *testing.T) {
	oct := NewOctree(Vector3{-1000, -1000, -1000}, Vector3{1000, 1000, 1000})

	var objs []Object
	for i := 0; i < 200; i++ {
		objs = append(objs, MockObject{ID: i + 1, Pos: Vector3{float64(i), 0, 0}})
	}

	oct.InsertBatch(objs)

	results := oct.QueryRange(Vector3{-1000, -1000, -1000}, Vector3{1000, 1000, 1000}, 0, 1e18)
	if len(results) != 200 {
		t.Errorf("Expected 200 objects after batch insert, got %d", len(results))
	}

	// Validate tree integrity
	if err := oct.Validate(); err != nil {
		t.Errorf("Validation failed: %v", err)
	}
}

func TestOctreeValidate(t *testing.T) {
	oct := NewOctree(Vector3{-100, -100, -100}, Vector3{100, 100, 100})

	// Empty tree should be valid
	if err := oct.Validate(); err != nil {
		t.Errorf("Empty tree should be valid: %v", err)
	}

	// Add valid objects
	for i := 1; i <= 10; i++ {
		oct.Insert(MockObject{ID: i, Pos: Vector3{float64(i), 0, 0}})
	}

	if err := oct.Validate(); err != nil {
		t.Errorf("Valid tree failed validation: %v", err)
	}
}
