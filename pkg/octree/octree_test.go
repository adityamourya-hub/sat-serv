package octree

import (
	"testing"
)

type MockObject struct {
	ID  int
	Pos Vector3
}

func (m MockObject) GetPosition() Vector3  { return m.Pos }
func (m MockObject) GetID() int            { return m.ID }
func (m MockObject) GetTimestamp() float64 { return 0.0 }

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
