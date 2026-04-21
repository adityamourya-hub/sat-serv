package spatial

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"ssis/pkg/types"
)

// TestBVH100KObjects tests BVH construction with 100k objects
// Validates the 2-second build target from the plan
func TestBVH100KObjects(t *testing.T) {
	// Generate 100k random objects
	objects := generateRandomObjects(100000)

	start := time.Now()
	bvh := NewBVH(objects)
	elapsed := time.Since(start)

	t.Logf("BVH construction time for 100k objects: %v", elapsed)

	if elapsed > 2*time.Second {
		t.Errorf("BVH construction exceeded 2-second target: %v", elapsed)
	}

	if bvh.Root == nil {
		t.Fatal("BVH root is nil")
	}

	if len(bvh.Objects) != 100000 {
		t.Errorf("Expected 100000 objects, got %d", len(bvh.Objects))
	}
}

// TestBVHSingleObject tests the fast path for single-object BVH
func TestBVHSingleObject(t *testing.T) {
	obj := &testObject{
		id:   1,
		pos:  types.Vector3{X: 0, Y: 0, Z: 0},
		time: 0.0,
	}

	bvh := NewBVH([]BVHObject{obj})

	if bvh.Root == nil {
		t.Fatal("BVH root is nil")
	}

	if !bvh.Root.IsLeaf {
		t.Error("Single object BVH should be a leaf")
	}

	if bvh.Root.Object != obj {
		t.Error("Leaf object mismatch")
	}
}

// TestBVHEmptyList tests empty BVH construction
func TestBVHEmptyList(t *testing.T) {
	bvh := NewBVH([]BVHObject{})

	if bvh.Root != nil {
		t.Error("Empty BVH should have nil root")
	}

	if len(bvh.Objects) != 0 {
		t.Errorf("Expected 0 objects, got %d", len(bvh.Objects))
	}
}

// TestAABBValidateBounds tests the bounds validation function
func TestAABBValidateBounds(t *testing.T) {
	tests := []struct {
		name    string
		aabb    AABB
		wantErr bool
	}{
		{
			name: "valid bounds",
			aabb: AABB{
				Min:  types.Vector3{X: -1000, Y: -1000, Z: -1000},
				Max:  types.Vector3{X: 1000, Y: 1000, Z: 1000},
				TMin: 0,
				TMax: 100,
			},
			wantErr: false,
		},
		{
			name: "degenerate spatial bounds",
			aabb: AABB{
				Min:  types.Vector3{X: 0, Y: 0, Z: 0},
				Max:  types.Vector3{X: 0, Y: 0, Z: 0},
				TMin: 0,
				TMax: 100,
			},
			wantErr: false, // Should auto-expand
		},
		{
			name: "inverted spatial bounds",
			aabb: AABB{
				Min:  types.Vector3{X: 1000, Y: 1000, Z: 1000},
				Max:  types.Vector3{X: -1000, Y: -1000, Z: -1000},
				TMin: 0,
				TMax: 100,
			},
			wantErr: false, // Should auto-correct
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid, err := tt.aabb.ValidateBounds()
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateBounds() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && !valid {
				t.Error("ValidateBounds() returned false for valid case")
			}
		})
	}
}

// TestMortonCodeNaNHandling tests NaN input handling
func TestMortonCodeNaNHandling(t *testing.T) {
	pos := types.Vector3{X: math.NaN(), Y: 0, Z: 0}
	bounds := AABB{
		Min:  types.Vector3{X: -1000, Y: -1000, Z: -1000},
		Max:  types.Vector3{X: 1000, Y: 1000, Z: 1000},
		TMin: 0,
		TMax: 100,
	}

	code := mortonCode(pos, 0, bounds)
	if code != 0 {
		t.Errorf("NaN input should return 0, got %d", code)
	}
}

// TestMortonCodeInfHandling tests Inf input handling
func TestMortonCodeInfHandling(t *testing.T) {
	pos := types.Vector3{X: math.Inf(1), Y: 0, Z: 0}
	bounds := AABB{
		Min:  types.Vector3{X: -1000, Y: -1000, Z: -1000},
		Max:  types.Vector3{X: 1000, Y: 1000, Z: 1000},
		TMin: 0,
		TMax: 100,
	}

	code := mortonCode(pos, 0, bounds)
	if code != 0xFFFFFFFFFFFFFFFF {
		t.Errorf("Inf input should return max uint64, got %d", code)
	}
}

// Helper functions

type testObject struct {
	id   int
	pos  types.Vector3
	time float64
}

func (t *testObject) GetPosition() types.Vector3 {
	return t.pos
}

func (t *testObject) GetID() int {
	return t.id
}

func (t *testObject) GetTimestamp() float64 {
	return t.time
}

func generateRandomObjects(count int) []BVHObject {
	rand.Seed(time.Now().UnixNano())
	objects := make([]BVHObject, count)

	for i := 0; i < count; i++ {
		objects[i] = &testObject{
			id: i,
			pos: types.Vector3{
				X: rand.Float64()*20000 - 10000, // LEO range
				Y: rand.Float64()*20000 - 10000,
				Z: rand.Float64()*20000 - 10000,
			},
			time: rand.Float64() * 3600, // 1 hour window
		}
	}

	return objects
}
