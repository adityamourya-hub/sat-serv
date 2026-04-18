package spatial

import (
	"testing"

	"ssis/pkg/octree"
)

func TestHashGridQueries(t *testing.T) {
	g := NewHashGrid(10)
	g.Rebuild([]Point{
		{ID: 1, Pos: octree.Vector3{X: 0, Y: 0, Z: 0}},
		{ID: 2, Pos: octree.Vector3{X: 9, Y: 9, Z: 9}},
		{ID: 3, Pos: octree.Vector3{X: 30, Y: 30, Z: 30}},
	})

	aabb := g.QueryAABB(octree.Vector3{X: -1, Y: -1, Z: -1}, octree.Vector3{X: 12, Y: 12, Z: 12})
	if len(aabb) != 2 {
		t.Fatalf("Expected 2 points in AABB query, got %d", len(aabb))
	}

	r := g.QueryRadius(octree.Vector3{X: 0, Y: 0, Z: 0}, 1.0)
	if len(r) != 1 || r[0] != 1 {
		t.Fatalf("Expected only ID 1 in radius query, got %v", r)
	}
}
