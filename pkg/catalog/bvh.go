package catalog

import (
	"fmt"
	"math"
	"sort"
	"time"

	"ssis/pkg/octree"
)

/**
 * BVHNode represents a volume in the Bounding Volume Hierarchy.
 * It uses a binary tree structure to provide O(log N) spatial-temporal 
 * pruning of the satellite catalog.
 */
type BVHNode struct {
	Min, Max octree.Vector3
	MinT, MaxT float64
	
	Left, Right *BVHNode
	Objects     []ObjectRecord
	IsLeaf      bool
}

/**
 * BVHIndex implements a high-performance spatial-temporal index.
 * It replaces the Octree for Phase 1.7 production requirements, providing 
 * superior performance for skewed orbital distributions.
 * 
 * <plan_complexity>High</plan_complexity><total_lines_estimate>300</total_lines_estimate>
 */
type BVHIndex struct {
	Root *BVHNode
}

/**
 * BuildBVH constructs the tree using Centroid Median splitting.
 * 
 * @param objects: The list of space objects to index.
 */
func BuildBVH(objects []ObjectRecord) *BVHIndex {
	// Entry Point: BVH Build
	fmt.Printf("DEBUG: Building BVH Index for %d objects at %v\n", len(objects), time.Now())
	
	if len(objects) == 0 {
		return &BVHIndex{Root: nil}
	}
	
	return &BVHIndex{
		Root: buildNode(objects, 0),
	}
}

func buildNode(objects []ObjectRecord, depth int) *BVHNode {
	node := &BVHNode{
		Min: octree.Vector3{X: 1e15, Y: 1e15, Z: 1e15},
		Max: octree.Vector3{X: -1e15, Y: -1e15, Z: -1e15},
		MinT: 1e15,
		MaxT: -1e15,
	}
	
	// Calculate Bounding Box
	for _, obj := range objects {
		expandNode(node, obj)
	}
	
	// Leaf condition
	if len(objects) <= 8 || depth > 12 {
		node.Objects = objects
		node.IsLeaf = true
		return node
	}
	
	// Split along longest axis
	axis := longestAxis(node)
	sort.Slice(objects, func(i, j int) bool {
		return getCoord(objects[i].Pos, axis) < getCoord(objects[j].Pos, axis)
	})
	
	mid := len(objects) / 2
	node.Left = buildNode(objects[:mid], depth+1)
	node.Right = buildNode(objects[mid:], depth+1)
	
	return node
}

func expandNode(n *BVHNode, obj ObjectRecord) {
	n.Min.X = math.Min(n.Min.X, obj.Pos.X)
	n.Min.Y = math.Min(n.Min.Y, obj.Pos.Y)
	n.Min.Z = math.Min(n.Min.Z, obj.Pos.Z)
	n.Max.X = math.Max(n.Max.X, obj.Pos.X)
	n.Max.Y = math.Max(n.Max.Y, obj.Pos.Y)
	n.Max.Z = math.Max(n.Max.Z, obj.Pos.Z)
	
	t := obj.Timestamp
	if t > 0 {
		n.MinT = math.Min(n.MinT, t)
		n.MaxT = math.Max(n.MaxT, t)
	}
}

func longestAxis(n *BVHNode) int {
	dx := n.Max.X - n.Min.X
	dy := n.Max.Y - n.Min.Y
	dz := n.Max.Z - n.Min.Z
	if dx >= dy && dx >= dz { return 0 }
	if dy >= dx && dy >= dz { return 1 }
	return 2
}

func getCoord(v octree.Vector3, axis int) float64 {
	if axis == 0 { return v.X }
	if axis == 1 { return v.Y }
	return v.Z
}

/**
 * QueryRange performs O(log N) spatial-temporal pruning.
 */
func (idx *BVHIndex) QueryRange(min, max octree.Vector3, tMin, tMax float64) []ObjectRecord {
	if idx.Root == nil { return nil }
	results := []ObjectRecord{}
	idx.Root.query(min, max, tMin, tMax, &results)
	return results
}

func (n *BVHNode) query(min, max octree.Vector3, tMin, tMax float64, res *[]ObjectRecord) {
	// Spatial Overlap Check
	if n.Min.X > max.X || n.Max.X < min.X ||
	   n.Min.Y > max.Y || n.Max.Y < min.Y ||
	   n.Min.Z > max.Z || n.Max.Z < min.Z {
		return
	}
	
	// Temporal Overlap Check (if timestamps available)
	if n.MinT != 1e15 && (n.MinT > tMax || n.MaxT < tMin) {
		return
	}
	
	if n.IsLeaf {
		for _, obj := range n.Objects {
			if obj.Pos.X >= min.X && obj.Pos.X <= max.X &&
			   obj.Pos.Y >= min.Y && obj.Pos.Y <= max.Y &&
			   obj.Pos.Z >= min.Z && obj.Pos.Z <= max.Z {
				*res = append(*res, obj)
			}
		}
		return
	}
	
	n.Left.query(min, max, tMin, tMax, res)
	n.Right.query(min, max, tMin, tMax, res)
}
