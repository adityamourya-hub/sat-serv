package spatial

import (
	"math"
	"sort"
)

/**
 * ADITI 4.0: 4D Space-Time Linear BVH (LBVH)
 *
 * Indexes RSOs in 3D position + 1D time/energy space.
 * Uses Morton curve (Z-order) sorting for fast hierarchy construction.
 *
 * Key Features:
 * - O(N log N) build time via sorting
 * - O(log N) query time with 4D reachability pruning
 * - Cache-friendly linear layout (no pointer chasing)
 * - Incremental updates for dynamic catalogs
 *
 * The 4th dimension represents either:
 * - Time (for conjunction prediction over horizon)
 * - Energy (for reachability-based threat assessment)
 * - Velocity magnitude (for kinetic threat prioritization)
 */

// MortonCode is a 64-bit spatial hash for Z-order curve
type MortonCode uint64

// BVH4DNode represents a node in the 4D BVH hierarchy
type BVH4DNode struct {
	// Bounding box in 4D: [x,y,z,w] where w is time/energy
	Min [4]float64
	Max [4]float64

	// Hierarchy links (linear layout)
	LeftChild  int32 // Index of left child (-1 if leaf)
	RightChild int32 // Index of right child (-1 if leaf)

	// Leaf data
	ObjectID uint32 // RSO ID (valid if leaf)
	IsLeaf   bool

	// Morton code for spatial ordering
	Morton MortonCode
}

// BVH4D is a 4D linear bounding volume hierarchy
type BVH4D struct {
	Nodes      []BVH4DNode // Linear node array (cache-friendly)
	RootIndex  int32
	NumObjects int32

	// World bounds for normalization
	WorldMin [4]float64
	WorldMax [4]float64

	// The 4th dimension interpretation
	WDimension WDimension
}

// WDimension specifies what the 4th dimension represents
type WDimension int

const (
	WDimensionTime     WDimension = iota // Time of closest approach
	WDimensionEnergy                     // Orbital energy (for reachability)
	WDimensionVelocity                   // Velocity magnitude
)

// LBVHBuilder constructs the 4D BVH using the LBVH algorithm
type LBVHBuilder struct {
	Objects []BVH4DObject
	Nodes   []BVH4DNode
}

// BVH4DObject represents an object to be inserted into the BVH
type BVH4DObject struct {
	ID       uint32
	Position [3]float64
	W        float64       // 4th dimension value
	Bounds   [4][2]float64 // Min/max in each dimension
}

// NewBVH4D creates an empty 4D BVH
func NewBVH4D(wDim WDimension) *BVH4D {
	return &BVH4D{
		Nodes:      make([]BVH4DNode, 0, 1024),
		RootIndex:  -1,
		WDimension: wDim,
		WorldMin:   [4]float64{-1e6, -1e6, -1e6, 0},
		WorldMax:   [4]float64{1e6, 1e6, 1e6, 1e6},
	}
}

// Build constructs the BVH from a set of objects
func (bvh *BVH4D) Build(objects []BVH4DObject) error {
	if len(objects) == 0 {
		bvh.RootIndex = -1
		bvh.NumObjects = 0
		return nil
	}

	// Calculate world bounds
	bvh.calculateWorldBounds(objects)

	// Compute Morton codes for all objects
	type mortonObject struct {
		obj    BVH4DObject
		morton MortonCode
	}

	mortonObjects := make([]mortonObject, len(objects))
	for i, obj := range objects {
		// Normalize to [0,1] and compute Morton code
		normalized := bvh.normalizePosition(obj.Position, obj.W)
		mortonObjects[i] = mortonObject{
			obj:    obj,
			morton: computeMortonCode64(normalized),
		}
	}

	// Sort by Morton code (Z-order curve)
	sort.Slice(mortonObjects, func(i, j int) bool {
		return mortonObjects[i].morton < mortonObjects[j].morton
	})

	// Build hierarchy using LBVH algorithm
	builder := &LBVHBuilder{
		Objects: make([]BVH4DObject, len(objects)),
		Nodes:   make([]BVH4DNode, 0, len(objects)*2),
	}

	for i, mo := range mortonObjects {
		builder.Objects[i] = mo.obj
	}

	// Recursively build tree
	rootIdx := builder.buildRecursive(0, len(objects), 0)

	bvh.Nodes = builder.Nodes
	bvh.RootIndex = int32(rootIdx)
	bvh.NumObjects = int32(len(objects))

	return nil
}

// QueryReachability finds all objects within a 4D reachability envelope
// The envelope is defined by a center point and radii in each dimension
func (bvh *BVH4D) QueryReachability(center [4]float64, radii [4]float64) []uint32 {
	if bvh.RootIndex < 0 {
		return nil
	}

	results := make([]uint32, 0)
	bvh.queryRecursive(bvh.RootIndex, center, radii, &results)
	return results
}

// QueryNearest finds k nearest neighbors in 4D space
func (bvh *BVH4D) QueryNearest(query [4]float64, k int) []NearestResult {
	if bvh.RootIndex < 0 {
		return nil
	}

	// Simple priority queue implementation
	pq := &nearestQueue{capacity: k}
	bvh.nearestRecursive(bvh.RootIndex, query, pq)
	return pq.results()
}

// Private methods

func (bvh *BVH4D) calculateWorldBounds(objects []BVH4DObject) {
	// Initialize with first object
	for i := 0; i < 4; i++ {
		bvh.WorldMin[i] = math.Inf(1)
		bvh.WorldMax[i] = math.Inf(-1)
	}

	// Expand to contain all objects
	for _, obj := range objects {
		for d := 0; d < 3; d++ {
			if obj.Position[d] < bvh.WorldMin[d] {
				bvh.WorldMin[d] = obj.Position[d]
			}
			if obj.Position[d] > bvh.WorldMax[d] {
				bvh.WorldMax[d] = obj.Position[d]
			}
		}
		if obj.W < bvh.WorldMin[3] {
			bvh.WorldMin[3] = obj.W
		}
		if obj.W > bvh.WorldMax[3] {
			bvh.WorldMax[3] = obj.W
		}
	}

	// Add some padding
	for i := 0; i < 4; i++ {
		range_ := bvh.WorldMax[i] - bvh.WorldMin[i]
		bvh.WorldMin[i] -= range_ * 0.01
		bvh.WorldMax[i] += range_ * 0.01
	}
}

func (bvh *BVH4D) normalizePosition(pos [3]float64, w float64) [4]float64 {
	var result [4]float64
	for i := 0; i < 3; i++ {
		range_ := bvh.WorldMax[i] - bvh.WorldMin[i]
		if range_ > 0 {
			result[i] = (pos[i] - bvh.WorldMin[i]) / range_
		} else {
			result[i] = 0.5
		}
	}
	range_ := bvh.WorldMax[3] - bvh.WorldMin[3]
	if range_ > 0 {
		result[3] = (w - bvh.WorldMin[3]) / range_
	} else {
		result[3] = 0.5
	}
	return result
}

func (b *LBVHBuilder) buildRecursive(start, end, depth int) int {
	nodeIdx := len(b.Nodes)
	b.Nodes = append(b.Nodes, BVH4DNode{})

	if end-start == 1 {
		// Leaf node
		obj := b.Objects[start]
		b.Nodes[nodeIdx] = BVH4DNode{
			Min:      [4]float64{obj.Bounds[0][0], obj.Bounds[1][0], obj.Bounds[2][0], obj.Bounds[3][0]},
			Max:      [4]float64{obj.Bounds[0][1], obj.Bounds[1][1], obj.Bounds[2][1], obj.Bounds[3][1]},
			ObjectID: obj.ID,
			IsLeaf:   true,
		}
		return nodeIdx
	}

	// Find split position using Morton code bits
	split := b.findSplit(start, end, depth)

	// Build children
	leftIdx := b.buildRecursive(start, split, depth+1)
	rightIdx := b.buildRecursive(split, end, depth+1)

	// Compute node bounds from children
	leftNode := &b.Nodes[leftIdx]
	rightNode := &b.Nodes[rightIdx]

	node := &b.Nodes[nodeIdx]
	node.LeftChild = int32(leftIdx)
	node.RightChild = int32(rightIdx)
	node.IsLeaf = false

	// Merge bounds
	for i := 0; i < 4; i++ {
		node.Min[i] = math.Min(leftNode.Min[i], rightNode.Min[i])
		node.Max[i] = math.Max(leftNode.Max[i], rightNode.Max[i])
	}

	return nodeIdx
}

func (b *LBVHBuilder) findSplit(start, end, depth int) int {
	// Simple middle split for now
	// Could be improved by checking Morton code bits
	return start + (end-start)/2
}

func (bvh *BVH4D) queryRecursive(nodeIdx int32, center [4]float64, radii [4]float64, results *[]uint32) {
	if nodeIdx < 0 || int(nodeIdx) >= len(bvh.Nodes) {
		return
	}

	node := &bvh.Nodes[nodeIdx]

	// Check if node intersects query region
	if !intersects4D(node.Min, node.Max, center, radii) {
		return
	}

	if node.IsLeaf {
		*results = append(*results, node.ObjectID)
		return
	}

	// Recurse to children
	bvh.queryRecursive(node.LeftChild, center, radii, results)
	bvh.queryRecursive(node.RightChild, center, radii, results)
}

func (bvh *BVH4D) nearestRecursive(nodeIdx int32, query [4]float64, pq *nearestQueue) {
	if nodeIdx < 0 || int(nodeIdx) >= len(bvh.Nodes) {
		return
	}

	node := &bvh.Nodes[nodeIdx]

	// Compute minimum possible distance from query to this node's bbox
	minDist := distancePointToBBox4D(query, node.Min, node.Max)

	// Prune if too far
	if !pq.canAccept(minDist) {
		return
	}

	if node.IsLeaf {
		// Compute actual distance to object center
		// (in practice, would use actual object position)
		center := [4]float64{
			(node.Min[0] + node.Max[0]) / 2,
			(node.Min[1] + node.Max[1]) / 2,
			(node.Min[2] + node.Max[2]) / 2,
			(node.Min[3] + node.Max[3]) / 2,
		}
		dist := distance4D(query, center)
		pq.insert(NearestResult{ObjectID: node.ObjectID, Distance: dist})
		return
	}

	// Recurse to children (closest first for better pruning)
	leftDist := distancePointToBBox4D(query, bvh.Nodes[node.LeftChild].Min, bvh.Nodes[node.LeftChild].Max)
	rightDist := distancePointToBBox4D(query, bvh.Nodes[node.RightChild].Min, bvh.Nodes[node.RightChild].Max)

	if leftDist < rightDist {
		bvh.nearestRecursive(node.LeftChild, query, pq)
		bvh.nearestRecursive(node.RightChild, query, pq)
	} else {
		bvh.nearestRecursive(node.RightChild, query, pq)
		bvh.nearestRecursive(node.LeftChild, query, pq)
	}
}

// Morton code calculation (64-bit)
func computeMortonCode64(pos [4]float64) MortonCode {
	// Expand 21-bit values to 63 bits with space-filling curve
	x := expandBits3D(uint32(pos[0] * 0x1FFFFF))
	y := expandBits3D(uint32(pos[1] * 0x1FFFFF))
	z := expandBits3D(uint32(pos[2] * 0x1FFFFF))
	w := expandBits3D(uint32(pos[3] * 0x1FFFFF))

	// Interleave: x,y,z,w in 4D
	return MortonCode(interleave4D(x, y, z, w))
}

func expandBits3D(v uint32) uint64 {
	// Expand 21-bit value to 63 bits (3*21) for 3D
	x := uint64(v) & 0x1FFFFF
	x = (x | x<<32) & 0x1F00000000FFFF
	x = (x | x<<16) & 0x1F0000FF0000FF
	x = (x | x<<8) & 0x100F00F00F00F00F
	x = (x | x<<4) & 0x10F0F0F0F0F0F0F
	x = (x | x<<2) & 0x1249249249249249
	return x
}

func interleave4D(x, y, z, w uint64) uint64 {
	// Interleave 4 16-bit values into 64 bits
	// Pattern: wzyx wzyx wzyx ...
	result := uint64(0)
	for i := uint(0); i < 16; i++ {
		result |= ((x >> i) & 1) << (4 * i)
		result |= ((y >> i) & 1) << (4*i + 1)
		result |= ((z >> i) & 1) << (4*i + 2)
		result |= ((w >> i) & 1) << (4*i + 3)
	}
	return result
}

// Geometry helpers

func intersects4D(min1, max1, center, radii [4]float64) bool {
	for i := 0; i < 4; i++ {
		if max1[i] < center[i]-radii[i] || min1[i] > center[i]+radii[i] {
			return false
		}
	}
	return true
}

func distancePointToBBox4D(p, min, max [4]float64) float64 {
	distSq := 0.0
	for i := 0; i < 4; i++ {
		if p[i] < min[i] {
			d := p[i] - min[i]
			distSq += d * d
		} else if p[i] > max[i] {
			d := p[i] - max[i]
			distSq += d * d
		}
	}
	return math.Sqrt(distSq)
}

func distance4D(a, b [4]float64) float64 {
	distSq := 0.0
	for i := 0; i < 4; i++ {
		d := a[i] - b[i]
		distSq += d * d
	}
	return math.Sqrt(distSq)
}

// NearestResult represents a nearest neighbor query result
type NearestResult struct {
	ObjectID uint32
	Distance float64
}

// nearestQueue is a simple max-heap for k-nearest neighbors
type nearestQueue struct {
	items    []NearestResult
	capacity int
}

func (q *nearestQueue) canAccept(dist float64) bool {
	if len(q.items) < q.capacity {
		return true
	}
	return dist < q.items[0].Distance // Max-heap, items[0] is worst
}

func (q *nearestQueue) insert(item NearestResult) {
	if len(q.items) < q.capacity {
		q.items = append(q.items, item)
		q.bubbleUp(len(q.items) - 1)
	} else if item.Distance < q.items[0].Distance {
		q.items[0] = item
		q.bubbleDown(0)
	}
}

func (q *nearestQueue) results() []NearestResult {
	// Return sorted by distance (ascending)
	result := make([]NearestResult, len(q.items))
	copy(result, q.items)
	sort.Slice(result, func(i, j int) bool {
		return result[i].Distance < result[j].Distance
	})
	return result
}

func (q *nearestQueue) bubbleUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if q.items[i].Distance <= q.items[parent].Distance {
			break
		}
		q.items[i], q.items[parent] = q.items[parent], q.items[i]
		i = parent
	}
}

func (q *nearestQueue) bubbleDown(i int) {
	n := len(q.items)
	for {
		left := 2*i + 1
		right := 2*i + 2
		largest := i

		if left < n && q.items[left].Distance > q.items[largest].Distance {
			largest = left
		}
		if right < n && q.items[right].Distance > q.items[largest].Distance {
			largest = right
		}

		if largest == i {
			break
		}

		q.items[i], q.items[largest] = q.items[largest], q.items[i]
		i = largest
	}
}
