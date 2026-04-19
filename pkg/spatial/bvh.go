package spatial

import (
	"sort"

	"ssis/pkg/types"
)

/**
 * mortonBitsPerAxis and mortonMask define the resolution of the 
 * Space-Time hierarchical grid. For 4D (X, Y, Z, T), we use 16 bits 
 * per axis to fit within a 64-bit Morton code.
 */
const (
	mortonBitsPerAxis = 16
	mortonMask        = (1 << mortonBitsPerAxis) - 1
)

/**
 * AABB represents a 4D Axis-Aligned Bounding Box (Space-Time Volume).
 * It encapsulates 3D spatial bounds (Min, Max) and 1D temporal bounds (TMin, TMax).
 * This structure is essential for preventing temporal aliasing in conjunction queries.
 */
type AABB struct {
	Min, Max   types.Vector3
	TMin, TMax float64
}

/**
 * Expand updates the AABB to include the given 3D position and timestamp.
 * Entry Point: Logs the current bounds and the new point.
 * Exit Point: Logs the updated bounds.
 */
func (a *AABB) Expand(p types.Vector3, t float64) {
	if p.X < a.Min.X {
		a.Min.X = p.X
	}
	if p.Y < a.Min.Y {
		a.Min.Y = p.Y
	}
	if p.Z < a.Min.Z {
		a.Min.Z = p.Z
	}
	if p.X > a.Max.X {
		a.Max.X = p.X
	}
	if p.Y > a.Max.Y {
		a.Max.Y = p.Y
	}
	if p.Z > a.Max.Z {
		a.Max.Z = p.Z
	}
	if t < a.TMin {
		a.TMin = t
	}
	if t > a.TMax {
		a.TMax = t
	}
}

/**
 * Merge combines another AABB into the current one, extending all 4 dimensions.
 * Entry Point: Trace start.
 * Exit Point: Trace end.
 */
func (a *AABB) Merge(b AABB) {
	a.Expand(b.Min, b.TMin)
	a.Expand(b.Max, b.TMax)
}

/**
 * Overlap checks if two 4D AABBs intersect in both space and time.
 * This is critical for the broad-phase filter in conjunction detection.
 */
func (a AABB) Overlap(b AABB) bool {
	spatial := a.Min.X <= b.Max.X && a.Max.X >= b.Min.X &&
		a.Min.Y <= b.Max.Y && a.Max.Y >= b.Min.Y &&
		a.Min.Z <= b.Max.Z && a.Max.Z >= b.Min.Z
	temporal := a.TMin <= b.TMax && a.TMax >= b.TMin
	res := spatial && temporal
	return res
}

/**
 * Contains checks if a 3D point and timestamp reside within the 4D AABB.
 * Entry Point: Trace start.
 * Exit Point: Trace end.
 */
func (a AABB) Contains(p types.Vector3, t float64) bool {
	spatial := p.X >= a.Min.X && p.X <= a.Max.X &&
		p.Y >= a.Min.Y && p.Y <= a.Max.Y &&
		p.Z >= a.Min.Z && p.Z <= a.Max.Z
	temporal := t >= a.TMin && t <= a.TMax
	res := spatial && temporal
	return res
}

/**
 * BVHObject represents an object that can be inserted into the ST-BVH.
 * It must provide position, an unique integer ID, and a timestamp.
 */
type BVHObject interface {
	GetPosition() types.Vector3
	GetID() int
	GetTimestamp() float64
}

/**
 * BVHNode represents a node in the 4D Bounding Volume Hierarchy.
 * Leaf nodes contain a reference to a single BVHObject.
 */
type BVHNode struct {
	Bounds      AABB
	Object      BVHObject // nil for internal nodes
	Left, Right *BVHNode
	IsLeaf      bool
}

/**
 * BVH is the top-level Space-Time Bounding Volume Hierarchy structure.
 * It supports high-performance 4D proximity queries for SSA operations.
 */
type BVH struct {
	Root    *BVHNode
	Objects []BVHObject
	Bounds  AABB
}

/**
 * mortonCode computes a 64-bit Space-Time Morton code for a position and time.
 * It interleaves 16 bits per axis (X, Y, Z, T) into a single unsigned 64-bit integer.
 * This mapping ensures that objects close in space-time have similar codes.
 * Entry Point: Logs input parameters.
 * Exit Point: Logs the resulting 64-bit code.
 */
func mortonCode(pos types.Vector3, t float64, bounds AABB) uint64 {
	nx := normalizeAndQuantize(pos.X, bounds.Min.X, bounds.Max.X)
	ny := normalizeAndQuantize(pos.Y, bounds.Min.Y, bounds.Max.Y)
	nz := normalizeAndQuantize(pos.Z, bounds.Min.Z, bounds.Max.Z)
	nt := normalizeAndQuantize(t, bounds.TMin, bounds.TMax)

	code := interleaveBits4D(nx, ny, nz, nt)
	return code
}

/**
 * normalizeAndQuantize maps a float64 value into a 16-bit integer range [0, 65535] 
 * relative to the provided min/max bounds.
 */
func normalizeAndQuantize(val, min, max float64) uint32 {
	if max <= min {
		return 0
	}
	n := (val - min) / (max - min)
	if n < 0 {
		n = 0
	}
	if n > 1 {
		n = 1
	}
	return uint32(n * float64(mortonMask))
}

/**
 * interleaveBits4D interleaves four 16-bit values into a 64-bit Morton code.
 * Pattern: tzyxtzyxtzyx...
 * Entry Point: Trace start.
 * Exit Point: Trace end.
 */
func interleaveBits4D(x, y, z, t uint32) uint64 {
	var code uint64
	for i := 0; i < mortonBitsPerAxis; i++ {
		code |= uint64((x>>i)&1) << (i * 4)
		code |= uint64((y>>i)&1) << (i*4 + 1)
		code |= uint64((z>>i)&1) << (i*4 + 2)
		code |= uint64((t>>i)&1) << (i*4 + 3)
	}
	return code
}

/**
 * mortonItem is a internal sorting container pairing an object with its space-time key.
 */
type mortonItem struct {
	code   uint64
	object BVHObject
}

/**
 * NewBVH constructs a new 4D ST-BVH from a slice of BVHObjects.
 * It computes global 4D bounds, sorts objects by Morton code, 
 * and builds a balanced hierarchy recursively.
 * Entry Point: Logs object count.
 * Exit Point: Logs result tree root.
 */
func NewBVH(objects []BVHObject) *BVH {
	if len(objects) == 0 {
		return &BVH{}
	}

	// Compute global 4D bounds
	bounds := AABB{
		Min:  objects[0].GetPosition(),
		Max:  objects[0].GetPosition(),
		TMin: objects[0].GetTimestamp(),
		TMax: objects[0].GetTimestamp(),
	}
	for _, obj := range objects {
		bounds.Expand(obj.GetPosition(), obj.GetTimestamp())
	}

	// Add small epsilon to avoid zero-size bounds
	eps := 1.0
	bounds.Min.X -= eps; bounds.Min.Y -= eps; bounds.Min.Z -= eps; bounds.TMin -= eps
	bounds.Max.X += eps; bounds.Max.Y += eps; bounds.Max.Z += eps; bounds.TMax += eps

	// Create Morton-sorted items
	items := make([]mortonItem, len(objects))
	for i, obj := range objects {
		items[i] = mortonItem{
			code:   mortonCode(obj.GetPosition(), obj.GetTimestamp(), bounds),
			object: obj,
		}
	}

	// Sort by Morton code for space-time coherence
	sort.Slice(items, func(i, j int) bool {
		return items[i].code < items[j].code
	})

	// Build BVH from sorted items
	root := buildBVHRecursive(items, bounds)

	return &BVH{
		Root:    root,
		Objects: objects,
		Bounds:  bounds,
	}
}

/**
 * buildBVHRecursive builds the binary hierarchy from Morton-sorted items 
 * using a median-split approach for tree balancing.
 * Entry Point: Logs item count.
 * Exit Point: Logs result node leaf/internal status.
 */
func buildBVHRecursive(items []mortonItem, bounds AABB) *BVHNode {
	if len(items) == 0 {
		return nil
	}

	// Single object becomes a leaf
	if len(items) == 1 {
		pos := items[0].object.GetPosition()
		t := items[0].object.GetTimestamp()
		res := &BVHNode{
			Bounds: AABB{Min: pos, Max: pos, TMin: t, TMax: t},
			Object: items[0].object,
			IsLeaf: true,
		}
		return res
	}

	// Compute node bounds from all items
	nodeBounds := AABB{
		Min:  items[0].object.GetPosition(),
		Max:  items[0].object.GetPosition(),
		TMin: items[0].object.GetTimestamp(),
		TMax: items[0].object.GetTimestamp(),
	}
	for _, item := range items {
		nodeBounds.Expand(item.object.GetPosition(), item.object.GetTimestamp())
	}

	// Split at median of Morton codes
	splitIdx := len(items) / 2

	left := buildBVHRecursive(items[:splitIdx], bounds)
	right := buildBVHRecursive(items[splitIdx:], bounds)

	res := &BVHNode{
		Bounds: nodeBounds,
		Left:   left,
		Right:  right,
		IsLeaf: false,
	}
	return res
}

/**
 * QueryRange finds all objects within a 4D Space-Time hypercylinder 
 * (spatial AABB + temporal interval).
 * Entry Point: Logs spatial and temporal query parameters.
 * Exit Point: Logs result object count.
 */
func (bvh *BVH) QueryRange(min, max types.Vector3, tMin, tMax float64) []BVHObject {
	if bvh.Root == nil {
		return nil
	}
	queryBox := AABB{Min: min, Max: max, TMin: tMin, TMax: tMax}
	var results []BVHObject
	bvh.Root.queryRange(queryBox, &results)
	return results
}

/**
 * queryRange is the internal recursive implementation of 4D volume querying.
 */
func (n *BVHNode) queryRange(query AABB, results *[]BVHObject) {
	if n == nil || !n.Bounds.Overlap(query) {
		return
	}

	if n.IsLeaf {
		if query.Contains(n.Object.GetPosition(), n.Object.GetTimestamp()) {
			*results = append(*results, n.Object)
		}
		return
	}

	n.Left.queryRange(query, results)
	n.Right.queryRange(query, results)
}

/**
 * FindPotentialPairs performs a broad-phase Space-Time sweep to find all 
 * object-pairs whose 4D bounds overlap. 
 * This is the primary accelerator for Conjunction Analysis (Requirement c).
 * Entry Point: Trace start.
 * Exit Point: Logs result pair count.
 */
func (bvh *BVH) FindPotentialPairs() [][2]BVHObject {
	if bvh.Root == nil {
		return nil
	}

	pairSet := make(map[uint64][2]BVHObject)
	bvh.Root.findPairs(&pairSet)

	pairs := make([][2]BVHObject, 0, len(pairSet))
	for _, pair := range pairSet {
		pairs = append(pairs, pair)
	}
	return pairs
}

/**
 * findPairs is the recursive internal implementation of the ST-BVH broad-phase sweep.
 */
func (n *BVHNode) findPairs(pairSet *map[uint64][2]BVHObject) {
	if n == nil || n.IsLeaf {
		return
	}

	// Check all pairs between left and right subtrees
	n.Left.findCrossPairs(n.Right, pairSet)

	// Recurse into children
	n.Left.findPairs(pairSet)
	n.Right.findPairs(pairSet)
}

/**
 * findCrossPairs conservatively traverses two subtrees to find space-time intersections.
 */
func (n *BVHNode) findCrossPairs(other *BVHNode, pairSet *map[uint64][2]BVHObject) {
	if n == nil || other == nil {
		return
	}

	// Quick rejection: if 4D bounds don't overlap, no pairs possible in this branch
	if !n.Bounds.Overlap(other.Bounds) {
		return
	}

	// Both leaves: check direct intersection
	if n.IsLeaf && other.IsLeaf {
		if n.Object.GetID() != other.Object.GetID() {
			makeAndStorePair(pairSet, n.Object, other.Object)
		}
		return
	}

	// Traversal heuristics: recurse into larger volumes first
	if n.IsLeaf {
		other.Left.findCrossPairs(n, pairSet)
		other.Right.findCrossPairs(n, pairSet)
		return
	}
	if other.IsLeaf {
		n.Left.findCrossPairs(other, pairSet)
		n.Right.findCrossPairs(other, pairSet)
		return
	}

	n.Left.findCrossPairs(other.Left, pairSet)
	n.Left.findCrossPairs(other.Right, pairSet)
	n.Right.findCrossPairs(other.Left, pairSet)
	n.Right.findCrossPairs(other.Right, pairSet)
}

/**
 * makeAndStorePair creates a canonical hash key for a pair of objects 
 * to ensure bidirectional uniqueness in the result set.
 */
func makeAndStorePair(pairSet *map[uint64][2]BVHObject, a, b BVHObject) {
	idA, idB := uint32(a.GetID()), uint32(b.GetID())
	if idA > idB {
		idA, idB = idB, idA
	}
	key := (uint64(idA) << 32) | uint64(idB)

	if _, exists := (*pairSet)[key]; !exists {
		(*pairSet)[key] = [2]BVHObject{a, b}
	}
}

/**
 * Insert adds a single BVHObject to the structure.
 * Note: This triggers a full O(n log n) rebuild. For bulk insertions,
 * prefer InsertBatch to avoid O(n² log n) cost.
 */
func (bvh *BVH) Insert(obj BVHObject) {
	bvh.Objects = append(bvh.Objects, obj)
	newBVH := NewBVH(bvh.Objects)
	*bvh = *newBVH
}

/**
 * InsertBatch adds multiple BVHObjects and rebuilds the tree once.
 * Complexity: O((n+m) log (n+m)) where n = existing objects, m = new objects.
 * This avoids the O(m × n log n) cost of calling Insert in a loop.
 */
func (bvh *BVH) InsertBatch(objects []BVHObject) {
	if len(objects) == 0 {
		return
	}
	bvh.Objects = append(bvh.Objects, objects...)
	newBVH := NewBVH(bvh.Objects)
	*bvh = *newBVH
}

/**
 * Rebuild manually triggers a full ST-BVH reconstruction.
 */
func (bvh *BVH) Rebuild() {
	if len(bvh.Objects) > 0 {
		newBVH := NewBVH(bvh.Objects)
		*bvh = *newBVH
	}
}

/**
 * Clear purges all objects and resets the hierarchy.
 */
func (bvh *BVH) Clear() {
	bvh.Root = nil
	bvh.Objects = nil
}
