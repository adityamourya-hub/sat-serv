package octree

import (
	"container/heap"
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"ssis/pkg/types"
)

// Vector3 is a backward compatibility alias for types.Vector3.
// New code should use types.Vector3 directly.
type Vector3 = types.Vector3

type Bound struct {
	Min, Max Vector3
}

func (b Bound) Contains(p Vector3) bool {
	return p.X >= b.Min.X && p.X <= b.Max.X &&
		p.Y >= b.Min.Y && p.Y <= b.Max.Y &&
		p.Z >= b.Min.Z && p.Z <= b.Max.Z
}

type Object interface {
	GetPosition() Vector3
	GetID() int
	GetTimestamp() float64
}

const (
	MaxObjects = 8
	MaxDepth   = 10 // 1 km^3 Resolution at LEO
)

type Node struct {
	Bounds     Bound
	Objects    []Object
	Children   [8]*Node
	IsLeaf     bool
	Depth      int
	MinT, MaxT float64
}

// Octree implements a Sparse Voxel Octree (SVO) with temporal support.
type Octree struct {
	Root *Node
	mu   sync.RWMutex
}

func NewOctree(min, max Vector3) *Octree {
	return &Octree{
		Root: &Node{
			Bounds: Bound{Min: min, Max: max},
			IsLeaf: true,
			Depth:  0,
			MinT:   1e15,
			MaxT:   -1e15,
		},
	}
}

func (n *Node) childIndex(p Vector3) int {
	midx := (n.Bounds.Min.X + n.Bounds.Max.X) / 2
	midy := (n.Bounds.Min.Y + n.Bounds.Max.Y) / 2
	midz := (n.Bounds.Min.Z + n.Bounds.Max.Z) / 2
	idx := 0
	if p.X > midx {
		idx |= 1
	}
	if p.Y > midy {
		idx |= 2
	}
	if p.Z > midz {
		idx |= 4
	}
	return idx
}

func (n *Node) Insert(obj Object) {
	if !n.Bounds.Contains(obj.GetPosition()) {
		return
	}
	t := obj.GetTimestamp()
	if t < n.MinT {
		n.MinT = t
	}
	if t > n.MaxT {
		n.MaxT = t
	}

	if n.IsLeaf {
		n.Objects = append(n.Objects, obj)
		if len(n.Objects) > MaxObjects && n.Depth < MaxDepth {
			n.subdivide()
		}
		return
	}
	idx := n.childIndex(obj.GetPosition())
	if n.Children[idx] == nil {
		// Delayed allocation for Sparsity
		n.Children[idx] = n.createChild(idx)
	}
	n.Children[idx].Insert(obj)
}

func (n *Node) createChild(idx int) *Node {
	midX := (n.Bounds.Min.X + n.Bounds.Max.X) / 2
	midY := (n.Bounds.Min.Y + n.Bounds.Max.Y) / 2
	midZ := (n.Bounds.Min.Z + n.Bounds.Max.Z) / 2
	var min, max Vector3
	if idx&1 == 0 {
		min.X, max.X = n.Bounds.Min.X, midX
	} else {
		min.X, max.X = midX, n.Bounds.Max.X
	}
	if idx&2 == 0 {
		min.Y, max.Y = n.Bounds.Min.Y, midY
	} else {
		min.Y, max.Y = midY, n.Bounds.Max.Y
	}
	if idx&4 == 0 {
		min.Z, max.Z = n.Bounds.Min.Z, midZ
	} else {
		min.Z, max.Z = midZ, n.Bounds.Max.Z
	}
	return &Node{Bounds: Bound{Min: min, Max: max}, IsLeaf: true, Depth: n.Depth + 1, MinT: 1e15, MaxT: -1e15}
}

func (n *Node) subdivide() {
	if !n.IsLeaf {
		return
	}
	n.IsLeaf = false
	objects := n.Objects
	n.Objects = nil
	for _, obj := range objects {
		idx := n.childIndex(obj.GetPosition())
		if n.Children[idx] == nil {
			n.Children[idx] = n.createChild(idx)
		}
		n.Children[idx].Insert(obj)
	}
}

// Remove deletes an object from the octree. Returns true if found and removed.
func (n *Node) Remove(obj Object) bool {
	if n == nil || !n.Bounds.Contains(obj.GetPosition()) {
		return false
	}
	if n.IsLeaf {
		for i, o := range n.Objects {
			if o.GetID() == obj.GetID() {
				n.Objects = append(n.Objects[:i], n.Objects[i+1:]...)
				n.updateTemporalBounds()
				return true
			}
		}
		return false
	}
	idx := n.childIndex(obj.GetPosition())
	if n.Children[idx] == nil {
		return false
	}
	removed := n.Children[idx].Remove(obj)
	if removed {
		n.updateTemporalBounds()
		n.tryCollapse()
	}
	return removed
}

func (n *Node) updateTemporalBounds() {
	n.MinT = 1e15
	n.MaxT = -1e15
	if n.IsLeaf {
		for _, obj := range n.Objects {
			t := obj.GetTimestamp()
			if t < n.MinT {
				n.MinT = t
			}
			if t > n.MaxT {
				n.MaxT = t
			}
		}
		return
	}
	for i := 0; i < 8; i++ {
		if n.Children[i] != nil {
			if n.Children[i].MinT < n.MinT {
				n.MinT = n.Children[i].MinT
			}
			if n.Children[i].MaxT > n.MaxT {
				n.MaxT = n.Children[i].MaxT
			}
		}
	}
}

func (n *Node) tryCollapse() {
	if n.IsLeaf {
		return
	}
	totalObjects := 0
	for i := 0; i < 8; i++ {
		if n.Children[i] != nil {
			if !n.Children[i].IsLeaf {
				return // Can't collapse if any child is internal
			}
			totalObjects += len(n.Children[i].Objects)
		}
	}
	if totalObjects <= MaxObjects {
		n.IsLeaf = true
		n.Objects = make([]Object, 0, totalObjects)
		for i := 0; i < 8; i++ {
			if n.Children[i] != nil {
				n.Objects = append(n.Objects, n.Children[i].Objects...)
				n.Children[i] = nil
			}
		}
	}
}

// QueryRangeTemporal performs a 4D spatial-temporal query on the SVO.
func (n *Node) queryRangeTemporal(min, max Vector3, tMin, tMax float64, results *[]Object) {
	if n == nil {
		return
	}

	// Temporal pruning (skip if node contains unspecified-timestamp objects: MinT==0)
	if n.MinT > 0 && (n.MinT > tMax || n.MaxT < tMin) {
		return
	}

	// Spatial pruning
	if !n.overlaps(min, max) {
		return
	}

	// Full containment optimization: if this node is entirely within query AABB and temporal range,
	// we just collect all objects without further spatial checks.
	fullSpatial := n.isContained(min, max)
	fullTemporal := n.MinT >= tMin && n.MaxT <= tMax

	if n.IsLeaf {
		for _, obj := range n.Objects {
			if fullSpatial && fullTemporal {
				*results = append(*results, obj)
				continue
			}
			pos := obj.GetPosition()
			ts := obj.GetTimestamp()
			tMatch := ts == 0 || (ts >= tMin && ts <= tMax)
			if pos.X >= min.X && pos.X <= max.X &&
				pos.Y >= min.Y && pos.Y <= max.Y &&
				pos.Z >= min.Z && pos.Z <= max.Z &&
				tMatch {
				*results = append(*results, obj)
			}
		}
		return
	}

	for i := 0; i < 8; i++ {
		if n.Children[i] != nil {
			n.Children[i].queryRangeTemporal(min, max, tMin, tMax, results)
		}
	}
}

func (n *Node) overlaps(min, max Vector3) bool {
	return n.Bounds.Min.X <= max.X && n.Bounds.Max.X >= min.X &&
		n.Bounds.Min.Y <= max.Y && n.Bounds.Max.Y >= min.Y &&
		n.Bounds.Min.Z <= max.Z && n.Bounds.Max.Z >= min.Z
}

func (n *Node) isContained(min, max Vector3) bool {
	return n.Bounds.Min.X >= min.X && n.Bounds.Max.X <= max.X &&
		n.Bounds.Min.Y >= min.Y && n.Bounds.Max.Y <= max.Y &&
		n.Bounds.Min.Z >= min.Z && n.Bounds.Max.Z <= max.Z
}

func (oct *Octree) Insert(obj Object) {
	oct.mu.Lock()
	defer oct.mu.Unlock()
	oct.Root.Insert(obj)
}

func (oct *Octree) Remove(obj Object) bool {
	oct.mu.Lock()
	defer oct.mu.Unlock()
	return oct.Root.Remove(obj)
}

func (oct *Octree) QueryRange(min, max Vector3, tMin, tMax float64) []Object {
	oct.mu.RLock()
	defer oct.mu.RUnlock()
	out := make([]Object, 0, MaxObjects*4)
	oct.Root.queryRangeTemporal(min, max, tMin, tMax, &out)
	return out
}

// QueryRangeIter performs a callback-based query without allocation.
// Return false from callback to stop iteration early.
func (oct *Octree) QueryRangeIter(min, max Vector3, tMin, tMax float64, callback func(Object) bool) {
	oct.mu.RLock()
	defer oct.mu.RUnlock()
	oct.Root.queryRangeIter(min, max, tMin, tMax, callback)
}

func (n *Node) queryRangeIter(min, max Vector3, tMin, tMax float64, callback func(Object) bool) bool {
	if n == nil {
		return true
	}
	if n.MinT > 0 && (n.MinT > tMax || n.MaxT < tMin) {
		return true
	}
	if !n.overlaps(min, max) {
		return true
	}
	fullSpatial := n.isContained(min, max)
	fullTemporal := n.MinT >= tMin && n.MaxT <= tMax

	if n.IsLeaf {
		for _, obj := range n.Objects {
			if fullSpatial && fullTemporal {
				if !callback(obj) {
					return false
				}
				continue
			}
			pos := obj.GetPosition()
			ts := obj.GetTimestamp()
			tMatch := ts == 0 || (ts >= tMin && ts <= tMax)
			if pos.X >= min.X && pos.X <= max.X &&
				pos.Y >= min.Y && pos.Y <= max.Y &&
				pos.Z >= min.Z && pos.Z <= max.Z &&
				tMatch {
				if !callback(obj) {
					return false
				}
			}
		}
		return true
	}
	for i := 0; i < 8; i++ {
		if n.Children[i] != nil {
			if !n.Children[i].queryRangeIter(min, max, tMin, tMax, callback) {
				return false
			}
		}
	}
	return true
}

// Stats holds octree statistics.
type Stats struct {
	Nodes       int
	Leafs       int
	Objects     int
	MaxDepth    int
	AvgDepth    float64
	MemoryBytes int64
}

// Stats returns octree statistics including memory estimate.
func (oct *Octree) Stats() Stats {
	oct.mu.RLock()
	defer oct.mu.RUnlock()
	return oct.Root.collectStats()
}

func (n *Node) collectStats() Stats {
	if n == nil {
		return Stats{}
	}
	s := Stats{
		Nodes:    1,
		MaxDepth: n.Depth,
	}
	if n.IsLeaf {
		s.Leafs = 1
		s.Objects = len(n.Objects)
	} else {
		totalDepth := 0
		for i := 0; i < 8; i++ {
			if n.Children[i] != nil {
				cs := n.Children[i].collectStats()
				s.Nodes += cs.Nodes
				s.Leafs += cs.Leafs
				s.Objects += cs.Objects
				if cs.MaxDepth > s.MaxDepth {
					s.MaxDepth = cs.MaxDepth
				}
				totalDepth += cs.MaxDepth * cs.Objects
			}
		}
		if s.Objects > 0 {
			s.AvgDepth = float64(totalDepth) / float64(s.Objects)
		}
	}
	// Estimate memory: Node struct + Object slice overhead
	const nodeSize = int64(unsafe.Sizeof(Node{}))
	s.MemoryBytes = int64(s.Nodes) * nodeSize
	return s
}

// objectHeap implements a min-heap for k-NN search.
type objectHeap struct {
	objects []Object
	origin  Vector3
	dists   []float64
}

func (h *objectHeap) Len() int           { return len(h.objects) }
func (h *objectHeap) Less(i, j int) bool { return h.dists[i] > h.dists[j] } // Max-heap for eviction
func (h *objectHeap) Swap(i, j int) {
	h.objects[i], h.objects[j] = h.objects[j], h.objects[i]
	h.dists[i], h.dists[j] = h.dists[j], h.dists[i]
}
func (h *objectHeap) Push(x interface{}) {
	obj := x.(Object)
	h.objects = append(h.objects, obj)
	h.dists = append(h.dists, dist2(h.origin, obj.GetPosition()))
}
func (h *objectHeap) Pop() interface{} {
	old := h.objects
	n := len(old)
	obj := old[n-1]
	h.objects = old[:n-1]
	h.dists = h.dists[:n-1]
	return obj
}

func dist2(a, b Vector3) float64 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	dz := a.Z - b.Z
	return dx*dx + dy*dy + dz*dz
}

func minDist2ToBound(p Vector3, b Bound) float64 {
	var dx, dy, dz float64
	if p.X < b.Min.X {
		dx = b.Min.X - p.X
	} else if p.X > b.Max.X {
		dx = p.X - b.Max.X
	}
	if p.Y < b.Min.Y {
		dy = b.Min.Y - p.Y
	} else if p.Y > b.Max.Y {
		dy = p.Y - b.Max.Y
	}
	if p.Z < b.Min.Z {
		dz = b.Min.Z - p.Z
	} else if p.Z > b.Max.Z {
		dz = p.Z - b.Max.Z
	}
	return dx*dx + dy*dy + dz*dz
}

// Nearest returns the k nearest objects to position, excluding excludeID.
func (oct *Octree) Nearest(pos Vector3, k int, excludeID int) []Object {
	oct.mu.RLock()
	defer oct.mu.RUnlock()
	if k <= 0 {
		return nil
	}
	h := &objectHeap{origin: pos}
	oct.Root.nearest(pos, k, excludeID, h)
	return h.objects
}

func (n *Node) nearest(pos Vector3, k int, excludeID int, h *objectHeap) {
	if n == nil {
		return
	}
	// Prune if this node cannot contain closer points than worst in heap
	if h.Len() == k {
		minDist2 := minDist2ToBound(pos, n.Bounds)
		if minDist2 > h.dists[0] {
			return
		}
	}
	if n.IsLeaf {
		for _, obj := range n.Objects {
			if obj.GetID() == excludeID {
				continue
			}
			d2 := dist2(pos, obj.GetPosition())
			if h.Len() < k {
				heap.Push(h, obj)
			} else if d2 < h.dists[0] {
				heap.Pop(h)
				heap.Push(h, obj)
			}
		}
		return
	}
	// Visit children sorted by distance for better pruning
	type childDist struct {
		idx  int
		dist float64
	}
	var children []childDist
	for i := 0; i < 8; i++ {
		if n.Children[i] != nil {
			d := minDist2ToBound(pos, n.Children[i].Bounds)
			children = append(children, childDist{i, d})
		}
	}
	// Sort by min distance (closest first)
	for i := 0; i < len(children); i++ {
		for j := i + 1; j < len(children); j++ {
			if children[j].dist < children[i].dist {
				children[i], children[j] = children[j], children[i]
			}
		}
	}
	for _, cd := range children {
		n.Children[cd.idx].nearest(pos, k, excludeID, h)
	}
}

// InsertBatch inserts multiple objects efficiently.
// Disables collapse during insertion, runs it once at end.
func (oct *Octree) InsertBatch(objs []Object) {
	oct.mu.Lock()
	defer oct.mu.Unlock()
	for _, obj := range objs {
		oct.Root.insertNoCollapse(obj)
	}
	oct.Root.collapseIfNeeded()
}

func (n *Node) insertNoCollapse(obj Object) {
	if !n.Bounds.Contains(obj.GetPosition()) {
		return
	}
	t := obj.GetTimestamp()
	if t < n.MinT {
		n.MinT = t
	}
	if t > n.MaxT {
		n.MaxT = t
	}
	if n.IsLeaf {
		n.Objects = append(n.Objects, obj)
		if len(n.Objects) > MaxObjects && n.Depth < MaxDepth {
			n.subdivide()
		}
		return
	}
	idx := n.childIndex(obj.GetPosition())
	if n.Children[idx] == nil {
		n.Children[idx] = n.createChild(idx)
	}
	n.Children[idx].insertNoCollapse(obj)
}

func (n *Node) collapseIfNeeded() {
	if n.IsLeaf {
		return
	}
	for i := 0; i < 8; i++ {
		if n.Children[i] != nil {
			n.Children[i].collapseIfNeeded()
		}
	}
	n.tryCollapse()
}

// Validation errors
var (
	ErrObjectOutOfBounds = errors.New("object outside node bounds")
	ErrInvalidTemporal   = errors.New("invalid temporal bounds (MinT > MaxT)")
	ErrDuplicateID       = errors.New("duplicate object ID")
	ErrTooManyObjects    = errors.New("leaf exceeds MaxObjects without being at MaxDepth")
	ErrEmptyInternalNode = errors.New("internal node has no children")
)

// Validate checks octree integrity. Returns first error found or nil.
func (oct *Octree) Validate() error {
	oct.mu.RLock()
	defer oct.mu.RUnlock()
	seen := make(map[int]struct{})
	return oct.Root.validate(seen)
}

func (n *Node) validate(seen map[int]struct{}) error {
	if n == nil {
		return nil
	}
	if n.MinT > n.MaxT && n.MaxT != -1e15 { // Allow initial empty state
		return fmt.Errorf("%w: node at depth %d has MinT=%f > MaxT=%f", ErrInvalidTemporal, n.Depth, n.MinT, n.MaxT)
	}
	if n.IsLeaf {
		if len(n.Objects) > MaxObjects && n.Depth < MaxDepth {
			return fmt.Errorf("%w: %d objects at depth %d", ErrTooManyObjects, len(n.Objects), n.Depth)
		}
		for _, obj := range n.Objects {
			pos := obj.GetPosition()
			if !n.Bounds.Contains(pos) {
				return fmt.Errorf("%w: object %d at (%f,%f,%f) outside (%+v to %+v)",
					ErrObjectOutOfBounds, obj.GetID(), pos.X, pos.Y, pos.Z, n.Bounds.Min, n.Bounds.Max)
			}
			id := obj.GetID()
			if _, ok := seen[id]; ok {
				return fmt.Errorf("%w: %d", ErrDuplicateID, id)
			}
			seen[id] = struct{}{}
		}
		return nil
	}
	childCount := 0
	for i := 0; i < 8; i++ {
		if n.Children[i] != nil {
			childCount++
			if err := n.Children[i].validate(seen); err != nil {
				return err
			}
		}
	}
	if childCount == 0 {
		return fmt.Errorf("%w at depth %d", ErrEmptyInternalNode, n.Depth)
	}
	return nil
}
