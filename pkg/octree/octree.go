package octree

type Vector3 struct {
	X, Y, Z float64
}

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
	if p.X >= midx {
		idx |= 1
	}
	if p.Y >= midy {
		idx |= 2
	}
	if p.Z >= midz {
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

func (oct *Octree) Insert(obj Object) { oct.Root.Insert(obj) }
func (oct *Octree) QueryRange(min, max Vector3, tMin, tMax float64) []Object {
	out := make([]Object, 0, 1024)
	oct.Root.queryRangeTemporal(min, max, tMin, tMax, &out)
	return out
}
