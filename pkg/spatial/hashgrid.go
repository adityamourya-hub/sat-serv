package spatial

import (
	"math"
	"ssis/pkg/octree"
)

type CellKey struct {
	X int
	Y int
	Z int
}

type Point struct {
	ID  int
	Pos octree.Vector3
}

type HashGrid struct {
	cellSize  float64
	invCell   float64
	cells     map[CellKey][]int
	positions map[int]octree.Vector3
}

func NewHashGrid(cellSize float64) *HashGrid {
	if cellSize <= 0 {
		cellSize = 25.0
	}
	return &HashGrid{
		cellSize:  cellSize,
		invCell:   1.0 / cellSize,
		cells:     make(map[CellKey][]int),
		positions: make(map[int]octree.Vector3),
	}
}

func (g *HashGrid) Rebuild(points []Point) {
	g.cells = make(map[CellKey][]int, len(points)/4+1)
	g.positions = make(map[int]octree.Vector3, len(points))
	for _, p := range points {
		g.Upsert(p)
	}
}

func (g *HashGrid) Upsert(p Point) {
	if oldPos, ok := g.positions[p.ID]; ok {
		oldKey := g.keyFor(oldPos)
		g.removeFromCell(oldKey, p.ID)
	}

	key := g.keyFor(p.Pos)
	g.cells[key] = append(g.cells[key], p.ID)
	g.positions[p.ID] = p.Pos
}

func (g *HashGrid) Delete(id int) {
	oldPos, ok := g.positions[id]
	if !ok {
		return
	}
	oldKey := g.keyFor(oldPos)
	g.removeFromCell(oldKey, id)
	delete(g.positions, id)
}

func (g *HashGrid) QueryAABB(min, max octree.Vector3) []int {
	minK := g.keyFor(min)
	maxK := g.keyFor(max)

	seen := make(map[int]struct{})
	ids := make([]int, 0)

	for x := minK.X; x <= maxK.X; x++ {
		for y := minK.Y; y <= maxK.Y; y++ {
			for z := minK.Z; z <= maxK.Z; z++ {
				bucket := g.cells[CellKey{X: x, Y: y, Z: z}]
				for _, id := range bucket {
					if _, ok := seen[id]; ok {
						continue
					}
					pos := g.positions[id]
					if pos.X < min.X || pos.X > max.X || pos.Y < min.Y || pos.Y > max.Y || pos.Z < min.Z || pos.Z > max.Z {
						continue
					}
					seen[id] = struct{}{}
					ids = append(ids, id)
				}
			}
		}
	}

	return ids
}

func (g *HashGrid) QueryRadius(center octree.Vector3, radius float64) []int {
	if radius <= 0 {
		return nil
	}
	min := octree.Vector3{X: center.X - radius, Y: center.Y - radius, Z: center.Z - radius}
	max := octree.Vector3{X: center.X + radius, Y: center.Y + radius, Z: center.Z + radius}
	candidates := g.QueryAABB(min, max)

	r2 := radius * radius
	out := make([]int, 0, len(candidates))
	for _, id := range candidates {
		p := g.positions[id]
		dx := p.X - center.X
		dy := p.Y - center.Y
		dz := p.Z - center.Z
		if dx*dx+dy*dy+dz*dz <= r2 {
			out = append(out, id)
		}
	}
	return out
}

func (g *HashGrid) keyFor(p octree.Vector3) CellKey {
	return CellKey{
		X: int(math.Floor(p.X * g.invCell)),
		Y: int(math.Floor(p.Y * g.invCell)),
		Z: int(math.Floor(p.Z * g.invCell)),
	}
}

func (g *HashGrid) removeFromCell(key CellKey, id int) {
	bucket := g.cells[key]
	for i := range bucket {
		if bucket[i] == id {
			bucket[i] = bucket[len(bucket)-1]
			bucket = bucket[:len(bucket)-1]
			break
		}
	}
	if len(bucket) == 0 {
		delete(g.cells, key)
		return
	}
	g.cells[key] = bucket
}
