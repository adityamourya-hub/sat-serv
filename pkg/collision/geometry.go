package collision

import (
	"math"
	"sort"

	"ssis/pkg/types"
)

// Vector2 represents a 2D point or vector in the encounter plane.
type Vector2 struct {
	X, Y float64
}

// ConvexHull2D computes the 2D convex hull of a set of points using the Monotone Chain algorithm.
// The resulting hull is returned in counter-clockwise order.
func ConvexHull2D(points []Vector2) []Vector2 {
	n := len(points)
	if n <= 3 {
		return points
	}

	// Sort points primarily by X, then by Y
	sort.Slice(points, func(i, j int) bool {
		if points[i].X != points[j].X {
			return points[i].X < points[j].X
		}
		return points[i].Y < points[j].Y
	})

	upper := make([]Vector2, 0, n)
	for _, p := range points {
		for len(upper) >= 2 {
			if crossProduct2D(upper[len(upper)-2], upper[len(upper)-1], p) > 0 {
				break
			}
			upper = upper[:len(upper)-1]
		}
		upper = append(upper, p)
	}

	lower := make([]Vector2, 0, n)
	for i := n - 1; i >= 0; i-- {
		p := points[i]
		for len(lower) >= 2 {
			if crossProduct2D(lower[len(lower)-2], lower[len(lower)-1], p) > 0 {
				break
			}
			lower = lower[:len(lower)-1]
		}
		lower = append(lower, p)
	}

	return append(upper[:len(upper)-1], lower[:len(lower)-1]...)
}

func crossProduct2D(a, b, c Vector2) float64 {
	return (b.X-a.X)*(c.Y-a.Y) - (b.Y-a.Y)*(c.X-a.X)
}

// MinkowskiSum2D computes the vertex set for the Minkowski sum of two 2D point sets.
// For convex hulls, the sum is the convex hull of the pairwise sums of vertices.
func MinkowskiSum2D(v1, v2 []Vector2) []Vector2 {
	res := make([]Vector2, 0, len(v1)*len(v2))
	for _, p1 := range v1 {
		for _, p2 := range v2 {
			res = append(res, Vector2{X: p1.X + p2.X, Y: p1.Y + p2.Y})
		}
	}
	return ConvexHull2D(res)
}

// ProjectToEncounterPlane projects 3D vertices into the 2D encounter plane.
// vRel is the relative velocity vector defining the normal to the plane.
func ProjectToEncounterPlane(vertices []types.Vector3, vRel types.Vector3) []Vector2 {
	vMag := math.Sqrt(vRel.X*vRel.X + vRel.Y*vRel.Y + vRel.Z*vRel.Z)
	if vMag < 1e-9 {
		// Fallback for zero relative velocity (should not happen in TCA)
		res := make([]Vector2, len(vertices))
		for i, v := range vertices {
			res[i] = Vector2{X: v.X, Y: v.Y}
		}
		return res
	}

	// Define orthonormal basis for the encounter plane (z-axis along vRel)
	unitZ := types.Vector3{X: vRel.X / vMag, Y: vRel.Y / vMag, Z: vRel.Z / vMag}
	
	// Choose an arbitrary reference for X
	ref := types.Vector3{X: 1, Y: 0, Z: 0}
	if math.Abs(unitZ.X) > 0.9 {
		ref = types.Vector3{X: 0, Y: 1, Z: 0}
	}

	// unitX = normalize(ref cross unitZ)
	ux := crossProduct3D(ref, unitZ)
	uxMag := math.Sqrt(ux.X*ux.X + ux.Y*ux.Y + ux.Z*ux.Z)
	unitX := types.Vector3{X: ux.X / uxMag, Y: ux.Y / uxMag, Z: ux.Z / uxMag}

	// unitY = unitZ cross unitX
	unitY := crossProduct3D(unitZ, unitX)

	res := make([]Vector2, len(vertices))
	for i, v := range vertices {
		res[i] = Vector2{
			X: dotProduct3D(v, unitX),
			Y: dotProduct3D(v, unitY),
		}
	}
	return res
}

func crossProduct3D(a, b types.Vector3) types.Vector3 {
	return types.Vector3{
		X: a.Y*b.Z - a.Z*b.Y,
		Y: a.Z*b.X - a.X*b.Z,
		Z: a.X*b.Y - a.Y*b.X,
	}
}

func dotProduct3D(a, b types.Vector3) float64 {
	return a.X*b.X + a.Y*b.Y + a.Z*b.Z
}
