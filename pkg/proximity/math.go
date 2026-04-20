package proximity

import (
	"math"

	"ssis/pkg/types"
)

// EuclideanDistance computes the Euclidean distance between two 3D points
func EuclideanDistance(p1, p2 types.Vector3) float64 {
	dx := p1.X - p2.X
	dy := p1.Y - p2.Y
	dz := p1.Z - p2.Z
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

// EuclideanDistance3D computes the Euclidean distance between two 3D points represented as arrays
func EuclideanDistance3D(p1, p2 [3]float64) float64 {
	dx := p1[0] - p2[0]
	dy := p1[1] - p2[1]
	dz := p1[2] - p2[2]
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

// SquaredDistance computes the squared Euclidean distance between two 3D points
// This avoids the sqrt operation and is useful when only comparing distances
func SquaredDistance(p1, p2 types.Vector3) float64 {
	dx := p1.X - p2.X
	dy := p1.Y - p2.Y
	dz := p1.Z - p2.Z
	return dx*dx + dy*dy + dz*dz
}

// SquaredDistance3D computes the squared Euclidean distance between two 3D points as arrays
// This avoids the sqrt operation and is useful when only comparing distances
func SquaredDistance3D(p1, p2 [3]float64) float64 {
	dx := p1[0] - p2[0]
	dy := p1[1] - p2[1]
	dz := p1[2] - p2[2]
	return dx*dx + dy*dy + dz*dz
}

// Clamp01 clamps a value between 0 and 1
func Clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// Clamp clamps a value between min and max
func Clamp(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
