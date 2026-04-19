package types

// Vector3 represents a 3D vector with X, Y, Z components.
// This is the canonical definition used throughout SSIS for spatial operations.
// The type was moved from pkg/octree to pkg/types to decouple spatial primitives
// from the legacy octree implementation.
type Vector3 struct {
	X, Y, Z float64
}
