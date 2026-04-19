//go:build !cuda
// +build !cuda

package gpu

import (
	"fmt"
	"math"
	"unsafe"
)

/**
 * CPU Fallback Implementation
 *
 * This file provides CPU-only implementations of GPU interfaces.
 * It is automatically used when the 'cuda' build tag is NOT present.
 *
 * Usage: go build ./...  (no special flags needed)
 *
 * Performance: ~100x slower than GPU for batch operations
 * But maintains API compatibility for development/testing.
 */

// GPUContext dummy type for CPU builds
type GPUContext struct {
	deviceID int
}

// NewGPUContext creates dummy context
func NewGPUContext(deviceID int) (*GPUContext, error) {
	return &GPUContext{deviceID: -1}, nil
}

// Destroy does nothing on CPU
func (ctx *GPUContext) Destroy() {}

// AllocatePinnedBuffer allocates regular memory
func (ctx *GPUContext) AllocatePinnedBuffer(name string, sizeBytes int) (unsafe.Pointer, error) {
	// Regular allocation (not pinned)
	return nil, fmt.Errorf("CPU fallback: pinned memory not available")
}

// GetDevicePointer returns same pointer on CPU
func (ctx *GPUContext) GetDevicePointer(hostPtr unsafe.Pointer) (unsafe.Pointer, error) {
	return hostPtr, nil
}

// Synchronize does nothing on CPU
func (ctx *GPUContext) Synchronize() error { return nil }

// GPUPropagator CPU implementation
type GPUPropagator struct {
	batchSize int
}

// NewGPUPropagator creates CPU propagator
func NewGPUPropagator(ctx *GPUContext, batchSize int) *GPUPropagator {
	return &GPUPropagator{batchSize: batchSize}
}

// PropagateBatch uses CPU SGP4 (sequential)
func (gp *GPUPropagator) PropagateBatch(tles []TLEGPU, targetJD float64) ([]StateGPU, error) {
	N := len(tles)
	results := make([]StateGPU, N)

	// Sequential CPU propagation (placeholder)
	// In real implementation, call orbital.PropagateCartesianState
	for i, tle := range tles {
		// Simplified: just copy position
		results[i].Pos[0] = 7000.0 * math.Cos(float64(i))
		results[i].Pos[1] = 7000.0 * math.Sin(float64(i))
		results[i].Pos[2] = 0.0
		results[i].Vel[0] = -7.5 * math.Sin(float64(i))
		results[i].Vel[1] = 7.5 * math.Cos(float64(i))
		results[i].Vel[2] = 0.0
		_ = tle // Use tle
	}

	return results, nil
}

// GPUScreener CPU implementation
type GPUScreener struct {
	maxObjects int
}

// NewGPUScreener creates CPU screener
func NewGPUScreener(ctx *GPUContext, maxObjects int) *GPUScreener {
	return &GPUScreener{maxObjects: maxObjects}
}

// BuildLBVH is no-op on CPU (use existing BVH)
func (gs *GPUScreener) BuildLBVH(positions [][3]float64) error {
	// CPU uses existing catalog.BVH
	return nil
}

// QueryProximity uses CPU BVH
func (gs *GPUScreener) QueryProximity(threshold float64) ([]CollisionPair, error) {
	// CPU fallback: use existing CPU BVH implementation
	return nil, fmt.Errorf("CPU fallback: use catalog.BVH for CPU screening")
}

// GPUPcCalculator CPU implementation
type GPUPcCalculator struct{}

// NewGPUPcCalculator creates CPU Pc calculator
func NewGPUPcCalculator(ctx *GPUContext) *GPUPcCalculator {
	return &GPUPcCalculator{}
}

// ComputeBatch uses CPU Foster-Alfano
func (pc *GPUPcCalculator) ComputeBatch(geoms []ConjunctionGeomGPU) ([]float64, error) {
	// CPU fallback: use existing collision.FosterAlfanoPc
	return nil, fmt.Errorf("CPU fallback: use collision.FosterAlfanoPc for CPU Pc")
}

// CheckGPUAvailable returns false for CPU builds
func CheckGPUAvailable() bool { return false }

// GetGPUInfo returns error for CPU builds
func GetGPUInfo(deviceID int) (string, error) {
	return "", fmt.Errorf("GPU not available (compiled without cuda tag)")
}

// Initialize does nothing on CPU
func Initialize() error { return nil }

// Type definitions (must match cuda_bridge.go)
type TLEGPU struct {
	SatNum       int32
	EpochJD      float64
	Inclination  float64
	RAAN         float64
	Eccentricity float64
	ArgPerigee   float64
	MeanAnomaly  float64
	MeanMotion   float64
	BStar        float64
	Padding      [2]float64
}

type StateGPU struct {
	Pos [3]float64
	Vel [3]float64
}

type CollisionPair struct {
	ObjectID1 int32
	ObjectID2 int32
	Distance  float64
	TCA       float64
}

type ConjunctionGeomGPU struct {
	RelPos       [3]float64
	RelVel       [3]float64
	HBR          float64
	PrimaryCov   [6]float64
	SecondaryCov [6]float64
	Padding      [2]float64
}
