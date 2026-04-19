//go:build cuda
// +build cuda

package gpu

/**
 * CUDA Bridge - Go to GPU Interface
 *
 * BUILD CONSTRAINT: This file requires CUDA toolkit.
 * Compile with: go build -tags cuda ./...
 * Without cuda tag, this file is excluded and CPU-only paths are used.
 *
 * Provides zero-copy integration between Go orchestration
 * and CUDA computation kernels.
 *
 * Architecture:
 * - Pinned memory buffers (accessible by CPU and GPU)
 * - Batch kernel launches for propagation/screening
 * - Async execution with callback synchronization
 *
 * Requires: CUDA toolkit, nvcc compiler
 *
 * Build: go build -tags cuda ./...
 */

import (
	"fmt"
	"runtime"
	"unsafe"
)

// #cgo LDFLAGS: -L${SRCDIR}/cuda -lsgp4_gpu -lcudart -lcuda
// #cgo CFLAGS: -I${SRCDIR}/cuda/include
// #include <cuda_runtime.h>
// #include "sgp4_gpu.h"
import "C"

// CUDAError represents CUDA runtime errors
type CUDAError struct {
	Code    CUresult
	Message string
}

func (e CUDAError) Error() string {
	return fmt.Sprintf("CUDA error %d: %s", e.Code, e.Message)
}

// CUresult is CUDA driver API result code
type CUresult int

const (
	CUDA_SUCCESS               CUresult = 0
	CUDA_ERROR_INVALID_VALUE            = 1
	CUDA_ERROR_OUT_OF_MEMORY            = 2
	CUDA_ERROR_NOT_INITIALIZED          = 3
)

// GPUContext manages CUDA device and memory
type GPUContext struct {
	deviceID      int
	stream        C.cudaStream_t
	pinnedBuffers map[string]unsafe.Pointer
}

// NewGPUContext initializes CUDA device
func NewGPUContext(deviceID int) (*GPUContext, error) {
	ctx := &GPUContext{
		deviceID:      deviceID,
		pinnedBuffers: make(map[string]unsafe.Pointer),
	}

	// Set device
	result := C.cudaSetDevice(C.int(deviceID))
	if result != C.cudaSuccess {
		return nil, CUDAError{Code: CUresult(result), Message: "failed to set device"}
	}

	// Create non-blocking stream for async operations
	result = C.cudaStreamCreate(&ctx.stream)
	if result != C.cudaSuccess {
		return nil, CUDAError{Code: CUresult(result), Message: "failed to create stream"}
	}

	return ctx, nil
}

// Destroy cleans up CUDA resources
func (ctx *GPUContext) Destroy() {
	// Free all pinned buffers
	for name, ptr := range ctx.pinnedBuffers {
		C.cudaFreeHost(ptr)
		delete(ctx.pinnedBuffers, name)
	}

	// Destroy stream
	C.cudaStreamDestroy(ctx.stream)
}

// AllocatePinnedBuffer creates page-locked memory for zero-copy
func (ctx *GPUContext) AllocatePinnedBuffer(name string, sizeBytes int) (unsafe.Pointer, error) {
	var ptr unsafe.Pointer

	result := C.cudaHostAlloc(&ptr, C.size_t(sizeBytes), C.cudaHostAllocMapped)
	if result != C.cudaSuccess {
		return nil, CUDAError{Code: CUresult(result), Message: "failed to allocate pinned memory"}
	}

	ctx.pinnedBuffers[name] = ptr
	return ptr, nil
}

// GetDevicePointer returns GPU pointer for pinned buffer
func (ctx *GPUContext) GetDevicePointer(hostPtr unsafe.Pointer) (unsafe.Pointer, error) {
	var devicePtr unsafe.Pointer

	result := C.cudaHostGetDevicePointer(&devicePtr, hostPtr, 0)
	if result != C.cudaSuccess {
		return nil, CUDAError{Code: CUresult(result), Message: "failed to get device pointer"}
	}

	return devicePtr, nil
}

// Synchronize waits for all GPU operations to complete
func (ctx *GPUContext) Synchronize() error {
	result := C.cudaStreamSynchronize(ctx.stream)
	if result != C.cudaSuccess {
		return CUDAError{Code: CUresult(result), Message: "stream synchronization failed"}
	}
	return nil
}

// GPUPropagator performs batch SGP4 propagation on GPU
type GPUPropagator struct {
	ctx       *GPUContext
	batchSize int
}

// NewGPUPropagator creates propagator for batch operations
func NewGPUPropagator(ctx *GPUContext, batchSize int) *GPUPropagator {
	return &GPUPropagator{
		ctx:       ctx,
		batchSize: batchSize,
	}
}

// PropagateBatch propagates multiple TLEs on GPU
// Input:  Array of TLEs and target epochs
// Output: Array of ECI states [x,y,z,vx,vy,vz]
func (gp *GPUPropagator) PropagateBatch(
	tles []TLEGPU,
	targetJD float64,
) ([]StateGPU, error) {

	N := len(tles)
	if N > gp.batchSize {
		return nil, fmt.Errorf("batch size %d exceeds maximum %d", N, gp.batchSize)
	}

	// Allocate pinned buffers if not already done
	bufferSize := N * int(unsafe.Sizeof(TLEGPU{}))
	inputPtr, err := gp.ctx.AllocatePinnedBuffer("sgp4_input", bufferSize)
	if err != nil {
		return nil, err
	}

	outputSize := N * int(unsafe.Sizeof(StateGPU{}))
	outputPtr, err := gp.ctx.AllocatePinnedBuffer("sgp4_output", outputSize)
	if err != nil {
		return nil, err
	}

	// Copy TLE data to pinned buffer
	inputSlice := (*[1 << 30]TLEGPU)(inputPtr)[:N:N]
	for i, tle := range tles {
		inputSlice[i] = tle
	}

	// Get device pointers (zero-copy)
	d_inputPtr, err := gp.ctx.GetDevicePointer(inputPtr)
	if err != nil {
		return nil, err
	}

	d_outputPtr, err := gp.ctx.GetDevicePointer(outputPtr)
	if err != nil {
		return nil, err
	}

	// Launch kernel
	threadsPerBlock := 256
	blocks := (N + threadsPerBlock - 1) / threadsPerBlock

	C.launch_sgp4_kernel(
		d_inputPtr,
		d_outputPtr,
		C.int(N),
		C.double(targetJD),
		C.int(threadsPerBlock),
		gp.ctx.stream,
	)

	// Synchronize
	if err := gp.ctx.Synchronize(); err != nil {
		return nil, err
	}

	// Read results from pinned buffer (CPU-accessible)
	outputSlice := (*[1 << 30]StateGPU)(outputPtr)[:N:N]
	results := make([]StateGPU, N)
	copy(results, outputSlice)

	return results, nil
}

// TLEGPU represents TLE data in GPU-friendly format
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
	Padding      [2]float64 // Align to 64 bytes
}

// StateGPU represents propagated state from GPU
type StateGPU struct {
	Pos [3]float64 // km
	Vel [3]float64 // km/s
}

// GPUScreener performs LBVH-based proximity screening
type GPUScreener struct {
	ctx        *GPUContext
	maxObjects int
	lbvhTree   unsafe.Pointer
}

// NewGPUScreener creates screening engine
func NewGPUScreener(ctx *GPUContext, maxObjects int) *GPUScreener {
	return &GPUScreener{
		ctx:        ctx,
		maxObjects: maxObjects,
	}
}

// BuildLBVH constructs Linear BVH from object positions
func (gs *GPUScreener) BuildLBVH(positions [][3]float64) error {
	N := len(positions)

	// Allocate buffer for positions
	bufferSize := N * 3 * 8 // 3 doubles × 8 bytes
	posPtr, err := gs.ctx.AllocatePinnedBuffer("lbvh_positions", bufferSize)
	if err != nil {
		return err
	}

	// Copy positions
	posSlice := (*[1 << 30]float64)(posPtr)[: N*3 : N*3]
	for i, pos := range positions {
		posSlice[i*3+0] = pos[0]
		posSlice[i*3+1] = pos[1]
		posSlice[i*3+2] = pos[2]
	}

	d_posPtr, err := gs.ctx.GetDevicePointer(posPtr)
	if err != nil {
		return err
	}

	// Launch LBVH construction kernel
	C.launch_lbvh_build(
		d_posPtr,
		C.int(N),
		&gs.lbvhTree,
		gs.ctx.stream,
	)

	return gs.ctx.Synchronize()
}

// QueryProximity finds all pairs within threshold
func (gs *GPUScreener) QueryProximity(threshold float64) ([]CollisionPair, error) {
	maxPairs := gs.maxObjects * 100 // Assume avg 100 neighbors per object

	// Allocate output buffer
	pairBufferSize := maxPairs * int(unsafe.Sizeof(CollisionPair{}))
	pairPtr, err := gs.ctx.AllocatePinnedBuffer("collision_pairs", pairBufferSize)
	if err != nil {
		return nil, err
	}

	countPtr, err := gs.ctx.AllocatePinnedBuffer("pair_count", 4)
	if err != nil {
		return nil, err
	}

	d_pairPtr, _ := gs.ctx.GetDevicePointer(pairPtr)
	d_countPtr, _ := gs.ctx.GetDevicePointer(countPtr)

	// Launch proximity query
	C.launch_proximity_query(
		gs.lbvhTree,
		C.double(threshold),
		d_pairPtr,
		d_countPtr,
		gs.ctx.stream,
	)

	if err := gs.ctx.Synchronize(); err != nil {
		return nil, err
	}

	// Read pair count
	count := *(*int32)(countPtr)
	if count > int32(maxPairs) {
		count = int32(maxPairs) // Truncated
	}

	// Read pairs
	pairSlice := (*[1 << 30]CollisionPair)(pairPtr)[:count:count]
	results := make([]CollisionPair, count)
	copy(results, pairSlice)

	return results, nil
}

// CollisionPair represents a potential conjunction candidate
type CollisionPair struct {
	ObjectID1 int32
	ObjectID2 int32
	Distance  float64 // km at TCA
	TCA       float64 // Julian date
}

// GPUPcCalculator performs batch Pc computation
type GPUPcCalculator struct {
	ctx *GPUContext
}

// NewGPUPcCalculator creates Pc computation engine
func NewGPUPcCalculator(ctx *GPUContext) *GPUPcCalculator {
	return &GPUPcCalculator{ctx: ctx}
}

// ComputeBatch calculates Pc for multiple conjunctions
func (pc *GPUPcCalculator) ComputeBatch(
	geoms []ConjunctionGeomGPU,
) ([]float64, error) {

	N := len(geoms)

	// Allocate buffers
	geomSize := N * int(unsafe.Sizeof(ConjunctionGeomGPU{}))
	geomPtr, err := pc.ctx.AllocatePinnedBuffer("pc_geoms", geomSize)
	if err != nil {
		return nil, err
	}

	resultSize := N * 8
	resultPtr, err := pc.ctx.AllocatePinnedBuffer("pc_results", resultSize)
	if err != nil {
		return nil, err
	}

	// Copy geometry data
	geomSlice := (*[1 << 30]ConjunctionGeomGPU)(geomPtr)[:N:N]
	copy(geomSlice, geoms)

	d_geomPtr, _ := pc.ctx.GetDevicePointer(geomPtr)
	d_resultPtr, _ := pc.ctx.GetDevicePointer(resultPtr)

	// Launch Pc kernel
	threadsPerBlock := 128
	blocks := (N + threadsPerBlock - 1) / threadsPerBlock

	C.launch_pc_kernel(
		d_geomPtr,
		d_resultPtr,
		C.int(N),
		C.int(threadsPerBlock),
		pc.ctx.stream,
	)

	if err := pc.ctx.Synchronize(); err != nil {
		return nil, err
	}

	// Read results
	resultSlice := (*[1 << 30]float64)(resultPtr)[:N:N]
	results := make([]float64, N)
	copy(results, resultSlice)

	return results, nil
}

// ConjunctionGeomGPU represents conjunction parameters for GPU
type ConjunctionGeomGPU struct {
	RelPos       [3]float64
	RelVel       [3]float64
	HBR          float64
	PrimaryCov   [6]float64 // Upper triangle of 3x3
	SecondaryCov [6]float64
	Padding      [2]float64 // Align to 64 bytes
}

// CheckGPUAvailable returns true if CUDA is available
func CheckGPUAvailable() bool {
	var deviceCount C.int
	result := C.cudaGetDeviceCount(&deviceCount)
	return result == C.cudaSuccess && deviceCount > 0
}

// GetGPUInfo returns device information
func GetGPUInfo(deviceID int) (string, error) {
	var props C.cudaDeviceProp
	result := C.cudaGetDeviceProperties(&props, C.int(deviceID))
	if result != C.cudaSuccess {
		return "", CUDAError{Code: CUresult(result), Message: "failed to get device properties"}
	}

	name := C.GoString(&props.name[0])
	return fmt.Sprintf("%s (CC %d.%d, %d MB)",
		name, props.major, props.minor, props.totalGlobalMem/1024/1024), nil
}

// Initialize CUDA runtime (call once at startup)
func Initialize() error {
	result := C.cudaFree(nil) // Dummy call to initialize runtime
	if result != C.cudaSuccess {
		return CUDAError{Code: CUresult(result), Message: "CUDA initialization failed"}
	}
	return nil
}

func init() {
	// Lock goroutine to OS thread for CUDA calls
	runtime.LockOSThread()
}
