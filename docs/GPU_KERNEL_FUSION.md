# GPU Kernel Fusion: SGP4 + Morton Code Single-Pass Pipeline

## ADITI 4.0: Geometric Mechanics Optimization #4

---

## Problem Statement

**Current Data Flow (Multi-Pass):**
```
CPU → SGP4(GPU) → PCIe → CPU → LBVH(CPU) → PCIe → Proximity(GPU)
```

**Bottleneck:** Round-trip over PCIe bus even with pinned memory
- PCIe latency: ~5-10 μs per transfer
- Two round-trips: ~20 μs overhead per frame
- 100k objects: data movement dominates compute

---

## Solution: Single-Pass Spatial Hashing

**Fused Data Flow:**
```
CPU → [SGP4 + Morton Code] (GPU) → Radix Sort (GPU) → Proximity (GPU) → Results
        ↑_________________________GPU ONLY_________________________↑
        Data NEVER leaves GPU until high-risk pairs identified
```

---

## Mathematical Foundation: Morton Codes

### Z-Order Curve (Z-Curve)
**Property:** Spatial locality preserved in 1D ordering
- Nearby objects in 3D have similar Morton codes
- Bitwise operations for neighbor finding

**Computation:**
```
Morton(x, y, z) = interleave_bits(x, y, z)

Example (2D for clarity):
x = 0001 (1), y = 0010 (2)
Morton = 00001010 (10)

x = 0001 (1), y = 0011 (3)  
Morton = 00001011 (11)
```

**Key Insight:** Objects within 1 cell in 3D differ by 1-3 bits in Morton code

### Morton Code for 3D Spatial Hashing

```cuda
__device__ uint64_t computeMortonCode(float x, float y, float z, 
                                       float gridMin, float gridMax) {
    // Normalize to [0, 1023] (10 bits per dimension)
    uint32_t ix = (uint32_t)(1023.0f * (x - gridMin) / (gridMax - gridMin));
    uint32_t iy = (uint32_t)(1023.0f * (y - gridMin) / (gridMax - gridMin));
    uint32_t iz = (uint32_t)(1023.0f * (z - gridMin) / (gridMax - gridMin));
    
    // Expand 10-bit to 30-bit with interleaving
    // x: 0000000000xxxxxxxxxx → 000...x0x0x0x0x0x0x0x0
    // y: 0000000000yyyyyyyyyy → 000...0y0y0y0y0y0y0y0y
    // z: 0000000000zzzzzzzzzz → 000...z0z0z0z0z0z0z0z0
    
    uint64_t mx = expandBits10(ix);
    uint64_t my = expandBits10(iy);
    uint64_t mz = expandBits10(iz);
    
    // Interleave: zxy zxy zxy ...
    return (mz << 2) | (my << 1) | mx;
}

__device__ uint64_t expandBits10(uint32_t v) {
    // Magic bit expansion for 10-bit to 30-bit
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8))  & 0x0300F00F;
    v = (v | (v << 4))  & 0x030C30C3;
    v = (v | (v << 2))  & 0x09249249;
    return v;
}
```

---

## Kernel Fusion Architecture

### Kernel 1: Propagate + Morton Code

```cuda
__global__ void sgp4_morton_kernel(
    const TLEGPU* __restrict__ tles,
    uint64_t* __restrict__ morton_codes,
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    int N,
    double targetJD,
    float gridMin, float gridMax
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // 1. SGP4 propagation
    TLEGPU tle = tles[idx];
    StateGPU state = sgp4_propagate(tle, targetJD);
    
    // 2. Compute Morton code inline (no memory write between)
    uint64_t code = computeMortonCode(
        state.pos.x, state.pos.y, state.pos.z,
        gridMin, gridMax
    );
    
    // 3. Write results
    morton_codes[idx] = code;
    positions[idx] = make_float3(state.pos.x, state.pos.y, state.pos.z);
    velocities[idx] = make_float3(state.vel.x, state.vel.y, state.vel.z);
}
```

**Benefits:**
- No intermediate CPU-GPU transfer
- Positions stay in GPU memory for next kernel
- Morton codes computed "for free" during propagation

### Kernel 2: Radix Sort

```cuda
// Use CUB (CUDA Unbound) or custom radix sort
// Sort keys (Morton codes) with values (object indices)

void radixSortMortonCodes(
    uint64_t* d_keys,
    int* d_values,
    int N
) {
    // CUB radix sort - O(N) complexity
    // Stable sort preserves temporal ordering
    cub::DeviceRadixSort::SortPairs(d_temp, temp_size, 
                                    d_keys, d_keys_out,
                                    d_values, d_values_out,
                                    N);
}
```

**Result:** Objects sorted by spatial locality
- Nearby objects contiguous in memory
- Coalesced access for proximity detection

### Kernel 3: Neighbor Detection via Bitwise XOR

```cuda
__global__ void proximity_detection_kernel(
    const uint64_t* __restrict__ morton_codes,
    const int* __restrict__ object_indices,
    const float3* __restrict__ positions,
    int N,
    float threshold_km,
    CollisionPair* __restrict__ pairs_out,
    int* __restrict__ pair_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    uint64_t myCode = morton_codes[idx];
    float3 myPos = positions[idx];
    
    // Check neighbors in sorted list (spatially nearby)
    // Look ahead/behind by 10-20 positions
    for (int j = max(0, idx - 10); j < min(N, idx + 10); j++) {
        if (j == idx) continue;
        
        // Fast rejection: XOR of Morton codes
        // XOR < small_threshold → spatially nearby
        uint64_t xorCode = myCode ^ morton_codes[j];
        
        // Count trailing zeros (rough distance estimate)
        int distanceEstimate = __clzll(xorCode); // Count leading zeros
        
        if (distanceEstimate > 50) { // Within ~1 cell
            // Precise distance check
            float3 otherPos = positions[j];
            float dx = myPos.x - otherPos.x;
            float dy = myPos.y - otherPos.y;
            float dz = myPos.z - otherPos.z;
            float dist = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (dist < threshold_km) {
                // Atomic add to pair list
                int pairIdx = atomicAdd(pair_count, 1);
                pairs_out[pairIdx] = make_collision_pair(
                    object_indices[idx], 
                    object_indices[j], 
                    dist
                );
            }
        }
    }
}
```

---

## Memory Layout Optimization

### SoA (Structure of Arrays) for GPU

```cuda
// AoS (bad for GPU):
struct TLEGPU { float a, e, i, raan, argp, m; };  // Strided access

// SoA (good for GPU):
struct TLEBatchGPU {
    float* semiMajorAxes;      // Contiguous
    float* eccentricities;     // Contiguous
    float* inclinations;       // Contiguous
    // ...
};
```

**Coalesced Memory Access:**
- Warp threads read consecutive floats
- 32 threads × 4 bytes = 128 bytes per transaction
- Maximum memory bandwidth utilization

### Zero-Copy with Pinned Memory

```go
// CPU allocates pinned memory accessible by GPU
func AllocatePinned(size int) unsafe.Pointer {
    var ptr unsafe.Pointer
    cudaHostAlloc(&ptr, size, cudaHostAllocMapped)
    return ptr
}

// Go slice backed by GPU memory
tles := (*[100000]TLEGPU)(pinnedPtr)

// Kernel reads/writes directly - no copies
```

---

## Performance Projections

### Baseline (Non-Fused)

| Operation | Time | PCIe Transfers |
|-----------|------|----------------|
| SGP4 (GPU) | 2.5 ms | CPU→GPU: 6.4 MB |
| PCIe round-trip | 10 ms | - |
| LBVH (CPU) | 50 ms | GPU→CPU: 4.8 MB |
| PCIe round-trip | 10 ms | - |
| Proximity (GPU) | 3 ms | CPU→GPU: 4.8 MB |
| Results (CPU) | - | GPU→CPU: 0.1 MB |
| **Total** | **~75 ms** | **3 transfers** |

### Fused Pipeline

| Operation | Time | PCIe Transfers |
|-----------|------|----------------|
| Kernel 1 (SGP4+Morton) | 2.8 ms | CPU→GPU: 6.4 MB (input only) |
| Kernel 2 (Radix Sort) | 1.2 ms | - |
| Kernel 3 (Proximity) | 2.5 ms | - |
| Results (CPU) | - | GPU→CPU: 0.1 MB (pairs only) |
| **Total** | **~6.5 ms** | **1 transfer** |

**Speedup:** ~11.5× faster

---

## Implementation Architecture

### Go Orchestrator

```go
type GPUFusionPipeline struct {
    // Buffers (pinned memory)
    tleInput     unsafe.Pointer  // 100k × 64 bytes
    mortonCodes  unsafe.Pointer  // 100k × 8 bytes
    positions    unsafe.Pointer  // 100k × 12 bytes
    pairOutput   unsafe.Pointer  // 1M × 16 bytes (max pairs)
    
    // CUDA handles
    kernel1, kernel2, kernel3 cudaStream_t
}

func (p *GPUFusionPipeline) ProcessBatch(tles []TLEGPU) []CollisionPair {
    // 1. Copy TLEs to pinned buffer (CPU)
    copyPinned(p.tleInput, tles)
    
    // 2. Launch fused kernel (async)
    C.launch_fused_pipeline(
        p.tleInput,
        p.mortonCodes,
        p.positions,
        p.pairOutput,
        len(tles),
    )
    
    // 3. CPU does other work while GPU processes...
    
    // 4. Synchronize and read results
    C.cudaStreamSynchronize(p.kernel1)
    pairs := readPairs(p.pairOutput)
    
    return pairs
}
```

### Single-Pass CUDA Kernel

```cuda
__global__ void fused_sda_pipeline(
    const TLEGPU* __restrict__ tles,
    uint64_t* __restrict__ morton_codes,
    float3* __restrict__ positions,
    CollisionPair* __restrict__ pairs_out,
    int* __restrict__ pair_count,
    int N,
    double targetJD,
    float threshold_km
) {
    // Grid-stride loop for large N
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += gridDim.x * blockDim.x) {
        
        // Stage 1: SGP4 propagation
        StateGPU state = sgp4_propagate(tles[idx], targetJD);
        positions[idx] = make_float3(state.pos.x, state.pos.y, state.pos.z);
        
        // Stage 2: Morton code
        morton_codes[idx] = computeMortonCode(state.pos.x, state.pos.y, state.pos.z);
    }
    
    __syncthreads(); // Wait for all threads
    
    // Stage 3: Radix sort (using shared memory or device-wide)
    // Implemented separately or use CUB
    
    __syncthreads();
    
    // Stage 4: Proximity detection
    // Each thread checks neighbors
    proximity_check(morton_codes, positions, pairs_out, pair_count, N, threshold_km);
}
```

---

## Bitwise Neighbor Finding

### XOR-Based Proximity

**Property:** Morton codes of nearby objects differ by small amount

```
Object A: 00101010101010101010101010101010
Object B: 00101010101010101010101010101011 (1 bit different)
XOR:      00000000000000000000000000000001

Result: XOR has few bits set → objects nearby
```

**Implementation:**
```cuda
// Fast neighbor rejection
uint64_t xorCode = myCode ^ neighborCode;
if (__popcll(xorCode) < 6) {  // Fewer than 6 differing bits
    // Do precise distance check
}
```

---

## Scalability Analysis

### Theoretical Limits

| Catalog Size | Kernel 1 | Kernel 2 | Kernel 3 | Total | PCIe |
|-------------|----------|----------|----------|-------|------|
| 10k | 0.28 ms | 0.12 ms | 0.25 ms | 0.65 ms | 1× |
| 30k | 0.84 ms | 0.36 ms | 0.75 ms | 1.95 ms | 1× |
| 100k | 2.8 ms | 1.2 ms | 2.5 ms | 6.5 ms | 1× |
| 500k | 14 ms | 6 ms | 12.5 ms | 32.5 ms | 1× |

**Conclusion:** 500k objects in ~33ms (well under 60-second target)

---

## Integration with ADITI 4.0

### Current → Fused Transition

```go
// Before (separate passes)
func OldScreening(tles []TLEGPU) []CollisionPair {
    states := gpu.SGP4Batch(tles)           // PCIe: Host→Device, Device→Host
    pairs := cpu.LBVH(states)               // CPU processing
    critical := gpu.ProximityCheck(pairs)     // PCIe: Host→Device, Device→Host
    return critical
}

// After (fused)
func NewScreening(tles []TLEGPU) []CollisionPair {
    pairs := gpu.FusedPipeline(tles)        // PCIe: Host→Device (input only)
    return pairs                            // PCIe: Device→Host (results only)
}
```

---

## Risk Mitigation

### Potential Issues & Solutions

| Issue | Risk | Mitigation |
|-------|------|------------|
| Grid size limits | Morton code precision | Use hierarchical grids |
| Sorting overhead | 100k elements | CUB radix sort is O(N) |
| Memory capacity | 500k objects | Streaming/chunking |
| Numerical precision | SGP4 accuracy | Keep double precision |

### Fallback Strategy

If kernel fusion fails:
```go
func ScreeningWithFallback(tles []TLEGPU) []CollisionPair {
    pairs, err := gpu.TryFusedPipeline(tles)
    if err != nil {
        // Fallback to multi-pass
        pairs = OldScreening(tles)
    }
    return pairs
}
```

---

## Summary

**GPU Kernel Fusion Strategy:**

1. **Morton codes** computed inline during SGP4 propagation
2. **Radix sort** orders objects by spatial locality
3. **Bitwise XOR** rapidly identifies neighbor candidates
4. **Single PCIe transfer** (input only) vs 3 in baseline
5. **11.5× speedup** enables 500k object processing

**Implementation Path:**
- Week 1: Morton code kernel (test standalone)
- Week 2: Integrate with SGP4 kernel
- Week 3: Add radix sort and proximity detection
- Week 4: Performance validation and optimization

**Verdict:** This is the key optimization to reach "dominant" (not just marginal) performance for May 4.
