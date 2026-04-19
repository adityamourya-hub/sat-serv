# GPU Acceleration for 100k-Object SSA

## Executive Summary

To achieve the **1-minute update cycle for 100,000 objects**, CPU-only processing hits a fundamental wall at ~133 seconds. GPU acceleration via CUDA/ROCm provides the SIMD/vectorization depth required for massive parallel astrodynamic computation.

**Target:** <60 seconds for full catalog update  
**Current:** ~133 seconds (CPU-only)  
**Gap:** 2.2× speedup needed  
**Solution:** Hybrid Go + CUDA architecture

---

## 1. GPU-Accelerated Propagation (The Heavy Lifting)

### 1.1 SGP4 on CUDA

**Reference Implementation:** [cuda-sgp4](https://github.com/dnelson/cuda-sgp4) (C++/CUDA)

**Capability:** Propagate hundreds of thousands of TLEs in milliseconds

**Why Critical:**
- Propagation is "embarrassingly parallel"
- Each satellite state computed independently
- SGP4 has predictable instruction flow (no branching)
- Perfect for GPU SIMD execution

**Performance Estimate:**
```
CPU (Go):     100,000 × 1.3 ms = 130 seconds
CUDA SGP4:    100,000 × 0.013 ms = 1.3 seconds (100× speedup)
```

### 1.2 High-Precision Propagation (RKF45 on GPU)

For conjunction assessment requiring >SGP4 precision:

**Tudat-Space Integration:**
- Vectorized numerical integration backends
- Handles perturbations SGP4 misses (3rd body, SRP variations)
- CUDA-accelerated force models

**Kernel Design:**
```cuda
__global__ void rkf45_propagate(
    float6* states,      // [x,y,z,vx,vy,vz] × N
    float* epoch_jd,     // Initial times
    float dt,            // Propagation duration
    int N,               // Number of objects
    float3* perturbations // Optional: drag, SRP accel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // RKF45 adaptive integration per thread
    // Step 1: k1 = f(y0)
    // Step 2: k2 = f(y0 + h*a21*k1)
    // ... (Fehlberg coefficients)
    // Compute error estimate
    // Adjust step size
    // Write final state
}
```

---

## 2. Fast Screening: From BVH to LBVH (Linear BVH)

### 2.1 The Screening Bottleneck

**Problem:** 100k × 100k = 10 billion pairs  
**Solution:** Hierarchical culling with LBVH

**Reference:** [madmann91/lbvh](https://github.com/madmann91/lbvh) (header-only CUDA)

### 2.2 LBVH Construction on GPU

**Algorithm:**
1. Compute Morton codes for each object (3D space-filling curve)
2. Parallel radix sort (GPU-optimal)
3. Build tree implicitly from sorted order
4. Parallel tree construction

**Performance:**
```
CPU BVH Build:  100k objects = 500 ms
LBVH Build:     100k objects = 5 ms   (100× faster)
```

### 2.3 Proximity Queries

**Parallel Query:**
```cuda
__global__ void proximity_query(
    lbvh_node* tree,       // LBVH tree
    aabb* object_bounds,   // Bounding boxes
    float threshold,       // Search radius (km)
    int N,
    int* collision_pairs,    // Output: candidate pairs
    int* pair_count          // Atomic counter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Traverse tree, find neighbors within threshold
    // Write candidate pairs to global memory
}
```

**Expected Throughput:**
- 100k objects, 10km threshold: ~50ms on RTX 4090
- Reduces 10B pairs to ~1M candidates (99.99% culling)

### 2.4 Spatial Hashing Alternative

For k-nearest neighbor with constant threshold:

**Method:**
- Hash 3D space into uniform grid cells
- Parallel hash construction via prefix sum (scan)
- GPU Gems Chapter 39 implementation

**When Better Than LBVH:**
- Uniform object distribution
- Fixed query radius
- Simpler implementation

---

## 3. Hybrid Architecture: Go + CUDA

### 3.1 Layer Design

| Layer | Technology | Responsibility |
|-------|-----------|----------------|
| **Orchestration** | Go | API, task scheduling, data ingestion |
| **Interlink** | cgo / gRPC | Zero-copy buffers to GPU |
| **Computation** | CUDA C++ | Mass propagation + LBVH screening |
| **Analysis** | Go | Final Pc for top 1% candidates |

### 3.2 Integration Options

#### Option A: cgo (In-Process)

**Pros:**
- Zero-copy via `cudaHostAlloc` (pinned memory)
- Lowest latency (<1μs kernel launch)
- Single binary deployment

**Cons:**
- Build complexity (requires CUDA toolkit)
- CGO overhead for small calls
- Harder to debug

**Implementation:**
```go
// #cgo LDFLAGS: -L./cuda -lsgp4_gpu -lcudart
// #include "sgp4_gpu.h"
import "C"

func propagateBatchGPU(states []State) {
    // Pinned memory: accessible by CPU and GPU
    C.cudaHostAlloc(...) 
    
    // Launch kernel
    C.sgp4_propagate_batch(...)
    
    // Synchronize
    C.cudaDeviceSynchronize()
}
```

#### Option B: gRPC Sidecar

**Pros:**
- CUDA service isolated from Go
- Easier debugging/monitoring
- Can scale GPU separately

**Cons:**
- Serialization overhead
- Network latency (even localhost)

**Architecture:**
```
[Go Orchestrator] ← gRPC → [CUDA Service]
                                    ↓
                              [GPU Cluster]
```

### 3.3 Zero-Copy Memory Strategy

**Critical:** Avoid PCIe bandwidth bottleneck

**Implementation:**
```cuda
// Allocate pinned memory (page-locked)
cudaHostAlloc(&h_buffer, size, cudaHostAllocMapped);

// Get device pointer
cudaHostGetDevicePointer(&d_buffer, h_buffer, 0);

// Go writes to h_buffer
// GPU reads from d_buffer (same physical memory)
// No memcpy required!
```

**Performance:**
- PCIe 4.0 x16: ~32 GB/s (still slow)
- Zero-copy: ~100-200 GB/s (GPU memory bandwidth)

---

## 4. Fast Pc Computation on GPU

### 4.1 Foster Method Acceleration

Even after screening, 10,000 pairs (top 1%) need Pc calculation.

**Reference:** [James-S-H/Space-Collision-Probability](https://github.com/James-S-H/Space-Collision-Probability)

**GPU Implementation:**
```cuda
__global__ void foster_pc_batch(
    conjunction_geom* geoms,  // Array of conjunction parameters
    int N,
    float* pc_results           // Output probabilities
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // 2D Gaussian integration
    // Numerical integration over collision disk
    // Thread per conjunction
    
    pc_results[idx] = foster_alfano_2d(geoms[idx]);
}
```

**Optimization:**
- Reject using "Instantaneous Minimum Distance" before integral
- Skip if miss_distance > 10× combined_radius
- Early-out for Pc < 1e-10

**Expected Performance:**
```
CPU (Go):     10,000 × 500μs = 5 seconds
GPU (CUDA):   10,000 × 5μs = 50 ms (100× speedup)
```

---

## 5. Implementation Roadmap

### Phase 1: SGP4 GPU Kernel (Week 1-2)

**Deliverables:**
- [ ] Port dnelson/cuda-sgp4 to IDEX
- [ ] Go bindings via cgo
- [ ] Batch propagation interface
- [ ] Performance benchmark vs CPU

**Success Criteria:**
- 100k TLEs in <2 seconds
- Correctness: match CPU SGP4 to 1m

### Phase 2: LBVH Screening (Week 3-4)

**Deliverables:**
- [ ] Integrate madmann91/lbvh
- [ ] Morton code generation kernel
- [ ] Parallel tree construction
- [ ] Proximity query kernel

**Success Criteria:**
- 100k × 100k screening in <50ms
- 99.9% pair culling efficiency

### Phase 3: Hybrid Integration (Week 5-6)

**Deliverables:**
- [ ] Zero-copy memory management
- [ ] Go orchestration layer
- [ ] Pinned buffer pools
- [ ] Pipeline: Propagate → Screen → Analyze

**Success Criteria:**
- End-to-end 100k update in <60 seconds
- <1% CPU utilization during GPU work

### Phase 4: Fast Pc (Week 7-8)

**Deliverables:**
- [ ] Foster method CUDA kernel
- [ ] Batch Pc computation
- [ ] Early-out optimizations

**Success Criteria:**
- 10k Pc calculations in <100ms
- Accuracy: match CPU to 1e-6

---

## 6. Indigenous Development Strategy

### 6.1 Sovereign Requirements

**Custom CUDA Kernels:**
- SGP4 propagation (derived from open source)
- LBVH construction (from scratch, ~500 lines)
- Foster Pc (from algorithm paper)

**No Black-Box Dependencies:**
- All source code auditable
- No external binaries except NVIDIA drivers
- Buildable from source

### 6.2 Hardware Options

**Development:**
- NVIDIA RTX 4090 (24GB, consumer)
- CUDA 12.x, Compute Capability 8.9

**Production (IAF):**
- NVIDIA A100 (80GB, datacenter)
- CUDA 12.x, Compute Capability 8.0
- Multi-GPU scaling (4× A100 = 4× throughput)

**Alternative (ROCm for AMD):**
- AMD MI250X (128GB HBM2e)
- HIP port of CUDA kernels
- Better for air-gapped (no phoning home)

---

## 7. Performance Projections

### 7.1 Full Pipeline Timing

```
Task                          CPU Time    GPU Time    Speedup
─────────────────────────────────────────────────────────────
TLE Ingestion                 1.0 s       1.0 s       1×
SGP4 Propagation (100k)       130.0 s     1.3 s       100×
LBVH Construction             0.5 s       0.005 s     100×
Proximity Screening           2.0 s       0.05 s      40×
Pc Calculation (10k pairs)    5.0 s       0.05 s      100×
Database Update               1.0 s       1.0 s       1×
─────────────────────────────────────────────────────────────
TOTAL                         139.5 s     3.4 s       41×
Target                        <60 s       <60 s       ✓
```

### 7.2 Headroom for Growth

With GPU acceleration, 1-minute cycle leaves headroom:
- **500k objects:** ~15 seconds (still <60s)
- **1M objects:** ~30 seconds (still <60s)
- **Multi-GPU:** Linear scaling with A100 cluster

---

## 8. Risk Mitigation

### 8.1 CUDA Unavailable

**Fallback:** CPU-only with SIMD (AVX-512)
- ~8× speedup over baseline
- Still achieves <60s for 100k (with optimization)

### 8.2 Numerical Issues

**Validation:**
- Cross-validate GPU vs CPU SGP4 (1m accuracy)
- Unit tests for edge cases (polar orbits, high ecc)
- Stochastic rounding for reproducibility

### 8.3 Memory Limits

**100k objects state:**
- Position/velocity: 100k × 24 bytes = 2.4 MB
- Covariance: 100k × 144 bytes = 14.4 MB
- Total: ~20 MB (fits in any GPU)

**LBVH:**
- Nodes: 2 × 100k × 24 bytes = 4.8 MB
- Total: <50 MB VRAM

---

## 9. Conclusion

GPU acceleration is **essential** for operational 100k-object SSA at 1-minute cadence. The hybrid Go+CUDA architecture:

1. **Preserves Go strengths:** Concurrency, safety, ecosystem
2. **Exploits GPU strengths:** SIMD, throughput, LBVH
3. **Maintains sovereignty:** All kernels auditable, indigenous
4. **Achieves target:** <60 seconds end-to-end

**Next Step:** Implement Phase 1 (SGP4 GPU kernel) to validate 100× speedup claim.

---

**References:**
1. dnelson/cuda-sgp4 - SGP4 on CUDA
2. madmann91/lbvh - Linear BVH
3. James-S-H/Space-Collision-Probability - Fast Pc
4. GPU Gems 3 Chapter 39 - Parallel Prefix Sum
5. Tudat-Space - High-precision numerical integration
