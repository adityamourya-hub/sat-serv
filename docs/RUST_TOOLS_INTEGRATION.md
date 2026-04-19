# Rust Scientific Computing Tools for ADITI 4.0

## Executive Summary

**Source:** contagon (CMU RPL) Rust ecosystem  
**Application:** Reference implementations for ADITI 4.0 Go-native porting  
**Alternative:** Direct integration via gRPC/FFI for rapid deployment

---

## Tool Mapping: Rust → Go

### 1. num-dual (Automatic Differentiation)

**Purpose:** Exact Jacobian computation for Orbit Determination  
**Rust Repo:** github.com/contagon/num-dual  
**Status:** Fork of itt-ustutt/num-dual

**ADITI 4.0 Need:**
- Batch LS Jacobian (AᵀWA matrix construction)
- CR3BP STM (State Transition Matrix) computation
- Covariance sensitivity analysis

**Go Implementation Strategy:**
```go
// Option A: Manual analytical derivatives (recommended)
// - Faster (no AD overhead)
// - Deterministic
// - Physics-grounded

// Example: CR3BP Jacobian (analytical)
func cr3bpJacobian(state CR3BPState) [6][6]float64 {
    var J [6][6]float64
    
    // ∂f/∂x computed analytically from equations of motion
    // More precise than numerical differentiation
    // ~10x faster than auto-diff
    
    return J
}
```

**Go Alternative:**
```go
// github.com/JuliaNLSolvers/ForwardDiff.jl concept in Go
// Or: github.com/gorgonia/gorgonia (graph-based AD)
```

**Recommendation:** Manual analytical Jacobians for production ADITI. Use num-dual as validation reference only.

---

### 2. tiny-solver-rs (Optimization)

**Purpose:** Levenberg-Marquardt for Batch Least Squares  
**Rust Repo:** github.com/contagon/tiny-solver-rs  
**Algorithm:** Trust-region Gauss-Newton with LM damping

**ADITI 4.0 Need:**
- OD convergence: (AᵀWA)δx = AᵀWb
- Parameter estimation (drag, SRP coefficients)
- Minimize residuals: Σ(O-C)²

**Current ADITI Implementation:**
```go
// pkg/od/batch_ls.go
// Gaussian elimination with partial pivoting
// Levenberg-Marquardt damping

func solveLinearSystem(A [6][6]float64, b [6]float64) [6]float64 {
    // Custom implementation
    // Less sophisticated than tiny-solver-rs
    // But: zero dependencies, fully indigenous
}
```

**Go Alternatives:**
```go
// gonum.org/v1/gonum/optimize
// - L-BFGS, CG, Newton
// - Not LM specifically

// github.com/c-bata/goptuna
// - Bayesian optimization
// - Overkill for OD
```

**Recommendation:** 
- **Short-term:** Keep current indigenous solver (validated, simple)
- **Medium-term:** Port LM algorithm from tiny-solver-rs to Go
- **Validation:** Compare convergence against tiny-solver-rs

---

### 3. sophus-rs (Lie Groups)

**Purpose:** SE(3), SO(3) for attitude/orbit dynamics  
**Rust Repo:** github.com/contagon/sophus-rs  
**Concept:** Lie algebra for rotation/orientation (no gimbal lock)

**ADITI 4.0 Need:**
- Spacecraft attitude quaternions
- SE(3) for orbit+attitude combined state
- Exponential map for rotation interpolation

**Current ADITI Implementation:**
```go
// pkg/physics/state.go
// - Simple 3D vectors
// - No formal Lie group structure
// - Quaternions: [4]float64 (basic ops)

// Limitation: No SE(3) composition, no exp/log maps
```

**Go Alternatives:**
```go
// github.com/ungerik/go3d
// - Vectors, matrices, quaternions
// - No Lie group formalism

// Custom implementation needed for sophus-level capabilities
```

**Recommendation:**
- **Critical for:** Formation flying, relative motion
- **Port priority:** Medium (not blocking May 4)
- **Reference:** Use sophus-rs algorithms for validation

**Key Algorithms to Port:**
```rust
// From sophus-rs:
// SO3::exp(ω)  // Rotation vector to quaternion
// SE3::exp(ξ)  // Twist to transformation matrix
// Adjoint action for covariance propagation
```

---

### 4. faer-ext (Linear Algebra)

**Purpose:** High-performance matrix operations  
**Rust Repo:** github.com/contagon/faer-ext  
**Features:** Sparse matrices, Cholesky, QR, SVD

**ADITI 4.0 Need:**
- Covariance matrix Cholesky decomposition
- Sparse STM for multiple shooting
- Large-scale OD (1000+ observations)

**Current ADITI Implementation:**
```go
// pkg/od/batch_ls.go
// Dense 6x6 matrices only
// Gaussian elimination (O(n³))
// No sparse support

// pkg/orbital/cr3bp_advanced.go
// gonum.org/v1/gonum/mat for eigenvalues
```

**Go Alternatives:**
```go
// gonum.org/v1/gonum/mat
// - Dense matrices
// - Cholesky, QR, SVD, Eigen
// - Good performance, pure Go

// github.com/james-bowman/sparse
// - CSR/CSC sparse matrices
// - Sparse Cholesky
```

**Performance Comparison:**

| Operation | faer-rs | gonum | Notes |
|-----------|---------|-------|-------|
| 6x6 Cholesky | ~1μs | ~5μs | faer SIMD-optimized |
| 1000x1000 SVD | ~10ms | ~50ms | gonum pure Go |
| Sparse solve | Optimized | Limited | faer has sparse QR |

**Recommendation:**
- **Current:** gonum/mat sufficient for 6x6 OD covariance
- **Future:** faer-ext patterns for large-scale problems
- **GPU:** Metal/cuda for 100k×100k BVH (not dense algebra)

---

## Integration Strategies

### Option A: Reference Implementation (Recommended)

**Approach:** Study Rust code, port algorithms to Go

**Pros:**
- Full indigenous control
- Physics-grounded (understand every line)
- No FFI complexity
- IAF auditability

**Cons:**
- Development time
- Validation burden

**Timeline:** Post-May 4 for non-critical paths

---

### Option B: gRPC Service Wrapper

**Approach:** Wrap Rust tools as microservices

```rust
// Rust service (tiny-solver)
#[tokio::main]
async fn main() {
    let server = SolverServer::new();
    server.serve(":50051").await;
}
```

```go
// Go client
client := solver.NewClient("localhost:50051")
result, _ := client.SolveOD(observations)
```

**Pros:**
- Use production-ready Rust tools
- Language-agnostic
- Easy to swap implementations

**Cons:**
- Network latency (~1-5ms)
- Extra service to maintain
- Not truly indigenous

**Verdict:** Too complex for May 4 deadline

---

### Option C: FFI/cgo (Advanced)

**Approach:** Link Rust static libraries via cgo

```go
// #cgo LDFLAGS: -ltiny_solver
// #include "solver.h"
import "C"

func rustSolver() {
    C.tiny_solve(...)
}
```

**Pros:**
- Zero-copy memory
- Native performance
- Single binary

**Cons:**
- Build complexity (Rust + Go toolchain)
- Unsafe code blocks
- Debugging nightmare

**Verdict:** Overkill for ADITI 4.0 scope

---

## Strategic Recommendation

### Immediate (T-15 days to May 4)

**Keep indigenous Go implementations:**
- `pkg/od/batch_ls.go` - Current solver is sufficient
- `gonum/mat` - Covers 6×6 covariance needs
- Manual Jacobians - Faster than AD for our use case

**Validation against Rust tools:**
```bash
# 1. Run identical OD problem in both
# 2. Compare solutions (should match within 1e-6)
# 3. Document any discrepancies
```

### Post-May 4 Roadmap

**Phase 1: Algorithm Validation**
- Port 3-5 key algorithms from Rust to Go
- Validate against original Rust implementations
- Performance benchmark

**Phase 2: Performance Optimization**
- SIMD optimizations (Go assembly or C intrinsics)
- Sparse matrix support for large OD batches
- GPU acceleration for specific kernels

**Phase 3: Feature Expansion**
- Sophus-level SE(3) support for formation flying
- Advanced optimization (interior point, global)
- Extended precision (quad-precision covariance)

---

## Priority Matrix

| Tool | ADITI Need | Current Status | Priority | Action |
|------|-----------|----------------|----------|--------|
| **num-dual** | OD Jacobian | Manual derivs | Low | Validate only |
| **tiny-solver** | Batch LS | Indigenous | Medium | Port post-May4 |
| **sophus-rs** | Attitude | Basic quat | Medium | Port for formation flying |
| **faer-ext** | Covariance | gonum/mat | Low | Use gonum |

---

## Code Samples: Rust → Go

### Example 1: Levenberg-Marquardt (tiny-solver pattern)

**Rust:**
```rust
// From tiny-solver-rs
let solver = LevenbergMarquardt::new()
    .with_tol(1e-6)
    .with_max_iter(50);

let result = solver.minimize(problem, initial_guess);
```

**Go (ADITI 4.0):**
```go
// pkg/od/batch_ls.go
estimator := NewBatchLSEstimator()
estimator.MaxIterations = 50
estimator.ConvergenceThreshold = 1e-6
estimator.DampingFactor = 1e-3  // LM damping

result, _ := estimator.Ensure(batch, initialState, jd, params)
```

### Example 2: SE(3) Composition (sophus pattern)

**Rust:**
```rust
// From sophus-rs
let T1 = SE3::from_xyz_quat(xyz1, quat1);
let T2 = SE3::from_xyz_quat(xyz2, quat2);
let T_composed = T1 * T2;  // Group composition
```

**Go (future):**
```go
// Potential ADITI addition
type SE3 struct {
    Rotation    *Quaternion
    Translation [3]float64
}

func (s *SE3) Compose(other *SE3) *SE3 {
    // T_composed = T1 ∘ T2
    return &SE3{
        Rotation: s.Rotation.Multiply(other.Rotation),
        Translation: add(s.Translation, 
            s.Rotation.Rotate(other.Translation)),
    }
}
```

### Example 3: Sparse Cholesky (faer pattern)

**Rust:**
```rust
// From faer
let L = sparse_cholesky(A);  // A = L*L'
let x = L.solve(b);
```

**Go (current):**
```go
// Dense fallback
func invertSymmetricMatrix(A [6][6]float64) [6][6]float64 {
    // Gaussian elimination on dense matrix
    // O(n³) but n=6 is tiny
}
```

---

## Conclusion

**The Verdict:**

Rust tools (num-dual, tiny-solver, sophus, faer) are excellent **reference implementations** but not critical path for May 4 deadline.

**ADITI 4.0's indigenous Go implementations are sufficient for:**
- 100k-object catalog (SGP4/RKF45)
- 6×6 Batch LS (Gaussian elimination)
- BVH screening (custom, optimized)
- Covariance realism (gonum/mat)

**Post-May 4:** Port selected algorithms from Rust to Go for:
- Formation flying (sophus-style SE(3))
- Large-scale OD (sparse solvers)
- Advanced optimization (global solvers)

**Key Insight:**
> "Don't optimize prematurely. Indigenous Go code that's validated against Rust gold standards is more valuable than complex FFI integrations for the May 4 deadline."

**Reference, don't integrate. Port, don't wrap. Validate, don't depend.** 🦀➡️🐹
