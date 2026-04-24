# Git-Hunt Julia Packages: Design Reference Analysis for IDEX

## Executive Summary

The `git-hunt/` Julia packages (`GeometricMachineLearning.jl`, `GeometricOptimizers.jl`, `SymbolicNeuralNetworks.jl`, etc.) are **valuable design references** but should **not be integrated** as runtime dependencies. IDEX's Go + Python stack already covers 90% of the same mathematical ground natively. The remaining 10% represents actionable design debt.

---

## 1. GeometricMachineLearning.jl

### 1.1 SympNet Architecture (Jin 2020)

**What Julia provides:**
- `LASympNet`: Linear-activation symplectic nets with alternating `LinearLayerQ`/`LinearLayerP` + bias + activation layers
- `GSympNet`: Gradient-based symplectic nets with `GradientLayerQ`/`GradientLayerP` and upscaling dimensions
- `SymplecticTransformer`: Attention-augmented symplectic architecture combining transformer blocks with symplectic layers
- `SymplecticAutoencoder`: Nonlinear dimensionality reduction preserving symplectic structure (Brantner 2023)

**What IDEX has:**
- `ai/v3/hamiltonian_dynamics.py:64-83`: A basic MLP HNN with `nn.Linear` + `nn.Softplus` → scalar Hamiltonian
- No symplectic weight constraints — the network is a generic function approximator, not a symplectic map

**Design Debt:**
- **CRITICAL**: The current HNN is a Greydanus-style " vanilla" HNN (Greydanus 2019). It learns `H(q,p)` but the network itself does not guarantee symplecticity of the learned flow. A maneuver could push the model into a region where the Jacobian no longer satisfies `J^T Ω J = Ω`, breaking long-term orbit stability.
- **ACTION**: Implement a PyTorch `SympNet` module following the LASympNet pattern:
  ```python
  class LinearLayerQ(nn.Module): ...  # Updates only q: q' = q + W_q p + b_q
  class LinearLayerP(nn.Module): ...  # Updates only p: p' = p + W_p q + b_p
  class LASympNet(nn.Module):
      # Alternates: LinearLayerQ → LinearLayerP → BiasLayer → Activation
      # Ensures each layer is a symplectic map, thus the composition is symplectic
  ```
- **MEDIUM**: The `SymplecticAutoencoder` (Brantner 2023) is highly relevant for HGV model reduction. The full 6-DOF state plus latent physics code `z` is high-dimensional. A symplectic autoencoder could compress to a 4D reduced manifold (2 position + 2 momentum) for real-time inference on M2 Air. **ACTION**: Prototype a `SymplecticAutoencoder` in `ai/v3/` for PAX Tier-2 fallback.

### 1.2 Hamiltonian Neural Network Variants

**What Julia provides:**
- `StandardHamiltonianArchitecture`: Greydanus HNN (implemented)
- `GeneralizedHamiltonianArchitecture`: GHNN (Horn 2025) — **explicitly unimplemented** in Julia (`error("GHNN still has to be implemented!")`)

**What IDEX has:**
- `ai/v3/hamiltonian_dynamics.py`: Standard HNN with Softplus activations
- `ai/v3/port_hamiltonian.py`: DPHNO with dissipative port `R(z)` and structure matrix `J(z)`

**Design Debt:**
- **LOW**: Julia's GHNN stub confirms this is bleeding-edge research. IDEX's DPHNO is actually *ahead* of the Julia package here — DPHNO includes dissipation, which GHNN does not. No action needed; IDEX should publish its DPHNO formulation.
- **MEDIUM**: Julia's `StandardHamiltonianArchitecture` uses `TanhActivation()` while IDEX uses `Softplus`. The Julia choice is better for orbital dynamics because:
  - `tanh` is bounded → prevents energy explosion during maneuvers
  - `softplus` is unbounded → can produce arbitrarily large Hamiltonians
  **ACTION**: A/B test `tanh` vs `softplus` in `hamiltonian_dynamics.py` with the CR3BP energy conservation test from `pkg/orbital/cr3bp_symplectic_test.go`.

### 1.3 Volume-Preserving Feedforward (Non-Hamiltonian)

**What Julia provides:**
- `VolumePreservingFeedforward`: For systems that are volume-preserving but not Hamiltonian (e.g., Lorenz, Navier-Stokes in divergence-free form)

**What IDEX has:**
- Nothing. All current models assume Hamiltonian structure.

**Design Debt:**
- **LOW**: HGV wake dynamics in the transitional flow regime (30-80km) are *not* purely Hamiltonian — they involve viscous dissipation, shock heating, and plasma chemistry. The DPHNO `R(z)` port captures dissipation but assumes a symplectic structure matrix `J(z)`.
- **ACTION**: For the FractionalAtmosphereAdjust spectral multiplier, consider whether the FNO kernel should be volume-preserving but non-Hamiltonian. If so, port `VolumePreservingFeedforward` to PyTorch as a reference.

---

## 2. GeometricOptimizers.jl

### 2.1 Manifold Adam on Stiefel/Grassmann

**What Julia provides:**
- `Adam` optimizer with retraction on Stiefel manifold: `Y^T Y = I`
- `𝔄(A)` modified exponential map: `∑ (1/n!) A^{n-1}` computed to machine precision via iterative Taylor expansion
- Riemannian gradient: `rgrad(Y, ∇L) = ∇L - Y (∇L^T Y)`
- Metric: `g_Y(Δ₁, Δ₂) = Tr(Δ₁^T (I - ½YY^T) Δ₂)`

**What IDEX has:**
- `ai/v3/manifold_projection.py:74-137`: Newton projection onto `H(q,p) = H_target` energy surface
- No Riemannian optimization — standard PyTorch `Adam` on Euclidean parameter space
- No orthonormality constraints on weights

**Design Debt:**
- **CRITICAL**: The `SymplecticRegularizer` in `manifold_projection.py:174-179` mentions "enforcing unitary constraints on e3nn layer weights" but the implementation is a TODO. The Julia `StiefelManifold` + `Adam` retraction provides a complete, tested implementation of exactly this.
- **ACTION**: Port the Julia Stiefel Adam to PyTorch:
  1. Replace `SymplecticRegularizer` stub with a `RiemannianAdam` optimizer
  2. For each e3nn layer weight matrix `W`, treat it as a point on Stiefel manifold
  3. Compute Euclidean gradient `∇L` via autograd
  4. Project to Riemannian gradient: `∇L - W (∇L^T W)`
  5. Update with Adam moments (on the tangent space)
  6. Retract back to manifold via QR decomposition: `W_new = qr(W + step)`
  7. The Julia `𝔄(A)` series gives a more accurate retraction than QR — port it for high-precision training on H100

- **MEDIUM**: The Julia `global_section(Y)` via QR decomposition (`qr!(A).Q`) is a clean implementation for computing the orthogonal complement `Y_⊥`. This is needed for the ASR filter's subspace alignment in `pkg/telescope/asr_filter.go`.

### 2.2 Retraction Methods

**What Julia provides:**
- Modified exponential `𝔄(A) = ∑_{n=1}^∞ (1/n!) A^{n-1}` — converges in ~10 iterations to machine eps
- `𝔄exp(X, Y) = I + X 𝔄(X, Y) Y^T` — used for Cayley/transformer retraction

**What IDEX has:**
- `manifold_projection.py`: Simple Newton projection, no retraction caching

**Design Debt:**
- **LOW**: The Newton projection in `manifold_projection.py` is `O(k n^3)` where `k` is iterations. The Julia retraction is `O(n^3)` with no iteration. For the 6D state space, this is negligible. But for the hypernetwork atlas weight space (potentially 10^4 dim), the Julia approach would be faster.
- **ACTION**: Benchmark both methods in `benchmarks/integrator_comparison.py` with a 1000-dim parameter vector.

---

## 3. SymbolicNeuralNetworks.jl

### 3.1 Symbolic Jacobian → Poisson Tensor

**What Julia provides:**
- `symbolic_hamiltonian_vector_field(nn::SymbolicNeuralNetwork)`:
  1. Compute symbolic Jacobian `□ = Jacobian(nn)`
  2. Construct Poisson tensor `𝕁 = [0, I; -I, 0]`
  3. Return `𝕁 * derivative(□)'` as symbolic expression
  4. Compile via `RuntimeGeneratedFunctions.jl` for fast execution

**What IDEX has:**
- `pkg/physics/sindy.go`: SINDy discovers `ẋ = Θ(x) Ξ` via sparse regression
- `ai/v3/manifold_projection.py`: JAX `grad(H_fn)` for autograd Hamiltonian vector field
- No symbolic vector field extraction from neural networks

**Design Debt:**
- **HIGH**: The SINDy layer in `pkg/physics/sindy.go` produces an `EquationString` with confidence. But there is **no bridge** between the DPHNO neural network and the SINDy symbolic certificate. The Julia pipeline shows exactly how to do this:
  1. Symbolize the DPHNO Hamiltonian `H_θ(q,p)` using `sympy` or `torch.fx`
  2. Compute symbolic `∇H`
  3. Apply Poisson tensor to get `ẋ = 𝕁 ∇H`
  4. Compare with SINDy-discovered equations for consistency grading
- **ACTION**: Implement `SymbolicDPHNO` in Python using `torch.fx.symbolic_trace` or `sympy` to extract the learned Hamiltonian as a symbolic expression. This is the missing link for the VSC (Verified Symbolic Certificate) mentioned in the architecture docs.

### 3.2 Second-Order Derivatives

**What Julia provides:**
- Direct computation of `∂²H/∂q∂p` via `SymbolicNeuralNetworks.Jacobian` composed with itself
- Avoids Zygote's AD limitations for Hessians

**What IDEX has:**
- JAX `grad(grad(H))` for Hessians, but this is not used in the current codebase

**Design Debt:**
- **MEDIUM**: For the covariance propagation in `physics.State.Covariance [36]float64`, the state transition matrix `Φ = ∂x/∂x₀` requires Jacobian of the flow. Currently computed numerically in `pkg/orbital/propagate.go`. A symbolic `Φ` from the DPHNO would be exact and faster.
- **ACTION**: Add `jax.jacfwd` or `torch.autograd.functional.jacobian` to compute `Φ` analytically from the DPHNO flow map, using the Julia approach as a reference for the Poisson-tensor structure.

---

## 4. RungeKutta.jl

### 4.1 Yoshida-4/8 Symplectic Integrators

**What Julia provides:**
- `Yoshida4`, `Yoshida8`: Higher-order symplectic integrators with optimized coefficients
- `ForestRuth4`: 4th-order symplectic integrator
- `RungeKutta.Tableau`: General tableau-based RK methods

**What IDEX has:**
- `pkg/physics/symplectic_engine.go:33-49`: Forest-Ruth coefficients (`γ = 1/(2-∛2)`)
- `pkg/physics/integrator.go`: Generic RK4, RKF45
- `pkg/orbital/lgvi_integrator.go`: Lie group variational integrator

**Design Debt:**
- **LOW**: IDEX already has a correct Yoshida-4 implementation. The Julia package offers higher-order methods (Yoshida-8) and optimized tableaus that could reduce error for long HGV propagation (skip-glide trajectories can be 30+ minutes).
- **ACTION**: Port Yoshida-8 coefficients to `symplectic_engine.go` as a `ForestRuth8Step` method for high-fidelity offline training data generation. Not needed for the <100ms real-time loop.

---

## 5. Priority-Ordered Action Items

| Priority | Item | Location | Effort | Impact |
|---|---|---|---|---|
| **P0** | Port Stiefel Adam retraction to PyTorch | `ai/v3/riemannian_adam.py` | 2 days | Enables e3nn weight orthonormality, critical for symplectic regularizer |
| **P0** | Implement symbolic DPHNO → SINDy bridge | `ai/v3/symbolic_dphno.py` | 3 days | Unblocks VSC certificate, sovereign handshake integrity |
| **P1** | Implement PyTorch LASympNet | `ai/v3/sympnet.py` | 2 days | Guarantees symplecticity of learned flow, replaces vanilla HNN |
| **P1** | A/B test tanh vs softplus in HNN | `benchmarks/hnn_ablation.py` | 1 day | Prevents energy explosion during high-dV maneuvers |
| **P2** | Prototype symplectic autoencoder | `ai/v3/symplectic_autoencoder.py` | 3 days | Enables Tier-2 model reduction for M2 Air thermal governor |
| **P2** | Port Yoshida-8 for offline training | `pkg/physics/symplectic_engine.go` | 1 day | Reduces long-trajectory integration error |
| **P3** | Volume-preserving feedforward for wake | `ai/v3/vp_ffn.py` | 3 days | Only if transitional flow regime requires non-Hamiltonian structure |

---

## 6. Integration Boundary

**What to NOT port:**
- Julia runtime / Package manager — IDEX is Go + Python
- `GeometricMachineLearning.DataLoader` — PyTorch DataLoader is sufficient
- `AbstractNeuralNetworks` type hierarchy — Python dataclasses + protocols are sufficient
- `GeometricOptimizers.BFGS` — JAX `jaxopt` already provides BFGS/L-BFGS

**What to keep as references only:**
- Mathematical formulations and coefficient tables (Yoshida-8, Forest-Ruth coefficients)
- Algorithm pseudocode for retraction methods
- Symplectic autoencoder architecture diagrams

---

*Generated: 2026-04-23*
*Scope: Design reference analysis for `git-hunt` Julia packages vs. IDEX native stack*
*Next review: After P0 items complete*
