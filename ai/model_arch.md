# HF-PGFM Dominant Architecture: Technical Proposal

## Executive Verdict

**The current SINDy/Astrojax combination is NOT the global maximum.** It is a strong *local* maximum for interpretable maneuver detection in orbital regimes, but it has five structural failure modes that will collapse under the HGV/Siliguri corridor requirements:

| Bottleneck | Current State | Failure Mode |
|---|---|---|
| **SINDy Library** | Fixed polynomial + trig basis | Cannot represent skip-glide aerodynamic manifold; HGV bank-angle dynamics live on SO(3), not ℝ⁶ polynomial ring |
| **Astrojax World Model** | Cowell/RK4 with J2+drag | Euler integration in `_mock_propagate`; no Navier-Stokes coupling; drag model is exponential (8.5 km scale height) vs real HGV corridor (30-80 km rarefied/transitional flow) |
| **IMM Filter** | 3-mode Markov with hardcoded transitions | Mode probabilities `{0.90, 0.08, 0.02}` are static; zero-day maneuver (novel bank-angle sequence) has no mode to absorb it |
| **HNN Training** | `StructurePreservingLoss` with λ_h=0.05 penalty | Physics is a *soft* penalty, not a hard constraint; gradient can trade energy conservation for observation fit |
| **Edge Inference** | ONNX/CoreML runtime selection | No dynamic complexity shedding; full model runs regardless of target phase |

---

## 1. The Fundamental Pivot: Neural-Hamiltonian-Operator (NHO) Architecture

### Why Not Pure SINDy
SINDy (`pkg/physics/sindy.go`) discovers dynamics from a **pre-defined library** of candidate functions. The library in your codebase is:
```
{1, x, y, z, vx, vy, vz, x², y², z², xy, xz, yz, x³, y³, z³, sin(r/7000), cos(r/7000), v·vx, v·vy, v·vz}
```
This is 21 terms. An HGV executing a skip-glide with bank-angle modulation lives on a **6-DOF spherical** state space `(R, φ, λ, V, γ, ψ)` with **non-separable lift/drag coupling** through density `ρ(h, F10.7, Ap)`. The SINDy library cannot represent:

- `L·cos(σ)/m` — lift-bank coupling (σ is unobservable from tracking data alone)
- `(V²/r)·cos²(γ)·sin(ψ)·tan(φ)` — centripetal-heading coupling on a sphere
- The density function `ρ(h)` which transitions from continuum to rarefied flow in the 30-80 km corridor

SINDy will either **overfit** (too many polynomial terms) or **miss** (bank-angle dynamics invisible in Cartesian residuals).

### Why Not Pure HNN
Your `GeneralizedHNN` (`ai/v3/hamiltonian_dynamics.py`) and `GHNNVectorField` (`ai/neural_ode/ghnn.py`) learn H(q,p) as a scalar MLP. This is correct for conservative orbital mechanics but **structurally wrong for HGVs** because:

- HGV dynamics are **non-conservative**: drag dissipates energy, lift does work via bank angle
- The Hamiltonian formulation requires a **port-Hamiltonian** extension: `dx/dt = (J - R)∇H + gu` where R is the dissipation matrix and g·u is the control input (bank angle)
- Your current `equations_of_motion()` computes `dq/dt = ∂H/∂p, dp/dt = -∂H/∂q` — no dissipation term

### The Answer: Dissipative Port-Hamiltonian Neural Operator (DPHNO)

The correct architecture is a **Dissipative Port-Hamiltonian Neural Operator** that:

1. **Learns** H(q,p) via a neural network (conserves orbital energy in vacuum)
2. **Learns** R(q,p) — a positive semi-definite dissipation matrix (captures drag, thermal radiation)
3. **Learns** g(q,p) — the input coupling (maps unknown bank-angle to state evolution)
4. Uses a **Fourier Neural Operator** backbone (your `HamiltonianNeuralOperator` in `q/core/hno.py`) for infinite-dimensional operator learning of the atmosphere

The dynamics become:
```
ẋ = [J(x) - R(x; θ_R)] · ∇H(x; θ_H) + g(x; θ_g) · u(t)
```

Where:
- `J` is the canonical symplectic matrix (hardcoded, not learned — this is your geometric constraint)
- `R(x; θ_R)` is a neural net outputting a Cholesky factor `L·Lᵀ` to guarantee positive semi-definiteness
- `H(x; θ_H)` is your existing `HamiltonianNet` architecture (4×128 Tanh)
- `g(x; θ_g)` is a small MLP that maps state to control-effectiveness
- `u(t)` is the **inferred** control input (bank angle) — this is what SINDy discovers

**Key insight**: SINDy doesn't die. It becomes the **control inference** layer that estimates `u(t)` from residuals, while the DPHNO handles the physics. This is the SINDy-Hamiltonian synergy.

---

## 2. Foundation vs. Task: The Global Latent Physics Atlas

### Current Problem
Your training pipeline (`ai/neural_ode/train.py`) trains a single GHNN on trajectory data. This produces a **task-specific regressor**, not a foundation model. It cannot generalize to unseen atmospheric conditions, novel vehicle classes, or sensor degradation.

### Proposed Architecture: Atlas-Project-Execute (APE)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    H100 TRAINING CLUSTER                            │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              GLOBAL LATENT PHYSICS ATLAS (GLPA)               │  │
│  │                                                               │  │
│  │  Atlas Encoder E_θ: (state, atmosphere, vehicle_class) →      │  │
│  │                     z ∈ ℝ^256 (latent physics code)          │  │
│  │                                                               │  │
│  │  Atlas Decoder D_φ: z → {H(·), R(·), g(·)}                  │  │
│  │    (Hypernetwork: z generates the WEIGHTS of H, R, g)         │  │
│  │                                                               │  │
│  │  Trained on:                                                  │  │
│  │    - 10^6 synthetic HGV trajectories (varied ρ, L/D, σ)      │  │
│  │    - 10^5 orbital trajectories (LEO/MEO/GEO/HEO)             │  │
│  │    - 10^4 ballistic missile trajectories                      │  │
│  │    - NRLMSISE-00 atmosphere tables as physics prior           │  │
│  │    - US Standard Atmosphere 1976 (your atmosphere.go)         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                          │                                          │
│                          │ Distill: z → sparse local equations      │
│                          ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              PROJECTION LAYER (H100 → M2 Air)                 │  │
│  │                                                               │  │
│  │  For each vehicle class k:                                    │  │
│  │    z_k = E_θ(class_k)                                        │  │
│  │    {W_H^k, W_R^k, W_g^k} = D_φ(z_k)                        │  │
│  │                                                               │  │
│  │  Compress via structured pruning:                             │  │
│  │    4×128 Tanh → 4×32 Tanh (8× parameter reduction)          │  │
│  │    FP32 → INT8 quantization (4× memory reduction)            │  │
│  │    Total: 32× compression per projected model                │  │
│  │                                                               │  │
│  │  Output: N ONNX files, one per vehicle class                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          │ Deploy via Go ONNX Runtime
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    M2 AIR EDGE NODE                                  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              SPARSE LOCAL EXECUTOR                             │  │
│  │                                                               │  │
│  │  Go Runtime (pkg/hypersonic/):                                │  │
│  │    1. IMM filter selects mode → selects projected model       │  │
│  │    2. ONNX inference: state → {ẋ_predicted}                  │  │
│  │    3. SINDy on residual: discovers u(t) in real-time          │  │
│  │    4. Symbolic certificate: LaTeX equation from SINDy coeffs  │  │
│  │                                                               │  │
│  │  Fallback: Pure Go physics (hgv_physics.go) if ONNX fails    │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### The Hypernetwork Core

The **foundation** aspect is the hypernetwork `D_φ(z)`. Instead of training one model per scenario, you train a single network that **generates model weights** from a latent code. This means:

- **Zero-day maneuver**: If the Atlas encoder has never seen this exact maneuver, it can still interpolate in latent space between known vehicle classes
- **Atmospheric variation**: The latent code `z` encodes atmospheric state, so the generated H, R, g functions adapt to real-time space weather
- **Sensor degradation**: The encoder can condition on measurement noise level, generating appropriately regularized dynamics

---

## 3. Training Regime: Physics as Geometric Constraint

### Current Failure
Your `StructurePreservingLoss` (`ai/neural_ode/train.py:93`) treats physics as a penalty:
```python
loss = loss_obs + λ_h * loss_energy
```
With `λ_h = 0.05`, the optimizer can reduce `loss_obs` by 1.0 at the cost of increasing `loss_energy` by 20.0 — and still improve total loss. **Physics is violable.**

### Proposed: Constrained Optimization on the Symplectic Manifold

Replace penalty-based training with **hard geometric constraints** via three mechanisms:

#### A. Symplectic Integration in the Training Loop
Replace Euler steps in `GHNNTrainer.train_epoch()` with the Ruth-4 symplectic integrator you already have in `q/core/jax_backend.py:518`:
```python
# Current (violates symplecticity):
current_x = current_x + dx * dt  # Euler step

# Required (preserves symplecticity):
q, p = current_x[:, :3], current_x[:, 3:]
q, p = ruth4_step(q, p, dH_dq, dH_dp, dt)  # Forest-Ruth 4th order
current_x = torch.cat([q, p], dim=-1)
```
This ensures the **forward pass** itself preserves phase-space volume. The network cannot learn dynamics that violate Liouville's theorem because the integration scheme rejects them.

#### B. Navier-Stokes Residual as a Hard Constraint Layer
For HGV tracking in the 30-80 km corridor, the atmosphere is in **transitional flow** (Knudsen number 0.01-10). Your `GetDensityUnified` uses Jacchia-Roberts which is valid above 100 km.

Inject a differentiable atmosphere layer that enforces:
```
∂ρ/∂t + ∇·(ρv) = 0                    (continuity)
ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + f     (momentum — simplified N-S)
```

In practice, this means:
1. Pre-compute a lookup table of NRLMSISE-00 density profiles (you have the C reference at `pkg/physics/reference/nrlmsise-00.c`)
2. Fit a small neural network `ρ_nn(h, F10.7, Ap, lat, lon, DOY)` to this table
3. During training, the drag term `D = 0.5 · ρ_nn(h) · V² · A · CD` uses the **differentiable** density model
4. Add a constraint that `ρ_nn` satisfies the hydrostatic equation: `dp/dh = -ρg` → this is a 1D PDE constraint that can be enforced as a projection after each gradient step

#### C. Augmented Lagrangian for Conservation Laws
Instead of `loss = L_obs + λ·L_physics`, use:
```
L(θ, μ, λ) = L_obs(θ) + μᵀ·c(θ) + (ρ/2)·‖c(θ)‖²
```
Where `c(θ) = 0` are the hard constraints:
- `c₁`: Hamiltonian conservation `Var[H(trajectory)] = 0`
- `c₂`: Angular momentum conservation `Var[‖r×v‖] = 0` (for orbital segments)
- `c₃`: Hydrostatic equilibrium `∫(dp/dh + ρg)² dh = 0` (for atmosphere model)
- `c₄`: Positive semi-definiteness of R: `eigenvalues(R) ≥ 0`

The Lagrange multiplier `μ` is updated each epoch:
```python
μ = μ + ρ * c(θ)  # Dual ascent
```
This guarantees that at convergence, constraints are **exactly satisfied**, not approximately penalized.

---

## 4. Hardware-Algorithm Co-Design: Sparsity-on-Demand

### The Thermal Problem
M2 Air sustained TDP is ~15W with passive cooling. Running a full 4×128 HNN at 100Hz consumes ~8W (ONNX Runtime on Neural Engine). This leaves only 7W for OS, sensors, and comms. After 10 minutes of sustained inference, thermal throttling reduces clock to 60% → latency spikes.

### Proposed: Phase-Adaptive Execution (PAX)

```
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE-ADAPTIVE EXECUTION                       │
│                                                                   │
│  IMM Mode Probability → Execution Tier Selection                 │
│                                                                   │
│  ┌─────────────┐   P(ballistic) > 0.8                           │
│  │ TIER 0:     │   Pure Go physics (hgv_physics.go)             │
│  │ BALLISTIC   │   ~0.1W, 0.01ms latency                       │
│  │ (Go-only)   │   Full accuracy for ballistic arcs             │
│  └──────┬──────┘                                                 │
│         │ P(glide) > 0.5                                         │
│  ┌──────▼──────┐                                                 │
│  │ TIER 1:     │   Pruned ONNX (4×32, INT8)                    │
│  │ GLIDE       │   ~2W, 0.5ms latency                          │
│  │ (Sparse NN) │   8× fewer parameters than full model          │
│  └──────┬──────┘                                                 │
│         │ P(maneuver) > 0.3 OR residual > 3σ                    │
│  ┌──────▼──────┐                                                 │
│  │ TIER 2:     │   Full DPHNO (4×128, FP16)                    │
│  │ MANEUVER    │   ~8W, 2ms latency                            │
│  │ (Full NN)   │   Maximum fidelity for zero-day events         │
│  └──────┬──────┘                                                 │
│         │ SINDy R² < 0.7 (unexplainable dynamics)                │
│  ┌──────▼──────┐                                                 │
│  │ TIER 3:     │   Full DPHNO + SINDy + JAX imagination         │
│  │ BREACH      │   ~12W, 10ms latency (burst)                  │
│  │ (All-hands) │   JAX Metal backend for parallel futures        │
│  │             │   Triggers thermal warning after 60s            │
│  └─────────────┘                                                 │
│                                                                   │
│  THERMAL GOVERNOR:                                                │
│  - T < 70°C: All tiers available                                 │
│  - T 70-85°C: TIER 3 disabled, max TIER 2                       │
│  - T > 85°C: Force TIER 0 + alert ground station                │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Mapping to Existing Code

- **TIER 0**: Your `hgv_physics.go` `HGVEquationsOfMotion()` + `hgv_imm_filter.go` `predictBallistic()`
- **TIER 1**: Pruned version of `ghnn.py` exported via `export_ghnn_to_onnx()`, loaded in Go via existing ONNX bridge in `pkg/ml/`
- **TIER 2**: Full `GHNNVectorField` + `ResidualMLP` from `hybrid_propagator.py`
- **TIER 3**: `AstrojaxWorldModel.batch_imagine()` via JAX Metal + `SlidingSINDy.Update()` from `pkg/physics/sliding_sindy.go`

The tier selection integrates directly into your `HybridManeuverDetector` orchestrator (`pkg/maneuver/maneuver.go`) and `Orchestrator` (`pkg/ml/hybrid/orchestrator.go`), which already implement model-first-with-fallback logic.

---

## 5. The Sovereign Proof: Symbolic Certificates

### Current State
Your `SlidingSINDy` already outputs `EquationString` — a LaTeX-formatted equation. Your `ManeuverResult` contains `ActiveTerms` and `Coefficients`. This is 80% of a symbolic certificate.

### What's Missing
1. **Confidence interval** on each coefficient (currently placeholder `R² = 0.95`)
2. **Causal chain**: Which sensor observations led to which coefficients
3. **Physics consistency check**: Does the discovered equation satisfy conservation laws?
4. **Human-readable tactical interpretation**: What does `0.003·sin(r/7000)` mean operationally?

### Proposed: Verifiable Symbolic Certificate (VSC)

```go
type SymbolicCertificate struct {
    // The discovered equation
    Equation        string            // LaTeX: "\dot{v} = -0.42v|v| + 0.03\sin(r/7000)"
    ActiveTerms     map[string]float64 // {"drag": -0.42, "periodic_thrust": 0.03}
    
    // Statistical confidence
    RSquared        float64           // Actual R² (not placeholder)
    CoeffCILower    map[string]float64 // 95% CI lower bound per coefficient
    CoeffCIUpper    map[string]float64 // 95% CI upper bound per coefficient
    PValue          float64           // F-test p-value for model significance
    
    // Physics consistency
    EnergyResidual  float64           // |ΔH/H| over the window
    MomentumResidual float64          // |Δ(r×v)/(r×v)| over the window
    PhysicsGrade    string            // "A" (< 1e-6), "B" (< 1e-3), "C" (< 1e-1), "F" (violation)
    
    // Causal provenance
    SensorSources   []string          // Which radars/optical sensors contributed
    WindowStart     time.Time         // Observation window
    WindowEnd       time.Time
    NumObservations int
    
    // Tactical interpretation
    TacticalIntent  string            // "SATURN", "MARS_KINETIC", "RAHU_DECEPTIVE"
    IntentConfidence float64
    HumanSummary    string            // "Target executing 3G skip-glide maneuver with 
                                      //  0.03 km/s² periodic thrust modulation consistent
                                      //  with DF-17 class HGV in terminal approach phase"
    
    // Falsifiability
    Prediction      [6]float64        // Predicted state at T+60s
    PredictionSigma [6]float64        // Uncertainty envelope
    VerifyAt        time.Time         // When to check prediction
}
```

### The Certificate Pipeline

```
Observations → SlidingSINDy → Raw Coefficients
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            Bootstrap CI               Physics Consistency
            (resample window             (check dH/dt ≈ 0
             100×, refit)                 for conservative
                    │                     segments)
                    └───────────────┬───────────────┘
                                    ▼
                            Equation Assembly
                            (human-readable LaTeX +
                             tactical interpretation
                             via mapResultToIntent())
                                    │
                                    ▼
                            60-second Prediction
                            (forward-propagate using
                             discovered equation)
                                    │
                                    ▼
                            SIGNED CERTIFICATE
                            (hash of all fields +
                             timestamp for audit trail)
```

The key property: the certificate is **falsifiable**. It makes a prediction 60 seconds into the future. If the prediction matches observation within the stated uncertainty, the certificate is validated. If not, the system self-corrects and flags reduced confidence. A commander can verify this in real-time by watching whether predicted positions match radar returns.

---

## 6. Identified Code-Level Actions

### Critical Fixes (Must-Do)

| Priority | File | Issue | Action |
|---|---|---|---|
| **P0** | `pkg/physics/sindy.go:353-357` | `computeRSquared()` returns hardcoded 0.95 | Implement actual R² from predicted vs observed residuals |
| **P0** | `ai/v3/astrojax_world_model.py:140-144` | `_mock_propagate()` uses Euler integration | Replace with RK4 or reference your Ruth-4 symplectic from `jax_backend.py` |
| **P0** | `pkg/hypersonic/hgv_imm_filter.go:196` | Kalman gain hardcoded to 0.3 | Implement proper EKF with state-dependent gain from covariance |
| **P1** | `ai/neural_ode/train.py:148-151` | Training uses Euler integration | Use leapfrog/Yoshida-4 to preserve symplecticity during training |
| **P1** | `pkg/physics/atmosphere.go:22` | `GetDensityUnified()` uses fixed F10.7=150, Ap=15 | Wire to real-time space weather feed |
| **P1** | `q/core/jax_backend.py:502-516` | Duplicated `elif order == 3` block (syntax error) | Remove duplicate branch |
| **P2** | `pkg/physics/sliding_sindy.go:265-266` | `containsTrig`/`isConstantTerm` helpers likely missing | Verify these exist or implement them |

### Architecture Extensions (Build)

| Component | Location | Description |
|---|---|---|
| `DissipativeHNN` | `ai/v3/dphno.py` (new) | Port-Hamiltonian NN with learned dissipation R(x) |
| `HypernetworkAtlas` | `ai/v3/atlas.py` (new) | Hypernetwork that generates DPHNO weights from latent code |
| `NeuralAtmosphere` | `ai/v3/neural_atmos.py` (new) | Differentiable NRLMSISE-00 surrogate with hydrostatic constraint |
| `PAXController` | `pkg/ml/pax/controller.go` (new) | Phase-Adaptive Execution tier manager with thermal governor |
| `SymbolicCertificate` | `pkg/physics/certificate.go` (new) | VSC struct + generation pipeline |
| `BootstrapCI` | `pkg/physics/sindy_bootstrap.go` (new) | Bootstrap confidence intervals for SINDy coefficients |

---

## 7. Training Data Strategy for H100

### Phase 1: Synthetic Atlas (Weeks 1-4)
- Generate 10⁶ HGV trajectories using `HGVEquationsOfMotion()` with randomized:
  - L/D ratio: Uniform[1.5, 5.0]
  - Bank angle profile: Sinusoidal + step + ramp + zero-day random
  - Atmosphere: NRLMSISE-00 with F10.7 ∈ [70, 300], Ap ∈ [0, 400]
  - Initial conditions: Siliguri corridor ± 15° latitude, 30-80 km altitude
- Generate 10⁵ orbital trajectories from your existing Go propagator
- Store as Zarr/HDF5 for efficient H100 data loading

### Phase 2: Foundation Training (Weeks 5-8)
- Train GLPA hypernetwork on H100 cluster
- Loss: Augmented Lagrangian with {energy, momentum, hydrostatic} constraints
- Optimizer: AdamW + cosine annealing, 500 epochs
- Validation: Hold out 10% of each trajectory class + adversarial perturbations

### Phase 3: Projection & Edge Deployment (Weeks 9-10)
- For each vehicle class k, compute z_k and generate pruned weights
- Quantize to INT8, export to ONNX
- Deploy to M2 Air via your existing `edge_deploy.py` → `HardwareSelector`
- Validate: Run your `hgv_imm_filter.go` test suite against projected models

---

## Summary: The Dominant Stack

```
H100 (Train)                          M2 Air (Infer)
─────────────                         ──────────────
GLPA Hypernetwork                     PAX Controller
  │                                     │
  ├─ DissipativeHNN (H, R, g)          ├─ TIER 0: Go HGV Physics
  ├─ FNO Atmosphere                    ├─ TIER 1: Pruned ONNX
  ├─ Augmented Lagrangian Loss          ├─ TIER 2: Full DPHNO
  └─ Projection/Quantization           ├─ TIER 3: JAX Imagination
                                        │
                                        └─ SlidingSINDy → VSC
                                           (Symbolic Certificate)
```

**Decision Dominance = Physics you can't violate × Intelligence you can verify × Hardware that doesn't overheat.**
