# Comprehensive Architecture Analysis: SSIS / Axiom-Prime / QIDT Stack

**Scope**: Full data flow, algorithms, ML/training, neural architecture, Go↔Python dependencies, theory, and math from edge (M2 Air) to cloud (H100 cluster).

---

## 1. Executive Summary

This is a **sovereign space-domain awareness (SDA) stack** built around a single mathematical insight: **physics must be a hard constraint, not a soft penalty**. The architecture spans:

- **Go runtime** (real-time, <100ms OODA loop): orbital propagation, HGV tracking, SINDy, IMM filtering, ONNX inference, PAX thermal governance
- **Python/JAX training** (offline, H100): Port-Hamiltonian Neural Operators, Fourier Neural Operators for atmosphere, hypernetwork atlas, symplectic integrators
- **Quantum-immune substrate** (QIDT): Majorana zero-mode active evasion for rad-hard compute

The stack is designed for **closed-loop kinetic custody** — transitioning from open-loop orbital propagation to real-time, physics-grounded tracking of hypersonic glide vehicles (HGVs) with interpretable symbolic certificates.

---

## 2. Full Data Flow (Sensor → Decision)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SENSOR INGESTION LAYER                              │
│  Radar / Optical / IR / ELINT → pkg/boundary/sensor.go → FlatBuffers       │
│  Sampling: 100Hz (HGV), 1Hz (orbital), 10Hz (passive photons)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GO RUNTIME — REAL-TIME PIPELINE                         │
│  pkg/physics/sindy.go          — Residual-Inverse SINDy (maneuver detect)   │
│  pkg/hypersonic/hgv_imm_filter.go — IMM 3-mode filter + Active Inference   │
│  pkg/physics/atmosphere.go     — Jacchia-Roberts + Neural Δ-stream         │
│  pkg/ml/hybrid/orchestrator.go — PAX tier selection + fallback routing        │
│  pkg/catalog/bvh.go            — Orbital catalog + collision screening       │
│  engine/main.go                — gRPC/REST API + TLS + OpenTelemetry         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        ┌──────────────────────┐           ┌──────────────────────────────┐
        │   ONNX RUNTIME       │           │   ZMQ / PLASMA BRIDGE        │
        │   (Edge: M2 Air)     │           │   (Train: H100 Cluster)      │
        │   Tier 1: INT8       │           │   IPC: /tmp/train_signal.ipc │
        │   Tier 2: FP16/FP32  │           │   Plasma: Apache Arrow 2GB │
        └──────────────────────┘           └──────────────────────────────┘
                    │                                   │
                    ▼                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │              PYTHON/JAX TRAINING BACKEND                      │
        │  ai/v3/port_hamiltonian.py    — DPHNO core (JAX + Equinox)    │
        │  q/core/hno.py                — ρ_nn atmosphere operator      │
        │  ai/neural_ode/train.py       — PyTorch ONNX export path      │
        │  sda-train/train_init.py      — ZMQ/Plasma orchestrator       │
        └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT: SYMBOLIC CERTIFICATE                          │
│  ManeuverResult (sindy.go): EquationString, R², Bootstrap CI, Physics Grade  │
│  HamiltonianHash: SHA-256 of binned H_avg + coefficients                     │
│  Falsifiable Prediction: T+60s state with σ envelope                         │
│  SITREP: Source→Target transition alerts for mesh dispatch                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Phase-by-Phase Breakdown

**Phase 0: Observation Encoding**
- Raw sensor data → `pkg/boundary/sensor.go` → FlatBuffers serialization (`proto/ssis/v1/`)
- SE(3)-equivariant preprocessing: position/velocity normalized to canonical units (DU = 6378.137 km, TU = 806.8116 s)

**Phase 1: Physics Baseline**
- `pkg/physics/propagate.go` (RKF45/DP54) → Keplerian prediction
- `pkg/hypersonic/hgv_physics.go` → Spherical EOM for HGV (drag + lift + bank angle σ)
- `pkg/physics/atmosphere.go` → `GetDensityUnified()` with live F10.7/Ap from NOAA SWPC

**Phase 2: Residual Computation**
- `r(t) = ẋ_observed - ẋ_keplerian` (central difference derivatives)
- This residual is the **pure maneuver signal**; SINDy operates only on residuals, not full dynamics

**Phase 3: Sparse Discovery (SINDy)**
- `pkg/physics/sindy.go:DiscoverManeuver()` builds Θ library (21 terms: polynomial + trigonometric + drag)
- STLSQ sparse regression → coefficients with bootstrap CI (1000 resamples)
- `ManeuverResult` contains: active terms, R², 95% CI, energy drift σ_H

**Phase 4: Mode Filtering (IMM)**
- `pkg/hypersonic/hgv_imm_filter.go` maintains 3 modes: Ballistic / Glide / Maneuver
- Dynamic covariance from SPHS diffusion model (not hardcoded R matrix)
- Kalman gain computed from covariance trace, clamped to [0.01, 0.95]
- Zero-day mode spawning when NIS > χ² threshold (11.34 for 3 DOF @ 99%)

**Phase 5: Tier Selection (PAX)**
- `pkg/ml/pax/controller.go` selects execution tier based on IMM mode probabilities:
  - **Tier 0 (Ballistic)**: Pure Go physics — ~0.1W, 0.01ms
  - **Tier 1 (Glide)**: Pruned INT8 ONNX — ~2W, 0.5ms
  - **Tier 2 (Maneuver)**: Full FP32 DPHNO — ~8W, 2ms
  - **Tier 3 (Breach)**: JAX imagination + SINDy — ~12W, 10ms burst
- Thermal governor: T > 85°C forces Tier 0 + alert

**Phase 6: Certificate Generation**
- `ManeuverResult.ValidateCertificate()` checks:
  1. `σ_H / |H_avg| < 1e-7` (spoof detection — self-invalidates if violated)
  2. All physics constraints pass (energy, momentum, hydrostatic)
  3. Bootstrap CI coverage > 80%
- `ComputeHamiltonianHash()` — tolerance-binned SHA-256 for sovereign handshake
- `GenerateFalsifiablePredictions()` — T+60s Keplerian propagation with σ envelope

---

## 3. Algorithms — Theory and Code Mapping

### 3.1 SINDy (Residual-Inverse) — `pkg/physics/sindy.go:90-141`

**Reference**: Brunton et al., *PNAS* 2016

**Mathematical Formulation**:
Given state matrix X ∈ ℝ^(N×6) and derivative Ẋ ∈ ℝ^(N×6), find sparse Ξ such that:

```
ẋ_obs = [J(x) - R(x)]∇H(x) + g(x)·u(t)
r(t) = ẋ_obs - [J-R]∇H      ← observed residual
STLSQ: min ‖Θ·ξ - r‖²_F + λ‖ξ‖₀
```

Library Θ: `{1, x, y, z, vx, vy, vz, x², y², z², xy, xz, yz, x³, y³, z³, sin(r/7000), cos(r/7000), v·vx, v·vy, v·vz}` (21 terms)

**Bootstrap CI**: 1000 resamples with replacement, percentile method for 95% CI. Actual `R²` computed (not hardcoded) at `sindy.go:604-668`.

### 3.2 IMM Filter — `pkg/hypersonic/hgv_imm_filter.go`

**Modes**: Ballistic (coast) / Glide (L/D steady) / Maneuver (high-G bank)

**Math**:
```
μ_j = (1/c_j) · Λ_j · Σ_i(p_ij · μ_i)     — mode probability update
NIS = νᵀ S⁻¹ ν                           — normalized innovation squared
```

- Kalman gain computed from covariance trace (was hardcoded 0.3, now clamped `[0.01, 0.95]`)
- Dynamic covariance from SPHS diffusion model when `EnableDynamicCovariance=true`
- **Zero-day detection**: NIS > χ²_{3,0.99} = 11.34 spawns new mode
- Active Inference: Expected Free Energy `EFE = -log(Likelihood) + Complexity + AdversarialPenalty`

### 3.3 HGV Physics — `pkg/hypersonic/hgv_physics.go:102-160`

**Spherical EOM** (Regan & Anandakrishnan):
```
V̇ = -D/m - g·sin(γ)
γ̇ = (1/V)·[L·cos(σ)/m - (g - V²/r)·cos(γ)]
ψ̇ = (1/V·cos(γ))·[L·sin(σ)/m + (V²/r)·cos²(γ)·sin(ψ)·tan(φ)]
ṙ = V·sin(γ)
φ̇ = V·cos(γ)·cos(ψ)/r
λ̇ = V·cos(γ)·sin(ψ)/(r·cos(φ))
```
Drag: `D = 0.5·ρ·V²·A·CD`, Lift: `L = 0.5·ρ·V²·A·CL`. Bank angle `σ` is the **surveillance nightmare variable**.

### 3.4 Atmosphere — `pkg/physics/atmosphere.go:613-627`

**Jacchia-Roberts** for >100km; **US Standard Atmosphere 1976** stratified layers for <86km. Sigmoidal blend with FNO neural correction:
```
ρ_effective = (1-α)·ρ_FNO + α·ρ_JR
α = sigmoid(k·(|ρ_FNO - ρ_JR|/ρ_JR - threshold))
```
Hydrostatic projection enforces `dP/dz = -ρg` to prevent FNO hallucination.

---

## 4. Neural Architecture (Training Backend)

### 4.1 Port-Hamiltonian System — `ai/v3/port_hamiltonian.py:128-176`

**Core ODE**:
```
dz/dt = [J(z) - R(z; W_R)]·∇_z H(z, z_latent; W_H) + g(z; W_g)·u
```

| Component | Architecture | Constraint |
|---|---|---|
| `H_net` | 4×128 Tanh MLP | Scalar energy, conditioned on latent `z` via FiLM |
| `R_net` | 2×128 MLP → Cholesky `L` | `R = LLᵀ` guarantees PSD (no energy creation) |
| `g_net` | 2×128 MLP | Control coupling matrix |
| `J(z)` | Canonical symplectic `[[0,I],[-I,0]]` | Hardcoded, not learned |

**FiLM conditioning**: `h' = γ(z) ⊙ h + β(z)` — latent code gates every hidden layer.

### 4.2 Neural Atmosphere Operator — `q/core/hno.py:180-261`

**Density Neural Operator ρ_nn**:
- Encoder: 3 → width (altitude, F10.7, Ap)
- Fourier Spectral Layers: `modes=16`, captures thermospheric high-frequency variations
- Decoder: width → 1
- **Hydrostatic Projector**: `projectHydrostatic()` enforces `dP/dz = -ρg` via autodiff, then blends 70% projected + 30% neural

### 4.3 Yoshida-4 Symplectic Integrator — `ai/v3/port_hamiltonian.py:874-898`

Forest-Ruth coefficients:
```
c1 = c4 = 1/(2-2^(1/3)),  c2 = c3 = -2^(1/3)/(2-2^(1/3))
d1 = d3 = 1/(2-2^(1/3)),  d2 = -2^(1/3)/(2-2^(1/3)), d4 = 0
```
q/p split preserves symplecticity. **Energy drift < 1e-8 over 10⁴ steps** on Keplerian orbits.

### 4.4 Training Loss — Augmented Lagrangian

**Hard physics constraints** (not soft penalties):
```
L_total = L_dynamics + μ₁ᵀ·c_energy + (ρ₁/2)·‖c_energy‖²
                         + μ₂ᵀ·c_momentum + (ρ₂/2)·‖c_momentum‖²
                         + μ₃ᵀ·c_hydrostatic + (ρ₃/2)·‖c_hydrostatic‖²
                         + α_supCon·L_contrastive + α_reg·‖z‖²
```

Dual ascent: `μ_{k+1} = μ_k + ρ·c(θ_k)` every epoch.

**Contrastive loss** (SupCon) organizes latent space by regime — orbital, ballistic, HGV clusters.

### 4.5 Hypernetwork Atlas — `ai/v3/hypernetwork.py` (implied)

**Architecture**: Hypernetwork `D_φ(z)` generates weights of task-specific DPHNO models from latent physics code `z`:
```
z = Encoder(context_window) ∈ ℝ^64
W_H, W_R, W_g = D_φ(z)
```
This enables **zero-day generalization**: never-before-seen HGV configurations get inferred `z` from a small context window, then a custom DPHNO is instantiated on the fly.

---

## 5. Go ↔ Python Dependency Bridge

### 5.1 ONNX Runtime Path (Inference)
```
JAX DPHNO → jax2onnx (via equinox) → .onnx file
Go: ort.NewSession(path) → onnx.Session
```

### 5.2 ZMQ / Plasma Path (Training)
```
Go engine → ZMQ PUSH (ipc:///tmp/train_signal.ipc) → sda-train/train_init.py
Plasma: Apache Arrow shared memory for image frames / trajectory arrays
```

### 5.3 Dependency Matrix

| Go Dependency | Purpose | Python Equivalent |
|---|---|---|
| `github.com/microsoft/onnxruntime-go` | ONNX inference | `jax`, `equinox` (JAX) |
| `gonum` | Matrix ops, Cholesky | `jax.numpy.linalg` |
| `github.com/apache/arrow/go/v17` | Plasma serialization | `pyarrow`, `pyarrow.plasma` |
| `github.com/gin-gonic/gin` | REST API | FastAPI (not used here) |
| `golang.org/x/net/websocket` | gRPC streaming | `zmq` |
| `github.com/tle` | TLE parsing | `skyfield` |

---

## 6. Training Regime and Data Pipeline

### 6.1 Data Generation
- **10⁶ HGV trajectories**: stochastic bank angle profiles, variable L/D, skip-glide
- **10⁵ orbital trajectories**: Orekit with J2/J3/SRP/drag perturbations
- **10⁴ ballistic trajectories**: atmospheric entry with varying B = CD·A/m
- **10⁷ atmosphere samples**: MSIS-00 + Jacchia baseline with F10.7/Ap sweeps

### 6.2 Curriculum Phases

**Phase 1** (Days 1-3): Data pipeline — MIT SPLID, CARA, synthetic Orekit
**Phase 2** (Days 4-7): PatchTST-JEPA pre-training (self-supervised, mask ratio 0.6)
**Phase 3** (Days 8-14): DPHNO training
- Epochs 1-50: Freeze R, g — train H on pure Keplerian (conservative)
- Epochs 51-300: Unfreeze R — train with atmospheric drag (dissipative)
- Epochs 301-500: Unfreeze g — train on maneuvers (non-conservative)

**Optimizer**: AdamW, lr=1e-4, cosine annealing, weight decay 0.05
**Hardware**: H100 cluster (8× A100/H100), Zarr storage for 10⁷ samples

### 6.3 Conformal Prediction & Uncertainty

- **Bootstrap CI**: SINDy coefficients get 95% CI via 1000 resamples
- **Conformal**: Calibration set → prediction sets with coverage guarantee
- **Coverage**: Target 90% nominal, validated on holdout

---

## 7. Symbolic Certificates (VSC)

**Verifiable Sovereign Certificate** extends the existing SINDy `EquationString`:

```go
type SymbolicCertificate struct {
    EquationString      string    // Human-readable: "0.42·x - 0.18·y + drag_term"
    BootstrapCILower    []float64 // Per-coefficient lower bound
    BootstrapCIUpper    []float64 // Per-coefficient upper bound
    PhysicsConsistency  float64   // Grade A/B/C/D based on constraint residuals
    CausalProvenance    []string  // Sensor IDs, processing chain hash
    HamiltonianHash     string    // SHA-256(binned H_avg + active terms)
    FalsifiablePredictions []State // T+60s states with σ envelope
}
```

**Spoof detection**: If `σ_H / |H_avg| >= 1e-7`, certificate self-invalidates. This is the **Sovereign Handshake** protocol.

---

## 8. Verification Metrics

| Metric | Target | How Verified |
|---|---|---|
| Energy drift (Keplerian, 1 orbit) | < 1e-6 | `tests/test_yoshida4_symplectic.py` |
| Position error (60s prediction) | < 50m | Go ↔ JAX trajectory comparison |
| SINDy R² | > 0.90 (actual) | Holdout set, not hardcoded |
| Hydrostatic residual | < 1e-6 | `verify_hydrostatic()` in `q/core/hno.py` |
| ONNX inference latency (M2 Air) | < 2ms | Benchmark suite |
| Thermal ceiling | < 85°C | PAX governor in `pkg/ml/pax/controller.go` |

---

## 9. Critical Code Fixes Identified

1. `sindy.go`: Hardcoded `R²=0.95` → replaced with actual computation
2. `astrojax mock`: Euler integration → Yoshida-4 symplectic
3. `hgv_imm_filter.go`: Hardcoded Kalman gain 0.3 → dynamic covariance with clamped gain
4. `jax_backend.py`: Duplicate branch in dual ascent → consolidated
5. `atmosphere.go`: Fixed parameters → space-weather-conditioned FNO blend

---

## 10. Theoretical Foundations

### 10.1 Port-Hamiltonian Systems
Reference: van der Schaft & Maschke. The PHS formulation `dz/dt = [J-R]∇H + gu` guarantees:
- **Passivity**: `dH/dt = -∇Hᵀ R ∇H + ∇Hᵀ G u ≤ supply_power` (since R is PSD)
- **Interconnection**: J encodes conservative structure (Poisson bracket)
- **Dissipation**: R encodes drag, friction, radiation pressure

### 10.2 Symplectic Integrators
Reference: Hairer et al., *Geometric Numerical Integration*. Yoshida-4 preserves:
- Modified Hamiltonian: `H̃ = H + Δt⁴·H_4 + ...` (no secular energy drift)
- Phase space volume: `det(∂z_{n+1}/∂z_n) = 1`

### 10.3 Lie Group SE(3)
Reference: Marsden & Ratiu. Position updates via exponential map:
- `exp(ω)` (Rodrigues' formula) for rotation
- Semi-implicit Euler on `so(3)` for angular momentum
- Avoids non-physical accumulation of angular errors

### 10.4 Koopman Operators (aspirational)
The DPHNO can be viewed as a finite-dimensional Koopman approximation: the latent code `z` acts as a Koopman eigenfunction conditioning the generator `[J-R]∇H`.

---

## 11. Strategic Advantage Summary

| Capability | Legacy Radar | This Stack |
|---|---|---|
| Detection modality | Active RF (jammable) | Passive photons + wake spectroscopy |
| HGV tracking | ~1-5s batch | <100ms Go-reflex |
| Physics grounding | Empirical tuning | Hard-constrained PHS + symplectic |
| Uncertainty | Ad-hoc | Bootstrap CI + conformal prediction |
| Zero-day adaptation | Manual retuning | Hypernetwork atlas inference |
| Edge thermal | Unmanaged | PAX 4-tier thermal governance |

---

**Status**: Architecture analyzed. Training data pipelines defined. Critical code fixes catalogued. ONNX export path documented. Next step: execute the 14-day sprint (ZMQ-INDI handshake, centroid-to-PHS feedback, Hamiltonian hash protocol).
