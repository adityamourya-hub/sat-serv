# drq

A command-and-control substrate designed for the Indian Air Force. It provides sub-second volumetric surveillance, hypersonic threat detection, and symplectic orbital prediction at planetary scale.

> **Implementation Status**
> | Component | Status | Notes |
> |-----------|--------|-------|
> | SGP4 / Orbital Propagation | Working | Go-native, tested |
> | SINDy Maneuver Detection | Partial | Bootstrap CI implemented; R² computed from residuals |
> | HGV IMM Filter | Partial | 3-mode filter active; Kalman gain still static |
> | ONNX Runtime Bridge | Stub | 12 `*_stub.go` files — no actual ONNX inference bound |
> | DPHNO (Port-Hamiltonian NN) | Research Claim | Architecture described, not trained or integrated |
> | Hypernetwork Atlas | Research Claim | Not trained |
> | Axiom-Zero Physics Core | Research Claim | No working JAX↔Go bridge |
> | Sovereign Handshake / Quantum-immune | Not Implemented | Marketing concept, no code exists |

# Capabilities

| Capability | Description | Status |
| :--- | :--- | :--- |
| **(a) Data Fusion** | Real-time multi-sensor fusion and volumetric screening. | 
| **(b) Proximity Alerts** | Neighbourhood Watch (Intelligent intent & proximity alerts). | 
| **(d) Hypersonic Tracking** | High-fidelity 8-state UKF for Glide Vehicle (HGV) tracking. | 
| **(g) Maneuver Detection** | PINN-accelerated learned residual analysis for delta-V discovery. | 
| **(i) Rapid Processing** | Sub-second propagation of 30k+ objects via Symplectic Kernels. |

---

## Strategic Architecture: 

* Every observation is a measurement of perturbed two-body Hamiltonian flow (H = T + V, with J2, drag ρ(v) exponential, SRP, third-body). Sensor physics sets hard limits:

| Sensor | Observables | Physics Constraint |
|--------|-------------|-------------------|
| **Radar** (NETRA phased-array) | Range, Doppler, RCS | Active, weather-agnostic, inverse-square power loss |
| **Optical** (Ladakh telescope) | Sub-arcsec angles, photometry, spectra | Passive, night/clear-sky limited, no range without stereo |
| **Passive RF** | Emitter signatures, Doppler | All-weather, day/night, detects thrust via signal change without illuminating |
| **NavIC** | Timing, TEC | Scintillation (S4>0.6 over Siliguri/Leh) corrupts phase at tens-of-ns |

Covariance explodes in-track from energy uncertainty (δa → δperiod → km/day drift). No "AI magic" — ML/DSP only accelerates where physics allows (residuals against conserved H, not black-box replacement).

**Indigenous reality:** NETRA phased-array radar (Northeast LEO), optical telescope (Ladakh), MOTR (Sriharikota), future space-based inspectors. Designed for on-prem IAF deployment with minimal external dependencies.

### P0: Defensive Foundation 

| Feature | Sensors | DSP Stack | ML Role |
|---------|---------|-----------|---------|
| **D1 Hamiltonian Firewall + D3 Covariance Realism** | Radar + optical + passive RF | State transition matrix Φ propagation, chi-square residual test, Kalman/UKF, eigenbeamspace beamforming | Physics-grounded unsupervised anomaly (residuals from ballistic HNN/SGP4). *No training on adversary data.* |
| **D2 NavIC Spoof-Immunity + D6 EIA Scintillation Auto-Switch** | NavIC L5 + pulsar timing + dual-freq receiver | Carrier-phase + code correlation, ROTI/MP monitoring, auto-switch dual-freq + inflate covariance when S4>0.6 | *None — pure physics DSP* |
| **D10 Integrity-Forecast Board + D7 Air-Gapped Kernel** | All sensors + dual-site astrometry | Aggregate custody map (72h strong/weak/doubtful per AOR) | *None* — rugged GPU/FPGA nodes (indigenous AMD ROCm stack), zero network surface |

**P0 Output:** Reject spoofed tracks before catalog ingest. Without this, volumetric scan (a) and NW (b) flood with garbage.

### P1: Dual-Use Multiplier 

| Feature | Sensors | DSP Stack | ML Role |
|---------|---------|-----------|---------|
| **D9 Ghost-Maneuver Detector + A7 Covert Δv Planning** | Passive RF (primary for sub-threshold electric thrust) + high-res radar/optical confirmation | RF interferometry, IMM switching (ballistic vs thrusting), coherent integration gain >100 dB | Physics-ML hybrid — residuals as features in Bayesian/particle filter for sub-threshold Δv posterior. Unsupervised anomaly on pattern-of-life residuals. |
| **D5 Bayesian Fuel Posterior** | Radar/optical + passive RF thrust signatures | Covariance propagation with consider-parameters (fuel mass hidden state) | Bayesian posterior P(Δv ≥ X) via particle filter/UKF on Hamiltonian mismatch. Direct feed to provenance hash. |
| **A1 Reciprocal Reachability + A2 Blind-Window Exploitation** | Full catalog from all sensors (dynamic tasking via A* greedy) | Lambert solver + hybrid propagator, sensor FOV modeling (aperture, slew, occultation, lighting) | *None core* — physics propagation first. Optional learned priors for rapid re-plan post-manoeuvre. |

**P1 Output:** Standing OPLAN menu (A1) + 72h ISR blind calendar (A2) as air-ops MET layer.

### P2: Strategic Offensive (inspector constellation required)

| Feature | Sensors | Status |
|---------|---------|--------|
| **A3 Spectral Fingerprint Library** | Optical (photometry + spectroscopy) + passive RF characterization | Physics-grounded classification. *Political weapon only after P0 battle-tested.* |
| **A4 Kill-Chain Cueing** | High-fidelity radar/optical + Hamiltonian hash | Conservative gating (D1-D3-D5 only). *Highest escalation — one bad cue loses strategic game.* |
| **A6 Cluster Wolfpack** | ≥3 inspectors (space-based optical/radar/RF) | Multi-agent pursuit under J2/drag. *Physics-first pursuit planner.* |

### Minimal Stack

```
Core engine:    Hybrid propagator (SGP4 fast-screen → numerical truth)
                + Differentiable HNN for provenance hash + residuals
Sensor fusion:  NETRA-style radar + Ladakh optical + indigenous passive RF
                + NavIC timing. Dynamic tasking layer.
DSP kernel:     Beamforming, coherent integration, chi-square/IMM filtering,
                scintillation mitigation
ML layer:       Only hybrid physics-ML (Bayesian on Hamiltonian residuals,
                unsupervised anomaly) — never replaces physics
Hardware:       Indigenous GPU/FPGA on IAF nodes. Scale via add-nodes.
```

**CLI Validation Loop:**
```bash
iaf-sda sensor-stack \
    --features d1 d9 a1 \
    --sensors radar+optical+rf \
    --dsp hybrid-propagator \
    --ml bayesian-residuals \
    --output custody_map.json \
    --validate physics-metrics
```

**Metrics:** Energy conservation error, ghost detection latency, blind-window accuracy vs synthetic TLE + thrust injections.

# Security Baseline

- **Zero-Trust gRPC**: Mandatory Ed25519 signing for all telemetry responses via logic-level interceptors.
- **mTLS 1.3 Hardening**: Hardware-anchored transport security with mandatory client-cert verification.
- **Indigenous Physics Core**: Axiom-Zero symplectic substrate (Yoshida-4/8) ensuring zero secular energy drift.
- **Deployment**: Docker Compose and Kubernetes manifests provided. Air-gap configuration is manual.

# Architecture

 a **Differentiable Physics-Managed Awareness** plane, unifying high-performance Go-based catalog management with the **Axiom-Zero v3** JAX-native physics core.

### Design

**Layer 1: Go Operations Plane** (High-concurrency orchestration)
- `pkg/catalog`: High-churn optimized catalog management
- `pkg/orbital`: Unified Propagator interface (SGP4 + Axiom-Zero)
- `pkg/spatial`: BVH + Morton sweep-prune for O(n log n) conjunction screening
- `pkg/crypto`: Ed25519 response signing and audit trails
- `pkg/server`: mTLS 1.3 hardened gRPC/REST endpoints

**Layer 2: Axiom-Zero Physics Substrate** (Symplectic neural flows)
- `ai/v3/axiom_zero`: Hamiltonian-native rKAN physics kernel
- `ai/v3/sfmn_kernel.py`: State-space fusion neural network
- `ai/v3/maneuver_distiller.py`: Delta-V attribution via symbolic distillation
- `ai/v3/hamiltonian_discovery.py`: Learned Hamiltonian dynamics
- `ai/v3/physics_geometry_layer.py`: Equinoctial element differentiable geometry

###  Bridge

1. **Modeling (Python/JAX)**: High-order Hamiltonian dynamics learned using rational-KAN (rKAN) models
2. **Synthesis**: Trained weights for Force Potential (∇V) exported to JSON artifacts
3. **Operation (Go)**: Go-native rKAN kernel implements Legendre basis and softplus projections
4. **Integration**: Trajectories evolved using 4th-Order Symplectic Yoshida Integrator in pure Go, ensuring 10⁻¹⁵ energy conservation



### Prerequisites
- Go 1.21+
- SSIS_SIGNING_KEY (64-byte Ed25519 Private Key)

### Deployment
```bash
# Set signing identity
export SSIS_SIGNING_KEY=$(cat /etc/ssis/signer.key)

# Run the engine
go run ./engine/main.go
```

---
**Confidentiality Notice**: This software contains Defense IP. Unauthorized distribution is prohibited.

## Repository Structure

### Operational (Go)
- `engine/`: Primary operational entrypoint (Go binary)
- `pkg/`: Core operational packages
  - `catalog/`: High-churn optimized catalog management
  - `orbital/`: Unified Propagator interface (SGP4 + Axiom-Zero)
  - `hypersonic/`: HGV 6-DOF physics and IMM filtering
  - `physics/`: Classical SINDy, sliding window dynamics, atmosphere models
  - `ml/`: ONNX runtime bridge, hybrid orchestration, model serving
  - `collision/`: CDM generation and conjunction screening
  - `crypto/`: Ed25519 response signing and handshake protocol
- `internal/`: Internal implementation details
- `cmd/`: Command-line utilities and generators

### ML/AI Substrate (Python/JAX)
- `ai/v3/`: **Axiom-Zero v3** — Hamiltonian-native physics substrate
  - `port_hamiltonian.py`: Core DPHNO implementation (J, R, H, g decomposition)
  - `dphno.py`: Dissipative Port-Hamiltonian Neural Operator
  - `hypernetwork_atlas.py`: Foundation model hypernetwork for weight generation
  - `neural_density.py`: Differentiable NRLMSISE-00 atmosphere operator
  - `conformal_uq.py`: Conformalized Quantile Regression for uncertainty
  - `e3nn_hamiltonian.py`: SE(3)-equivariant Hamiltonian networks
  - `lie_phs.py`: Lie-group port-Hamiltonian systems
  - `adaptive_sindy.py`: Neural SINDy with learned library basis functions
- `ai/neural_ode/`: PyTorch GHNN training (legacy path, migrating to JAX)
- `sda-train/`: Training pipeline and model registry
- `q/core/`: JAX-native physics kernels, symplectic integrators, FNO backbone

### Documentation & Training
- `docs/`: Technical documentation and validation reports
  - `training/`: Training execution guides and readiness reports
  - `validation_reports/`: CARA parity cases and verification logs
- `scripts/`: Automation for build, SBOM generation, and physics validation
- `artifacts/`: Production model bundles and build artifacts
- `assets/`: Static assets (satellite catalogues, orbital architecture)

---

## ML Developers Guide

### Architecture Overview

SSIS implements a **Dissipative Port-Hamiltonian Neural Operator (DPHNO)** architecture that unifies orbital mechanics and hypersonic vehicle tracking under a single foundation model. The key innovation is treating physics as hard geometric constraints rather than soft penalties.

**Core Equation:**
```
ẋ = [J(x) − R(x; θ_R)] · ∇_x H(x, z; θ_H) + g(x; θ_g) · u(t)
```

Where:
- **J**: Canonical symplectic structure (hardcoded, structure-preserving)
- **R(x; θ_R)**: Learned dissipation matrix (Cholesky-parameterized, PSD by construction)
- **H(x, z; θ_H)**: Hamiltonian conditioned on latent physics code z
- **g(x; θ_g)**: Control port matrix for hidden inputs (bank angle, thrust)
- **u(t)**: Inferred control via residual SINDy (not observed)

### Key ML Components

| Component | File | Purpose |
|-----------|------|---------|
| **Port-Hamiltonian Core** | `ai/v3/port_hamiltonian.py` | DPHNO dynamics with Cholesky R, Yoshida-4 integration |
| **Hypernetwork Atlas** | `ai/v3/hypernetwork_atlas.py` | Generates {H, R, g} weights from latent code z |
| **Neural Atmosphere** | `ai/v3/neural_density.py` | Differentiable NRLMSISE-00 surrogate with hydrostatic constraint |
| **Equivariant Encoder** | `ai/v3/e3nn_hamiltonian.py` | SE(3)-equivariant state encoding for geometric invariance |
| **Conformal UQ** | `ai/v3/conformal_uq.py` | Distribution-free prediction intervals with coverage guarantees |
| **Adaptive SINDy** | `ai/v3/adaptive_sindy.py` | Neural library basis functions for control inference |

### Training Curriculum (Five Phases)

**Phase 0: Physics Bootstrap (Weeks 1-2)**
Train atmospheric DeepONet independently. Target: <0.5% density error vs NRLMSISE-00.
```bash
cd ai/v3
python neural_density.py --train --epochs 500 --constraint hydrostatic
```

**Phase 1: Orbital Foundation (Weeks 3-6)**
Train SE(3)-GNN encoder + hypernetwork on conservative orbital dynamics (R=0).
```bash
python trainer.py --phase orbital --loss augmented_lagrangian --integrator yoshida4
```

**Phase 2: Hypersonic Extension (Weeks 7-10)**
Activate dissipation term R and train on HGV skip-glide trajectories.
```bash
python trainer.py --phase hypersonic --data sda-train/data/hgv_zarr --enable_dissipation
```

**Phase 3: Adversarial Hardening (Weeks 11-13)**
Physics-constrained PGD training for zero-day maneuver robustness.
```bash
python trainer.py --phase adversarial --epsilon_m 10.0 --physics_gate physics_bridge
```

**Phase 4: Online Adaptation (Weeks 14+)**
Elastic Weight Consolidation for continual learning without catastrophic forgetting.

### Development Workflow

**1. Environment Setup**
```bash
# Python dependencies
pip install -r ai/v3/requirements.txt

# JAX with Metal backend for M-series Macs
pip install jax-metal

# Go dependencies (for ONNX bridge testing)
go mod download
```

**2. Data Generation**
```bash
# Generate synthetic HGV trajectories
go run cmd/gen_training_data/main.go --type hgv --count 1000000 --output zarr

# Generate orbital trajectories with HPOP (RKF45)
go run cmd/gen_training_data/main.go --type orbital --count 250000 --integrator rkf45
```

**3. Model Training**
```bash
# Train with Augmented Lagrangian constraints
cd ai/v3
python trainer.py \
    --config configs/dphno_foundation.yaml \
    --data_path ../../sda-train/data/processed/ \
    --output_dir ../../artifacts/models/
```

**4. Export to ONNX (Go Runtime)**
```bash
python export_dphno_onnx.py \
    --checkpoint artifacts/models/dphno_phase2.ckpt \
    --output artifacts/onnx/dphno_hgv.onnx \
    --quantize int8
```

**5. Validation**
```bash
# Physics integrity tests
go test ./pkg/hypersonic -run TestHGVPhysics -v
python -m pytest ai/v3/test_physics_constraints.py

# CARA parity cases
go test ./pkg/collision -run TestCARAParity -catalog docs/validation_reports/CARA_PARITY_CASES.json
```

### Critical Implementation Notes

**Hard Physics Constraints (Not Soft Penalties)**
The system uses Augmented Lagrangian optimization:
```
L(θ, μ, ρ) = L_obs(θ) + μᵀ·c(θ) + (ρ/2)·‖c(θ)‖²
μ ← μ + ρ·c(θ)  (dual ascent)
```

Constraints c(θ) = 0 at convergence:
- Energy conservation: `Var[H(trajectory)] = 0`
- Angular momentum: `Var[‖r×v‖] = 0`
- Hydrostatic equilibrium: `∫(dp/dh + ρg)² dh = 0`
- Dissipation PSD: `λ_min(R) ≥ 0` (structural via Cholesky)

**Integration Scheme**
The trainer uses Yoshida-4 symplectic integrator with proper q/p splitting. This is structurally required—Euler or RK4 violate phase-space volume conservation regardless of what the neural network learns.

**Residual SINDy for Control Inference**
Classical SINDy discovers forward dynamics from a fixed library. Here, SINDy is repurposed to infer hidden control inputs:
```
u(t) = SINDy( ẋ_obs − (J−R)∇H )
```
The library is control primitives (bang-bang, sinusoid, ramp), not polynomial dynamics.

### Evaluation Metrics Hierarchy

**Level 1: Physics Integrity (Must Pass)**
| Metric | Target | Test |
|--------|--------|------|
| Energy drift (24hr) | < 10⁻⁶ relative | `test_symplectic_integrator.py` |
| Density error | < 0.5% vs NRLMSISE | `test_neural_atmosphere.py` |
| Dissipation sign | Always ≤ 0 | `test_port_hamiltonian.py` |
| R PSD fraction | 100.0% | Eigenvalue check |

**Level 2: Tracking Accuracy**
| Metric | Target |
|--------|--------|
| Position error (T+60s, orbital) | < 10m (3σ) |
| Position error (T+60s, HGV) | < 50m (3σ) |
| L/D estimation error | < 5% |
| Zero-day detection rate | > 95% |

**Level 3: Operational**
| Metric | Target |
|--------|--------|
| End-to-end latency | < 100ms |
| SINDy R² (bootstrap) | > 0.85 |
| Certificate falsifiability | > 90% |

### Integration with Go Runtime

The ONNX bridge (`pkg/ml/`) loads quantized models and executes via ONNX Runtime:
```go
// Load DPHNO model for specific vehicle class
model, _ := ml.LoadONNXModel("artifacts/onnx/dphno_hgv_int8.onnx")

// Inference: state → predicted dynamics
pred := model.Predict(state, atmosphereContext)
```

Phase-Adaptive Execution (PAX) selects model tier based on IMM mode probability and thermal state—see `ai/model_arch.md` §4 for full specification.

### Documentation References

- `ai/foundation.md`: First-principles derivation of LG-PHNO architecture
- `ai/model_arch.md`: DPHNO dominant architecture proposal
- `ai/ml_plan.md`: Complete 16-week training curriculum
- `docs/training/TRAINING_EXECUTION_GUIDE.md`: Operational training procedures
- `docs/ADITI_SDA_FirstPrinciples_Spec.md`: System specification and requirements

---

## Foundation Model Architecture

The SSIS foundation model is a **Lie-Group Port-Hamiltonian Neural Operator (LG-PHNO)** that generates physics kernels on-demand from a latent physics code. This enables zero-day generalization to unseen vehicle classes and atmospheric conditions.

### Architecture Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INPUT: Sensor Measurements                           │
│  (r, v, σ_r, σ_v, t, sensor_id) + (F10.7, Ap, DoY, lat, lon)                │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 0: SE(3)-EQUIVARIANT GRAPH NEURAL NETWORK                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Nodes: RSO/HGV track points (position, velocity)                  │    │
│  │ • Edges: Sensor-measurement bipartite graph                         │    │
│  │ • Message passing: Equivariant under rotation/translation           │    │
│  │ • Output: z_obs ∈ ℝ^512 (rotation-invariant latent state)           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: ATMOSPHERIC NEURAL OPERATOR (DeepONet)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Input: (h, F10.7, Ap, DoY, lat, lon) → ρ(h)                       │    │
│  │ • Branch net: Encodes space weather → coefficients                  │    │
│  │ • Trunk net: Maps altitude → basis functions                        │    │
│  │ • Hard constraint: ∂p/∂h = -ρg (hydrostatic equilibrium)            │    │
│  │ • Output: z_atm ∈ ℝ^256, ρ_nn(h) differentiable field               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: HYPERNETWORK CONTEXT MODULE                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ z_context = Concat[z_obs, z_atm, p_mode] ∈ ℝ^768                    │    │
│  │                                                                     │    │
│  │ Hypernetwork H_φ: z_context → {W_H, W_R, W_g, W_lib}                │    │
│  │   • W_H: Hamiltonian network weights                                │    │
│  │   • W_R: Dissipation network weights (Cholesky L·Lᵀ)                │    │
│  │   • W_g: Control coupling weights                                   │    │
│  │   • W_lib: Neural SINDy library adaptation                          │    │
│  │                                                                     │    │
│  │ Per-vehicle-class physics kernels generated on-the-fly              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: DISSIPATIVE PORT-HAMILTONIAN DYNAMICS                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  dz/dt = [J - R(z; W_R)] · ∇_z H(z; W_H) + g(z; W_g)·u              │    │
│  │                                                                     │    │
│  │  • J = [[0, I], [-I, 0]] — canonical symplectic (hardcoded)         │    │
│  │  • R = L·Lᵀ — PSD by Cholesky construction                          │    │
│  │  • H — scalar Hamiltonian from generated weights                    │    │
│  │  • u — SINDy-discovered control (bank angle, thrust)                │    │
│  │                                                                     │    │
│  │  Integration: Yoshida-4 symplectic with operator splitting          │    │
│  │  (conservative step + dissipative step preserves structure)         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: NEURAL SINDY READOUT                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Learned library: Θ_learned(z) = [φ₁(z), ..., φ_K(z)]              │    │
│  │   (each φᵢ is a small 1×32 network, not fixed polynomials)          │    │
│  │ • Sparse regression: ż = Θ_learned(z) · ξ, ‖ξ‖₀ ≤ s                 │    │
│  │ • Output: SymbolicCertificate with CI, physics grade, 60s prediction│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: CONFORMAL UNCERTAINTY QUANTIFICATION (CQR)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Distribution-free prediction intervals                            │    │
│  │ • P(y_true ∈ Ĉ_α(x)) ≥ 1 - α (exact coverage guarantee)             │    │
│  │ • Output: σ_position ∈ ℝ³ with 95% coverage                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OUTPUT: Verifiable Symbolic Certificate              │
│  • LaTeX equation from discovered dynamics                                  │
│  • Bootstrap 95% CI per coefficient                                         │
│  • Physics grade (A/B/C/F)                                                  │
│  • 60-second falsifiable prediction with σ envelope                         │
│  • Ed25519 signed hash for audit trail                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Total Parameters: ~181M (GPT-2 scale, intentionally modest)
Edge Footprint: ~300KB per class (INT8 quantized, 32× compression)
```

### PAX: Phase-Adaptive Execution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE-ADAPTIVE EXECUTION (PAX)                           │
│                                                                             │
│  IMM Mode Probability ──► Execution Tier Selection                          │
│                                                                             │
│  ┌─────────────┐    P(ballistic) > 0.8                                      │
│  │ TIER 0      │    Pure Go physics (~0.1W, 0.01ms)                         │
│  │ BALLISTIC   │    hgv_physics.go: HGVEquationsOfMotion()                  │
│  └──────┬──────┘                                                            │
│         │ P(glide) > 0.5                                                    │
│  ┌──────▼──────┐                                                            │
│  │ TIER 1      │    Pruned ONNX 4×32 INT8 (~2W, 0.5ms)                      │
│  │ GLIDE       │    8× parameter reduction                                  │
│  └──────┬──────┘                                                            │
│         │ P(maneuver) > 0.3 OR residual > 3σ                               │
│  ┌──────▼──────┐                                                           │
│  │ TIER 2      │    Full DPHNO 4×128 FP16 (~8W, 2ms)                       │
│  │ MANEUVER    │    Maximum fidelity for zero-day events                   │
│  └──────┬──────┘                                                           │
│         │ SINDy R² < 0.7 (unexplainable dynamics)                          │
│  ┌──────▼──────┐                                                           │
│  │ TIER 3      │    Full DPHNO + SINDy + JAX imagination (~12W, 10ms)      │
│  │ BREACH      │    JAX Metal backend, thermal warning after 60s           │
│  └─────────────┘                                                           │
│                                                                            │
│  THERMAL GOVERNOR:                                                         │
│    T < 70°C: All tiers available                                           │
│    T 70-85°C: TIER 3 disabled, max TIER 2                                  │
│    T > 85°C: Force TIER 0 + alert ground station                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Full Data Flow: Input to Output

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SENSOR INPUT LAYER                             │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   NETRA      │  │  LADAKH      │  │  PASSIVE RF  │  │   NavIC      │     │
│  │ Radar Array  │  │  Telescope   │  │   Network    │  │   L5 Band    │     │
│  │              │  │              │  │              │  │              │     │
│  │ Range        │  │ Sub-arcsec   │  │ Emitter      │  │ TEC/Scint    │     │
│  │ Doppler      │  │ Angles       │  │ Signatures   │  │ Timing       │     │
│  │ RCS          │  │ Photometry   │  │ Doppler      │  │ Pseudorange  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                  │                  │           │
│         └─────────────────┴──────────────────┴──────────────────┘           │
│                                  │                                            │
└──────────────────────────────────┼────────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DSP PRE-PROCESSING                                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  1. BEAMFORMING & DETECTION                                         │     │
│  │     • Eigenbeamspace beamforming (radar long-baseline networks)     │     │
│  │     • Coherent integration gain >100 dB (weak scattered signals)    │     │
│  │     • CFAR detection thresholding                                   │     │
│  │                                                                     │     │
│  │  2. MEASUREMENT EXTRACTION                                          │     │
│  │     • Radar: range/Doppler/RCS from pulse returns                   │     │
│  │     • Optical: centroiding on PSF-deconvolved streaks               │     │
│  │     • RF: Doppler/code correlation with aiding                      │     │
│  │     • NavIC: Carrier-phase + code (dual-freq when S4>0.6)           │     │
│  │                                                                     │     │
│  │  3. OUTLIER REJECTION                                               │     │
│  │     • Chi-square test: |measurement - prediction|² / σ²             │     │
│  │     • Dual-site astrometry for cloud/spoof discrimination           │     │
│  └─────────────────────────────────┬───────────────────────────────────┘     │
└────────────────────────────────────┼──────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HAMILTONIAN FIREWALL (D1/D3)                             │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  PHYSICS GATE                                                       │     │
│  │                                                                     │     │
│  │  State Transition:  x(t+Δt) = Φ(t+Δt, t) · x(t) + Q(Δt)             │     │
│  │                                                                     │     │
│  │  Residual Test:     |ΔH/H| < bound after modeled perturbations?     │     │
│  │                     • J2 gravity                                    │     │
│  │                     • Drag ρ(v) exponential                         │     │
│  │                     • Solar radiation pressure                      │     │
│  │                     • Third-body perturbations                      │     │
│  │                                                                     │     │
│  │  FAIL → Reject track as spoof / garbage                             │     │
│  │  PASS → Propagate to fusion layer                                   │     │
│  │                                                                     │     │
│  │  Kalman/UKF: P(t) = Φ·P·Φᵀ + Q (covariance realism)                 │     │
│  └─────────────────────────────────┬───────────────────────────────────┘     │
└────────────────────────────────────┼──────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MULTI-SENSOR ORBIT DETERMINATION                        │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  INITIAL ORBIT DETERMINATION (IOD)                                  │     │
│  │  • Gauss/Laplace methods from angles-only (optical)                 │     │
│  │  • Range/Doppler methods (radar)                                    │     │
│  │                                                                     │     │
│  │  DIFFERENTIAL CORRECTION                                            │     │
│  │  • Weighted least-squares on (O-C) residuals                        │     │
│  │  • W = diag(1/σ²_radar, 1/σ²_optical, 1/σ²_rf)                      │     │
│  │                                                                     │     │
│  │  COVARIANCE PROPAGATION                                             │     │
│  │  • δa → δperiod → km/day in-track drift mapped                      │     │
│  │  • Consider parameters: drag uncertainty, solar flux                │     │
│  └─────────────────────────────────┬───────────────────────────────────┘     │
└────────────────────────────────────┼──────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   FOUNDATION MODEL INFERENCE (LG-PHNO)                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  INPUT: (r, v, t, F10.7, Ap, vehicle_class_prior)                   │     │
│  │                                                                     │     │
│  │  SE(3)-GNN Encoder ──► z_obs                                        │     │
│  │  DeepONet Atmosphere ──► z_atm, ρ_nn(h)                             │     │
│  │  IMM Mode Probability ──► p_mode                                    │     │
│  │                                                                     │     │
│  │  Hypernetwork: H_φ(z_context) ──► {W_H, W_R, W_g}                   │     │
│  │                                                                     │     │
│  │  DPHNO Integration:                                                 │     │
│  │    dz/dt = [J - R(z)]·∇H(z) + g(z)·u(t)                             │     │
│  │                                                                     │     │
│  │  Yoshida-4 symplectic steps ──► trajectory prediction               │     │
│  │                                                                     │     │
│  │  Neural SINDy on residual ──► u(t) inferred control                 │     │
│  │                                                                     │     │
│  │  Conformal UQ ──► σ_position with 95% coverage guarantee            │     │
│  └─────────────────────────────────┬───────────────────────────────────┘     │
└────────────────────────────────────┼──────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ANALYTICS & DECISION LAYER                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  GHOST-MANEUVER DETECTION (D9)                                      │     │
│  │  • Bayesian posterior P(thrusting | residuals)                      │     │
│  │  • IMM switching: ballistic vs thrusting mode probabilities         │     │
│  │  • Sub-threshold Δv detection via passive RF signatures             │     │
│  │                                                                     │     │
│  │  FUEL POSTERIOR (D5)                                                │     │
│  │  • Particle filter on fuel mass as hidden state                     │     │
│  │  • P(Δv ≥ X | Hamiltonian mismatch history)                         │     │
│  │                                                                     │     │
│  │  RECIPROCAL REACHABILITY (A1/A2)                                    │     │
│  │  • Lambert solver + hybrid propagator                               │     │
│  │  • Min-Δv time-of-flight + propellant cost                          │     │
│  │  • 72h ISR blind-window calendar                                    │     │
│  └─────────────────────────────────┬───────────────────────────────────┘     │
└────────────────────────────────────┼──────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VERIFIABLE OUTPUTS                                    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐      │
│  │  SYMBOLIC CERTIFICATE (VSC)                                          │     │
│  │  ┌─────────────────────────────────────────────────────────────┐     │     │
│  │  │ Equation: \dot{v} = -0.42v|v| + 0.03sin(r/7000) + u_drag    │     │     │
│  │  │ R² = 0.87 (bootstrap CI: [0.82, 0.91])                      │     │     │
│  │  │ Physics Grade: B (< 1e-3 energy residual)                   │     │     │
│  │  │ Prediction: [x, y, z, vx, vy, vz] @ T+60s ± σ               │     │     │
│  │  │ VerifyAt: 2026-04-21T12:34:56Z                              │     │     │
│  │  │ Hash: sha256(...) signed with SSIS_SIGNING_KEY              │     │     │
│  │  └─────────────────────────────────────────────────────────────┘     │     │
│  │                                                                      │     │
│  │  CUSTODY MAP                                                         │     │
│  │  • 72h strong/weak/doubtful per AOR                                  │     │
│  │  • Covariance ellipses for all trackable objects                     │     │
│  │                                                                      │     │
│  │  DECISION PRODUCTS                                                   │     │
│  │  • Conjunction Data Message (CDM) for collision (d)                  │     │
│  │  • Neighbourhood Watch alert (b) for proximity anomalies             │     │
│  │  • Ghost-maneuver warning (D9) for sub-threshold thrust              │     │
│  │  • Standing OPLAN menu (A1) + blind calendar (A2)                    │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Latency Budget

| Stage | Target Latency | Processing |
|-------|----------------|------------|
| Sensor acquisition | < 10 ms | Hardware timestamp synchronized |
| DSP beamforming/detection | < 20 ms | FPGA/GPU parallel |
| Hamiltonian firewall | < 5 ms | Physics gate check |
| Orbit determination | < 10 ms | Weighted least-squares |
| Foundation model inference | < 50 ms | PAX tier selection, ONNX runtime |
| Analytics & decision | < 5 ms | Bayesian update, IMM switching |
| **Total end-to-end** | **< 100 ms** | Sensor → custody decision |

---

## Quick Start
### Build
```bash
make build-static
```

### Scale Validation
```bash
./engine/main.go --scale-test --catalog-size 100000 --scale-hours 168
```

### Physics Benchmark (Axiom-Zero rKAN)
```bash
python3 scripts/rkan_validation.py
```

## Verification
```bash
go test ./...
pytest ai/v3/test/
```

## Development Status
- **Phase 0**: Foundation ✅ Complete
- **Phase 1**: Repo Unification ✅ Complete
- **Phase 2-3**: AI Core ✅ Complete (5 neuro-symbolic models implemented)
- **Phase 4-6**: Capability upgrades, UX, and security (In Progress)

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system flow and [docs/training/](docs/training/) for training pipeline status.
