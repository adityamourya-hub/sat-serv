# SSIS Fallback Decision Logic

This document specifies the autonomous fallback protocols for the Sovereign Space Intelligence Substrate (SSIS). The substrate is designed to maintain operational state even when high-fidelity ML models or complex propagation kernels fail, ensuring "Always-On" situational awareness.

## 1. Probability of Collision (Pc): Foster vs. Alfano

The system dynamically selects between the Foster (1992) adaptive quadrature and the Alfano (2005) single-integral series based on the geometric encounter parameters.

| Input Condition | Logic Result | Method Selected | Bias/Reasoning |
| :--- | :--- | :--- | :--- |
| Aspect Ratio $\le 500$ AND $v_{rel} \le 12$ km/s | False | Foster (1992) | High-precision Adaptive Gauss-Legendre quadrature (32-256 pts). |
| Aspect Ratio $> 500$ | True | Alfano (2005) | Singular/Skinny covariance requires series approximation for stability. |
| Relative Velocity $> 12$ km/s | True | Alfano (2005) | High-speed encounters favor single-integral series convergence. |
| Input Validation Fail ($R \le 0$, $\sigma \le 0$) | Fail | `invalid_input` | Domain violation prevention; returns 0 Pc with alert. |

**Source**: `pkg/collision/foster.go` -> `AdaptiveFosterPcWithMetadata`

---

## 2. ML Inference Engine: ONNX vs. Heuristic (Dummy)

The orchestration layer manages the lifecycle of ONNX Runtime sessions. If the model substrate is unavailable or corrupt, the system falls back to a deterministic heuristic engine.

| Condition | Fallback Action | Reasoning |
| :--- | :--- | :--- |
| Model files missing (`maneuver_detector.onnx`, etc.) | Switch to `DummyInferenceEngine` | Prevents startup failure in incomplete environments. |
| ONNX Runtime init failure (missing shared libs) | Switch to `DummyInferenceEngine` | Graceful degradation on unsupported OS/Hardware. |
| Inference Run Error (Runtime crash/NaN) | Heuristic Output | Preserves system availability over model purity. |
| `SSIS_STRICT_MODE=1` | Fail-Fast (No Fallback) | Mandatory for formal certification and high-trust audits. |

**Source**: `pkg/ml/inference.go` -> `NewInferenceEngineWithFallback`

---

## 3. Orbit Propagation: Differentiable vs. Classical

SSIS prioritizes differentiable physics for Orbit Determination (OD) while maintaining a high-fidelity numerical backup.

| State | Primary Mode | Fallback Mode |
| :--- | :--- | :--- |
| Model Present | `DifferentiablePropagator` (dSGP4) | Numerical RK4 (NRLMSISE-00) |
| Model Missing | Numerical RK4 | SGP4 (Analytical) |
| Real-time Stream | Hybrid PINN-Refiner | SGP4 Standard |

**Source**: `pkg/orbital/differentiable.go` & `pkg/orbital/fusion.go`

---

## 4. Truth Table Summary

| Feature | Primary Method | Fallback Condition | Fallback Method |
| :--- | :--- | :--- | :--- |
| **Pc Calc** | Foster (1992) | Aspect Ratio $> 500$ OR $v_{rel} > 12$ | Alfano (2005) |
| **Maneuver ID** | ONNX (Transformer) | Model/Runtime Unavailable | Heuristic Variance Analysis |
| **Threat Triage** | ONNX (RapidRanker) | Model/Runtime Unavailable | Ballistic Heuristics |
| **OD Correction** | Gauss-Newton (dSGP4) | Jacobian Missing | Standard SGP4 Iteration |

---

## 5. Axiom-Zero Physics Core Integration

The Sovereign Physics Core (Axiom-Zero) provides learned Hamiltonian dynamics via rKAN. Integration follows a strict fallback hierarchy.

### 5.1 Propagation Engine Selection

| Condition | Engine | Energy Drift | Notes |
| :--- | :--- | :--- | :--- |
| `x-ssis-physics-engine: AXIOM` + weights loaded | Axiom-Zero (Yoshida-8 + rKAN) | $< 10^{-14}$ | Sovereign symplectic propagation |
| `x-ssis-physics-engine: AXIOM` + weights missing | SGP4 Analytical | $\sim 10^{-6}$ | Graceful fallback with warning |
| `x-ssis-physics-engine: SGP4` (default) | SGP4 Analytical | $\sim 10^{-6}$ | Standard operational mode |
| Air-gap mode (no ONNX runtime) | SGP4 Analytical | $\sim 10^{-6}$ | Zero external dependencies |

**Source**: `pkg/server/grpc.go` → `ScanGroundRegion`, `pkg/orbital/axiom_propagator.go`

### 5.2 Covariance Inflation (Physics-Informed Uncertainty)

| Condition | Covariance Source | Realism Score |
| :--- | :--- | :--- |
| rKAN residual variance available | Learned perturbation structure | 0.8 |
| rKAN unavailable, static SNC | Process noise matrix (tuned) | 0.5 |
| Emergency fallback | Diagonal inflation (conservative) | 0.3 |

**Source**: `ai/active/v3/axiom_zero/sda/conjunction.py` → `PhysicsInformedConjunction`

### 5.3 Maneuver Attribution

| Condition | Attribution Method | Output |
| :--- | :--- | :--- |
| Delta-rKAN model loaded | Symbolic distillation | LaTeX formula + type |
| Delta-rKAN unavailable | CUSUM + RTN decomposition | Type + magnitude |
| All ML unavailable | Energy/momentum spike detection | Binary detection only |

**Source**: `ai/active/v3/axiom_zero/sda/maneuver_detect.py` → `ManeuverDetector`

---

## 6. Complete Decision Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SSIS FALLBACK DECISION TREE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [START] → Check Physics Engine Header                                      │
│     │                                                                       │
│     ├─► AXIOM requested?                                                    │
│     │      │                                                                │
│     │      ├─► YES → Axiom weights loaded?                                  │
│     │      │           │                                                    │
│     │      │           ├─► YES → Use Yoshida-8 + rKAN (Primary)            │
│     │      │           │                                                    │
│     │      │           └─► NO → Log warning, use SGP4 (Fallback)           │
│     │      │                                                                │
│     │      └─► NO → Use SGP4 Analytical (Default)                          │
│     │                                                                       │
│  [Pc CALCULATION]                                                           │
│     │                                                                       │
│     ├─► Aspect Ratio ≤ 500 AND v_rel ≤ 12 km/s?                            │
│     │      │                                                                │
│     │      ├─► YES → Foster 2D (Gauss-Legendre 32-256 pts)                 │
│     │      │                                                                │
│     │      └─► NO → Alfano Series (Single-integral)                        │
│     │                                                                       │
│     ├─► Pc < 1e-7 AND high-value asset?                                    │
│     │      │                                                                │
│     │      └─► YES → Monte Carlo 10⁶ samples (verification)                │
│     │                                                                       │
│  [MANEUVER DETECTION]                                                       │
│     │                                                                       │
│     ├─► ONNX Transformer available?                                        │
│     │      │                                                                │
│     │      ├─► YES → ML classification + CUSUM confirmation                │
│     │      │                                                                │
│     │      └─► NO → CUSUM on energy/momentum invariants                    │
│     │                                                                       │
│     ├─► Delta-rKAN available?                                              │
│     │      │                                                                │
│     │      ├─► YES → Symbolic attribution with LaTeX                       │
│     │      │                                                                │
│     │      └─► NO → RTN decomposition + heuristic classification           │
│     │                                                                       │
│  [SIGNING & AUDIT]                                                          │
│     │                                                                       │
│     └─► Ed25519 signer available?                                          │
│            │                                                                │
│            ├─► YES → Sign all responses                                    │
│            │                                                                │
│            └─► NO → FAIL (no unsigned responses allowed)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Strict Mode Behavior

When `SSIS_STRICT_MODE=1` is set:

| Component | Normal Mode | Strict Mode |
| :--- | :--- | :--- |
| Missing ONNX model | Fallback to heuristic | **FAIL** |
| Missing Axiom weights | Fallback to SGP4 | **FAIL** |
| Unsigned response | Never allowed | Never allowed |
| NaN/Inf in Pc | Return 0 with alert | **FAIL** |
| Cholesky failure | SafeCholesky regularization | **FAIL** |

Strict mode is mandatory for formal certification (DO-178C DAL-C, IEC 61508 SIL-2).
