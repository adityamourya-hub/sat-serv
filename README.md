# idex — Space Domain Awareness Platform

Go-native orbital catalog and HGV tracking with Python/JAX ML research substrate.

> **Implementation Status**
> | Component | Status | Notes |
> |-----------|--------|-------|
> | SGP4 / Orbital Propagation | Working | Go-native, tested |
> | SINDy Maneuver Detection | Working | Bootstrap CI implemented; R² computed from residuals (SS_res/SS_tot) |
> | HGV IMM Filter | Partial | 3-mode filter active; Kalman gain configurable (MinGain=0.01, MaxGain=0.95) |
> | ONNX Runtime Bridge | Stub | 12 `*_stub.go` files — no actual ONNX inference bound |
> | DPHNO (Port-Hamiltonian NN) | Research Claim | Architecture described, not trained or integrated |
> | Hypernetwork Atlas | Research Claim | Not trained |
> | Axiom-Zero Physics Core | Research Claim | No working JAX↔Go bridge |
> | Sovereign Handshake / Quantum-immune | Not Implemented | Marketing concept, no code exists |

## What Works

- **Catalog management** (`pkg/catalog/`): High-churn RSO catalog with WAL replay.
- **Orbital propagation** (`pkg/orbital/`): SGP4 fast-screen + Akhenakh numerical truth.
- **Conjunction screening** (`pkg/collision/`): Foster-Alfano/Monte-Carlo surrogate.
- **HGV tracking** (`pkg/hypersonic/`): 3-mode IMM filter with static transition matrix.
- **SINDy analysis** (`pkg/physics/sindy.go`): Fixed 21-term library, real R² from residuals.
- **Atmosphere** (`pkg/physics/atmosphere.go`): Jacchia-Roberts >100 km; hardcoded F10.7=150, Ap=15.
- **gRPC server** (`pkg/server/`): mTLS 1.3 + Ed25519 signing stubs.

## What Is Stubbed / Research

- **ONNX bridge** (`pkg/ml/onnx/*_stub.go`): Build stubs return `fmt.Errorf("not yet implemented")`. No CGO-bound runtime exists.
- **DPHNO** (`ai/v3/port_hamiltonian.py`, `ai/v3/dphno.py`): Correct JAX port-Hamiltonian implementation exists but is **not wired** to the PyTorch training pipeline (`ai/neural_ode/train.py`), whose "symplectic" integrator is actually explicit RK-like (not symplectic).
- **Hypernetwork Atlas**: Not trained.
- **Axiom-Zero Bridge**: No working JAX→Go weight transfer.
- **Ghost-maneuver / fuel-posterior / reciprocal-reachability**: Architecture docs only; no runtime.

## Build

```bash
# Go
make build-static

# Python (JAX research only — not wired to production)
pip install -r ai/v3/requirements.txt
```

## Test

```bash
go test ./pkg/orbital ./pkg/collision ./pkg/catalog ./pkg/physics
# Many tests in pkg/ml/onnx/ and pkg/integration/ fail because they exercise stubs.
```

## Repository Layout

- `pkg/` — Go operational code (what compiles).
- `ai/v3/` — Python/JAX research substrate (correct PHS, broken trainer bridge).
- `cmd/` — CLI utilities.
- `docs/` — Architecture proposals and forensic analysis.

## Known Issues

1. **Python AI/ML files** (71 files in `q/core/` and `ai/v3/`): Flat indentation — method/class bodies at same indentation as definitions. These files need complete rewrites, not indentation fixes.
2. `astrojax` dependency was hallucinated; module renamed to `orbital_world_model.py`. Stale `.pyc` cache and Go comment references remain.
3. `tier1_pruned.onnx` / `tier2_full.onnx` were **synthetic dummy files** (random bytes with ONNX header) — deleted.

## Recent Fixes (2026-04-27)

- `randFloat()` in `pkg/physics/integrator.go`: Replaced constant LCG (always returned ~0.2361) with `rand.Float64()`.
- `t.Skip` → `t.Fatal`: Converted unimplemented tests to surface gaps in CI (preserved `testing.Short()` performance guards).
- `TestSymplecticReversibility`: Fixed test bug — compared against `-p0` instead of `+p0` after time-reversal cycle.
- `TestForestRuthSymplecticError`: Relaxed threshold from 1e-10 to 1e-4 (finite-difference Jacobian numerical floor).
- `TestSymplecticManifoldNormalization`: Relaxed tolerance from 1e-10 to 1e-9 for 1000-step accumulation.
- `SlidingSINDy.UpdateCount`: Moved to top of method to count every call (not just stride-filtered calls), fixing thread safety test.

See `docs/AI_GENERATION_FORENSIC_ANALYSIS.md` for full forensic report.

**Confidentiality Notice**: This software contains Defense IP. Unauthorized distribution is prohibited.
