# Full ML Check + End-to-End PINN Integration for `idex`

## Context

`idex` is a Space Domain Awareness platform: Go-native orbital catalog/conjunction/HGV stack (`pkg/`) plus a 71-file Python physics-ML research substrate (`ai/v3/`). Per [README.md](README.md), three real gaps exist:

1. **ONNX bridge is stubbed**: 12 `pkg/ml/onnx/*_stub.go` files return `"not yet implemented"`. No CGO-bound runtime exists. The previous `tier1_pruned.onnx` / `tier2_full.onnx` were synthetic dummy bytes and were deleted.
2. **Trainer / model bridge is broken**: correct JAX port-Hamiltonian code lives in [ai/v3/port_hamiltonian.py](ai/v3/port_hamiltonian.py), but the legacy PyTorch trainer in `ai/neural_ode/train.py` uses an explicit-RK integrator that is not actually symplectic. Mature v3 modules (FiLM, manifold projection, SE(3) MBPO, SPHS diffusion, trifurcated MLP, Hamiltonian curriculum) are not wired to the production Go path.
3. **Code-quality debt in `ai/v3/`**: README states the 71 Python files have flat indentation (method bodies at class-definition column) and "need complete rewrites, not indentation fixes."

The user asked for a *full check* of the ML stack with the framework choice driven by what `idex` already has, plus the four deliverables from their original prompt (architecture overview, production code, data pipeline, evaluation metrics). This plan does both: a structured audit of every layer, then a concrete end-to-end PINN trajectory-prediction slice that wires one ONNX stub from JAX training → ONNX export → Go inference and produces all four deliverables against it.

---

## Recommended Stack (what fits `idex`)

| Layer | Choice | Why |
|---|---|---|
| Research / training | **JAX + Equinox** | Matches existing v3 mature modules: [film_hamiltonian.py](ai/v3/film_hamiltonian.py), [manifold_projection.py](ai/v3/manifold_projection.py), [mbpo_se3_agent.py](ai/v3/mbpo_se3_agent.py), [hamiltonian_pretraining.py](ai/v3/hamiltonian_pretraining.py). Forest-Ruth symplectic integrator and FiLM-PHS are already JAX. |
| ONNX-export-critical components | **PyTorch** where present | [sphs_diffusion.py](ai/v3/sphs_diffusion.py) and [trifurcated_mlp.py](ai/v3/trifurcated_mlp.py) are already PyTorch and already include precision-aware heads (FP32 H-net / BF16 R-net / INT8 g-net). Do not rewrite these. |
| Serving boundary | **ONNX via `onnx` build tag** | `pkg/ml/onnx/` already defines `Inferrer` interface and stub. JAX path: `jax2tf` → `tf2onnx`. PyTorch path: `torch.onnx.export`. |
| Physics libraries | **Use what's in-tree** | SGP4 truth is already in [pkg/orbital/](pkg/orbital/). Atmosphere is [pkg/physics/atmosphere.go](pkg/physics/atmosphere.go) (Jacchia-Roberts). Do **not** add `poliastro`/`skyfield` — they would duplicate existing Go ground-truth. Only `sgp4` PyPI may be used for synthetic-dataset generation. NRLMSISE wrapper already exists at [ai/v3/nrlmsise_wrapper.py](ai/v3/nrlmsise_wrapper.py). |
| Training data | Synthetic + TLE | Reuse `pkg/ml/dphno/synthetic_generator.py` and the unified [ai/v3/dataloader.py](ai/v3/dataloader.py) (40/30/30 HGV/Orbital/Ballistic mix). |
| Performance | **Edge-target M2 Air, <10μs/step** | Per [ai/v3/film_hamiltonian.py:17](ai/v3/film_hamiltonian.py#L17) the deployment envelope is already set; vectorize via `jax.vmap` + `jit`; quantize per `trifurcated_mlp` precision plan. |

---

## Phase 0 — Static Audit (read-only)

Goal: produce `/Users/adi/idex/docs/ML_AUDIT_REPORT.md` mapping every Python module to {compiles, imports, tests pass, physics-claim verified, ONNX stub mapped}.

Steps:
1. Run `python -m py_compile` over every file in `ai/v3/` to enumerate indentation/syntax breakage README warns about. Capture a clean list of files that need rewriting vs. files that pass.
2. Run `ruff check ai/v3/ --select=E,F,W` for lint summary.
3. Run `pytest ai/v3/ -x --collect-only` to find all collectable tests; then `pytest ai/v3/ -k "not slow"` for the actual pass/fail matrix.
4. Cross-reference each `pkg/ml/onnx/*_stub.go` to its intended Python source. Mapping table goes into the audit report. Likely candidates:
   - `maneuver_model_stub.go` ↔ [ai/v3/film_hamiltonian.py](ai/v3/film_hamiltonian.py) + [ai/v3/port_hamiltonian.py](ai/v3/port_hamiltonian.py)
   - `pol_transformer_stub.go` ↔ pattern-of-life transformer (no clear Python source — flag as stub-only)
   - `rkan_covariance_stub.go` ↔ [ai/v3/sphs_diffusion.py](ai/v3/sphs_diffusion.py) (state-dependent Σ(x))
   - `delta_rkan_attributor_stub.go` ↔ [ai/v3/maya_detector.py](ai/v3/maya_detector.py) or similar
   - `track_association_stub.go` ↔ pkg/ml/lightgbm path (not Python v3)
   - `blindspot_forecaster_stub.go`, `collision_model_stub.go`, `rapid_ranker_stub.go`, `axiom_zero_integration_stub.go`, `runtime_stub.go` — categorize per the audit
5. Categorize each module: `mature` / `stub-only` / `needs-rewrite` / `delete-me`. The git-status-deleted files (`astrojax_world_model.py`, `requirements.txt` duplicates, the `pkg/ml/onnx/*_onnx.go` files) need a `git rm` decision recorded.

Out of this phase: the audit report is the user-facing checklist of what's real, what's fake, what's next.

---

## Phase 1 — Physics Correctness Verification (read-only + small test additions)

Goal: numerically confirm the four headline physics claims in the v3 stack, so we can trust them as the substrate for Phase 2.

Tests to add or run (in `ai/v3/tests/`):

1. **Energy conservation** — drive [ai/v3/manifold_projection.py](ai/v3/manifold_projection.py) with a perturbed Keplerian state, run 10⁴ Forest-Ruth steps, assert `|H_t - H_0| < 1e-8` (the budget claimed in `train_phs_jax.py:18`). If the existing test at [ai/v3/manifold_projection.py:329](ai/v3/manifold_projection.py#L329) only covers a single projection, extend it to the long-horizon case.
2. **Symplectic preservation** — compute Jacobian of the Forest-Ruth flow over one step, assert `‖JᵀΩJ - Ω‖_F < 1e-4` (matches the relaxed tolerance from the recent commit `7dbc360`). Reuse coefficients from `train_phs_jax.py:46-52`.
3. **SE(3) equivariance** — apply random `SO(3)` rotation + translation to the orbital state, run [ai/v3/mbpo_se3_agent.py](ai/v3/mbpo_se3_agent.py)'s encoder, assert scalar invariants differ by <1e-5 (existing claim at [ai/v3/mbpo_se3_agent.py:470](ai/v3/mbpo_se3_agent.py#L470)).
4. **Stratonovich Milstein correction** — compare [ai/v3/sphs_diffusion.py](ai/v3/sphs_diffusion.py) Milstein step against an Itô Euler-Maruyama baseline on a known SDE; assert weak-order error is O(dt) not O(√dt).

Failures here are blockers for Phase 2 and get logged into the audit report.

---

## Phase 2 — End-to-End PINN: Tumbling-Satellite Trajectory from Sparse Optical Tracks

This phase delivers everything in the user's original "deliverables" list, scoped to one concrete model wired all the way to Go inference. It exercises the JAX → ONNX → Go path end-to-end, killing one stub and proving the bridge works.

### 2.1 Mathematical & Architecture Overview

Equations modeled (all already partially present in the codebase — reuse, do not reinvent):

```
Hamiltonian:    H(q,p) = ‖p‖²/2 - μ/‖q‖ + H_J2(q) + H_drag(q,p)
J2 perturbation: H_J2 = (μJ₂R_E²)/(2‖q‖³) · (3sin²φ - 1)         # geopotential
Atmospheric:    dp/dt += -½ ρ(h) C_D A/m ‖v_rel‖ v_rel           # NRLMSISE-00
Tumbling:       attitude on SO(3), angular velocity ω, body-fixed inertia I
Observation:    z_t = h(q_t) + n_t,  n_t ~ N(0, R(elevation, cloud_mask))
```

Loss decomposition:

```
L = λ_data · MSE(q̂, q_obs)                  # likelihood on observed optical fixes
  + λ_H    · ‖dH/dt‖²                        # Hamiltonian conservation residual
  + λ_sym  · ‖J^T Ω J - Ω‖_F                 # symplectic penalty (soft, in addition to projection)
  + λ_man  · ‖H(q̂,p̂) - H_target‖²           # manifold-projection residual
  + λ_J2   · J2-perturbation consistency
  + λ_lat  · contrastive on FiLM latent z (orbital/ballistic/HGV separation)
```

Architecture (call-graph, all modules already in v3):

```
[sparse optical fixes, sensor metadata]
        │
        ▼
   regime_classifier ──► z ∈ R³² (FiLM latent)
        │                       │
        ▼                       ▼
 trifurcated_mlp(state) ──► (H, R, g) heads
        │                       │
        ▼                       ▼
  film_hamiltonian.H(x,z) ──► ∇_x H
        │
        ▼
  Forest-Ruth symplectic integrator (4th order, Yoshida 1990)
        │
        ▼
  manifold_projection.project()  ── hard energy constraint
        │
        ▼
  predicted trajectory over 24h horizon
```

### 2.2 Production Code

Modules to add (small, focused, each <300 LOC):

- [ai/v3/pinn_tumbling_predictor.py](ai/v3/pinn_tumbling_predictor.py) **(new)** — top-level PINN that composes `trifurcated_mlp` + `film_hamiltonian` + Forest-Ruth + `manifold_projection`. PyTorch front-end (so ONNX export is `torch.onnx.export`-clean) calling the JAX physics layers via `jax2torch` bridge — OR a pure-JAX implementation with `jax2tf` → `tf2onnx` export. Default to **pure JAX** because every physics primitive is already JAX; only the trifurcated MLP heads need a wrapper.
- [ai/v3/pinn_loss.py](ai/v3/pinn_loss.py) **(new)** — implements the loss decomposition above as `pure_callable(params, batch) -> dict` so each term can be logged separately to W&B.
- [ai/v3/data/optical_tracks_synth.py](ai/v3/data/optical_tracks_synth.py) **(new)** — synthetic optical-track generator (see 2.3).
- [ai/v3/eval/physics_metrics.py](ai/v3/eval/physics_metrics.py) **(new)** — domain metrics (see 2.4).
- [ai/v3/export/pinn_to_onnx.py](ai/v3/export/pinn_to_onnx.py) **(new)** — JAX → ONNX export pipeline using `jax2tf` + `tf2onnx`. Include opset-checking and shape-pinning.
- [pkg/ml/onnx/maneuver_model.go](pkg/ml/onnx/maneuver_model.go) **(new, build-tagged `onnx`)** — companion to the existing `maneuver_model_stub.go`. Uses `onnxruntime-go` (or whatever dependency is in `go.mod`; **verify before committing**) to load the exported model and implement the `Inferrer` interface declared alongside [pkg/ml/onnx/runtime_stub.go](pkg/ml/onnx/runtime_stub.go).
- [pkg/ml/onnx/maneuver_model_test.go](pkg/ml/onnx/maneuver_model_test.go) **(new)** — round-trip test: known JAX prediction == Go ONNX prediction within 1e-4.

PEP-8, type hints, and module docstrings throughout. **No** comments restating what code does — only why-comments where the physics is non-obvious (e.g., the Stratonovich correction term in the SDE step). Do **not** patch the 71-file indentation issue here — that is its own Phase 4 effort.

### 2.3 Data Pipeline

Synthetic generator at [ai/v3/data/optical_tracks_synth.py](ai/v3/data/optical_tracks_synth.py) produces a mock dataset of (orbital_state_history, sparse_optical_observations) tuples:

1. **Orbit truth** — sample TLEs from public NORAD set or synthesize `(a, e, i, Ω, ω, M₀)` over realistic LEO ranges. Propagate with SGP4 (`sgp4` PyPI package) for 24 h at 1 s.
2. **Tumbling truth** — random body-frame inertia tensor; integrate Euler's rigid-body equations on SO(3) using existing [ai/v3/lie_integrator.py](ai/v3/lie_integrator.py).
3. **Observation simulation** — pick a random ground-station lat/lon; convert ECI→AzEl; mark `visible_t` only when elevation > 10° AND sun-elevation < -6° (astronomical twilight). Apply Gaussian angular noise σ ~ 2 arcsec.
4. **Edge cases**:
   - **Cloud-cover dropout**: Bernoulli mask with p_clear depending on synthetic Markov-chain weather state.
   - **Sensor dropout**: random gaps with rate 5 %/min.
   - **Telemetry latency**: append per-fix latency draw from log-normal(μ=200ms, σ=400ms); the model receives `(fix, latency)` so the loss can down-weight stale data.
5. **Regime balance** — 40/30/30 HGV/Orbital/Ballistic via [ai/v3/dataloader.py](ai/v3/dataloader.py)'s sampler.

Output: `tfrecord` or `webdataset` shards; loader uses `jax.tree_util` for batching.

### 2.4 Evaluation Metrics

Implemented in [ai/v3/eval/physics_metrics.py](ai/v3/eval/physics_metrics.py):

ML metrics (table-stakes):
- MSE on `(q, p)` per-step.
- F1 on regime classifier (orbital / ballistic / HGV).
- NLL under predictive Σ(x) head (`sphs_diffusion`).

Domain physics metrics (the ones the user explicitly asked for):
- **Ephemeris error in km** — RIC-frame (radial / in-track / cross-track) RMSE at T+1h, T+6h, T+24h. Standard SDA reporting.
- **Energy drift** — `|H(t) − H(0)| / |H(0)|` over 24 h. Target <1e-6.
- **Spatial resolution limit** — minimum ground separation at which two close-conjunction objects can be distinguished given σ.
- **Probability-of-collision (Pc)** comparison — feed predicted covariance to `pkg/collision/surrogate.go` Foster-Alfano and compare Pc estimates against SGP4-truth Pc on a held-out set of conjunction events.
- **SE(3) equivariance under random rotation** (regression test from Phase 1).
- **Onboard latency** — wall-clock per single forward pass on M2 Air; assert <10 μs claim.

### 2.5 ONNX Round-Trip & Go Wire-Up

1. Export trained JAX model via `jax2tf` (with `enable_xla=False` for portability) → `tf2onnx` → fixed-shape ONNX (opset 17).
2. Validate with `onnxruntime` Python that JAX vs ONNX output matches to 1e-5.
3. Implement `pkg/ml/onnx/maneuver_model.go` behind the `onnx` build tag, mirroring the stub's signature.
4. Run `go test -tags=onnx ./pkg/ml/onnx/...` round-trip test asserting Go inference matches the captured Python reference within 1e-4.

This kills one stub. The same pattern then applies to `rkan_covariance` (sphs_diffusion) and `delta_rkan_attributor` in follow-up work.

---

## Phase 3 — Architecture Overview Document

Single document at [docs/ML_PINN_ARCHITECTURE.md](docs/ML_PINN_ARCHITECTURE.md) (new) consolidating:
- Equations, loss decomposition, architecture diagram (above).
- Performance envelope per module: parameter count, FLOPs, INT8/FP32 split (numbers already exist in `trifurcated_mlp.py`).
- Deployment topology: H100 train → ONNX → Go runtime → M2 Air edge.
- Failure modes and graceful-degradation policy when sensor inputs are missing (cloud / dropout / latency).

---

## Phase 4 — Cleanup (must NOT block Phases 0–3)

Strictly opt-in, do this only for files we *touch*. Do not perform a 71-file rewrite in this plan.

- For every v3 file we modify: bring it to PEP-8, add type hints, fix the indentation issue.
- Resolve currently-deleted files in `git status` (decide: restore, delete, or move).
- Remove stale `astrojax` references in Go comments and `.pyc` cache.
- Note: `--no-verify` etc. are forbidden; if pre-commit hooks fail, fix root cause.

---

## Critical Files (paths — read these before editing)

Existing, must read:
- [ai/v3/film_hamiltonian.py](ai/v3/film_hamiltonian.py)
- [ai/v3/manifold_projection.py](ai/v3/manifold_projection.py)
- [ai/v3/port_hamiltonian.py](ai/v3/port_hamiltonian.py)
- [ai/v3/sphs_diffusion.py](ai/v3/sphs_diffusion.py)
- [ai/v3/trifurcated_mlp.py](ai/v3/trifurcated_mlp.py)
- [ai/v3/hamiltonian_pretraining.py](ai/v3/hamiltonian_pretraining.py)
- [ai/v3/dataloader.py](ai/v3/dataloader.py)
- [ai/v3/train_phs_jax.py](ai/v3/train_phs_jax.py)
- [ai/v3/lie_integrator.py](ai/v3/lie_integrator.py)
- [ai/v3/nrlmsise_wrapper.py](ai/v3/nrlmsise_wrapper.py)
- [pkg/ml/onnx/runtime_stub.go](pkg/ml/onnx/runtime_stub.go)
- [pkg/ml/onnx/maneuver_model_stub.go](pkg/ml/onnx/maneuver_model_stub.go)
- [pkg/ml/dphno/synthetic_generator.py](pkg/ml/dphno/synthetic_generator.py)
- [pkg/orbital/](pkg/orbital/) (SGP4 truth)
- [pkg/collision/surrogate.go](pkg/collision/surrogate.go) (Pc evaluator we reuse)

To add:
- [ai/v3/pinn_tumbling_predictor.py](ai/v3/pinn_tumbling_predictor.py)
- [ai/v3/pinn_loss.py](ai/v3/pinn_loss.py)
- [ai/v3/data/optical_tracks_synth.py](ai/v3/data/optical_tracks_synth.py)
- [ai/v3/eval/physics_metrics.py](ai/v3/eval/physics_metrics.py)
- [ai/v3/export/pinn_to_onnx.py](ai/v3/export/pinn_to_onnx.py)
- [pkg/ml/onnx/maneuver_model.go](pkg/ml/onnx/maneuver_model.go)
- [pkg/ml/onnx/maneuver_model_test.go](pkg/ml/onnx/maneuver_model_test.go)
- [docs/ML_AUDIT_REPORT.md](docs/ML_AUDIT_REPORT.md)
- [docs/ML_PINN_ARCHITECTURE.md](docs/ML_PINN_ARCHITECTURE.md)

---

## Verification — End-to-End Test Recipe

1. `pytest ai/v3/ -k "not slow"` — green; physics regression tests from Phase 1 pass.
2. `python -m ai.v3.pinn_tumbling_predictor --train --epochs=10 --synthetic` — short-horizon train smoke run; final loss decomposition logged to W&B; energy drift <1e-6 over a 1h held-out propagation.
3. `python -m ai.v3.eval.physics_metrics --checkpoint=...` — ephemeris RMSE in RIC frame at T+1h, T+6h, T+24h emitted; Pc cross-check against `pkg/collision/surrogate.go` truth.
4. `python -m ai.v3.export.pinn_to_onnx --checkpoint=... --out=models/maneuver_model.onnx` — clean export at opset 17; Python `onnxruntime` cross-check ‖Δ‖∞ < 1e-5.
5. `go test -tags=onnx ./pkg/ml/onnx/... -run TestManeuverModelRoundTrip` — Go inference matches captured Python reference within 1e-4.
6. `go test ./pkg/orbital ./pkg/collision ./pkg/catalog ./pkg/physics` — pre-existing green tests still green (no regressions).
7. M2-Air latency benchmark: `python -m ai.v3.eval.latency --hw=m2air` — assert <10 μs/forward-pass on encoder + Hamiltonian gradient.

---

## Out of Scope (explicitly)

- Hypernetwork Atlas — separate research effort, not needed for this slice.
- "Sovereign Handshake" / "Quantum-immune" — README marks these as marketing concepts with no code.
- SAR / multispectral CV models — different deliverable, doesn't reuse v3 PHS stack.
- Wholesale rewrite of all 71 v3 files for indentation — bounded to files we touch.
- Adding `poliastro` / `skyfield` / `Orekit` — duplicate of existing Go SGP4 + Jacchia-Roberts.
